import os, argparse
import re, json
from keras.models import load_model
from bilinear_sampler import *
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Perform test on adversarial monocular depth estimation model.

class Adversarial_Depth_Tests(object):

    def __init__(self):
        # Initialize model
        args = self.parse_args()
        self.setup_parameters(args)


    def parse_args(self):
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Perform tests on adversarial depth model.')
        parser.add_argument('--test',         action='store_true', help='Test model on input set.', default=False)
        parser.add_argument('--infer',        action='store_true', help='Infer depth (displayed as disparity) of single image.', default=False)
        parser.add_argument('--save',         action='store_true', help='Save test results.', default=False)
        parser.add_argument('--plot_results', action='store_true', help='Plot test results.', default=False)
        parser.add_argument('--img_dir',      type=str,            help='Directory containing testing data. Must be in left-right directory format.',  default=None)
        parser.add_argument('--img_file',     type=str,            help='Inference image file.', default=None)
        parser.add_argument('--save_output',  type=str,            help='Directory/File to save results.',  default=None)
        parser.add_argument('--sampling',     type=str,            help='Sampling method of test data [random/all].', default='random')
        parser.add_argument('--test_samples', type=int,            help='Number of test samples.', default=1)
        parser.add_argument('--model_file',   type=str,            help='Trained model file (.h5).', default='adversarial_depth_model.h5')
        args = parser.parse_args()
        return args


    def setup_parameters(self,args):
        # Usage flags
        self.test = args.test
        self.infer = args.infer
        self.save = args.save
        # Image directories/files
        self.img_dir = args.img_dir
        self.img_file = args.img_file
        self.save_output = args.save_output
        self.plot_results = args.plot_results
        # Model testing parameters
        self.sampling = args.sampling
        self.test_samples = args.test_samples
        self.model_file = args.model_file 
        # Load depth estimation model
        self.depth_model = load_model(self.model_file, 
                custom_objects={'bilinear_sampling_L': bilinear_sampling_L, 'bilinear_sampler_1d_h': bilinear_sampler_1d_h}, compile=False)
        if (self.depth_model is None):
            print '[!] Depth estimation model could not be loaded. Exiting.'
            exit()
        # Model image size
        self.img_rows = self.depth_model.get_layer(index=0).input_shape[1]
        self.img_cols = self.depth_model.get_layer(index=0).input_shape[2]
        self.input_channels = 3 # Arbitrarily defined as an RGB image

        # Setup weight matrices for depth inference
        def w1(x,cols):
            x = x/float(cols)
            if (x < 0.05):
                return 10*x
            elif (x < 0.95):
                return 0.5
            else:
                return 10*(x-1)+1
        def w2(x,cols):
            x = x/float(cols)
            if (x < 0.05):
                return -10*x+1
            elif (x < 0.95):
                return 0.5
            else:
                return -10*(x-1)
        # Create weight matrices
        self.weights1 = [w1(x,self.img_cols) for x in range(0,self.img_cols)]
        self.weights1 = np.tile(self.weights1, (self.img_rows,1))
        self.weights2 = [w2(x,self.img_cols) for x in range(0,self.img_cols)]
        self.weights2 = np.tile(self.weights2, (self.img_rows,1))


    def __get_model_output(self, img_L, img_R=None, flip=False, depth_est_params=None):
        # Convert input images to model format
        img_L = cv.cvtColor(img_L, cv.COLOR_BGR2RGB) # imread returns BGR image
        img_L = cv.resize(img_L, (self.img_cols, self.img_rows), interpolation=cv.INTER_LINEAR)
        img_L = img_L / 255.
        img_L = img_L.reshape(1,self.img_rows,self.img_cols,self.input_channels)
        if (img_R is None):
            img_R = np.zeros(img_L.shape)
        else:
            img_R = cv.cvtColor(img_R, cv.COLOR_BGR2RGB) # imread returns BGR image
            img_R = cv.resize(img_R, (self.img_cols, self.img_rows), interpolation=cv.INTER_LINEAR)
            img_R = img_R / 255.
            img_R = img_R.reshape(1,self.img_rows,self.img_cols,self.input_channels)

        # Predict model outputs
        test_input = [img_L, img_R]
        test_result = self.depth_model.predict(test_input)
        # Gather model outputs
        disparity_L = test_result[0][0].reshape(self.img_rows,self.img_cols)
        imageL_rec = test_result[1][0].reshape(self.img_rows,self.img_cols,3)

        # Combine with disparity of flipped input
        if (flip):
            img_L_flipped = np.flip(img_L, axis=2)
            # Predict flipped outputs
            test_input_flipped = [img_L_flipped, img_R] # Reconstruction does not concern us
            test_result_flipped = self.depth_model.predict(test_input_flipped)
            disparity_L_flipped = test_result_flipped[0][0].reshape(self.img_rows,self.img_cols)
            disparity_L_flipped = np.flip(disparity_L_flipped, axis=1)
            # Compute total disparity from normal and flipped outputs
            disparity_L = np.multiply(disparity_L, self.weights1) + np.multiply(disparity_L_flipped, self.weights2)

        # Estimate depth from disparity map
        if (depth_est_params is None):
            depth_estimation_L = None
        else:
            # Estimate depth
            disparity_L_clipped = np.clip(disparity_L, 0.0001, None)
            depth_estimation_L = np.clip(depth_est_params['baseline'] * depth_est_params['focal_length_L'] / (1242 * disparity_L_clipped), None, 50)

        return (disparity_L, imageL_rec, img_L, img_R, depth_estimation_L)
        

    def test_model(self):
        # Predict for all or randomly sampled images
        filenames = os.listdir(self.img_dir + '/left/data/')
        if (self.sampling == 'random'):
            filenames = list(np.random.choice(filenames, self.test_samples, replace=False))

        for sample_file in filenames:
            # Load images from file
            img_L = cv.imread(self.img_dir + '/left/data/' + sample_file, cv.IMREAD_COLOR)
            img_R = cv.imread(self.img_dir + '/right/data/' + sample_file, cv.IMREAD_COLOR)
            # Estimate disparity
            (disparity_L, imageL_rec, img_L, img_R, _) = self.__get_model_output(img_L, img_R)

            # Save results if specified
            if (self.save):
                disparity_L_file = disparity_L / np.amax(disparity_L) * 255.0
                disparity_L_file = disparity_L_file.astype(np.uint8)
                save_path = self.save_output + '/left_disparity/data/'
                if (not os.path.isdir(save_path)):
                    os.makedirs(save_path)
                cv.imwrite(save_path + sample_file, disparity_L_file)

            # Plot results if specified
            if (self.plot_results):
                fig = plt.figure()
                fig.add_subplot(221)
                plt.imshow(img_L[0,:,:,:])
                plt.title('Left image input')

                fig.add_subplot(222)
                plt.imshow(img_R[0,:,:,:])
                plt.title('Right image input')

                fig.add_subplot(223)
                plt.imshow(disparity_L)
                plt.title('Left disparity')

                fig.add_subplot(224)
                plt.imshow(imageL_rec)
                plt.title('Left image reconstructed')

                plt.show()       


    def infer_depth(self):
        # Load image from file
        inference_input = cv.imread(self.img_file)
        if (inference_input is None):
            print '[!] Could not load image.'
            exit()

        # Predict disparity
        (disparity_L, _, inference_input, _, depth_estimation_L) = self.__get_model_output(inference_input, flip=True)
        
        # Plot results
	if (self.plot_results):
            fig = plt.figure()
            fig.add_subplot(211)
            plt.imshow(inference_input[0,:,:,:])
            plt.title('Left image input')

            fig.add_subplot(212)
            plt.imshow(disparity_L)
            plt.title('Depth estimation')

            plt.show()       

	# Save to file if specified
        if (self.save):
            disparity_L = disparity_L / np.amax(disparity_L) * 255.0
            disparity_L = disparity_L.astype(np.uint8)
	    disparity_L = cv.applyColorMap(disparity_L, cv.COLORMAP_PARULA)
            cv.imwrite(self.save_output, disparity_L)



if __name__ == '__main__':
    tests = Adversarial_Depth_Tests()
    if (tests.test):
        # Test model
        tests.test_model()
    elif (tests.infer):
        # Infer depth of single image
        tests.infer_depth()
    else:
        print '[!] No usage flags asserted. Exiting.'


