import os, argparse
import re, json
from keras.models import load_model
from bilinear_sampler import *
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Adversarial depth model KITTI evaluation.

# TODO: Remove param_dir and parameter dependence of selection set evaluation.

class Adversarial_Depth_KITTI_Eval(object):

    def __init__(self):
        # Initialize model
        args = self.parse_args()
        self.setup_parameters(args)


    def parse_args(self):
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Evaluate adversarial depth model on KITTI test data.')
        parser.add_argument('--evaluate_on_data',      action='store_true', help='Evaluate model on source data set and save results. Must be in left-right format.', default=False)
        parser.add_argument('--evaluate_on_selection', action='store_true', help='Evaluate model on validation data selection (as provided by KITTI) and save results.', default=False)
        parser.add_argument('--upsample',              action='store_true', help='Upsample results to original image size.', default=False)
        parser.add_argument('--source_dir',            type=str, help='Data source directory.',  default=None)
        parser.add_argument('--save_dir',              type=str, help='Output directory.',  default=None)
        parser.add_argument('--param_dir',             type=str, help='Parameter directory. Must be in JSON format.',  default=None)
        parser.add_argument('--model_file',            type=str, help='Trained model file (.h5).', default='adversarial_depth_model.h5')
        args = parser.parse_args()
        return args


    def setup_parameters(self,args):
        # Usage flags
        self.evaluate_on_data = args.evaluate_on_data
        self.evaluate_on_selection = args.evaluate_on_selection
        self.upsample = args.upsample
        # Image directories/files
        self.source_dir = args.source_dir
        self.save_dir = args.save_dir
        self.param_dir = args.param_dir
        # Load depth estimation model
        self.model_file = args.model_file 
        self.depth_model = load_model(self.model_file, custom_objects={'bilinear_sampling_L': bilinear_sampling_L, 'bilinear_sampler_1d_h': bilinear_sampler_1d_h}, compile=False)
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


    def eval_on_data(self):
        # Load depth estimation parameters
        depth_est_param_dates = dict()
        depth_est_params = dict()
        for f in os.listdir(self.source_dir):
            if (str(f).endswith('.json')):
                with open(self.source_dir + '/' + f, 'r') as param_file:
                    param_date = re.match(r"^[0-9]{4}_[0-9]{2}_[0-9]{2}", f)
                    depth_est_param_dates[param_date.group(0)] = json.load(param_file)

        # Evaluate model for all test images
        for f in os.listdir(self.source_dir + '/left/data/'):
            # Load left/right images from file
            img_L = cv.imread(self.source_dir + '/left/data/' + f, cv.IMREAD_COLOR)
            img_R = cv.imread(self.source_dir + '/right/data/' + f, cv.IMREAD_COLOR)
            # Extract date and record from img filename
            result_path_L = self.save_dir + '/left_pred/' + f
            result_path_R = self.save_dir + '/right_pred/' + f
            # Create save directory if needed
            if ((not os.path.isdir(self.save_dir + '/left_pred')) or (not os.path.isdir(self.save_dir + '/right_pred'))):
                os.makedirs(self.save_dir + '/left_pred')
                os.makedirs(self.save_dir + '/right_pred')

            # Select depth estimation parameters from file date
            param_date = re.match(r"^[0-9]{4}_[0-9]{2}_[0-9]{2}", f)
            depth_est_params = depth_est_param_dates[param_date.group(0)]
            
            # Estimate left image depth
            (disparity_L, _, _, _, depth_estimation_L) = self.__get_model_output(img_L, img_R=None, flip=True, depth_est_params=depth_est_params)
            # Save result in output directory
            if (self.upsample):
                depth_estimation_L = cv.resize(depth_estimation_L, (img_L.shape[1],img_L.shape[0]), interpolation=cv.INTER_CUBIC)
            depth_estimation_L = depth_estimation_L * 256.0
            depth_estimation_L = depth_estimation_L.astype(np.uint16)
            cv.imwrite(result_path_L, depth_estimation_L)

            # Estimate right image depth
            (disparity_R, _, _, _, depth_estimation_R) = self.__get_model_output(img_R, img_R=None, flip=True, depth_est_params=depth_est_params)
            # Save result in output directory
            if (self.upsample):
                depth_estimation_R = cv.resize(depth_estimation_R, (img_R.shape[1],img_R.shape[0]), interpolation=cv.INTER_CUBIC)
            depth_estimation_R = depth_estimation_R * 256.0
            depth_estimation_R = depth_estimation_R.astype(np.uint16)
            cv.imwrite(result_path_R, depth_estimation_R)


    def eval_on_selection(self):
        # Load depth estimation parameters
        depth_est_param_dates = dict()
        depth_est_params = dict()
        for f in os.listdir(self.param_dir):
            if (str(f).endswith('.json')):
                with open(self.param_dir + '/' + f, 'r') as param_file:
                    param_date = re.match(r"^[0-9]{4}_[0-9]{2}_[0-9]{2}", f)
                    depth_est_param_dates[param_date.group(0)] = json.load(param_file)

        # Create save directory
        if (not os.path.isdir(self.save_dir + '/depth_estimation/')):
            os.makedirs(self.save_dir + '/depth_estimation/')

        # Iterate over all validation images
        for f in os.listdir(self.source_dir + '/image/'):
            # Load image
            img = cv.imread(self.source_dir + '/image/' + f, cv.IMREAD_COLOR)
            # Create result path
            img_date = re.match(r"^[0-9]{4}_[0-9]{2}_[0-9]{2}_drive_[0-9]{4}_sync_", f)
            img_record = re.search(r"_[0-9]{10}_image_[0-9]{2}.png", f)
            result_path = self.save_dir + '/depth_estimation/' + img_date.group(0) + 'groundtruth_depth' + img_record.group(0)
            # Select depth estimation parameters
            param_date = re.match(r"^[0-9]{4}_[0-9]{2}_[0-9]{2}", f)
            depth_est_params = depth_est_param_dates[param_date.group(0)]

            # Estimate depth
            (_, _, _, _, depth_estimation) = self.__get_model_output(img, img_R=None, flip=True, depth_est_params=depth_est_params)
            # Save result
            depth_estimation = cv.resize(depth_estimation, (img.shape[1],img.shape[0]), interpolation=cv.INTER_CUBIC)
            depth_estimation = depth_estimation * 256.0
            depth_estimation = depth_estimation.astype(np.uint16)
            cv.imwrite(result_path, depth_estimation)



if __name__ == '__main__':
    evals  = Adversarial_Depth_KITTI_Eval()
    if (evals.evaluate_on_data):
        evals.eval_on_data()
    elif (evals.evaluate_on_selection):
        evals.eval_on_selection()
    else:
        print '[!] No usage flags asserted. Exiting'


