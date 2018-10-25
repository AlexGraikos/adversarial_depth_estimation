import os, signal, argparse
from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, concatenate, UpSampling2D, ActivityRegularization
from keras.layers import BatchNormalization, LeakyReLU, Cropping2D
from keras import optimizers, losses
from keras import callbacks
import keras.preprocessing.image as pre
from bilinear_sampler import *
import numpy as np
import matplotlib.pyplot as plt

# Creates and trains adversarial depth estimation model.

class Adversarial_Depth_Model(object):

    def __init__(self):
        # Initialize model
        args = self.parse_args()
        self.setup_parameters(args)
        # Instantiate models
        self.generator_model = None
        self.discriminator_model = None
        self.adversarial_model = None
        self.depth_model = None


    def parse_args(self):
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Keras implementation of adversarial monocular depth estimation model.')
        parser.add_argument('--img_dir',      type=str,   help='Directory containing training data. Must be in left-right directory format',  default=None)
        parser.add_argument('--img_rows',     type=int,   help='Model image rows.', default=256)
        parser.add_argument('--img_cols',     type=int,   help='Model image columns.', default=512)
        parser.add_argument('--gen_type',     type=str,   help='Generator type [autoencoder/dense].', default='dense')
        parser.add_argument('--batch_size',   type=int,   help='Images per batch.', default=8)
        parser.add_argument('--epochs',       type=int,   help='Number of training epochs.', default=5)
        parser.add_argument('--lr',           type=float, help='Learning rate.', default=0.0001)
        parser.add_argument('--model_name',   type=str,   help='Model name.', default='adversarial_depth_model')
        args = parser.parse_args()
        return args


    def setup_parameters(self,args):
        # Image directories
        self.img_dir_L = args.img_dir + '/left'
        self.img_dir_R = args.img_dir + '/right'
        # Model image size
        self.img_rows = args.img_rows
        self.img_cols = args.img_cols
        self.input_channels = 3 # Arbitrarily defined as an RGB image
        # Model parameters
        self.gen_type = args.gen_type
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.learning_rate = args.lr
        self.training_set_size = len(os.listdir(self.img_dir_L + '/data/'))
        self.steps_per_epoch = self.training_set_size/self.batch_size
        self.loss_lambda = 1000
        self.crop_size = 25
        self.validation_size = (self.img_rows/16, self.img_cols/16-3) # Discriminator validation patch size
        self.model_name = args.model_name
        # Optimizer 
        self.optimizer = optimizers.adam(lr=self.learning_rate)
        # Input generators
        self.discriminator_input_generator = self.mixed_input_generator(self.img_dir_L, self.img_dir_R, 
                self.img_rows, self.img_cols, self.batch_size)
        self.adversarial_input_generator = self.input_generator(self.img_dir_L, self.img_dir_R, 
                self.img_rows, self.img_cols, self.batch_size)


    def create_generator(self):
        if (self.gen_type == 'dense'):
            # --- DenseNet generator ---
            # Layer generating functions
            def dense_block(prev_layer, depth, skip_layer=None, growth_rate=16, kernel=3, concat_input=True):
                # Create layer stack
                if (skip_layer != None):
                    prev_layer = concatenate([skip_layer, prev_layer])
                stacking_layers = [prev_layer]
                
                # Add layers until depth is reached
                for i in range(depth):
                    if (len(stacking_layers) > 1):
                        d = concatenate(stacking_layers, axis=-1)
                    else:
                        d = stacking_layers[0]

                    d = Conv2D(filters=growth_rate, kernel_size=kernel, strides=1, padding='same', activation='relu')(d)
                    stacking_layers.append(d)

                # Total dense-block output
                if (not concat_input):
                    stacking_layers = stacking_layers[1:]
                d = concatenate(stacking_layers, axis=-1)

                return d

            def transition_down(prev_layer):
                td = prev_layer
                td = Conv2D(filters=int(prev_layer.shape[-1]), kernel_size=1, strides=2, padding='same', activation='relu')(td)
                return td
     
            def transition_up(prev_layer):
                tu = prev_layer
                tu = UpSampling2D(size=(2,2))(tu)
                tu = Conv2D(filters=int(prev_layer.shape[-1]), kernel_size=3, strides=1, padding='same', activation='relu')(tu)
                return tu 
           

            # Model input
            imageL = Input(shape=(self.img_rows,self.img_cols,self.input_channels))
            imageR = Input(shape=(self.img_rows,self.img_cols,self.input_channels))
            # Encoder
            enc_block1  = dense_block(imageL,      2, kernel=7)
            td1         = transition_down(enc_block1)
            enc_block1b = dense_block(td1,         2, kernel=7)
            enc_block2  = dense_block(enc_block1b, 2, kernel=5)
            td2         = transition_down(enc_block2)
            enc_block2b = dense_block(td2,         2, kernel=5)
            enc_block3  = dense_block(enc_block2b, 2)
            td3         = transition_down(enc_block3)
            enc_block3b = dense_block(td3,         2)
            enc_block4  = dense_block(enc_block3b, 4)
            td4         = transition_down(enc_block4)
            enc_block4b = dense_block(td4,         4)
            enc_block5  = dense_block(enc_block4b, 8)
            td5         = transition_down(enc_block5)
            enc_block5b = dense_block(td5,         8)
            # Decoder
            dec_block5  = dense_block(enc_block5b, 8)
            tu5         = transition_up(dec_block5)
            dec_block5b = dense_block(tu5,         8, skip_layer=enc_block4b, concat_input=False)
            dec_block4  = dense_block(dec_block5b, 4)
            tu4         = transition_up(dec_block4)
            dec_block4b = dense_block(tu4,         4, skip_layer=enc_block3b, concat_input=False)
            dec_block3  = dense_block(dec_block4b, 2)
            tu3         = transition_up(dec_block3)
            dec_block3b = dense_block(tu3,         2, skip_layer=enc_block2b, concat_input=False)
            dec_block2  = dense_block(dec_block3b, 2)
            tu2         = transition_up(dec_block2)
            dec_block2b = dense_block(tu2,         2, skip_layer=enc_block1b, concat_input=False)
            dec_block1  = dense_block(dec_block2b, 2)
            tu1         = transition_up(dec_block1)
            dec_block1b = dense_block(tu1,         2, skip_layer=None, concat_input=False)
            # Disparity maps
            disparity_L = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', activation='relu')(dec_block1b)
            disparity_R = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', activation='relu')(dec_block1b)
            # Left-right image reconstruction
            reconstructed_image_L = Lambda(bilinear_sampling_L, output_shape=(self.img_rows,self.img_cols,self.input_channels))([imageR, disparity_L])
            reconstructed_image_R = Lambda(bilinear_sampling_R, output_shape=(self.img_rows,self.img_cols,self.input_channels))([imageL, disparity_R])
            # Disparity consistency regularization
            disparity_lr_consistency_layer = Lambda(disparity_lr_consistency, output_shape=(self.img_rows,self.img_cols,1))([disparity_L, disparity_R])
            disparity_rl_consistency_layer = Lambda(disparity_rl_consistency, output_shape=(self.img_rows,self.img_cols,1))([disparity_R, disparity_L])
            regularization_lr = ActivityRegularization(l1=1.0, l2=0.0)(disparity_lr_consistency_layer)
            regularization_rl = ActivityRegularization(l1=1.0, l2=0.0)(disparity_rl_consistency_layer)

            # Training and inference models
            self.generator_model = Model(inputs=[imageL, imageR], outputs=[reconstructed_image_L, reconstructed_image_R])
            self.depth_model = Model(inputs=[imageL, imageR], outputs=[disparity_L, reconstructed_image_L])

        elif (self.gen_type == 'autoencoder'):
            # --- Autoencoder generator ---
            # Layer generating functions
            def enc_layer(prev_layer, filters, stride, kernel=3):
                enc = Conv2D(filters=filters, kernel_size=kernel, strides=stride, padding='same', activation='relu')(prev_layer) 
                return enc

            def dec_layer(prev_layer, skip_layer, filters, upsample):
                dec = prev_layer
                if (upsample):
                    dec = UpSampling2D(size=(2,2))(dec)
                if (skip_layer != None):
                    dec = concatenate([skip_layer, dec], axis=-1)
                dec = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', activation='relu')(dec)
                return dec

            # Model input
            imageL = Input(shape=(self.img_rows,self.img_cols,self.input_channels), name='imageL')
            imageR = Input(shape=(self.img_rows,self.img_cols,self.input_channels), name='imageR')
            # Encoder
            enc1 =  enc_layer(imageL, 32,  stride=2, kernel=7)
            enc1b = enc_layer(enc1,   32,  stride=1, kernel=7)
            enc2 =  enc_layer(enc1b,  64,  stride=2, kernel=5)
            enc2b = enc_layer(enc2,   64,  stride=1, kernel=5)
            enc3 =  enc_layer(enc2b,  128, stride=2)
            enc3b = enc_layer(enc3,   128, stride=1)
            enc4 =  enc_layer(enc3b,  256, stride=2)
            enc4b = enc_layer(enc4,   256, stride=1)
            enc5 =  enc_layer(enc4b,  512, stride=2)
            enc5b = enc_layer(enc5,   512, stride=1)
            enc6 =  enc_layer(enc5b,  512, stride=2)
            enc6b = enc_layer(enc6,   512, stride=1)
            enc7 =  enc_layer(enc6b,  512, stride=2)
            enc7b = enc_layer(enc7,   512, stride=1)
            # Decoder
            dec7  = dec_layer(enc7b, None, 512, upsample=True)
            dec7b = dec_layer(dec7, enc6b, 512, upsample=False)
            dec6  = dec_layer(dec7b, None, 512, upsample=True)
            dec6b = dec_layer(dec6, enc5b, 512, upsample=False)
            dec5  = dec_layer(dec6b, None, 256, upsample=True)
            dec5b = dec_layer(dec5, enc4b, 256, upsample=False)
            dec4  = dec_layer(dec5b, None, 128, upsample=True)
            dec4b = dec_layer(dec4, enc3b, 128, upsample=False)
            dec3  = dec_layer(dec4b, None, 64,  upsample=True)
            dec3b = dec_layer(dec3, enc2b, 64,  upsample=False)
            dec2  = dec_layer(dec3b, None, 32,  upsample=True)
            dec2b = dec_layer(dec2, enc1b, 32,  upsample=False)
            dec1  = dec_layer(dec2b, None, 16,  upsample=True)
            dec1b = dec_layer(dec1,  None, 16,  upsample=False)
            # Left image reconstruction
            disparity_L = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', activation='relu')(dec1b) 
            disparity_R = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', activation='relu')(dec1b) 
            # Left-right image reconstruction
            reconstructed_image_L = Lambda(bilinear_sampling_L, output_shape=(self.img_rows,self.img_cols,self.input_channels))([imageR, disparity_L])
            reconstructed_image_R = Lambda(bilinear_sampling_R, output_shape=(self.img_rows,self.img_cols,self.input_channels))([imageL, disparity_R])
            # Disparity consistency regularization
            disparity_lr_consistency_layer = Lambda(disparity_lr_consistency, output_shape=(self.img_rows,self.img_cols,1))([disparity_L, disparity_R])
            disparity_rl_consistency_layer = Lambda(disparity_rl_consistency, output_shape=(self.img_rows,self.img_cols,1))([disparity_R, disparity_L])
            regularization_lr = ActivityRegularization(l1=1.0, l2=0.0)(disparity_lr_consistency_layer)
            regularization_rl = ActivityRegularization(l1=1.0, l2=0.0)(disparity_rl_consistency_layer)

            # Training and inference models
            self.generator_model = Model(inputs=[imageL, imageR], outputs=[reconstructed_image_L, reconstructed_image_R])
            self.depth_model = Model(inputs=[imageL, imageR], outputs=[disparity_L, reconstructed_image_L])

        else:
            print '[!] Generator type not supported.'
            exit()


    def create_discriminator(self):
        # Layer generating function
        def discr_layer(prev_layer, filters, kernel=4, batch_norm=True):
            discr = Conv2D(filters=filters, kernel_size=kernel, strides=2, padding='same')(prev_layer) 
            if (batch_norm):
                discr = BatchNormalization()(discr)
            discr = LeakyReLU(alpha=0.2)(discr)
            return discr
         
        # Model input
        imageL = Input(shape=(self.img_rows, self.img_cols, self.input_channels))
        imageR = Input(shape=(self.img_rows, self.img_cols, self.input_channels))
        imageL_cropped = Cropping2D(cropping=((0,0),(self.crop_size,self.crop_size)))(imageL)
        imageR_cropped = Cropping2D(cropping=((0,0),(self.crop_size,self.crop_size)))(imageR)
        discriminator_input = concatenate([imageL_cropped, imageR_cropped], axis=-1)
        # Discriminator network
        conv1 = discr_layer(discriminator_input, 64, batch_norm=False)
        conv2 = discr_layer(conv1, 128)
        conv3 = discr_layer(conv2, 256)
        conv4 = discr_layer(conv3, 512)
        validation_layer = Conv2D(filters=1, kernel_size=4, strides=1, padding='same', activation='sigmoid')(conv4)

        # Create and compile model
        self.discriminator_model = Model(inputs=[imageL, imageR], outputs=validation_layer)
        self.discriminator_model.compile(optimizer=self.optimizer, loss='binary_crossentropy')


    def create_adversarial_model(self):
        # Model Input
        imageL = Input(shape=(self.img_rows,self.img_cols,self.input_channels))
        imageR = Input(shape=(self.img_rows,self.img_cols,self.input_channels))
        # Combine generator and discriminator models to create adversarial model
        reconstructed_images = self.generator_model([imageL, imageR]) # LR reconstructed images
        discriminator_output = self.discriminator_model([reconstructed_images[0], reconstructed_images[1]])

        # Adversarial model does not train discriminator
        self.discriminator_model.trainable = False
        # Create model and compile
        self.adversarial_model = Model(inputs=[imageL, imageR], outputs=[reconstructed_images[0], reconstructed_images[1], discriminator_output])
        self.adversarial_model.compile(optimizer=self.optimizer, loss=['mae','mae','binary_crossentropy'], loss_weights=[self.loss_lambda, self.loss_lambda, 1])


    def train_model(self):
        # Keras bug - model is not loaded on graph otherwise
        self.discriminator_model.trainable = True
        self.discriminator_model.predict(self.discriminator_input_generator.next()[0])

        # Learning rate halving callback function
        def halve_lr(epoch, curr_lr):
            # Epochs are zero-indexed
            if (epoch < 30):
                return curr_lr
            else:
                if (epoch % 10 == 0):
                    print '[*] Halving learning rate (=' + str(curr_lr) + ' -> ' + str(curr_lr / 2.0) + ')'
                    return curr_lr / 2.0
                else:
                    return curr_lr
        callback = [callbacks.LearningRateScheduler(schedule=halve_lr, verbose=0)]

        # Train discriminator and generator successively
        for i in range(self.epochs):
            print '[-] Adversarial model training epoch [' + str(i+1) + '/' + str(self.epochs) + ']'

            # Train discriminator
            self.discriminator_model.trainable = True
            self.discriminator_model.fit_generator(self.discriminator_input_generator, steps_per_epoch=self.steps_per_epoch, epochs=i+1, initial_epoch=i, callbacks=callback, verbose=1)
            # Train adversarial model
            self.discriminator_model.trainable = False
            self.adversarial_model.fit_generator(self.adversarial_input_generator, steps_per_epoch=self.steps_per_epoch, epochs=i+1, initial_epoch=i, verbose=1)

            # Save model every 5 epochs
            if (i % 5 == 0):
                postfix = '_epoch_' + str(i+1)
                print '[*] Saving model at epoch ' + str(i+1)
                self.save_model(postfix)


    def save_model(self, postfix=''):
        # Save inference model to file
        if (self.depth_model):
            if (not os.path.isdir('trained_models/')):
                os.makedirs('trained_models/')
            self.depth_model.save('trained_models/' + self.model_name + postfix + '.h5')
        else:
            print '[!] Model not defined. Abandoning saving.'
            exit()


    def mixed_input_generator(self, dir1, dir2, m, n, batch_size):
        # Image preprocessor
        datagen = pre.ImageDataGenerator(
                rescale = 1./255,
                channel_shift_range=0.1,
                fill_mode='nearest')
        p_flip = 0.5

        # L-R image generators
        # Use same random seed on generators to apply same shuffling/transformations
        seed = int(np.random.rand(1,1)*1000)
        train_L_generator = datagen.flow_from_directory(dir1, target_size=(m,n), interpolation='bilinear',
                batch_size=batch_size, class_mode=None, seed=seed)
        train_R_generator = datagen.flow_from_directory(dir2, target_size=(m,n), interpolation='bilinear',
                batch_size=batch_size, class_mode=None, seed=seed)
    
        while True:
            # L-R images
            input_imageL = train_L_generator.next()
            input_imageR = train_R_generator.next()

            # Randomly flip inputs horizontally
            if (np.random.rand() < p_flip):
                temp = input_imageR
                input_imageR = np.flip(input_imageL, axis=2)
                input_imageL = np.flip(temp, axis=2)
            
            half_samples = np.ceil(input_imageL.shape[0]/2.).astype(int)
            # Generator output for second half of batch
            reconstructed_images = self.generator_model.predict([input_imageL[half_samples:], input_imageR[half_samples:]])
            reconstructed_L = reconstructed_images[0]
            reconstructed_R = reconstructed_images[1]
            
            # Combine real and predicted images with labels 1s and 0s respectively
            total_imageL = np.concatenate((input_imageL[:half_samples], reconstructed_L), axis=0)
            total_imageR = np.concatenate((input_imageR[:half_samples], reconstructed_R), axis=0)
            labels = np.concatenate((np.ones((half_samples,self.validation_size[0],self.validation_size[1],1)),
                np.zeros((reconstructed_L.shape[0],self.validation_size[0],self.validation_size[1],1))), axis=0)

            # Returns (inputs,outputs) tuple
            yield ([total_imageL, total_imageR], [labels]) 


    def input_generator(self, dir1, dir2, m, n, batch_size):
        # Image preprocessor
        datagen = pre.ImageDataGenerator(
                rescale = 1./255,
                channel_shift_range=0.1,
                fill_mode='nearest')
        p_flip = 0.5

        # L-R image generators
        # Use same random seed on generators to apply same shuffling/transformations
        seed = int(np.random.rand(1,1)*1000)
        train_L_generator = datagen.flow_from_directory(dir1, target_size=(m,n), interpolation='bilinear',
                batch_size=batch_size, class_mode=None, seed=seed)
        train_R_generator = datagen.flow_from_directory(dir2, target_size=(m,n), interpolation='bilinear',
                batch_size=batch_size, class_mode=None, seed=seed)
    
        while True:
            # L-R images
            input_imageL = train_L_generator.next()
            input_imageR = train_R_generator.next()

            # Randomly flip inputs horizontally
            if (np.random.rand() < p_flip):
                temp = input_imageR
                input_imageR = np.flip(input_imageL, axis=2)
                input_imageL = np.flip(temp, axis=2)

            # Adversarial model training sets perdicted image labels to 1 
            samples = input_imageL.shape[0]
            labels = np.ones((samples,self.validation_size[0],self.validation_size[1],1))

            # Returns (inputs,outputs) tuple
            yield ([input_imageL, input_imageR], [input_imageL, input_imageR, labels]) 



if __name__ == '__main__':
    model = Adversarial_Depth_Model()

    # Setup termination signal handler
    def signal_handler(signum, frame):
        print '\n[!] Termination signal received. Saving model.'
        model.save_model()
        exit()
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create model
    model.create_generator()
    model.create_discriminator()
    model.create_adversarial_model()
    # Train model
    model.train_model()
    # Save model
    model.save_model()


