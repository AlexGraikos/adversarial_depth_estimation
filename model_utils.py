import os, argparse
import subprocess as sub
import re, json
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Adversarial depth model utilities.

class Adversarial_Depth_Utils(object):

    def __init__(self):
        args = self.parse_args()
        self.setup_parameters(args)


    def parse_args(self):
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Adversarial depth model utilities.')
        parser.add_argument('--subsample',           action='store_true', help='Subsample images in source to desired size and save to target. Must be in left-right directory format', default=False)
        parser.add_argument('--create_train_val',    action='store_true', help='Create training and validation sets defined in source file, of data in target file. Data must be in raw format.', default=False)
        parser.add_argument('--convert_KITTI_raw',   action='store_true', help='Converts KITTI raw source directory format to target left-right format.', default=False)
        parser.add_argument('--convert_KITTI_depth', action='store_true', help='Converts KITTI depth source directory format to target left-right format.', default=False)
        parser.add_argument('--delete_unmatched',    action='store_true', help='Deletes unmatched samples from source and target directories.', default=False)
        parser.add_argument('--merge',               action='store_true', help='Merge left-right formatted source data into a single target directory.', default=False)
        parser.add_argument('--source',              type=str,            help='Source directory/file.',  default=None)
        parser.add_argument('--target',              type=str,            help='Target directory.', default=None)
        parser.add_argument('--target_rows',         type=int,            help='Target image rows.', default=256)
        parser.add_argument('--target_cols',         type=int,            help='Target image columns.', default=512)
        parser.add_argument('--sampling_method',     type=str,            help='Sampling method to use for resizing [nearest_neighbor/bilinear]', default='bilinear')
        args = parser.parse_args()
        return args


    def setup_parameters(self,args):
        # Usage flags
        self.subsample = args.subsample
        self.create_train_val = args.create_train_val
        self.convert_KITTI_raw = args.convert_KITTI_raw
        self.convert_KITTI_depth = args.convert_KITTI_depth
        self.delete_unmatched = args.delete_unmatched
        self.merge = args.merge
        # Image directories
        self.source = args.source
        self.target = args.target
        # Target image size
        self.target_rows = args.target_rows
        self.target_cols = args.target_cols
        # Subsampling method
        self.sampling_method = args.sampling_method


    def subsample_images(self):
        # Directories to use
        source_img_L = self.source + '/left/data'
        source_img_R = self.source + '/right/data'
        target_img_L = self.target + '/left/data'
        target_img_R = self.target + '/right/data'

        # Create directories
        if (not os.path.isdir(target_img_L)):
            os.makedirs(target_img_L)
            os.makedirs(target_img_R)
            
        # Determine interpolation method
        if (self.sampling_method == 'nearest_neighbor'):
            interp_method = cv.INTER_NEAREST
        elif (self.sampling_method == 'bilinear'):
            interp_method = cv.INTER_LINEAR
        else:
            print '[!] Unknown sampling method. Exiting.'
            exit()

        # Iterate over all images in source folder (Left images)
        for filename in os.listdir(source_img_L):
            # Load image from file
            img_filename = source_img_L + '/' + filename
            img = cv.imread(img_filename, cv.IMREAD_COLOR)
            # Resize image
            img_resized = cv.resize(img, (self.target_cols,self.target_rows), interp_method)
            # Write result to target location
            target_filename = target_img_L + '/' + filename
            cv.imwrite(target_filename, img_resized)

        # Iterate over all images in source folder (Right images)
        for filename in os.listdir(source_img_R):
            # Load image from file
            img_filename = source_img_R + '/' + filename
            img = cv.imread(img_filename, cv.IMREAD_COLOR)
            # Resize image
            img_resized = cv.resize(img, (self.target_cols,self.target_rows), interp_method)
            # Write result to target location
            target_filename = target_img_R + '/' + filename
            cv.imwrite(target_filename, img_resized)

        # Copy depth estimation parameter files
        sub.call(["cp " + self.source + "/*.json " + self.target], shell=True)


    def create_training_validation_directories(self):
        training_dirs = []
        validation_dirs = []
        train_flag = False
        valid_flag = False

        # Load training and validation directories
        with open(self.source, 'r') as set_file:
            for line in set_file:
                if (line.startswith('training')):
                    train_flag = True
                    valid_flag = False
                    continue
                elif (line.startswith('validation')):
                    train_flag = False
                    valid_flag = True
                    continue

                if (train_flag):
                    training_dirs.append(line.rstrip())
                elif (valid_flag):
                    validation_dirs.append(line.rstrip())

        # Separate training and validation data
        os.chdir(self.target)
        os.makedirs('training_data/')
        os.makedirs('validation_data/')
        for directory in training_dirs:
            sub.call(["mv " + directory + " training_data/"], shell=True)
        for directory in validation_dirs:
            sub.call(["mv " + directory + " validation_data/"], shell=True)
        # Copy calibration data to each set
        for directory in os.listdir(self.target):
            if (directory not in ['training_data','validation_data']):
                sub.call(["cp " + directory + " training_data/ -r"], shell=True)
                sub.call(["mv " + directory + " validation_data/"], shell=True)


    # Taken from https://github.com/hunse/kitti
    def read_calib_file(self,path):
        float_chars = set("0123456789.e+- ")
        data = {}
        with open(path, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                value = value.strip()
                data[key] = value
                if float_chars.issuperset(value):
                    # try to cast to float array
                    try:
                        data[key] = np.array(map(float, value.split(' ')))
                    except ValueError:
                        # casting error: data[key] already eq. value, so pass
                        pass

        return data


    def convert_KITTI_raw_directory_format(self):
        # Copy files to target directory
        if (not os.path.isdir(self.target)):
            os.makedirs(self.target)
        os.chdir(self.target)
        sub.call(["rm * -rf"], shell=True)
        sub.call(["cp " + self.source + "/*" + " . -r"], shell=True)

        # Parse calibration files to generate depth estimation parameters
        cwd = os.getcwd()
        dirs = os.listdir(cwd)
        for f in dirs:
            if (re.match(r"^[0-9]{4}_[0-9]{2}_[0-9]{2}$", f)):
                calib_file = self.read_calib_file(self.target + '/' + f + '/calib_cam_to_cam.txt') 
                P2_rect = calib_file['P_rect_02'].reshape(3,4)
                P3_rect = calib_file['P_rect_03'].reshape(3,4)

                # cam 2 is left of camera 0  -6cm
                # cam 3 is to the right  +54cm
                b2 = P2_rect[0,3] / -P2_rect[0,0]
                b3 = P3_rect[0,3] / -P3_rect[0,0]

                # Save generated parameters to JSON file
                depth_est_params = dict()
                depth_est_params['baseline'] = b3-b2
                depth_est_params['focal_length_L'] = P2_rect[0,0]
                depth_est_params['focal_length_R'] = P3_rect[0,0]

                with open(f + '_depth_estimation_parameters.json', "w") as param_file:
                    json.dump(depth_est_params, param_file)

                # Delete calibration file
                sub.call(['rm', self.target + '/' + f, '-r'])


        # Delete unecessary directories
        sub.call(['rm */image_00 -rf'], shell=True)
        sub.call(['rm */image_01 -rf'], shell=True)
        sub.call(['rm */oxts -rf'], shell=True)
        sub.call(['rm */velodyne_points -rf'], shell=True)
        # Rename to left/right directories
        sub.call(["find . * | rename 's\\image_02\\left\\' 2> /dev/null"], shell=True)
        sub.call(["find . * | rename 's\\image_03\\right\\' 2> /dev/null"], shell=True)
        
        # Add prefixes to images to avoid duplicates
        # Get list of files in directory
        dirs = os.listdir(cwd)
        # Iterate over all files
        for f in dirs:
            if (os.path.isdir(self.target + '/' + f)):
                # Use directory name as prefix
                prefix = str(f)
                # Rename left data
                os.chdir(f+'/left/data/')
                sub.call(['rename \'s/(.*)/' + str(prefix) +'_$1/\' *.png'], shell=True)
                os.chdir(cwd)
                # Rename right data
                os.chdir(f+'/right/data/')
                sub.call(['rename \'s/(.*)/' + str(prefix) +'_$1/\' *.png'], shell=True)
                os.chdir(cwd)

        # Aggregate data into directories 
        # Left directory
        sub.call(['mkdir', 'left/data', '-p'])
        # Right directory
        sub.call(['mkdir', 'right/data', '-p'])
        # Aggregate
        sub.call(['find */left/data/ -type f | xargs -i mv "{}" left/data'], shell=True)
        sub.call(['find */right/data/ -type f | xargs -i mv "{}" right/data'], shell=True)
        # Delete all directories excluding left/right and parameter files
        sub.call(["ls | grep -v 'left' | grep -v 'right' | grep -v '.json' | xargs rm -rf"], shell=True)


    def convert_KITTI_depth_directory_format(self):
        # Copy files to target directory
        if (not os.path.isdir(self.target)):
            os.makedirs(self.target)
        os.chdir(self.target)
        sub.call(["rm * -rf"], shell=True)
        sub.call(["cp " + self.source + "/*" + " . -r"], shell=True)

        # Rename to left/right directories
        sub.call(["find . * | rename 's\\image_02\\left_gt\\' 2> /dev/null"], shell=True)
        sub.call(["find . * | rename 's\\image_03\\right_gt\\' 2> /dev/null"], shell=True)
        
        # Add prefixes to images to avoid duplicates
        # Get list of files in directory
        cwd = os.getcwd()
        dirs = os.listdir(cwd)
        # Iterate over all files
        for f in dirs:
            if (os.path.isdir(self.target + '/' + f)):
                # Use directory name as prefix
                prefix = str(f)
                # Rename left data
                os.chdir(f+'/proj_depth/groundtruth/left_gt/')
                sub.call(['rename \'s/(.*)/' + str(prefix) +'_$1/\' *.png'], shell=True)
                os.chdir(cwd)
                # Rename right data
                os.chdir(f+'/proj_depth/groundtruth/right_gt/')
                sub.call(['rename \'s/(.*)/' + str(prefix) +'_$1/\' *.png'], shell=True)
                os.chdir(cwd)

        # Aggregate data into directories 
        # Left directory
        sub.call(['mkdir', 'left_gt/', '-p'])
        # Right directory
        sub.call(['mkdir', 'right_gt/', '-p'])
        # Aggregate
        sub.call(['find */proj_depth/groundtruth/left_gt -type f | xargs -i mv "{}" left_gt/'], shell=True)
        sub.call(['find */proj_depth/groundtruth/right_gt -type f | xargs -i mv "{}" right_gt/'], shell=True)
        # Delete all directories excluding left/right and parameter files
        sub.call(["ls | grep -v 'left_gt' | grep -v 'right_gt' | xargs rm -rf"], shell=True)


    def delete_unmatched_samples(self):
        # Get list of all files in source and target directory
        samples1 = os.listdir(self.source)
        samples2 = os.listdir(self.target)

        # Delete samples that have no pairing sample 
        non_paired_samples = list(set(samples1).symmetric_difference(set(samples2)))
        for sample in non_paired_samples:
            try:
                os.remove(self.source + '/' + sample)
            except OSError:
                print '[!] Could not delete file ' + self.source + '/' + sample
            try:
                os.remove(self.target + '/' + sample)
            except OSError:
                print '[!] Could not delete file ' + self.target + '/' + sample


    def merge_left_right_depth(self):
        # Copy files to target directory
        if (not os.path.isdir(self.target)):
            os.makedirs(self.target)
        os.chdir(self.target)
        sub.call(["rm * -rf"], shell=True)
        sub.call(["cp " + self.source + "/*" + " . -r"], shell=True)

        # Create merge directory
        dir_names = filter(lambda x: x != '', self.source.split('/'))
        merge_dir = dir_names[-1] + '_merged'
        os.makedirs(self.target + '/' + merge_dir)

        # Parse all directories and rename data
        cwd = os.getcwd()
        dirs = os.listdir(cwd)
        for f in dirs:
            if (os.path.isdir(self.target + '/' + f)):
                # Use left/right as prefix
                prefix = ''
                if (f.startswith('left_')):
                    prefix = 'left'
                elif (f.startswith('right_')):
                    prefix = 'right'
                else:
                    continue
                # Rename 
                os.chdir(self.target + '/' + f)
                sub.call(['rename \'s/(.*)/' + str(prefix) +'_$1/\' *.png'], shell=True)
                # Copy to merged directory
                sub.call(["cp * " + self.target + '/' + merge_dir], shell=True)

        # Delete unnecessary directories
        os.chdir(cwd)
        sub.call(["ls | grep -v " + merge_dir + " | xargs rm -rf"], shell=True)



if __name__ == '__main__':
    utils = Adversarial_Depth_Utils()
    if (utils.subsample):
        utils.subsample_images()
    elif (utils.create_train_val):
        utils.create_training_validation_directories()
    elif (utils.convert_KITTI_raw):
        utils.convert_KITTI_raw_directory_format()
    elif (utils.convert_KITTI_depth):
        utils.convert_KITTI_depth_directory_format()
    elif (utils.delete_unmatched):
        utils.delete_unmatched_samples()
    elif (utils.merge):
        utils.merge_left_right_depth()
    else:
        print '[!] No usage flags asserted. Exiting.'

