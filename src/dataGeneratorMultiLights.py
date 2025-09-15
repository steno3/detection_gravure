from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import os
import gc
import numpy as np
import tensorflow as tf
import cv2

import src.rotate_dup as rot
import src.lowfreq_img as lfnoise
import src.dataGenerator
#from src.add_normals import add_noise_to_normals

class DataGeneratorMultiLights(Sequence):
    """ Generates data for Keras models.
    Inputs a tensor of shape (batch_size, height, width, nb_lights*3) where nb_lights is the number of light directions used as input.
    Args:
        data_names (list): List of image file names.
        batch_size (int): Size of the batches to be generated.
        epoch_size (int): Number of batches to be generated per epoch.
        patch_size (int): Size of the patches to be extracted.
        nb_imgs (int): Number of images (light directions) to use for a gt image.
        img_folder (str): Path to the folder containing the input images.
        groundtruth_folder (str): Path to the folder containing the ground truth images.
        patch_ratio (float): Ratio of the image to be used for patch extraction. In (0, 1]
        rotation (int): Degree of random rotation to apply to the patches. If -1, no rotation is applied.
        noise_scale (int): Scale of the gaussian filter for the noise. If -1, no noise is added.
        noise_max_angle (float): Maximum angle in degree for the noise rotation. Used only if noise_scale > 0.
        rescale (float): Rescaling factor for the images (patches are (size*rescale) when selected and then rescaled to patch_size).
        flip (bool): Whether to apply random horizontal flipping to the images.
        inputs_are_normals (bool): Whether the input images are normal maps (affects rotation and flipping).
        add_padding_needed (bool): Whether to add padding to the images if they are smaller than ground truth (useful for cr2 to jpg conversion).
        fun_img (function): Function to apply to the image data (default normalizes the image in [-1, 1]).
        fun_gt (function): Function to apply to the ground truth data (default normalizes the image in [0, 1]).
    """
    def __init__(self, data_names, batch_size, epoch_size, patch_size, nb_imgs, img_folder, groundtruth_folder, 
                patch_ratio=0.6,
                rotation_step=10, 
                noise_scale=-1, noise_max_angle=5,
                rescale=1.0,
                flip=False,
                inputs_are_normals=True,
                add_padding_needed=False,
                fun_img=lambda x: (x - 127.5) / 127.5,
                fun_gt=lambda x: 1 - x / 255.0):
        self.data_names = data_names
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.patch_size = patch_size
        self.nb_imgs = nb_imgs
        # Load images and adding padding if needed
        self.groundtruth_data = src.dataGenerator.DataGenerator.load_data_from_folder(groundtruth_folder, data_names, color_mode="grayscale", fun_traitement=fun_gt)
        self.img_folder = img_folder
        
        self.rotation_step = rotation_step
        self.noise = (noise_scale, noise_max_angle)
        self.rescale = rescale
        self.flip = flip
        self.patch_coords, self.patch_coords_size = src.dataGenerator.DataGenerator.get_patch_coords_and_counts(data_names, patch_size, groundtruth_folder, patch_ratio)
        self.add_padding_needed = add_padding_needed
        self.inputs_are_normals = inputs_are_normals
        self.fun_img = fun_img
        self.fun_gt = fun_gt
        self.on_epoch_end()

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, index):
        """
        Generate random batchs of data of size (batch_size, patch_size, patch_size, channels*nb_lights).
        where channels is 3 for RGB images and 1 for grayscale images.
        and nb_lights is the number of light directions = len(self.img_data[0]).
        """
        images = []
        groundtruths = []

        for _ in range(self.batch_size):
            rand = np.random.randint(0, len(self.data_names))
            img_name = self.data_names[rand]
            coords_pool = self.patch_coords[rand]
            coords_pool_size = self.patch_coords_size[rand]
            gt = self.groundtruth_data[rand]
            # Load all images for this ground truth
            # img_list = self.img_data[rand] # TODO: choisir X (aléatoire parmis celles du dossier) image ici
            img_list = []
            sub_imgs_name = os.listdir(os.path.join(self.img_folder, img_name[:-4])) # remove extension of 4 chars (may cause issues if not 4 chars like .png or .jpg)
            rand_imgs = np.random.choice(sub_imgs_name, self.nb_imgs, replace=False)
            for img_name2 in rand_imgs:
                img_path = os.path.join(self.img_folder, img_name[:-4], img_name2)
                img = load_img(img_path, color_mode="rgb")
                img_array = img_to_array(img)
                img_list.append(img_array)
            if self.add_padding_needed:
                img_list = self.add_padding([img_list], [gt])[0]

            if coords_pool_size == 0:
                print(f"No valid patch found for {img_name}. Skipping this image.")
                continue

            # Randomly select a patch coordinate
            idx = np.random.randint(0, coords_pool_size)
            y, x = coords_pool[idx]

            # Apply random scaling if specified and extract the patch
            if self.rescale > 1.0:
                # random scale in [1.0, self.rescale)
                scale_factor = np.random.uniform(1.0, self.rescale)
                n_y, n_x = int(y + self.patch_size * scale_factor), int(x + self.patch_size * scale_factor)
                patch_img = [img[y:n_y, x:n_x] for img in img_list]
                patch_gt = gt[y:n_y, x:n_x]
                # print(img.shape)
                # print(patch_img.shape, patch_gt.shape, scale_factor)
                # print(x, y, n_x, n_y)
                # Resize back to patch_size
                patch_img = [cv2.resize(img, (self.patch_size, self.patch_size)) for img in patch_img]
                patch_gt = cv2.resize(patch_gt, (self.patch_size, self.patch_size))
                
            else:
                patch_img = [img[y:y+self.patch_size, x:x+self.patch_size] for img in img_list]
                patch_gt = gt[y:y+self.patch_size, x:x+self.patch_size]
            
            # Sizes
            # print(f"* Patch size: {patch_img.shape}, Ground truth size: {patch_gt.shape}")

            # Apply flipping if specified (with a 50% chance)
            if self.flip and np.random.rand() > 0.5:
                if self.inputs_are_normals:
                    patch_img = [rot.flip_normal(img) for img in patch_img]
                else:
                    patch_img = [np.flip(img, axis=1) for img in patch_img]
                patch_gt = np.flip(patch_gt, axis=1)

            # Apply random rotation if specified
            if self.rotation_step > 0:
                range_angle = np.arange(0, 360, self.rotation_step)
                angle = np.random.choice(range_angle)
                patch_img = [rot.rotate_matrix(img, angle, normals=self.inputs_are_normals) for img in patch_img]
                patch_gt = rot.rotate_matrix(patch_gt, angle, normals=False)
            
            # Here to avoid issues with format changes in rotation
            patch_img = [self.fun_img(img) for img in patch_img]

            # Apply low-frequency angular noise if specified
            if self.noise[0] > 0 and self.inputs_are_normals:
                patch_img = [lfnoise.normal_rotation_noise(img, scale=self.noise[0], angle_max=self.noise[1]) for img in patch_img]

            patch_img = np.concatenate(patch_img, axis=-1)  # Concatenate along the channel dimension
            # Append the patch to the batch
            images.append(patch_img)
            groundtruths.append(patch_gt)

            # free the memory of img_list
            # del img_list # it slows down the training but avoids memory issues by a lot
            # del patch_img
        # gc.collect() # force garbage collection to free memory of images if above doesnt work
        

        X = tf.convert_to_tensor(np.array(images), dtype=tf.float32)
        Y = tf.cast(np.array(groundtruths), tf.float32)
        return X, Y

    def on_epoch_end(self):
        pass


    @staticmethod
    def are_same_size(data_list1, data_list2):
        """
        Check if all images in two lists have the same dimensions.
        list1 is a list of lists of images. Each sublist corresponds to images from one light direction
            and should have the same dimensions as the corresponding image in list2 (not in a sublist).
        Args:
            data_list1 (list): First list of list of image arrays.
            data_list2 (list): Second list of image arrays.
        Returns:
            bool: True if all images have the same xy dimensions, False otherwise.
        """
        for sublist, img2 in zip(data_list1, data_list2):
            if len(sublist) != len(data_list2):
                print(f"* Different number of images: {len(data_list1)} and {len(data_list2)}")
                return False
            for img1 in sublist:
                if img1.shape[:2] != img2.shape[:2]:
                    return False
        return True
    
    @staticmethod
    def add_padding(data_list, reference_list):
        """
        Add padding to images in data_list to match the dimensions of images in reference_list.
        list1 is a list of lists of images. Each sublist corresponds to images from one light direction
            and should have the same dimensions as the corresponding image in list2 (not in a sublist).
        Args:
            data_list (list): List of image arrays to be padded.
            reference_list (list): List of reference image arrays for target dimensions.
        Returns:
            list: List of list of padded image arrays.
        """
        padded_data = []
        for sublist, ref in zip(data_list, reference_list):
            padded_data_sublist = []
            for img in sublist:
                if img.shape[:2] == ref.shape[:2]:
                    padded_data_sublist.append(img)
                    continue
                h_diff = max(0, ref.shape[0] - img.shape[0])
                w_diff = max(0, ref.shape[1] - img.shape[1])
                # print(f"* Padding image from {img.shape} to {ref.shape} (h_diff={h_diff}, w_diff={w_diff})")
                top = h_diff // 2
                bottom = h_diff - top
                left = w_diff // 2
                right = w_diff - left
                padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT)
                padded_data_sublist.append(padded_img)
            padded_data.append(padded_data_sublist)
        return padded_data