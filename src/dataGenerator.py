from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import os
import numpy as np
import tensorflow as tf

import src.select_patch as sp
import src.rotate_dup as rot
from src.add_normals import add_noise_to_normals

class DataGenerator(Sequence):
    """ Generates data for Keras models.
    Args:
        data_names (list): List of image file names.
        batch_size (int): Size of the batches to be generated.
        epoch_size (int): Total number of samples to be generated per epoch.
        patch_size (int): Size of the patches to be extracted.
        img_folder (str): Path to the folder containing the input images.
        groundtruth_folder (str): Path to the folder containing the ground truth images.
        mask_folder (str): Path to the folder containing the mask images.
        rotation (int): Degree of random rotation to apply to the patches. If -1, no rotation is applied.
        noise_scale (int): Scale of the low-frequency noise to be added to the images. If -1, no noise is added.
        noise_intensity (float): Intensity of the low-frequency noise [0, 1].
        noise_width (float): Width of the noise band [0, 1] (scale from center).
        fun_img (function): Function to apply to the image data (default normalizes the image in [-1, 1]).
        fun_gt (function): Function to apply to the ground truth data (default normalizes the image in [0, 1]).
    """
    def __init__(self, data_names, batch_size, epoch_size, patch_size, img_folder, groundtruth_folder, mask_folder,
                pas_rotation=10, 
                noise_scale=-1, noise_intensity=1, noise_width=0.5, # TODO changer le fonctionnement du bruit -> bruit angulaire
                fun_img=lambda x: (x - 127.5) / 127.5,
                fun_gt=lambda x: 1 - x / 255.0):
        self.data_names = data_names
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.patch_size = patch_size
        self.img_data = DataGenerator.load_data_from_folder(img_folder, data_names, color_mode="rgb")
        self.groundtruth_data = DataGenerator.load_data_from_folder(groundtruth_folder, data_names, color_mode="grayscale", fun_traitement=fun_gt)
        self.pas_rotation = pas_rotation
        self.noise = (noise_scale, noise_intensity, noise_width)
        self.patch_coords, self.patch_coords_size = DataGenerator.get_patch_coords_and_counts(data_names, patch_size, mask_folder)
        self.fun_img = fun_img
        self.fun_gt = fun_gt
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.epoch_size / self.batch_size))

    def __getitem__(self, index):
        """
        Generate random batchs of data
        """
        images = []
        groundtruths = []

        for _ in range(self.batch_size):
            rand = np.random.randint(0, len(self.data_names))
            img_name = self.data_names[rand]
            coords_pool = self.patch_coords[rand]
            coords_pool_size = self.patch_coords_size[rand]
            img = self.img_data[rand]
            gt = self.groundtruth_data[rand]

            if coords_pool_size == 0:
                print(f"No valid patch found for {img_name}. Skipping this image.")
                continue
            # Randomly select a patch coordinate
            idx = np.random.randint(0, coords_pool_size)
            x, y = coords_pool[idx]

            # Extract the patch
            patch_img = img[y:y+self.patch_size, x:x+self.patch_size]
            patch_gt = gt[y:y+self.patch_size, x:x+self.patch_size]
            # Sizes
            # print(f"* Patch size: {patch_img.shape}, Ground truth size: {patch_gt.shape}")

            # Apply random rotation if specified
            if self.pas_rotation > 0:
                range_angle = np.arange(0, 360, self.pas_rotation)
                angle = np.random.choice(range_angle)
                patch_img = rot.rotate_matrix(patch_img, angle, normals=True)
                patch_gt = rot.rotate_matrix(patch_gt, angle, normals=False)
            
            # Here to avoid issues with format changes in rotation
            patch_img = self.fun_img(patch_img)

            # Apply low-frequency noise if specified
            if self.noise[0] > 0:
                patch_img = add_noise_to_normals(patch_img, size=self.patch_size, scale=self.noise[0], intensity=self.noise[1], width=self.noise[2])

            # Append the patch to the batch
            images.append(patch_img)
            groundtruths.append(patch_gt) 

        X = tf.convert_to_tensor(np.array(images), dtype=tf.float32)
        Y = tf.cast(np.array(groundtruths), tf.float32)
        return X, Y

    def on_epoch_end(self):
        # We can shuffle the data here if needed
        pass

    @staticmethod
    def load_data_from_folder(folder, data_names, color_mode="rgb", fun_traitement=lambda x: x):
        """
        Load all image file names from a given folder.
        Args:
            folder (str): Path to the folder containing images.
            data_names (list): List of image file names to load.
            color_mode (str): Color mode for loading images ('rgb' or 'grayscale').
            fun_traitement (function): Function to apply to the loaded image data.
        """
        img_data = []
        for f in data_names:
            path = os.path.join(folder, f)
            if os.path.isfile(path) and f.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_data.append(fun_traitement(img_to_array(load_img(path, color_mode=color_mode))))
                print(f"* Loaded image: {f} from {folder}")
                print(f"    Image shape: {img_data[-1].shape}")
                print(f"    Image min: {img_data[-1].min()}, max: {img_data[-1].max()}")
        return img_data

    @staticmethod
    def get_patch_coords_and_counts(data_names, patch_size, mask_folder):
        """
        Get the coordinates of patches from the mask image.
        Args:
            data_names (list): List of image file names.
            patch_size (int): Size of the patches to be extracted.
            mask_folder (str): Path to the folder containing the mask images.
        """
        coords = []
        counts = []
        for name in data_names:
            mask_path = os.path.join(mask_folder, name)
            mask = load_img(mask_path, color_mode="grayscale")
            mask_arr = np.array(mask)
            coo, count = sp.find_white_zones(mask_arr, patch_size)
            coords.append(coo)
            counts.append(count)
            print(f"* Found {count} valid patches in mask: {name}")
        return coords, counts