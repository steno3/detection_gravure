from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import os
import numpy as np
import tensorflow as tf
import cv2
import scipy.ndimage as ndi

import src.rotate_dup as rot
import src.lowfreq_img as lfnoise
#from src.add_normals import add_noise_to_normals

class DataGenerator(Sequence):
    """ Generates data for Keras models.
    Args:
        data_names (list): List of image file names.
        batch_size (int): Size of the batches to be generated.
        epoch_size (int): Number of batches to be generated per epoch.
        patch_size (int): Size of the patches to be extracted.
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
    def __init__(self, data_names, batch_size, epoch_size, patch_size, img_folder, groundtruth_folder, 
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
        # Load images and adding padding if needed
        self.groundtruth_data = DataGenerator.load_data_from_folder(groundtruth_folder, data_names, color_mode="grayscale", fun_traitement=fun_gt)
        self.img_data = DataGenerator.load_data_from_folder(img_folder, data_names, color_mode="rgb")
        if not DataGenerator.are_same_size(self.img_data, self.groundtruth_data):
            if add_padding_needed:
                self.img_data = DataGenerator.add_padding(self.img_data, self.groundtruth_data)
                print("* Added padding to input images to match ground truth size.")
            else:
                raise ValueError("Input images and ground truth images must have the same dimensions.")
        
        self.rotation_step = rotation_step
        self.noise = (noise_scale, noise_max_angle)
        self.rescale = rescale
        self.flip = flip
        self.patch_coords, self.patch_coords_size = DataGenerator.get_patch_coords_and_counts(data_names, patch_size, groundtruth_folder, patch_ratio)
        self.inputs_are_normals = inputs_are_normals
        self.fun_img = fun_img
        self.fun_gt = fun_gt
        self.on_epoch_end()

    def __len__(self):
        return self.epoch_size

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
            y, x = coords_pool[idx]

            # Apply random scaling if specified and extract the patch
            if self.rescale > 1.0:
                # random scale in [1.0, self.rescale)
                scale_factor = np.random.uniform(1.0, self.rescale)
                n_y, n_x = int(y + self.patch_size * scale_factor), int(x + self.patch_size * scale_factor)
                patch_img = img[y:n_y, x:n_x]
                patch_gt = gt[y:n_y, x:n_x]
                # print(img.shape)
                # print(patch_img.shape, patch_gt.shape, scale_factor)
                # print(x, y, n_x, n_y)
                # Resize back to patch_size
                patch_img = cv2.resize(patch_img, (self.patch_size, self.patch_size))
                patch_gt = cv2.resize(patch_gt, (self.patch_size, self.patch_size))
                
            else:
                patch_img = img[y:y+self.patch_size, x:x+self.patch_size]
                patch_gt = gt[y:y+self.patch_size, x:x+self.patch_size]
            
            # Sizes
            # print(f"* Patch size: {patch_img.shape}, Ground truth size: {patch_gt.shape}")

            # Apply flipping if specified (with a 50% chance)
            if self.flip and np.random.rand() > 0.5:
                if self.inputs_are_normals:
                    patch_img = rot.flip_normal(patch_img)
                else:
                    patch_img = np.flip(patch_img, axis=1)
                patch_gt = np.flip(patch_gt, axis=1)

            # Apply random rotation if specified
            if self.rotation_step > 0:
                range_angle = np.arange(0, 360, self.rotation_step)
                angle = np.random.choice(range_angle)
                patch_img = rot.rotate_matrix(patch_img, angle, normals=self.inputs_are_normals)
                patch_gt = rot.rotate_matrix(patch_gt, angle, normals=False)
            
            # Here to avoid issues with format changes in rotation
            patch_img = self.fun_img(patch_img)

            # Apply low-frequency angular noise if specified
            if self.noise[0] > 0 and self.inputs_are_normals:
                patch_img = lfnoise.normal_rotation_noise(patch_img, scale=self.noise[0], angle_max=self.noise[1])

            # Append the patch to the batch
            images.append(patch_img)
            groundtruths.append(patch_gt) 

        X = tf.convert_to_tensor(np.array(images), dtype=tf.float32)
        Y = tf.cast(np.array(groundtruths), tf.float32)
        return X, Y

    def on_epoch_end(self):
        # We can do funny stuff here
        pass

    @staticmethod
    def load_data_from_folder(folder, data_names, color_mode="rgb", fun_traitement=lambda x: x):
        """
        Load all image file names from a given folder, matching by base name and any valid extension.
        Args:
            folder (str): Path to the folder containing images.
            data_names (list): List of image file names to load (base names or with any extension).
            color_mode (str): Color mode for loading images ('rgb' or 'grayscale').
            fun_traitement (function): Function to apply to the loaded image data.
        """
        img_data = []
        valid_exts = ('.png', '.jpg', '.jpeg', '.JPG', '.JPEG')
        files_in_folder = os.listdir(folder)
        for f in data_names:
            base, _ = os.path.splitext(f)
            found = False
            for ext in valid_exts:
                candidate = base + ext
                if candidate in files_in_folder:
                    path = os.path.join(folder, candidate)
                    img_data.append(fun_traitement(img_to_array(load_img(path, color_mode=color_mode))))
                    print(f"* Loaded image: {candidate} from {folder}")
                    print(f"    Image shape: {img_data[-1].shape}")
                    print(f"    Image min: {img_data[-1].min()}, max: {img_data[-1].max()}")
                    found = True
                    break
            if not found:
                print(f"! Warning: No image found for base name '{base}' in {folder} with extensions {valid_exts}")
        return img_data

    @staticmethod
    def are_same_size(data_list1, data_list2):
        """
        Check if all images in two lists have the same dimensions.
        Args:
            data_list1 (list): First list of image arrays.
            data_list2 (list): Second list of image arrays.
        Returns:
            bool: True if all images have the same xy dimensions, False otherwise.
        """
        if len(data_list1) != len(data_list2):
            print(f"* Different number of images: {len(data_list1)} and {len(data_list2)}")
            return False
        for img1, img2 in zip(data_list1, data_list2):
            if img1.shape[:2] != img2.shape[:2]:
                return False
        return True
    
    @staticmethod
    def add_padding(data_list, reference_list):
        """
        Add padding to images in data_list to match the dimensions of images in reference_list.
        Args:
            data_list (list): List of image arrays to be padded.
            reference_list (list): List of reference image arrays for target dimensions.
        Returns:
            list: List of padded image arrays.
        """
        padded_data = []
        for img, ref in zip(data_list, reference_list):
            if img.shape[:2] == ref.shape[:2]:
                padded_data.append(img)
                continue
            h_diff = max(0, ref.shape[0] - img.shape[0])
            w_diff = max(0, ref.shape[1] - img.shape[1])
            # print(f"* Padding image from {img.shape} to {ref.shape} (h_diff={h_diff}, w_diff={w_diff})")
            top = h_diff // 2
            bottom = h_diff - top
            left = w_diff // 2
            right = w_diff - left
            padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT)
            padded_data.append(padded_img)
        return padded_data

    @staticmethod
    def get_patch_coords_and_counts(data_names, patch_size, gt_folder, patch_ratio):
        """
        Get the coordinates of patches from the ground truth image, matching by base name and any valid extension.
        Args:
            data_names (list): List of image file names.
            patch_size (int): Size of the patches to be extracted.
            gt_folder (str): Path to the folder containing the ground truth images.
            patch_ratio (float): Ratio of the image to be used for patch extraction. In (0, 1]
        Returns:
            Tuple[List[np.ndarray], List[int]]: A tuple containing the coordinates of the patches and their counts by image.
        """
        coords = []
        counts = []
        valid_exts = ('.png', '.jpg', '.jpeg', '.JPG', '.JPEG')
        files_in_folder = os.listdir(gt_folder)
        for name in data_names:
            base, _ = os.path.splitext(name)
            found = False
            for ext in valid_exts:
                candidate = base + ext
                if candidate in files_in_folder:
                    gt_path = os.path.join(gt_folder, candidate)
                    gt = load_img(gt_path, color_mode="grayscale")
                    gt_arr = np.array(gt)
                    arr = (1 - gt_arr / 255.0).astype(np.uint8)
                    nb_iter = int(patch_size * patch_ratio)

                    # Dilation - we could change structure connectivity to match the data augmentation
                    arr = ndi.binary_dilation(arr, iterations=nb_iter)
                    coo = np.argwhere(arr)
                    coo = coo - (patch_size // 2, patch_size // 2)
                    # Filter out-of-bounds coordinates
                    coo = coo[(coo[:, 0] >= 0) & (coo[:, 1] >= 0)]
                    coo = coo[(coo[:, 0] <= gt_arr.shape[0] - patch_size) & (coo[:, 1] <= gt_arr.shape[1] - patch_size)]

                    coords.append(coo)
                    counts.append(len(coo))
                    print(f"* Found {len(coo)} valid patches in {candidate}")
                    found = True
                    break
            if not found:
                print(f"! Warning: No ground truth image found for base name '{base}' in {gt_folder} with extensions {valid_exts}")
                coords.append(np.empty((0, 2), dtype=int))
                counts.append(0)
        return coords, counts