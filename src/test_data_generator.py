import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import array_to_img

from src.dataGenerator import DataGenerator

def main():
    data_names = ["general.png"]
    batch_size = 2
    epoch_size = 10
    patch_size = 512
    img_folder = "./mur"
    groundtruth_folder = "./gravure"
    mask_folder = "./masque"
    pas_rotation = 10
    noise_scale = 64
    noise_intensity = 1
    noise_width = 0.5

    gen = DataGenerator(
        data_names, 
        batch_size, 
        epoch_size, 
        patch_size, 
        img_folder, 
        groundtruth_folder, 
        mask_folder, 
        pas_rotation=pas_rotation, 
        noise_scale=noise_scale,
        noise_intensity=noise_intensity,
        noise_width=noise_width
    )
    # Tester le générateur
    # for i in range(len(gen)):
    #     X, y = gen[i]
    #     print(f"Batch {i}:")
    #     print("  X shape:", X.shape)
    #     print("  y shape:", y.shape)
    
    # Afficher la premiere image et son masque
    first_img_batch, first_mask_batch = gen[0]
    first_img = (first_img_batch[0] + 1) / 2  # Revert normalization
    first_mask = first_mask_batch[0] # Revert normalization

    plt.subplot(1, 2, 1)
    plt.title("Image")
    plt.imshow(first_img)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title("Mask")
    plt.imshow(first_mask, cmap='gray')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
