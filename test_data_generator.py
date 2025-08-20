import cv2
from tensorflow.keras.preprocessing.image import array_to_img

from src.dataGenerator import DataGenerator

def main():
    data_names = ["os_0.png"]
    batch_size = 2
    epoch_size = 10
    patch_size = 512
    img_folder = "./dataset/data_nino_gen/normale"
    groundtruth_folder = "./dataset/data_nino_gen/gravure"
    mask_folder = "./dataset/data_nino_gen/masque"
    pas_rotation = 10
    noise_scale = 64
    noise_intensity = 0.2
    noise_width = 0.4

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

    # Convert images to uint8 for saving
    img_to_save = (first_img.numpy() * 255).astype('uint8')
    mask_to_save = (first_mask.numpy() * 255).astype('uint8')
    cv2.imwrite('first_img.png', cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR))
    cv2.imwrite('first_mask.png', mask_to_save)
    print("Images saved as first_img.png and first_mask.png")

if __name__ == "__main__":
    main()
