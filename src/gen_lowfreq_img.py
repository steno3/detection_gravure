import numpy as np
from PIL import Image

def generate_low_freq_noise_image(size=512, scale=64, intensity=1, width=1):
    """    Generate a low-frequency noise image.
    Args:
        size (int): Size of the image (size x size).
        scale (int): Scale of the noise (pixel-wise).
        intensity (float): Intensity of the noise [0, 1].
        width (float): Width of the noise band [0, 1].
    """
    d = int(127 * width)
    # Crée une image vide
    img = np.zeros((size, size, 3), dtype=np.float32)
    import scipy.ndimage
    # Génère un bruit basse fréquence différent pour chaque canal
    for c in range(3):
        noise = np.random.randn(size, size)
        low_freq_noise = scipy.ndimage.gaussian_filter(noise, sigma=scale)
        low_freq_noise = (low_freq_noise - low_freq_noise.min()) / (low_freq_noise.max() - low_freq_noise.min())
        img[..., c] = low_freq_noise * intensity

    # recadrage dynamique canal par canal dans [127 - d, 127 + d]
    min_val = 127 - d
    max_val = 127 + d
    min_img = img.min()
    max_img = img.max()
    img = (img - min_img) / (max_img - min_img) * (max_val - min_val) + min_val

    # Convertit en uint8
    img = img.clip(0, 255).astype(np.uint8)
    return img


if __name__ == "__main__":
    img = generate_low_freq_noise_image()
    im = Image.fromarray(img)
    im.save("lowfreq_noise.png")
    im.show()
