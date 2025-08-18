import cv2
import numpy as np
from matplotlib import cm
import os
import matplotlib.pyplot as plt

normal_map_path = './omoplate_1_unimsps.png'
output_murs_path = './img_a_pred/omoplate.png'

def load_normal_map(path):
    # Charge l'image et la convertit en float32
    normal_map = cv2.imread(path, cv2.IMREAD_COLOR)
    normal_map = cv2.cvtColor(normal_map, cv2.COLOR_BGR2RGB)
    normal_map = normal_map.astype(np.float32) / 255.0
    # Remap de [0,1] à [-1,1]
    normal_map = normal_map * 2.0 - 1.0
    return normal_map

def integrate_normals(normals):
    h, w, _ = normals.shape
    dzdx = -normals[..., 0] / (normals[..., 2] + 1e-8)
    dzdy = -normals[..., 1] / (normals[..., 2] + 1e-8)
    # Intégration par méthode de Frankot-Chellappa
    fx = np.fft.fft2(dzdx)
    fy = np.fft.fft2(dzdy)
    y, x = np.mgrid[0:h, 0:w]
    wx = 2 * np.pi * (x - w//2) / w
    wy = 2 * np.pi * (y - h//2) / h
    wx = np.fft.fftshift(wx)
    wy = np.fft.fftshift(wy)
    denom = wx**2 + wy**2
    denom[denom == 0] = 1  # éviter division par zéro
    F = (-1j * wx * fx - 1j * wy * fy) / denom
    F[0,0] = 0  # composante DC à zéro
    height_map = np.real(np.fft.ifft2(F))
    return height_map

def render_height_map(height_map, light_dir=np.array([1,1,1])):
    # Normalisation du vecteur lumière
    light_dir = light_dir / np.linalg.norm(light_dir)
    # Calcul des normales à partir du height_map
    dzdx = np.gradient(height_map, axis=1)
    dzdy = np.gradient(height_map, axis=0)
    normals = np.dstack((-dzdx, -dzdy, np.ones_like(height_map)))
    normals /= np.linalg.norm(normals, axis=2, keepdims=True)
    # Calcul de l'intensité lumineuse
    intensity = np.clip(np.dot(normals, light_dir), 0, 1)
    return intensity

def random_light_direction():
    # random x,y vecteur avec une norme = 2 pour un éclairage rasant
    x = np.random.uniform(-1, 1)
    y = np.random.uniform(-1, 1)
    norm = np.sqrt(x**2 + y**2 + 1)
    x = x / norm
    y = y / norm
    z = 1 / norm
    if x == 0 and y == 0:
        x = 1
        
    return np.array([x, y, z])  # z = 1 pour un éclairage rasant

## Processing Functions
def process_image(input_path, output_path):
    """
    Process the normal map and save the resulting shaded image.
    """
    normal_map = load_normal_map(input_path)
    height_map = integrate_normals(normal_map)
    light_dir = random_light_direction()
    print(f"Using light direction: {light_dir}")
    shaded = render_height_map(height_map, light_dir=light_dir)
    
    plt.imsave(output_path, shaded, cmap=cm.gray)

def process_image_stereo(input_path, output_path, nb_images=10, z_height=0.7):
    """
    Process the normal map and save multiple shaded images with light directions from angles in a circle around the image.
    nb_images: number of images to generate with different light directions.
    z_height: height of the light source above the image plane (fixed).
    """
    normal_map = load_normal_map(input_path)
    height_map = integrate_normals(normal_map)
    
    for i in range(nb_images):
        angle = 2 * np.pi * i / nb_images
        x = np.cos(angle)
        y = np.sin(angle)
        norm = np.sqrt(x**2 + y**2 + z_height**2)
        x /= norm
        y /= norm
        light_dir = np.array([x, y, z_height])
        #print(f"Using light direction: {light_dir}")
        shaded = render_height_map(height_map, light_dir=light_dir)
        output_file = f"{output_path[:-4]}_{i+1}.png"
        plt.imsave(output_file, shaded, cmap=cm.gray)


## Main
if __name__ == "__main__":
    # Remplacez 'normal_map.png' par le chemin de votre image de champ de normales
    shaded = process_image_stereo(normal_map_path, output_murs_path)
    print(f"Shaded image saved to {output_murs_path}")