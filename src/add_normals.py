import numpy as np
import sys
import cv2
import os

import lowfreq_img as lfnoise

def load_normals_png(file_path):
    # Charge un champ de normales depuis un fichier PNG (format RGB, valeurs [0,255] -> [-1,1])
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Fichier introuvable: {file_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    normals = img.astype(np.float32) / 255.0  # [0,1]
    normals = normals * 2.0 - 1.0  # [-1,1]
    return normals

def save_normals_png(normals, file_path):
    # Sauvegarde un champ de normales dans un fichier PNG (remap [-1,1] -> [0,255])
    normals = (normals + 1.0) / 2.0  # [-1,1] -> [0,1]
    normals = np.clip(normals * 255.0, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(normals, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_path, img)

def add_normals(normals1, normals2):
    """ Additionne deux champs de normales pixel par pixel et normalise le résultat.
    Args:
        normals1 (np.ndarray): Premier champ de normales.
        normals2 (np.ndarray): Deuxième champ de normales.
    """
    h1, w1, _ = normals1.shape
    h2, w2, _ = normals2.shape
    if h2 > h1 or w2 > w1:
        normals2 = normals2[:h1, :w1, :]
    elif h2 < h1 or w2 < w1:
        normals1 = normals1[:h2, :w2, :]
    summed = normals1 + normals2
    norms = np.linalg.norm(summed, axis=2, keepdims=True)
    norms[norms == 0] = 1
    return np.clip(summed / norms, -1, 1)

def add_noise_to_normals(normals, size=512, scale=64, intensity=1, width=1):
    """ Ajoute du bruit basse fréquence à un champ de normales.
    Args:
        normals (np.ndarray): Champ de normales à traiter.
        size (int): Taille de l'image (size x size).
        scale (int): Échelle du bruit (pixel-wise).
        intensity (float): Intensité du bruit [0, 1].
        width (float): Largeur de la bande de bruit [0, 1].
    Returns:
        np.ndarray: Champ de normales avec bruit ajouté.
    """
    noise = lfnoise.generate_low_freq_noise_image(size=size, scale=scale, intensity=intensity, width=width)
    nn = noise.astype(np.float32) / 255.0  # [0,1]
    nn = nn * 2.0 - 1.0  # [-1,1]
    return add_normals(normals, nn)

def process_folder(input_fldr_normals1, input_fldr_normals2, output_fldr, nb_images=10):
    # Traite tous les fichiers dans le dossier input_fldr_normals1
    # pour chaque normals1 additionne tous les fichiers de normals2
    # nb_images limite le nombre d'images de normals2 à additionner
    os.makedirs(output_fldr, exist_ok=True)
    files1 = [f for f in os.listdir(input_fldr_normals1) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    files2 = [f for f in os.listdir(input_fldr_normals2) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for file1 in files1:
        normals1 = load_normals_png(os.path.join(input_fldr_normals1, file1))
        for i in range(nb_images):
            file2 = np.random.choice(files2)
            normals2 = load_normals_png(os.path.join(input_fldr_normals2, file2))
            result = add_normals(normals1, normals2)
            output_file = f"{os.path.splitext(file1)[0]}_{i+1}.png"
            save_normals_png(result, os.path.join(output_fldr, output_file))
        print(f"Traitement de {file1} terminé")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python add_normales.py normals1.png normals2.png output_folder")
        sys.exit(1)

    # process_folder(sys.argv[1], sys.argv[2], sys.argv[3])
    ret = add_noise_to_normals(load_normals_png(sys.argv[1]), size=512, scale=64, intensity=1, width=1)
    save_normals_png(ret, sys.argv[3])

    print(f"Addition terminée. Résultat sauvegardé dans {sys.argv[3]}")