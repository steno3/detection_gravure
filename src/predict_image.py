import sys
import argparse
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from tensorflow.keras.preprocessing import image
from tqdm import tqdm


def get_normalization_function(norm_type):
    """Retourne la fonction de normalisation correspondante."""
    if norm_type == 'standard':
        return lambda x: (x - 127.5) / 127.5
    elif norm_type == 'simple':
        return lambda x: x / 255.0
    elif norm_type == 'none':
        return lambda x: x
    else:
        raise ValueError(f"Type de normalisation inconnu: {norm_type}. Choix possibles: standard, simple, none")

# Losses
def dice_loss(y_true, y_pred, smooth=1e-4):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(y_true*y_pred, axis=[1,2,3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1,2,3])

    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - tf.reduce_mean(dice)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice

def focal_dice_loss(y_true, y_pred, lambda_bfce=1.0):
    bfce = tf.keras.losses.BinaryFocalCrossentropy()(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return lambda_bfce * bfce + dice

# Metrics
def f1_score_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return f1

# Main
def main(img_path, model_path, mask_path=None, stride=256, patch_size=512, fun_img=None, output_dir='out', mask_threshold=127):
    # Charge le modèle Keras sauvegardé au format .h5
    try:
        model = tf.keras.models.load_model(model_path, custom_objects={'bce_dice_loss':bce_dice_loss, 'focal_dice_loss': focal_dice_loss, 'dice_loss': dice_loss, 'f1_score_metric': f1_score_metric})
        target_size = model.input_shape[1:3]
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        sys.exit(1)

    # Vérifie si le chemin d'entrée est un dossier ou un fichier
    if os.path.isdir(img_path):
        img_files = [os.path.join(img_path, f) for f in os.listdir(img_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
    else:
        img_files = [img_path]

    for img_file in img_files:
        print(f"Traitement de l'image : {img_file} ({img_files.index(img_file)+1} sur {len(img_files)})")
        process_image(img_file, model, mask_path, stride, patch_size, fun_img, output_dir, mask_threshold)


def process_image(img_path, model, mask_path=None, stride=256, patch_size=512, fun_img=None, output_dir='out', mask_threshold=127):
    # Charger l'image d'entrée complète
    orig_img = image.load_img(img_path)
    orig_img_array = fun_img(image.img_to_array(orig_img))
    orig_height, orig_width = orig_img_array.shape[:2]

    # Charger le masque si fourni
    if mask_path:
        mask_img = image.load_img(mask_path, color_mode='grayscale', target_size=(orig_height, orig_width))
        mask_array = image.img_to_array(mask_img) > mask_threshold  # seuil binaire
        if mask_array.shape[-1] == 3:
            mask_array = mask_array[..., 0:1]

    # Préparer une image de sortie vide (1 canal pour noir et blanc)
    result_img = np.zeros((orig_height, orig_width, 1), dtype=np.float32)
    count_map = np.zeros((orig_height, orig_width, 1), dtype=np.float32)

    patch_coords = [(y, x) for y in range(0, orig_height, stride) for x in range(0, orig_width, stride)]
    for idx, (y, x) in enumerate(tqdm(patch_coords, desc="Progression", unit="patch")):
        # passer la boucle si le patch est entièrement en dehors du masque
        if mask_path:
            h = min(patch_size, orig_height - y)
            w = min(patch_size, orig_width - x)
            if not np.all(mask_array[y:y+h, x:x+w, 0]):
                continue

        patch = orig_img_array[y:y+patch_size, x:x+patch_size, :]
        pad_h = patch_size - patch.shape[0]
        pad_w = patch_size - patch.shape[1]
        if pad_h > 0 or pad_w > 0:
            patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0,0)), mode='constant')
        patch = np.expand_dims(patch, axis=0)
        pred = model.predict(patch, verbose=0)[0]
        if pred.shape[-1] > 1:
            pred = pred[..., 0:1]
        h = min(patch_size, orig_height - y)
        w = min(patch_size, orig_width - x)
        result_img[y:y+h, x:x+w, 0] += (pred[:h, :w, 0] * 255).astype(np.float32)
        count_map[y:y+h, x:x+w, 0] += 1
    
    count_map[count_map == 0] = 1  # éviter division par zéro
    result_img = result_img / count_map
    result_img = np.clip(result_img, 0, 255).astype(np.uint8)

    # Convertir en image PIL en mode 'L' (noir et blanc)
    im = Image.fromarray(result_img[:, :, 0], mode='L')
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(output_dir, f"{base_name}_result.png")
    im.save(out_path)
    print(f"Image résultat sauvegardée dans : {out_path}")

def parse_args():
    parser = argparse.ArgumentParser(description='Prédire les gravures sur des images en utilisant un modèle entraîné.')
    parser.add_argument('image_path', type=str, help='Chemin vers l\'image ou le dossier contenant les images')
    parser.add_argument('model_path', type=str, help='Chemin vers le modèle .h5')
    parser.add_argument('--mask-path', type=str, default=None, help='Chemin vers le masque (optionnel)')
    parser.add_argument('--stride', type=int, default=256, help='Stride pour le découpage en patchs (défaut: 256)')
    parser.add_argument('--patch-size', type=int, default=512, help='Taille des patchs (défaut: 512)')
    parser.add_argument('--norm', type=str, default='standard', choices=['standard', 'simple', 'none'], 
                        help='Type de normalisation: standard ((x-127.5)/127.5), simple (x/255), none (défaut: standard)')
    parser.add_argument('--output-dir', type=str, default='out', help='Dossier de sortie pour les résultats (défaut: out)')
    parser.add_argument('--mask-threshold', type=int, default=127, help='Seuil pour le masque binaire (défaut: 127)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Obtenir la fonction de normalisation
    fun_img = get_normalization_function(args.norm)
    
    # Appeler la fonction main avec tous les paramètres
    main(
        img_path=args.image_path,
        model_path=args.model_path,
        mask_path=args.mask_path,
        stride=args.stride,
        patch_size=args.patch_size,
        fun_img=fun_img,
        output_dir=args.output_dir,
        mask_threshold=args.mask_threshold
    )
