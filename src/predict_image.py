import sys
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from tensorflow.keras.preprocessing import image
from tqdm import tqdm

# hyperparamètres
stride = 256  # recouvrement de 50%
patch_size = 512
fun_img = lambda x: (x - 127.5) / 127.5 # Normalisation - doit être identique à l'entraînement

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
def main(img_path, model_path):
    # Charge le modèle Keras sauvegardé au format .h5
    try:
        model = tf.keras.models.load_model(model_path, custom_objects={'bce_dice_loss':bce_dice_loss, 'focal_dice_loss': focal_dice_loss, 'dice_loss': dice_loss, 'f1_score_metric': f1_score_metric})
        target_size = model.input_shape[1:3]
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        sys.exit(1)

    # Charger l'image d'entrée complète
    orig_img = image.load_img(img_path)
    orig_img_array = fun_img(image.img_to_array(orig_img))
    orig_height, orig_width = orig_img_array.shape[:2]

    # Préparer une image de sortie vide (1 canal pour noir et blanc)
    result_img = np.zeros((orig_height, orig_width, 1), dtype=np.float32)
    count_map = np.zeros((orig_height, orig_width, 1), dtype=np.float32)

    patch_coords = [(y, x) for y in range(0, orig_height, stride) for x in range(0, orig_width, stride)]
    for idx, (y, x) in enumerate(tqdm(patch_coords, desc="Progression", unit="patch")):
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
    os.makedirs('out', exist_ok=True)
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join('out', f"{base_name}_result.png")
    im.save(out_path)
    print(f"Image résultat sauvegardée dans : {out_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python predict_image.py <image_path> <model_path.h5>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])