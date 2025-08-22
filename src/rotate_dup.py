import os
from PIL import Image
import sys
import numpy as np

def rotate_and_save(image_path, output_dir):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    img = Image.open(image_path)
    angles = [0, 90, 180, 270]
    for angle in angles:
        rotated = img.rotate(-angle, expand=True)
        out_name = f"{base_name}_rot{angle}.png"
        rotated.save(os.path.join(output_dir, out_name))

def rotate_and_save_normals(image_path, output_dir):
    # specific function to handle normal maps
    # which are typically in the format of RGB but the rotation need to keep the directions of the vectors coherent
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    img = Image.open(image_path)
    angles = [0, 90, 180, 270]
    # Corrected rotation matrices for each angle
    rotation_matrices = {
        0: np.array([[1, 0], [0, 1]]),
        90: np.array([[0, 1], [-1, 0]]),
        180: np.array([[-1, 0], [0, -1]]),
        270: np.array([[0, -1], [1, 0]])
    }
    for angle in angles:
        rotated = img.rotate(-angle, expand=True)

        matrix = rotation_matrices[angle]
        arr = np.array(rotated).astype(np.float32)
        # Normalize RGB from [0,255] to [-1,1]
        normals = arr / 255.0 * 2 - 1

        # Apply rotation to X (R) and Y (G) channels
        xy = normals[..., :2]
        xy_rot = xy @ matrix.T
        normals[..., :2] = xy_rot

        # Re-normalize to [0,255]
        normals = ((normals + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        rotated_img = Image.fromarray(normals, mode=img.mode)

        out_name = f"{base_name}_rot{angle}.png"
        rotated_img.save(os.path.join(output_dir, out_name))

def rotate_image_by_angle(image_path, output_dir, angle, normals=False):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    img = Image.open(image_path)
    out_name = f"{base_name}_rot{angle}.png"
    
    if normals:
        # Rotation matrix for arbitrary angle
        rad = np.deg2rad(angle)
        matrix = np.array([
            [np.cos(rad), np.sin(rad)],
            [-np.sin(rad), np.cos(rad)]
        ])
        arr = np.array(img).astype(np.float32)
        normals_arr = arr / 255.0 * 2 - 1
        xy = normals_arr[..., :2]
        xy_rot = xy @ matrix.T
        normals_arr[..., :2] = xy_rot
        normals_arr = ((normals_arr + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        rotated = Image.fromarray(normals_arr, mode=img.mode)
    else:
        rotated = img.copy()

    rotated_img = rotated.rotate(-angle, expand=True)
    # Resize rotated image back to original size if dimensions changed
    if rotated_img.size != img.size:
        # Crop the center of the rotated image to match the original size
        w, h = rotated_img.size
        orig_w, orig_h = img.size
        left = (w - orig_w) // 2
        top = (h - orig_h) // 2
        right = left + orig_w
        bottom = top + orig_h
        rotated_img = rotated_img.crop((left, top, right, bottom))

    rotated_img.save(os.path.join(output_dir, out_name))

def rotate_matrix(arr, angle, normals=False):
    """ Matrix input version of function above.
    Rotate an image or normal map by a given angle.
    If normals=True, it applies a rotation matrix to the normal vectors.
    """
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    # Handle grayscale images with shape (H, W, 1) or (1, 1, 1)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr.squeeze(axis=2)
    img = Image.fromarray(arr)
    
    if normals:
        # Rotation matrix for arbitrary angle
        rad = np.deg2rad(angle)
        matrix = np.array([
            [np.cos(rad), np.sin(rad)],
            [-np.sin(rad), np.cos(rad)]
        ])
        normals_arr = arr / 255.0 * 2 - 1
        xy = normals_arr[..., :2]
        xy_rot = xy @ matrix.T
        normals_arr[..., :2] = xy_rot
        normals_arr = ((normals_arr + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        rotated = Image.fromarray(normals_arr) #, mode=img.mode)
    else:
        rotated = img.copy()

    rotated_img = rotated.rotate(-angle, expand=True)
    # Resize rotated image back to original size if dimensions changed
    if rotated_img.size != img.size:
        # Crop the center of the rotated image to match the original size
        w, h = rotated_img.size
        orig_w, orig_h = img.size
        left = (w - orig_w) // 2
        top = (h - orig_h) // 2
        right = left + orig_w
        bottom = top + orig_h
        rotated_img = rotated_img.crop((left, top, right, bottom))

    return np.array(rotated_img, dtype=np.float32)

def flip_normal(arr):
    """ Flip the normal map horizontally.
    mirroring the x-axis of the array.
    """
    # if arr.ndim == 3 and arr.shape[2] == 3:
    #     arr = arr[..., :2]
    arr = np.flip(arr, axis=1)

    # Mirror the normal vectors with the vertical-up (yz) plane
    arr[..., 0] = -arr[..., 0]

    return arr


def process_images(input_dir, output_dir, normals=False):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            image_path = os.path.join(input_dir, fname)
            if normals:
                rotate_and_save_normals(image_path, output_dir)
            else:
                rotate_and_save(image_path, output_dir)

def process_images_stereo(input_dir, output_dir, normals=False, teta_step=10):
    """
    Process images for each angle between 0 and 360 degrees in teta_step steps degrees.
    """
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            image_path = os.path.join(input_dir, fname)
            for angle in range(0, 360, teta_step):
                rotate_image_by_angle(image_path, output_dir, angle, normals=normals)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rotate_dup.py <is_normals> <input_dir> [output_dir]")
        sys.exit(1)
    is_normals = sys.argv[1].lower() == 'true'
    if not is_normals and sys.argv[1].lower() != 'false':
        print("First argument must be 'true' or 'false' to indicate if processing normals.")
        sys.exit(1)
    input_dir = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "out"

    # process_images(input_dir, output_dir, normals=is_normals)
    process_images_stereo(input_dir, output_dir, normals=is_normals)