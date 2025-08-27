import sys
import numpy as np
from PIL import Image
from numba import njit, prange

@njit(parallel=True)
def find_white_zones(mask_arr, patch_size):
    """
    Returns a list of coordinates where white patches of size patch_size
    can be found in the mask array. 
    """
    h, w = mask_arr.shape
    black = (mask_arr != 255)
    n_rows = h - patch_size + 1
    n_cols = w - patch_size + 1
    # Each thread will store up to n_cols results (worst case)
    thread_results = np.empty((n_rows, n_cols, 2), dtype=np.int32)
    thread_counts = np.zeros(n_rows, dtype=np.int32)
    for y in prange(n_rows):
        local_count = 0
        for x in range(n_cols):
            all_white = True
            # Check top and bottom borders
            for j in range(patch_size):
                if black[y, x+j] or black[y+patch_size-1, x+j]:
                    all_white = False
                    break
            if not all_white:
                continue
            # Check left and right borders
            for i in range(1, patch_size-1):
                if black[y+i, x] or black[y+i, x+patch_size-1]:
                    all_white = False
                    break
            if all_white:
                thread_results[y, local_count, 0] = x
                thread_results[y, local_count, 1] = y
                local_count += 1
        thread_counts[y] = local_count
    # Now concatenate all results
    total = np.sum(thread_counts)
    coords = np.empty((total, 2), dtype=np.int32)
    idx = 0
    for y in range(n_rows):
        for i in range(thread_counts[y]):
            coords[idx, 0] = thread_results[y, i, 0]
            coords[idx, 1] = thread_results[y, i, 1]
            idx += 1
    return coords, total

def _verify_white_zones(mask_arr, patch_size):
    """
    OUTPUT an image with position of all white patches marked gray.
    """
    zones, count = find_white_zones(mask_arr, patch_size)
    if count == 0:
        print("Aucune zone blanche suffisante trouvée dans le masque.")
        return

    for i in range(count):
        x, y = zones[i]
        mask_arr[y, x] = 128
    mask_img = Image.fromarray(mask_arr)
    mask_img.save("mask_with_zones.png")
    print("Image de masque avec zones sauvegardée sous 'mask_with_zones.png'.")

def main(image_path, mask_path, output_path, patch_size=512):
    image = Image.open(image_path)
    mask = Image.open(mask_path).convert('L')
    image_arr = np.array(image)
    mask_arr = np.array(mask)
    # Convert mask to binary if it has more than one channel
    if mask_arr.max() > 1:
        mask_arr = (mask_arr > 127).astype(np.uint8) * 255
    zones, count = find_white_zones(mask_arr, patch_size)
    if count == 0:
        print("Aucune zone blanche suffisante trouvée dans le masque.")
        return
    else:
        print(f"{count} zones blanches trouvées dans le masque.")
    # Randomly select a zone
    i = np.random.randint(0, count)
    x, y = zones[i]
    patch = image_arr[y:y+patch_size, x:x+patch_size]
    patch_img = Image.fromarray(patch)
    patch_img.save(output_path)
    print(f"Patch sauvegardé à {output_path} (x={x}, y={y})")

    # _verify_white_zones(mask_arr, patch_size)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python select_patch.py <image_path> <mask_path> <output_path> [patch_size]")
        sys.exit(1)
    image_path = sys.argv[1]
    mask_path = sys.argv[2]
    output_path = sys.argv[3]
    patch_size = int(sys.argv[4]) if len(sys.argv) > 4 else 512
    main(image_path, mask_path, output_path, patch_size)
