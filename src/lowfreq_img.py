import numpy as np
from PIL import Image
import scipy.ndimage

def noise3D(size=512, scale=64, intensity=1):
    """    Generate a noise array with 3 channels.
    Args:
        size (int): Size of the image (size x size).
        scale (int): Scale of the noise (pixel-wise). The higher the scale, the smoother the noise.
        intensity (float): Intensity of the noise [0, 1].
    Returns:
        np.ndarray of (size, size, 3) _float32 : A 3D array representing the noise.
    """
    img = np.zeros((size, size, 3), dtype=np.float32)
    # Génère un bruit basse fréquence différent pour chaque canal
    for c in range(3):
        noise = np.random.randn(size, size)
        low_freq_noise = scipy.ndimage.gaussian_filter(noise, sigma=scale)
        low_freq_noise = (low_freq_noise - low_freq_noise.min()) / (low_freq_noise.max() - low_freq_noise.min())
        img[..., c] = low_freq_noise * intensity
    return img

def angle_noise(size=512, scale=64, angle=5):
    """    Generate a low-frequency noise array of 2D.
    Args:
        size (int): Size of the image (size x size).
        scale (int): Scale of the noise (pixel-wise).
        angle (float): Angle of rotation for the noise.
    """
    img = np.zeros((size, size), dtype=np.float32)
    # Génère un bruit basse fréquence différent pour chaque canal
    noise = np.random.randn(size, size)
    low_freq_noise = scipy.ndimage.gaussian_filter(noise, sigma=scale)
    low_freq_noise = (low_freq_noise - low_freq_noise.min()) / (low_freq_noise.max() - low_freq_noise.min())
    img[...] = low_freq_noise * angle * 2 - angle

    return img

def generate_low_freq_noise_image(size=512, scale=64, intensity=1, width=1):
    """    Generate a low-frequency noise array with centered values (width).
    Args:
        size (int): Size of the image (size x size).
        scale (int): Scale of the noise (pixel-wise).
        intensity (float): Intensity of the noise [0, 1].
        width (float): Width of the noise band [0, 1].
    """
    img = noise3D(size=size, scale=scale, intensity=intensity)

    # recadrage dynamique canal par canal dans [127 - d, 128 + d]
    d = int(127 * width)

    min_val = 127 - d
    max_val = 128 + d
    min_img = img.min()
    max_img = img.max()
    img = (img - min_img) / (max_img - min_img) * (max_val - min_val) + min_val

    # Convertit en uint8
    img = img.clip(0, 255).astype(np.uint8)
    return img


def cross_product_matrix(v):
    """
    Constructs the cross-product matrix (also known as the skew-symmetric matrix) 
    for a given 3D vector `v`. The cross-product matrix is a 3x3 matrix that can 
    be used to compute the cross product of `v` with another vector via matrix 
    multiplication.
    Parameters:
        v (...,3): Array of shape (3,) representing a 3D vector.
    Returns:
        Array ...,3,3: A 3x3 skew-symmetric matrix corresponding to the input vector `v`.
    """

    z, u1, u2, u3 = np.broadcast_arrays(*((0,) + tuple(np.moveaxis(v, -1, 0))))
    result = np.stack([
        np.stack([z, -u3,  u2], axis=-1),
        np.stack([ u3, z, -u1], axis=-1),
        np.stack([-u2,  u1, z], axis=-1)
    ], axis=-2)
    return result

def axis_angle_to_matrix(axis, angle):
    """
    Converts an axis-angle representation to a rotation matrix.
    Parameters:
        axis (array-like): A 3D vector representing the axis of rotation. 
                            It should be normalized to have a unit length.
        angle (float or array-like): The angle of rotation in radians. 
                                    Can be a scalar or an array for batch processing.
    Returns:
        numpy.ndarray: A 3x3 rotation matrix (or a batch of 3x3 matrices) 
                        corresponding to the given axis and angle.
    """
    K = cross_product_matrix(axis)
    R = np.eye(3) + np.expand_dims(np.sin(angle), axis=(-1,-2)) * K + np.expand_dims(1 - np.cos(angle), axis=(-1,-2)) * np.matmul(K, K)
    return R


def normal_rotation_noise(normal, scale=64, angle_max=5):
    """ Applies a small rotation noise to the normal map.
    Avoid added noise directional bias.
    Args:
        normal (np.ndarray): The normal map to modify.
        angle (float): The rotation angle in degrees.
        scale (int): The scale of the noise.
    Returns:
        np.ndarray: The modified normal map with each normal rotated by 
    """
    # direction matrix
    n = noise3D(size=normal.shape[0], scale=scale, intensity=1)
    n = n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-6)

    # angle matrix
    a = angle_noise(size=normal.shape[0], scale=scale, angle=angle_max)

    # Rotation
    R = axis_angle_to_matrix(n, np.deg2rad(a))
    # print(normal.shape)
    # print(R[normal.shape[0]//2, normal.shape[1]//2])
    # print(a[normal.shape[0]//2, normal.shape[1]//2])
    # print(n[normal.shape[0]//2, normal.shape[1]//2])

    normal = np.einsum('...ij,...j->...i', R, normal)

    return normal

# Main
if __name__ == "__main__":
    img = generate_low_freq_noise_image()
    im = Image.fromarray(img)
    im.save("lowfreq_noise.png")
    im.show()
    # a = angle_noise()
    # print(a)  # Test angle noise
    # print(np.max(a), np.min(a))
    # print(a.dtype)
    # print(np.mean(a))
    # print(a.shape)
