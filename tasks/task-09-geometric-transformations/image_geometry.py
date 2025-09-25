# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale width by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""

import numpy as np

import numpy as np

def apply_geometric_transformations(img: np.ndarray) -> dict:
    height, width = img.shape

    translated_img = np.zeros_like(img)
    dx, dy = width // 4, height // 4
    translated_img[dy:, dx:] = img[:height - dy, :width - dx]

    rotated_img = np.rot90(img, k=-1)

    new_width = int(width * 1.5)
    x_indices = np.arange(new_width)
    original_x_indices = (x_indices / 1.5).astype(int)
    stretched_img = img[:, original_x_indices]

    mirrored_img = img[:, ::-1]

    y_coords, x_coords = np.indices((height, width))
    center_y, center_x = (height - 1) / 2.0, (width - 1) / 2.0

    x_from_center = x_coords - center_x
    y_from_center = y_coords - center_y
    
    radius = np.sqrt(x_from_center**2 + y_from_center**2)
    
    max_radius = np.sqrt(center_x**2 + center_y**2)
    radius_normalized = radius / max_radius
    
    k = 0.5
    
    distortion_factor = 1 + k * radius_normalized**2
    src_x = (center_x + x_from_center / distortion_factor).round().astype(int)
    src_y = (center_y + y_from_center / distortion_factor).round().astype(int)
    
    src_x = np.clip(src_x, 0, width - 1)
    src_y = np.clip(src_y, 0, height - 1)

    distorted_img = img[src_y, src_x]

    return {
        "translated": translated_img,
        "rotated": rotated_img,
        "stretched": stretched_img,
        "mirrored": mirrored_img,
        "distorted": distorted_img
    }