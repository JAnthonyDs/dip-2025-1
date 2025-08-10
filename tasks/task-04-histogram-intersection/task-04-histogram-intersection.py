import numpy as np

def compute_histogram_intersection(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute the histogram intersection similarity score between two grayscale images.

    This function calculates the similarity between the grayscale intensity 
    distributions of two images by computing the intersection of their 
    normalized 256-bin histograms.

    The histogram intersection is defined as the sum of the minimum values 
    in each corresponding bin of the two normalized histograms. The result 
    ranges from 0.0 (no overlap) to 1.0 (identical histograms).

    Parameters:
        img1 (np.ndarray): First input image as a 2D NumPy array (grayscale).
        img2 (np.ndarray): Second input image as a 2D NumPy array (grayscale).

    Returns:
        float: Histogram intersection score in the range [0.0, 1.0].

    Raises:
        ValueError: If either input is not a 2D array (i.e., not grayscale).
    """    
    if img1.ndim != 2 or img2.ndim != 2:
        raise ValueError("Both input images must be 2D grayscale arrays.")

    ### START CODE HERE ###
    # Histograms (256 bins) normalizados no intervalo [0, 256)
    counts1, _ = np.histogram(img1, bins=256, range=(0, 256))
    counts2, _ = np.histogram(img2, bins=256, range=(0, 256))

    h1 = counts1.astype(np.float64)
    h2 = counts2.astype(np.float64)

    s1 = h1.sum()
    s2 = h2.sum()
    if s1 > 0:
        h1 /= s1
    if s2 > 0:
        h2 /= s2

    intersection = float(np.minimum(h1, h2).sum())
    ### END CODE HERE ###


    return float(intersection)
