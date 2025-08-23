# image_similarity_solution.py
# REFERENCE SOLUTION — FOR INSTRUCTOR USE

"""
Image Similarity Metrics (NumPy-only)

This module implements four classic full-reference image similarity metrics for
grayscale, normalized images (values in [0, 1]):

1) Mean Squared Error (MSE)
2) Peak Signal-to-Noise Ratio (PSNR)
3) Structural Similarity Index (SSIM) — simplified global version
4) Normalized Pearson Correlation Coefficient (NPCC)

All metrics are implemented with NumPy only (no OpenCV/scikit-image/etc.).
Use `compare_images(i1, i2)` as the main entry point.

Notes
-----
- SSIM here is a *global* formulation (single value for the whole image) that
  uses means/variances/covariance over the entire image. It is not the
  windowed, perceptual version popularized in the original SSIM paper.
- Constants C1 and C2 default to 1e-8 for numerical stability, matching the
  provided README examples.
"""

from __future__ import annotations
import numpy as np


def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:
    """
    Compare two grayscale images using different similarity metrics.

    Parameters
    ----------
    i1 : np.ndarray
        First grayscale image as a 2D array of shape (H, W), values in [0, 1].
    i2 : np.ndarray
        Second grayscale image as a 2D array of shape (H, W), values in [0, 1].

    Returns
    -------
    dict
        Dictionary with:
            - "mse"  : float
            - "psnr" : float (dB; np.inf if images are identical)
            - "ssim" : float (simplified global SSIM)
            - "npcc" : float (Pearson correlation in [-1, 1])

    Raises
    ------
    ValueError
        If inputs are not 2D arrays with the same shape.
    """
    _validate_images(i1, i2)

    # Cast to float64 for numerical stability across all metrics
    i1 = i1.astype(np.float64, copy=False)
    i2 = i2.astype(np.float64, copy=False)

    mse_val = _mse(i1, i2)
    psnr_val = _psnr(i1, i2, data_range=1.0)  # images are normalized
    ssim_val = _ssim(i1, i2, C1=1e-8, C2=1e-8)
    npcc_val = _npcc(i1, i2)

    return {
        "mse": float(mse_val),
        "psnr": float(psnr_val),
        "ssim": float(ssim_val),
        "npcc": float(npcc_val),
    }


# ----------------------------- Helpers --------------------------------- #

def _validate_images(i1: np.ndarray, i2: np.ndarray) -> None:
    """
    Validate that both images are 2D ndarrays with identical shapes.

    Parameters
    ----------
    i1, i2 : np.ndarray
        Images to validate.

    Raises
    ------
    ValueError
        If dimensionality or shapes are invalid.
    """
    if not (isinstance(i1, np.ndarray) and isinstance(i2, np.ndarray)):
        raise ValueError("Inputs must be NumPy arrays.")
    if i1.ndim != 2 or i2.ndim != 2:
        raise ValueError("Inputs must be 2D arrays representing grayscale images.")
    if i1.shape != i2.shape:
        raise ValueError(f"Images must have the same shape. Got {i1.shape} vs {i2.shape}.")


def _mse(i1: np.ndarray, i2: np.ndarray) -> float:
    """
    Mean Squared Error (MSE).

    MSE = mean( (i1 - i2)^2 )

    Parameters
    ----------
    i1, i2 : np.ndarray
        Grayscale images with identical shapes.

    Returns
    -------
    float
        Mean squared error.
    """
    ### START CODE HERE ###
    ### TODO
    mse = np.mean(np.square(i1 - i2))
    ### END CODE HERE ###

    return mse


def _psnr(i1: np.ndarray, i2: np.ndarray, data_range: float = 1.0) -> float:
    """
    Peak Signal-to-Noise Ratio (PSNR) in dB.

    PSNR = 10 * log10( (L^2) / MSE ), where L = data_range (peak value).
    If MSE == 0, returns np.inf.

    Parameters
    ----------
    i1, i2 : np.ndarray
        Grayscale images with identical shapes.
    data_range : float, optional
        Peak possible value L for the image data. Default is 1.0 for normalized inputs.

    Returns
    -------
    float
        PSNR in decibels (dB), or np.inf if images are identical.
    """
    ### START CODE HERE ###
    ### TODO

    # Primeiro, calcula o MSE
    mse_val = _mse(i1, i2)

    # Se o MSE for 0, as imagens são idênticas e o PSNR é infinito.
    if mse_val == 0:
        return np.inf

    # Aplica a fórmula do PSNR.
    psnr = 10 * np.log10((data_range**2) / mse_val)
    ### END CODE HERE ###

    return psnr


def _ssim(i1: np.ndarray, i2: np.ndarray, *, C1: float = 1e-8, C2: float = 1e-8) -> float:
    """
    Simplified global Structural Similarity Index (SSIM).

    SSIM = ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) /
           ((mu1^2 + mu2^2 + C1) * (sigma1^2 + sigma2^2 + C2))

    This is a *global* (single-value) SSIM computed over the whole image,
    not the windowed version used in many libraries.

    Parameters
    ----------
    i1, i2 : np.ndarray
        Grayscale images with identical shapes, values in [0, 1].
    C1, C2 : float, optional
        Small positive constants to stabilize divisions. Defaults (1e-8) match the README.

    Returns
    -------
    float
        SSIM in approximately [-1, 1] (often near [0, 1] for natural images).
    """
    ### START CODE HERE ###
    ### TODO

    mu1 = np.mean(i1)
    mu2 = np.mean(i2)

    var1 = np.var(i1)
    var2 = np.var(i2)

    # Calcula a covariância entre as duas imagens.
    # É preciso achatar os arrays para 1D para a função np.cov.
    # O valor que queremos está na posição [0, 1] da matriz de covariância.
    cov12 = np.cov(i1.ravel(), i2.ravel())[0, 1]

    # Monta o numerador e o denominador da fórmula SSIM.
    numerator = (2 * mu1 * mu2 + C1) * (2 * cov12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (var1 + var2 + C2)

    ssim = numerator / denominator
    ### END CODE HERE ###

    return ssim


def _npcc(i1: np.ndarray, i2: np.ndarray) -> float:
    """
    Normalized Pearson Correlation Coefficient (NPCC).

    NPCC = sum( (i1 - mu1) * (i2 - mu2) ) /
           sqrt( sum( (i1 - mu1)^2 ) * sum( (i2 - mu2)^2 ) )

    Parameters
    ----------
    i1, i2 : np.ndarray
        Grayscale images with identical shapes.

    Returns
    -------
    float
        Correlation in [-1, 1]. Returns:
          - 1.0 if images are exactly identical constants,
          - 0.0 if correlation is undefined due to zero variance in at least one image
            (and images are not identical).
    """

    ### START CODE HERE ###
    mu1 = np.mean(i1)
    mu2 = np.mean(i2)
    i1_centered = i1 - mu1
    i2_centered = i2 - mu2

    # O numerador é a soma do produto das imagens centralizadas.
    numerator = np.sum(i1_centered * i2_centered)

    # O denominador é a raiz quadrada do produto da soma dos quadrados de cada imagem centralizada.
    sum_sq_i1 = np.sum(np.square(i1_centered))
    sum_sq_i2 = np.sum(np.square(i2_centered))
    denominator = np.sqrt(sum_sq_i1 * sum_sq_i2)

    # Caso especial: se o denominador for 0, a correlação é indefinida.
    # Isso acontece se uma das imagens for constante (variância zero).
    if denominator == 0:
        # Se forem constantes idênticas, a correlação é 1. Senão, é 0.
        npcc = 1.0 if np.array_equal(i1, i2) else 0.0
    else:
        npcc = numerator / denominator
    ### END CODE HERE ###

    return npcc

# -------------------------- Self-check (optional) ----------------------- #

if __name__ == "__main__":
    # Quick numeric check using the README's 2x2 example
    i1 = np.array([[0.0, 0.5],
                   [0.5, 1.0]], dtype=np.float64)
    i2 = np.array([[0.0, 0.4],
                   [0.6, 1.0]], dtype=np.float64)

    out = compare_images(i1, i2)
    # Rounded print
    pretty = {
        "mse": round(out["mse"], 6),
        "psnr": round(out["psnr"], 2) if np.isfinite(out["psnr"]) else float("inf"),
        "ssim": round(out["ssim"], 6),
        "npcc": round(out["npcc"], 6),
    }
    print(pretty)
