from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.ndimage import gaussian_laplace
from numpy.linalg import norm
import numpy as np


def hfen(pred, gt, filter_size=15, sigma=1.5):
    """
    Calculate the High Frequency Error Norm (HFEN) between two images.
    
    Parameters:
    - pred: np.ndarray, the reconstructed image (3D).
    - gt: np.ndarray, the ground truth (GT) image (3D).
    - sigma: float, the standard deviation of the Gaussian filter (default 1.5).
    
    Returns:
    - hfen_value: float, the calculated HFEN value.
    """
    # Apply LoG filter to both images
    log_pred = gaussian_laplace(pred, sigma=sigma)
    log_gt = gaussian_laplace(gt, sigma=sigma)
    
    # Flatten the images to 1D for norm calculation
    log_pred_flat = log_pred.ravel()
    log_gt_flat = log_gt.ravel()
    
    # Compute the L2 norm of the difference and reference
    numerator   = norm(log_pred_flat - log_gt_flat)
    denominator = norm(log_gt_flat)
    
    hfen_value = numerator / denominator if denominator != 0 else float('inf')
    return hfen_value


def fastmri_ssim_qvl(gt: np.ndarray, pred: np.ndarray) -> float:
    """Compute SSIM compatible with the FastMRI challenge."""
    assert gt.shape == pred.shape, "Shape mismatch between gt and pred."
    assert gt.dtype == pred.dtype, "Data type mismatch between gt and pred."

    if gt.ndim == 2:
        gt = np.expand_dims(gt, axis=0)
        pred = np.expand_dims(pred, axis=0)

    assert len(gt.shape) == 3, "Expecting 3D arrays."
    
    return structural_similarity(
        gt,
        pred,
        channel_axis=0,
        data_range=gt.max()
    )


def fastmri_psnr_qvl(gt: np.ndarray, pred: np.ndarray) -> float:
    """Compute PSNR compatible with the FastMRI challenge."""
    assert gt.shape == pred.shape, "Shape mismatch between gt and pred."
    assert gt.dtype == pred.dtype, "Data type mismatch between gt and pred."

    if gt.ndim == 2:
        gt   = np.expand_dims(gt, axis=0)
        pred = np.expand_dims(pred, axis=0)

    assert len(gt.shape) == 3, "Expecting 3D arrays."

    return psnr(
        image_true=gt,
        image_test=pred,
        data_range=gt.max()
    )


def fastmri_nmse_qvl(gt: np.ndarray, pred: np.ndarray) -> float:
    """Compute NMSE compatible with the FastMRI challenge."""
    assert gt.shape == pred.shape, "Shape mismatch between gt and pred."
    assert gt.dtype == pred.dtype, "Data type mismatch between gt and pred."

    if gt.ndim == 2:
        gt = np.expand_dims(gt, axis=0)
        pred = np.expand_dims(pred, axis=0)

    assert len(gt.shape) == 3, "Expecting 3D arrays."

    return np.linalg.norm(gt - pred)**2 / np.linalg.norm(gt)**2


def blurriness_metric(image: np.ndarray) -> float:
    """Compute a blurriness metric based on the Laplacian.
    We call this the variance of the Laplacian (VoFL).
    """

    # assert len(image.shape) == 3, "Expecting 3D arrays."
    
    # Compute the Laplacian of the image
    laplacian = gaussian_laplace(image, sigma=1)
    
    # The variance of the Laplacian is used as the metric
    return np.var(laplacian)
