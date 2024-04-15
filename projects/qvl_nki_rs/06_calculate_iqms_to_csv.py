import argparse
import pandas as pd
import numpy as np
import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from multiprocessing import Pool
from functools import partial

from pathlib import Path
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.ndimage import gaussian_laplace
from typing import Tuple


def parse_args():
    print("\n\n\n")
    parser = argparse.ArgumentParser(description='Create H5 files for all patients')
    parser.add_argument('--dataset_dir', type=str, default='/scratch/p290820/datasets/003_umcg_pst_ksps/', help='Path to the dataset directory')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("\n\n\n")
    return args


def setup_logger(log_dir: Path, use_time: bool = True) -> logging.Logger:
    """
    Configure logging to both console and file.
    This function sets up logging based on the specified logging directory.
    It creates a log file named with the current timestamp and directs log 
    messages to both the console and the log file.
    Parameters:
    - log_dir (Path): Directory where the log file will be stored.
    Returns:
    - logging.Logger: Configured logger instance.
    """
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if use_time: 
        log_file = log_dir / f"log_{current_time}.log"
    else:
        log_file = log_dir / "log.log"

    l = logging.getLogger()
    l.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    l.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    l.addHandler(console_handler)

    return l


def get_relevant_pat_dirs(rootdir: Path, include_list: list, logger: logging.Logger = None) -> list:
    """
    Get the patient directories that are relevant for the analysis.
    
    Parameters:
    - rootdir (Path): The root directory where the patient directories are stored.
    - include_list (list): The list of strings to include in the patient directory names.
    
    Returns:
    - patients_dirs (list): The list of patient directories that are relevant for the analysis.
    """
    
    # Get all the patient directories
    patients_dirs = [x for x in rootdir.iterdir() if x.is_dir()]

    # Filter out the patients that are not in the include_list
    patients_dirs = [x for x in patients_dirs if any([y in x.name for y in include_list])]
    
    
    if logger:
        logger.info(f"Found {len(patients_dirs)} patient directories in {rootdir}")
        logger.info(f"Patients directories: {patients_dirs}")

    return patients_dirs


def resample_to_reference(
    image: sitk.Image, 
    ref_img: sitk.Image, 
    interpolator = sitk.sitkNearestNeighbor, 
    default_pixel_value: float = 0
) -> sitk.Image:
    """
    Automatically aligns, resamples, and crops an SITK image to a
    reference image. Can be either from .mha or .nii.gz files.

    Parameters:
    `image`: The moving image to align
    `ref_img`: The reference image with desired spacing, crop size, etc.
    `interpolator`: SITK interpolator for resampling
    `default_pixel_value`: Pixel value for voxels outside of original image
    """
    resampled_img = sitk.Resample(image, ref_img, 
            sitk.Transform(), 
            interpolator, default_pixel_value, 
            ref_img.GetPixelID())
    return resampled_img


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


def calculate_bounding_box(roi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the bounding box around the lesion.
    """
    # Find the coordinates where ROI has non-zero values
    non_zero_coords = np.argwhere(roi)
    
    # Calculate the bounding box
    min_coords = non_zero_coords.min(axis=0)
    max_coords = non_zero_coords.max(axis=0)
    
    return min_coords, max_coords


def extract_sub_volume_with_padding(image: np.ndarray, min_coords: np.ndarray, max_coords: np.ndarray, padding: int) -> np.ndarray:
    """
    Extract the sub-volume from the image with optional padding around the bounding box.
    The padding will be reduced if it exceeds the image dimensions.
    Padding is not applied in the slice (first) dimension.
    Parameters:
    - image (np.ndarray): The image from which to extract the sub-volume
    - min_coords (np.ndarray): The minimum coordinates of the bounding box
    - max_coords (np.ndarray): The maximum coordinates of the bounding box
    - padding (int): The padding to apply around the bounding box
    Returns:
    - sub_volume (np.ndarray): The sub-volume with padding
    """
    # Initialize padded min and max coordinates
    padded_min_coords = min_coords.copy()
    padded_max_coords = max_coords.copy()
    
    # Apply padding to x and y dimensions (1 and 2), not slice (0)
    for dim in [1, 2]:
        # Reduce padding if it goes out of the image dimensions
        while True:
            if padded_min_coords[dim] - padding < 0 or padded_max_coords[dim] + padding >= image.shape[dim]:
                padding = max(padding // 2, 0)  # Halve the padding, minimum is 0
            else:
                break  # Exit the loop if padding is within limits
        
        # Apply padding
        padded_min_coords[dim] -= padding
        padded_max_coords[dim] += padding

    # Extract the sub-volume with padding
    sub_volume = image[padded_min_coords[0]:padded_max_coords[0]+1, 
                       padded_min_coords[1]:padded_max_coords[1]+1, 
                       padded_min_coords[2]:padded_max_coords[2]+1]
    
    return sub_volume


def save_slices_as_images(
    seg_bb: np.ndarray,
    recon_bb: np.ndarray,
    target_bb: np.ndarray,
    pat_dir: Path,
    output_dir: str,
    acceleration: int,
    lesion_num: int,
    is_mirror: bool = False,
):
    """
    Save each slice from seg_bb, recon_bb, and target_bb as images side by side.
    
    Parameters:
        seg_bb (np.ndarray): The bounding box extracted from the segmentation.     # 3d array
        recon_bb (np.ndarray): The bounding box extracted from the reconstruction. # 3d array
        target_bb (np.ndarray): The bounding box extracted from the target.        # 3d array
        pat_dir (Path): The patient directory.
        output_dir (str): The directory where the images will be saved.
        acceleration (int): The acceleration factor.
        lesion_num (int): The lesion number.
        is_mirror (bool): Whether the bounding box is mirrored.

    Returns:
        None
    """
    assert recon_bb.shape == target_bb.shape == seg_bb.shape, "Mismatch in shape among bounding boxes"
    assert seg_bb.ndim == 3, "Bounding box should be a 3D array"
    assert recon_bb.ndim == 3, "Bounding box should be a 3D array"
    assert target_bb.ndim == 3, "Bounding box should be a 3D array"

    # make the dirs if they don't exist
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"\t\t\tROI shape: {seg_bb.shape}, mean: {round(seg_bb.mean(), 4)}")
    logger.info(f"\t\t\tRecon shape: {recon_bb.shape}, mean: {round(recon_bb.mean(), 4)}")
    logger.info(f"\t\t\tTarget shape: {target_bb.shape}, mean: {round(target_bb.mean(), 4)}")

    # Number of slices should be the same for all bounding boxes
    
    for slice_idx in range(seg_bb.shape[0]):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plotting each slice side by side
        axes[0].imshow(seg_bb[slice_idx], cmap='gray')
        axes[0].axis('off')
        axes[0].set_title(f'Segmentation Slice {slice_idx}')
        
        axes[1].imshow(recon_bb[slice_idx], cmap='gray')
        axes[1].axis('off')
        axes[1].set_title(f'Reconstruction Slice {slice_idx}')

        axes[2].imshow(target_bb[slice_idx], cmap='gray')
        axes[2].axis('off')
        axes[2].set_title(f'Target Slice {slice_idx}')
        
        # Save the figure
        if not is_mirror:
            fpath = os.path.join(output_dir, f"RIM_R{acceleration}_{pat_dir.name}_lesion{lesion_num}_slice{slice_idx+1}.png")
        else:
            fpath = os.path.join(output_dir, f"RIM_R{acceleration}_{pat_dir.name}_lesion{lesion_num}_slice{slice_idx+1}_mirrored.png")

        plt.savefig(fpath)
        plt.close(fig)
        logger.info(f"\t\t\tSaved slice {slice_idx+1}/{seg_bb.shape[0]} to {fpath}")


def calculate_image_quality_metrics(
    recon: np.ndarray,
    target: np.ndarray,
    pat_dir: Path,
    acceleration: int,
    decimals: int = 3,
    fov: str = None,
    logger: logging.Logger = None,
) -> dict:
    
    # Compute image quality metrics (IQMs)
    ssim_iqm = fastmri_ssim_qvl(gt=target, pred=recon)
    psnr_iqm = fastmri_psnr_qvl(gt=target, pred=recon)
    nmse_iqm = fastmri_nmse_qvl(gt=target, pred=recon)
    vofl_iqm = blurriness_metric(image=recon)

    logger.info(f"\t\tSSIM: {ssim_iqm:.3f}, PSNR: {psnr_iqm:.3f}, NMSE: {nmse_iqm:.3f}, VOFL: {vofl_iqm:.3f}\n")

    # Prepare data for DataFrame
    data = {
        'pat_id':       pat_dir.name,
        'acceleration': acceleration,
        'ssim':         round(ssim_iqm, decimals),
        'psnr':         round(psnr_iqm, decimals),
        'nmse':         round(nmse_iqm, decimals),
        'vofl':         round(vofl_iqm, decimals),
        'roi':          fov,
    }
    return data


def calculate_iqm_and_add_to_df(
    df: pd.DataFrame,
    recon: np.ndarray,
    target: np.ndarray,
    pat_dir: Path,
    acc: int,
    fov: str,
    decimals: int = 3,
    logger: logging.Logger = None,
) -> pd.DataFrame:
    iqms_dict = calculate_image_quality_metrics(
        recon        = recon,
        target       = target,
        pat_dir      = pat_dir,
        acceleration = acc,
        fov          = fov,
        decimals     = decimals,
        logger       = logger,
    )
    new_row = pd.DataFrame([iqms_dict])
    return pd.concat([df, new_row], ignore_index=True)


def load_seg_from_dcm_like(seg_fpath: Path, ref_nifti: sitk.Image, pat_dir: Path, acc: int) -> tuple:
    """
    Load the segmentation from a .dcm file.
    """
    # Load images from file as SimpleITK images
    seg = sitk.ReadImage(str(seg_fpath))

    # Resample first to the same size as the recon before obtaining an array
    seg = resample_to_reference(image=seg, ref_img=ref_nifti)

    # Convert to NumPy arrays
    seg = sitk.GetArrayFromImage(seg)

    return seg


def load_nifti_as_array(nifti_path: Path) -> tuple:
    img  = sitk.ReadImage(nifti_path)
    img  = sitk.GetArrayFromImage(img)
    return img


def process_lesion_fov(
    df: pd.DataFrame,
    seg_idx: int,
    recon: np.ndarray,
    target: np.ndarray,
    seg_fpath: Path,
    ref_nifti: sitk.Image,
    pat_dir: Path,
    acc: int,
    decimals: int = 3,
    is_mirror: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Process the lesion FOV. This includes:
    - Extracting the sub-volume around the lesion
    - Saving the slices as images
    - Calculating the IQMs on each slice
    - Adding the IQMs to the DataFrame
    Parameters:
    - df (DataFrame): The DataFrame to which the IQMs will be added
    - seg_idx (int): The index of the segmentation file
    - recon (np.ndarray): The reconstruction volume
    - target (np.ndarray): The target volume
    - seg_fpath (Path): The path to the segmentation file
    - ref_nifti (sitk.Image): The reference NIfTI file
    - pat_dir (Path): The patient directory
    - acc (int): The acceleration factor
    - decimals (int): The number of decimals to round the IQMs to
    Returns:
    - df (DataFrame): The updated DataFrame with the IQMs
    - seg_bb (np.ndarray): The bounding box around the lesion
    """ 
    seg = load_seg_from_dcm_like(seg_fpath=seg_fpath, ref_nifti=ref_nifti, pat_dir=pat_dir, acc=acc)

    # Bounding box around the lesion
    min_coords, max_coords = calculate_bounding_box(roi=seg)
    seg_bb    = extract_sub_volume_with_padding(seg, min_coords, max_coords, padding=10)
    recon_bb  = extract_sub_volume_with_padding(recon, min_coords, max_coords, padding=10)
    target_bb = extract_sub_volume_with_padding(target, min_coords, max_coords, padding=10)

    save_slices_as_images(
        seg_bb       = seg_bb,
        recon_bb     = recon_bb,
        target_bb    = target_bb,
        pat_dir      = pat_dir,
        output_dir   = str(pat_dir / "lesion_bbs"),
        acceleration = acc,
        lesion_num   = seg_idx+1,
        is_mirror    = is_mirror,
    )

    # Each slice with a lesion IQM calculation and add to the dataframe
    for slice_idx in range(seg_bb.shape[0]):
        data = {
            'pat_id':       pat_dir.name,
            'acceleration': acc,
            'ssim':         round(fastmri_ssim_qvl(gt=target_bb[slice_idx], pred=recon_bb[slice_idx]), decimals),
            'psnr':         round(fastmri_psnr_qvl(gt=target_bb[slice_idx], pred=recon_bb[slice_idx]), decimals),
            'nmse':         round(fastmri_nmse_qvl(gt=target_bb[slice_idx], pred=recon_bb[slice_idx]), decimals),
            'vofl':         round(blurriness_metric(image=recon_bb[slice_idx]), decimals),
            'roi':          f"lsfov" if not is_mirror else f"lsfov_mirrored",
        }
        new_row = pd.DataFrame([data])
        df = pd.concat([df, new_row], ignore_index=True)

    # # Calculate mirrored bounding boxes
    # mirrored_min_coords = np.array(target.shape[:3]) - max_coords - 1
    # mirrored_max_coords = np.array(target.shape[:3]) - min_coords - 1

    # mirrored_recon = recon[mirrored_min_coords[0]:mirrored_max_coords[0], 
    #                        mirrored_min_coords[1]:mirrored_max_coords[1], 
    #                        mirrored_min_coords[2]:mirrored_max_coords[2]]

    # mirrored_target = target[mirrored_min_coords[0]:mirrored_max_coords[0], 
    #                          mirrored_min_coords[1]:mirrored_max_coords[1], 
    #                          mirrored_min_coords[2]:mirrored_max_coords[2]]

    # mirrored_seg = seg[mirrored_min_coords[0]:mirrored_max_coords[0], 
    #                mirrored_min_coords[1]:mirrored_max_coords[1], 
    #                mirrored_min_coords[2]:mirrored_max_coords[2]]

    # save_slices_as_images(
    #     seg_bb       = mirrored_seg,
    #     recon_bb     = mirrored_recon,
    #     target_bb    = mirrored_target,
    #     pat_dir      = pat_dir,
    #     output_dir   = str(pat_dir / "lesion_bbs"),
    #     acceleration = acc,
    #     lesion_num   = seg_idx+1,
    #     is_mirror    = True,
    # )

    # # Calculate IQMs for mirrored region
    # for slice_idx in range(mirrored_recon.shape[0]):
    #     data = {
    #         'pat_id':       pat_dir.name,
    #         'acceleration': acc,
    #         'ssim':         round(fastmri_ssim_qvl(gt=mirrored_target[slice_idx], pred=mirrored_recon[slice_idx]), decimals),
    #         'psnr':         round(fastmri_psnr_qvl(gt=mirrored_target[slice_idx], pred=mirrored_recon[slice_idx]), decimals),
    #         'nmse':         round(fastmri_nmse_qvl(gt=mirrored_target[slice_idx], pred=mirrored_recon[slice_idx]), decimals),
    #         'vofl':         round(blurriness_metric(image=mirrored_recon[slice_idx]), decimals),
    #         'roi':          f"lsfov_ml{seg_idx+1}_s{slice_idx+1}",
    #     }
    #     new_row = pd.DataFrame([data])
    #     df = pd.concat([df, new_row], ignore_index=True)

    return df, seg_bb


# def find_regions_of_interest_3d(cv_map: np.ndarray, window_size: int = 11, num_regions: int = 2, boundary: int = 20):
#     """
#     Find homogeneous and inhomogeneous regions based on the CV map in 3D.

#     Parameters:
#     - cv_map (np.ndarray): 3D Coefficient of Variation map.
#     - window_size (int): The size of the window to compute local mean CV.
#     - num_regions (int): The number of regions to identify for low and high variance.
#     - boundary (int): Number of pixels from the edge to exclude from ROI selection.

#     Returns:
#     - low_var_coords (np.ndarray): Coordinates of regions with low variance.
#     - high_var_coords (np.ndarray): Coordinates of regions with high variance.
#     """

#     # Focus on the center 10 slices
#     depth = cv_map.shape[0]
#     start_slice = max(depth // 2 - 5, 0)
#     end_slice = min(depth // 2 + 5, depth)
#     center_slices = cv_map[start_slice:end_slice, :, :]
    
#     # Compute the local mean CV for the center slices
#     local_mean_cv = uniform_filter(center_slices, size=(1, window_size, window_size))

#     # Block out the boundary by setting the CV values to extremes
#     local_mean_cv[:, :boundary, :] = np.inf
#     local_mean_cv[:, :, :boundary] = np.inf
#     local_mean_cv[:, -boundary:, :] = np.inf
#     local_mean_cv[:, :, -boundary:] = np.inf

#     # Flatten the local mean CV map and find the indices of the top num_regions for low and high variance
#     local_mean_cv_flat = local_mean_cv.ravel()
#     low_var_indices = np.argpartition(local_mean_cv_flat, num_regions)[:num_regions]
#     high_var_indices = np.argpartition(-local_mean_cv_flat, num_regions)[:num_regions]

#     # Convert the indices back to 3D coordinates
#     low_var_coords = np.column_stack(np.unravel_index(low_var_indices, local_mean_cv.shape)) + [start_slice, 0, 0]
#     high_var_coords = np.column_stack(np.unravel_index(high_var_indices, local_mean_cv.shape)) + [start_slice, 0, 0]

#     return low_var_coords, high_var_coords


# def compute_cv_map_3d(image: np.ndarray, window_size: int) -> np.ndarray:
#     """
#     Compute the coefficient of variation (CV) map of a 3D image.
    
#     Parameters:
#     - image (np.ndarray): The input 3D image array (num_slices, height, width).
#     - window_size (int): The size of the window to compute local statistics.
    
#     Returns:
#     - cv_map (np.ndarray): The computed 3D CV map.
#     """
#     depth, height, width = image.shape
#     cv_map = np.zeros((depth, height, width), dtype=np.float32)
    
#     for slice_idx in range(depth):
#         # Calculate the local mean
#         local_mean = uniform_filter(image[slice_idx], size=window_size)
        
#         # Calculate the local variance
#         local_var = uniform_filter(image[slice_idx]**2, size=window_size) - (local_mean ** 2)
        
#         # Calculate the local standard deviation
#         local_std = np.sqrt(local_var)
        
#         # Calculate the coefficient of variation (CV)
#         cv_map[slice_idx] = np.where(local_mean > 0, local_std / local_mean, 0)
    
    return cv_map


# def visualize_cv_map(cv_map_2d: np.ndarray, colormap: str = 'viridis', save_path: str = None):
#     """
#     Visualize the Coefficient of Variation (CV) map.
    
#     Parameters:
#     - cv_map_2d (np.ndarray): The CV map to visualize.
#     - colormap (str): The colormap to use for visualization.
#     - save_path (str): The path to save the image. If None, the image won't be saved.
    
#     Returns:
#     None
#     """
#     plt.figure(figsize=(10, 10))
#     plt.imshow(cv_map_2d, cmap=colormap)
#     plt.colorbar(label='Coefficient of Variation')
#     plt.title('Coefficient of Variation Map')
#     plt.axis('off')
    
#     if save_path:
#         plt.savefig(save_path)
#         logger.info(f"Saved CV map to {save_path}")
#     else:
#         plt.show()


# def generate_random_coordinates(grid_shape: tuple, bbox_dims: tuple, num_locations: int = 6) -> np.ndarray:
#     """
#     Generate random coordinates within a grid while accommodating a bounding box and avoiding edges.

#     Parameters:
#     - grid_shape (tuple): The shape of the grid (height, width).
#     - bbox_dims (tuple): The dimensions of the bounding box (height, width).
#     - num_locations (int): The number of random locations to generate.

#     Returns:
#     - coordinates (np.ndarray): An array of shape (num_locations, 2) containing the (x, y) coordinates.
#     """
    
#     # Calculate the buffer needed to avoid placing the bounding box too close to the edge
#     edge_buffer_x = bbox_dims[0] // 2
#     edge_buffer_y = bbox_dims[1] // 2

#     # Generate random x and y coordinates within the adjusted grid
#     random_x = np.random.randint(edge_buffer_x, grid_shape[0] - edge_buffer_x - bbox_dims[0], num_locations)
#     random_y = np.random.randint(edge_buffer_y, grid_shape[1] - edge_buffer_y - bbox_dims[1], num_locations)

#     # Combine x and y coordinates into a single array
#     coordinates = np.column_stack((random_x, random_y))
    
#     return coordinates


def write_patches_to_file(
    gt_patch: np.ndarray,
    pred_patch: np.ndarray,
    slice_idx: int,
    y: int,
    x: int,
    output_dir: Path,
    logger: logging.Logger
):
    """
    Save patches with low SSIM to file for visual inspection.

    Parameters:
    - gt_patch (np.ndarray): Ground truth image patch.
    - pred_patch (np.ndarray): Predicted image patch.
    - slice_idx (int): Index of the current slice.
    - y (int): y-coordinate of the top-left corner of the patch.
    - x (int): x-coordinate of the top-left corner of the patch.
    - output_dir (Path): Directory to save the output images.
    - logger (logging.Logger): Logger instance for logging messages.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(gt_patch, cmap='gray')
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')

    axes[1].imshow(pred_patch, cmap='gray')
    axes[1].set_title('Prediction')
    axes[1].axis('off')

    # Save the figure
    patch_output_path = output_dir / f"low_ssim_patch_slice{slice_idx}_y{y}_x{x}.png"
    fig.savefig(patch_output_path)
    plt.close(fig)

    logger.info(f"Saved a pair of patches with low SSIM to {patch_output_path}")


# def generate_ssim_map_3d(
#     gt: np.ndarray,
#     pred: np.ndarray,
#     window_size: int,
#     stride: int,
#     logger: logging.Logger = None
# ) -> np.ndarray:
#     """
#     Generate an SSIM map for 3D data (slices first) using a sliding window approach.
#     Also known as a Structural Similarity Index Map (SSIM map).

#     Parameters:
#     - gt (np.ndarray): Ground truth 3D volume (slices first).
#     - pred (np.ndarray): Predicted 3D volume (slices first).
#     - window_size (int): Size of the square window to calculate SSIM.
#     - stride (int): Stride for the sliding window.
#     - logger (logging.Logger): Logger instance for logging messages.

#     Returns:
#     - ssim_map (np.ndarray): SSIM map with the same dimensions as the input volume.
#     """
#     assert gt.shape == pred.shape, "Shapes of ground truth and prediction must match."

#     logger.info(f"Generating SSIM map for 3D data (slices first) with window size {window_size} and stride {stride}.")
#     logger.info(f"\tGround truth shape: {gt.shape} and reconstruction shape: {pred.shape}")
#     start_time = time.time()  # Start timing

#     # Initialize SSIM map for the 3D volume
#     ssim_map = np.zeros_like(gt)
    
#     nslices, height, width = gt.shape

#     # Iterate over each slice in the 3D volume
#     for slice_idx in range(nslices):
#         logger.info(f"\t\tGenerating SSIM-Map Slice {slice_idx+1}/{nslices}")
#         for y in range(0, height - window_size + 1, stride):
#             for x in range(0, width - window_size + 1, stride):
                
#                 gt_patch   = gt[slice_idx, y:y + window_size, x:x + window_size]
#                 pred_patch = pred[slice_idx, y:y + window_size, x:x + window_size]

#                 ssim = structural_similarity(
#                     gt_patch, 
#                     pred_patch, 
#                     data_range=gt.max() - gt.min()
#                 )
                
#                 # Save patches with low SSIM to file for visual inspection. An ssim value lower than 0.2 is considered too low, so something must be up.
#                 if ssim < 0.2:
#                     write_patches_to_file(
#                         gt_patch    = gt_patch,
#                         pred_patch  = pred_patch,
#                         slice_idx   = slice_idx,
#                         y           = y,
#                         x           = x,
#                         output_dir  = Path("/home1/p290820/tmp/low_ssim_patches"),
#                         logger      = logger,
#                     )

#                 ssim_map[slice_idx, y:y + window_size, x:x + window_size] = ssim

#     duration = time.time() - start_time
#     if logger:
#         logger.info(f"SSIM map generation for 3D data (slices first) took {duration:.2f} seconds.")

#     return ssim_map


def calculate_ssim_for_slice(
    slice_index: int, 
    target_volume: np.ndarray, 
    pred_volume: np.ndarray, 
    window_size: int, 
    stride: int
) -> Tuple[int, np.ndarray]:
    """
    Calculate the SSIM for a single slice of the 3D volume.

    Parameters:
        slice_index (int): Index of the slice in the volume.
        target_volume (np.ndarray): Ground truth volume.
        pred_volume (np.ndarray): Predicted volume.
        window_size (int): Size of the window for SSIM calculation.
        stride (int): Stride of the sliding window.

    Returns:
        Tuple[int, np.ndarray]: Index of the slice and its SSIM map.
    """
    height, width = target_volume.shape[1], target_volume.shape[2]
    ssim_map_slice = np.zeros((height, width))

    for y in range(0, height - window_size + 1, stride):
        for x in range(0, width - window_size + 1, stride):
            gt_patch = target_volume[slice_index, y:y + window_size, x:x + window_size]
            pred_patch = pred_volume[slice_index, y:y + window_size, x:x + window_size]

            # Compute SSIM for the current patch. The data_range is the maximum possible value of the image.
            ssim_value = structural_similarity(gt_patch, pred_patch, data_range=target_volume.max() - target_volume.min())

            # Fill the SSIM map for the current window location
            ssim_map_slice[y:y + window_size, x:x + window_size] = ssim_value

    return slice_index, ssim_map_slice


def generate_ssim_map_3d_parallel(
    target: np.ndarray, 
    pred: np.ndarray, 
    window_size: int, 
    stride: int, 
    num_workers: int, 
    logger: logging.Logger
) -> np.ndarray:
    """
    Generate an SSIM map for a 3D volume in parallel across multiple slices.

    Parameters:
        target (np.ndarray): Ground truth 3D volume.
        pred (np.ndarray): Predicted 3D volume.
        window_size (int): The size of the window for SSIM calculation.
        stride (int): The stride of the sliding window.
        num_workers (int): The number of worker processes to use.
        logger (logging.Logger): Logger for logging messages.

    Returns:
        np.ndarray: The SSIM map for the 3D volume.
    """
    logger.info("\t\t\tStarting parallel SSIM map generation.")
    ssim_map = np.zeros_like(target)

    with Pool(processes=num_workers) as pool:
        func = partial(calculate_ssim_for_slice, target_volume=target, pred_volume=pred, window_size=window_size, stride=stride)
        results = pool.map(func, range(target.shape[0]))

    for slice_index, ssim_map_slice in results:
        ssim_map[slice_index, :, :] = ssim_map_slice

    logger.info("\t\t\tCompleted parallel SSIM map generation.")
    return ssim_map


def calculate_and_save_ssim_map_3d(
    target: np.ndarray,
    recon: np.ndarray,
    output_dir: Path,
    patient_id: str,
    acc_factor: int,
    window_size: int = 11,
    stride: int = 4,
    logger: logging.Logger = None,
    ref_nifti: sitk.Image = None,
    metric: str = 'ssim',
):
    """
    Calculates an SSIM map for 3D volumes and saves it as a NIfTI file.

    Parameters:
    - target: The ground truth 3D volume.
    - reconstruction: The predicted 3D volume.
    - output_dir: Directory where the SSIM map NIfTI file will be saved.
    - patient_id: Identifier for the patient. In this case like 0001_ANON0123456
    - acceleration_factor: Acceleration factor (e.g., R2, R4, etc.) used in the file name.
    - window_size: The size of the window to compute SSIM, default is 11.
    - stride: The stride with which to apply the sliding window, default is 4.
    - logger: Logger for logging information.
    """
    if logger:
        logger.info(f"\t\t\tCalculating SSIM map with window size {window_size} and stride {stride}.")

    # Call to a function to generate the SSIM map
    # ssim_map = generate_ssim_map_3d(target, recon, window_size, stride, logger)
    ssim_map = generate_ssim_map_3d_parallel(target, recon, window_size, stride, num_workers=8, logger=logger)

    # Convert the SSIM map to a NIfTI image
    ssim_map_nifti = sitk.GetImageFromArray(ssim_map)

    # Define the output file path
    output_file_path = output_dir / f"RIM_R{acc_factor}_{patient_id}_dcml_{metric}_map_stride{stride}.nii.gz"

    # copy data from the ref nitfi to the ssim map nifti
    if ref_nifti:
        ssim_map_nifti.CopyInformation(ref_nifti)

    # Write the SSIM map NIfTI image to file
    sitk.WriteImage(ssim_map_nifti, str(output_file_path))
    
    if logger:
        logger.info(f"\t\t\tSaved SSIM map to {output_file_path}")


def calculate_iqms_on_all_patients(
    df: pd.DataFrame,
    patients_dir: Path,
    include_list: list,
    accelerations: list,
    do_ssim_map: bool = False,
    logger: logging.Logger = None,
) -> pd.DataFrame:
    """
    Calculate the IQMs for all patients in the patients_dir. On three FOVs: abfov, prfov, and lsfov.
    We add the IQMs to the DataFrame and return it.

    Parameters:
    - df (pd.DataFrame): The DataFrame to which the IQMs will be added
    - patients_dir (Path): The path to the directory containing the patient directories
    - include_list (list): The list of patients to include
    - accelerations (list): The list of accelerations to process
    - logger (logging.Logger): The logger instance
    - do_ssim_map (bool): Whether to calculate and save the SSIM map
    Returns:
    - df (pd.DataFrame): The updated DataFrame with the IQMs
    """
    pat_dirs = get_relevant_pat_dirs(patients_dir, include_list, logger)
    for pat_idx, pat_dir in enumerate(pat_dirs):
        logger.info(f"Processing patient {pat_idx+1}/{len(pat_dirs)}: {pat_dir.name}")
        os.makedirs(pat_dir / "temp", exist_ok=True)
        os.makedirs(pat_dir / "metric_maps", exist_ok=True)
        

        seg_files = [x for x in pat_dir.iterdir() if "roi" in x.name.lower()]
        for acc in accelerations:
            logger.info(f"\tProcessing acceleration: {acc}")

            # ABDOMINAL FOV (abfov) - Add the IQMs to the DataFrame
            recon_abfov  = load_nifti_as_array(nifti_path = str(pat_dir / "recons" / f"RIM_R{acc}_recon_{pat_dir.name}_pst_T2_recon.nii.gz"))
            target_abfov = load_nifti_as_array(nifti_path = str(pat_dir / "recons" / f"RIM_R{acc}_recon_{pat_dir.name}_pst_T2_target.nii.gz"))
            df = calculate_iqm_and_add_to_df( 
                df           = df,
                recon        = recon_abfov,
                target       = target_abfov,
                pat_dir      = pat_dir,
                acc          = acc,
                fov          = "abfov",
                logger       = logger,
            )
            
            # PROSTATE FOV (prfov) - Add the IQMs to the DataFrame
            recon_prfov  = load_nifti_as_array(nifti_path = str(pat_dir / "recons" / f"RIM_R{acc}_recon_{pat_dir.name}_pst_T2_dcml_recon.nii.gz"))
            target_prfov = load_nifti_as_array(nifti_path = str(pat_dir / "recons" / f"RIM_R{acc}_recon_{pat_dir.name}_pst_T2_dcml_target.nii.gz"))
            df = calculate_iqm_and_add_to_df(
                df           = df,
                recon        = recon_prfov,
                target       = target_prfov,
                pat_dir      = pat_dir,
                acc          = acc,
                fov          = "prfov",
                logger       = logger,
            )
            # Calculate an SSIM map of the reconstruction versus the target
            ref_nifti = sitk.ReadImage(str(pat_dir / "recons" / f"RIM_R{acc}_recon_{pat_dir.name}_pst_T2_dcml_target.nii.gz"))
            if do_ssim_map:
                calculate_and_save_ssim_map_3d(
                    target      = target_prfov,
                    recon       = recon_prfov,
                    output_dir  = pat_dir / "metric_maps",
                    patient_id  = pat_dir.name,
                    acc_factor  = acc,
                    window_size = 10,
                    stride      = 10,
                    logger      = logger,
                    ref_nifti   = ref_nifti,
                    metric      = 'ssim',
                )

            # LESION FOV (lsfov) - Add the IQMs to the DataFrame
            ref_nifti = sitk.ReadImage(str(pat_dir / "recons" / f"RIM_R{acc}_recon_{pat_dir.name}_pst_T2_dcml_target.nii.gz"))
            for seg_idx, seg_fpath in enumerate(seg_files):
                logger.info(f"\t\tProcessing ROI {seg_idx+1}/{len(seg_files)}: {seg_fpath.name}")
                df, seg_bb = process_lesion_fov(
                    df        = df,
                    seg_idx   = seg_idx,
                    recon     = recon_prfov,
                    target    = target_prfov,
                    seg_fpath = seg_fpath,
                    ref_nifti = ref_nifti,
                    pat_dir   = pat_dir,
                    acc       = acc,
                    is_mirror = False,
                )

            # LESION FOV MIRRORED (lsfov_mirrored) - Add the IQMs to the DataFrame
            for seg_idx, seg_fpath in enumerate(seg_files):
                logger.info(f"\t\tProcessing control mirrored ROI {seg_idx+1}/{len(seg_files)}: {seg_fpath.name}")
                df, seg_bb = process_lesion_fov(
                    df        = df,
                    seg_idx   = seg_idx,
                    recon     = np.flip(recon_prfov, axis=2),
                    target    = np.flip(target_prfov, axis=2),
                    seg_fpath = seg_fpath,
                    ref_nifti = ref_nifti,
                    pat_dir   = pat_dir,
                    acc       = acc,
                    is_mirror = True,
                )

    return df


def plot_quality_metric(df, metric='ssim', save_path=None, palette='bright'):
    # Dictionary for renaming ROIs
    rename_dict = {'abfov': 'Abdominal FOV', 'prfov': 'Prostate FOV', 'lsfov': 'Lesion FOV', 'lsfov_mirrored': 'Control Lesion FOV'}

    # Rename the 'roi' categories for better readability
    df['roi_grouped'] = df['roi'].apply(lambda x: rename_dict.get(x, x))

    # Count the number of datapoints for each trendline, grouped by 'roi_grouped' and 'acceleration'
    datapoints_count = df.groupby(['roi_grouped', 'acceleration']).size().reset_index(name='Number of Datapoints')

    # Create a dictionary to map ROI and acceleration to the number of datapoints
    datapoints_dict = {(roi, acc): count for roi, acc, count in zip(datapoints_count['roi_grouped'], datapoints_count['acceleration'], datapoints_count['Number of Datapoints'])}

    # Create a new legend label incorporating the number of datapoints
    df['legend_label'] = df.apply(lambda row: f"{row['roi_grouped']} (n={datapoints_dict[(row['roi_grouped'], row['acceleration'])]})", axis=1)

    # Create the scatter plot with trend lines
    plt.figure(figsize=(12, 8))
    color_palette = sns.color_palette(palette, n_colors=len(df['roi_grouped'].unique()))

    # Scatter plot
    sns.scatterplot(data=df, x='acceleration', y=metric, hue='legend_label', palette=color_palette, s=100)

    # Trend lines
    sns.lineplot(data=df, x='acceleration', y=metric, hue='roi_grouped', palette=color_palette, estimator=np.mean, legend=None)

    # Add titles and labels
    plt.title(f'DLRecon Image Quality: ({metric.upper()}) Degradation over Acceleration for Different Prostate ROIs')
    plt.xlabel('Acceleration')
    plt.ylabel(metric.upper())

    # Enhance the legend
    plt.legend(title='ROIs', title_fontsize='16', labelspacing=1.2, fontsize='12')
    plt.grid(True)

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(Path(save_path), dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {save_path}")


# Modified function to plot all four quality metrics in the same figure
def plot_all_quality_metrics(
    df: pd.DataFrame,
    metrics=['ssim', 'psnr', 'nmse', 'vofl'],
    save_path=None,
    palette='bright',
    logger: logging.Logger = None,
) -> None:
    """
    Create a 2x2 grid of subplots for the given quality metrics.

    Parameters:
    - df (DataFrame): The data to plot
    - metrics (list): The list of quality metrics to use
    - save_path (str): Optional path to save the generated plot
    """

    # Dictionary for renaming ROIs
    rename_dict = {'abfov': 'Abdominal FOV', 'prfov': 'Prostate FOV', 'lsfov': 'Lesion FOV', 'lsfov_mirrored': 'Control Lesion FOV'}
    df['roi_grouped'] = df['roi'].apply(lambda x: rename_dict.get(x, x))

    # Create the scatter plot with trend lines 
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    color_palette = sns.color_palette(palette, n_colors=len(df['roi_grouped'].unique()))

    # Count the number of datapoints for each trendline, grouped by 'roi_grouped' and 'acceleration'
    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Create scatter plot
        sns.scatterplot(data=df, x='acceleration', y=metric, hue='roi_grouped', style='roi_grouped', s=100, ax=ax, palette=palette)
        
        # Adding trend lines for each ROI
        for jdx, roi_grouped in enumerate(df['roi_grouped'].unique()):
            roi_grouped_data = df[df['roi_grouped'] == roi_grouped]
            sns.regplot(data=roi_grouped_data, x='acceleration', y=metric, scatter=False, ax=ax, color=color_palette[jdx])
        
        # Add title and labels
        ax.set_title(f"IQM: {metric.upper()}")
        ax.set_xlabel("Acceleration Factor")
        ax.set_ylabel(metric.upper())
        ax.grid(True)
        ax.legend().remove()
    
    # Increase the legend size
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', title='ROI Types', fontsize='12')
    fig.suptitle("Image Quality Metrics Across Accelerations and FOVs", fontsize=16)
    
    if save_path:
        plt.savefig(Path(save_path), dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {save_path}") if logger else None


##########################################################################################
if __name__ == "__main__":

    # Arguments
    args  = parse_args()
    DEBUG = True if args.debug else False
    TEMPDIR = Path("/home1/p290820/tmp")

    cfgs = {
        'do_ssim_map':           True,
        'dataset_dir':           Path(args.dataset_dir),
        'patients_dir':          Path(args.dataset_dir) / 'pat_data',
        'log_dir':               Path(args.dataset_dir) / 'logs',
        'fig_dir':               Path(args.dataset_dir) / 'figures',
        'df_fpath':              Path(args.dataset_dir) / 'results' / "iqms_per_fovs_on_accs_with_RIM_results_v2.csv",
        'accelerations':         [2,4,6,8] if not DEBUG else [2,4], # Accelerations included for post-processing.
        'image_quality_metrics': ['ssim', 'psnr', 'nmse', 'vofl'],
        'decimals':              3,
        'exclusion_list':        [],
        'include_list': [
            '0002',  # pat0002
            '0003',  # pat0003
            '0004',  # pat0004
            '0005',  # pat0005
            # '0006',  # pat0006     # This patient is excluded because it has a different number of slices
            '0007',  # pat0007
            '0008',  # pat0008
            '0009',  # pat0009
            '0010',   # pat0010
        ]
    }
    logger = setup_logger(cfgs['log_dir'], use_time=False)

    if DEBUG:
        logger.info(f"DEBUG MODE: Accelerations: {cfgs['accelerations']}")
        cfgs['df_fpath'] = Path(args.dataset_dir) / 'results' / "iqms_per_fovs_on_accs_with_RIM_with_other_rois_v2.csv"
        logger.info(f"DEBUG MODE: Saving DataFrame to {cfgs['df_fpath']}")

    df = pd.DataFrame(columns=['pat_id', 'acceleration', 'ssim', 'psnr', 'nmse', 'vofl', 'roi']).astype({
        'pat_id':       'str',
        'acceleration': 'int',
        'ssim':         'float64',
        'psnr':         'float64',
        'nmse':         'float64',
        'vofl':         'float64',
        'roi':          'str'
    })

    # if not os.path.exists(df_fpath):
    if True:
        df = calculate_iqms_on_all_patients(
            df            = df,
            patients_dir  = cfgs['patients_dir'],
            include_list  = cfgs['include_list'],
            accelerations = cfgs['accelerations'],
            do_ssim_map   = cfgs['do_ssim_map'],
            logger        = logger,
        )
        df.to_csv(cfgs['df_fpath'], index=False)
        logger.info(f"Saved DataFrame to {cfgs['df_fpath']}")
    else: # load from file
        df = pd.read_csv(cfgs['df_fpath'])
        logger.info(f"Loaded DataFrame from {cfgs['df_fpath']}")

    if True:
        for iqm in cfgs['image_quality_metrics']:
            plot_quality_metric(
                df         = df,
                metric     = iqm,
                save_path  = cfgs['fig_dir'] / f"{iqm}_vs_accs.png",
            )

        plot_all_quality_metrics(
            df         = df,
            metrics    = ['ssim', 'psnr', 'nmse', 'vofl'],
            save_path  = cfgs['fig_dir'] / "all_metrics_vs_accs_other_rois.png",
            palette    = 'bright',
        )

    # Group the data by 'roi' and 'acceleration' and calculate the mean and standard deviation of SSIM
    grouped_df = df.groupby(['roi', 'acceleration'])['ssim'].agg(['mean', 'std']).reset_index()

    # Display the table
    logger.info(f"These are the mean and standard deviation of SSIM for each ROI and acceleration factor:\n")
    logger.info(grouped_df)
