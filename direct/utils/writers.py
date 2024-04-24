# coding=utf-8
# Copyright (c) DIRECT Contributors

import logging
import pathlib
from typing import Callable, DefaultDict, Dict, Optional, Union

import h5py  # type: ignore
import numpy as np
import SimpleITK as sitk


logger = logging.getLogger(__name__)


def write_output_to_h5(
    output: Union[Dict, DefaultDict],
    output_directory: pathlib.Path,
    volume_processing_func: Optional[Callable] = None,
    output_key: str = "reconstruction",
    create_dirs_if_needed: bool = True,
    modelname = None,
    acceleration_factor = None,
    also_write_nifti = False,
) -> None:
    """Write dictionary with keys filenames and values torch tensors to h5 files.

    Parameters
    ----------
    output: dict
        Dictionary with keys filenames and values torch.Tensor's with shape [depth, num_channels, ...]
        where num_channels is typically 1 for MRI.
    output_directory: pathlib.Path
    volume_processing_func: callable
        Function which postprocesses the volume array before saving.
    output_key: str
        Name of key to save the output to.
    create_dirs_if_needed: bool
        If true, the output directory and all its parents will be created.
    modelname: str
        Name of the model.
    acceleration_factor: float
        Acceleration factor.
    also_write_nifti: bool
        If true, also write nifti files.

    Notes
    -----
    Currently only num_channels = 1 is supported. If you run this function with more channels the first one
    will be used.
    """
    if create_dirs_if_needed:
        # Create output directory
        output_directory.mkdir(exist_ok=True, parents=True)
        
    logger.info(f"length of output in write output to h5: {len(output)}")
    for idx, (volume, target, _, filename, samp_mask, pat_id) in enumerate(output):
        
        # Create a per patient a directory
        pat_dir = output_directory / pat_id
        pat_dir.mkdir(exist_ok=True, parents=True)
        
        # The output has shape (slice, 1, height, width)
        if isinstance(filename, pathlib.PosixPath):
            filename = filename.name
        logger.info(f"({idx + 1}/{len(output)}): Writing {pat_dir / filename}...")
            
        reconstruction = volume.numpy()[:, 0, ...].astype(np.float32)
        if target is not None:
            target = target.numpy()[:, 0, ...].astype(np.float32)
        if samp_mask is not None:
            samp_mask = samp_mask.numpy()[:, 0, ...].astype(np.float32)

        if volume_processing_func:
            reconstruction = volume_processing_func(reconstruction)

        # out_fname = pat_dir / f"{modelname}_R{int(acceleration_factor)}_recon.nii.gz"
        # logger.info(f"({idx + 1}/{len(output)}): Writing {out_fname}...")
        
        if also_write_nifti:
            # fname_recon_str  = str(out_fname).replace('.h5', '.nii.gz')
            fname_recon_str = pat_dir / f"{modelname}_R{int(acceleration_factor)}_recon.nii.gz"
            # fname_target_str = str(out_fname).replace('.h5', '_target.nii.gz')
            fname_target_str = pat_dir / f"{modelname}_R{int(acceleration_factor)}_target.nii.gz"
            write_numpy_to_nifti(reconstruction, out_fname=fname_recon_str)
            if target is not None:
                write_numpy_to_nifti(target, out_fname=fname_target_str)
            
        if samp_mask is not None:
            # fname_mask_str = str(out_fname).replace('.h5', '_mask.nii.gz')
            fname_mask_str = pat_dir / f"{modelname}_R{int(acceleration_factor)}_mask.nii.gz"
            write_numpy_to_nifti(samp_mask, out_fname=fname_mask_str, do_round=False)
            
        with h5py.File(pat_dir / filename, "w") as f:
            f.create_dataset(output_key, data=reconstruction)
            f.create_dataset("target", data=target)
            f.create_dataset("mask", data=samp_mask)
            f.attrs["acceleration_factor"] = acceleration_factor
            f.attrs["modelname"] = modelname


def write_numpy_to_nifti(img: np.ndarray, out_fname: str, do_round = True):
    """Write numpy array to nifti."""
    
    if do_round:
        img = np.round(img*1000, decimals=3)

    img = sitk.GetImageFromArray(img)
    sitk.WriteImage(img, out_fname)
    logger.info(f"Writing {out_fname}...")
