# coding=utf-8
# Copyright (c) DIRECT Contributors

import logging
import pathlib
from typing import Callable, DefaultDict, Dict, Optional, Union, List

import h5py  # type: ignore
import numpy as np
import SimpleITK as sitk

from sqlite3 import connect


logger = logging.getLogger(__name__)


# def write_output_to_h5(
#     output: Union[Dict, DefaultDict],
#     output_directory: pathlib.Path,
#     volume_processing_func: Optional[Callable]               = None,
#     output_key: str                                          = "reconstruction",
#     create_dirs_if_needed: bool                              = True,
#     modelname                                                = None,
#     acceleration_factor: Optional[Union[float, List[float]]] = None,
#     also_write_nifti                                         = False,
#     fnamepart                                                = None,
#     do_round                                                 = False,
# ) -> None:
#     """Write dictionary with keys filenames and values torch tensors to h5 files.

#     Parameters
#     ----------
#     output: dict
#         Dictionary with keys filenames and values torch.Tensor's with shape [depth, num_channels, ...]
#         where num_channels is typically 1 for MRI.
#     output_directory: pathlib.Path
#     volume_processing_func: callable
#         Function which postprocesses the volume array before saving.
#     output_key: str
#         Name of key to save the output to.
#     create_dirs_if_needed: bool
#         If true, the output directory and all its parents will be created.
#     modelname: str
#         Name of the model.
#     acceleration_factor: Can be a float or a list of floats.
#         Acceleration factor(s) for the reconstruction.
#     also_write_nifti: bool
#         If true, also write nifti files.

#     Notes
#     -----
#     Currently only num_channels = 1 is supported. If you run this function with more channels the first one
#     will be used.
#     """
#     if create_dirs_if_needed:
#         output_directory.mkdir(exist_ok=True, parents=True)
#         logger.info(f"Output directory: {output_directory}")

#     logger.info(f"length of output in write output to h5: {len(output)}")
#     for idx, (volume, target, _, filename, samp_mask, pat_id) in enumerate(output):
        
#         # some logging for debugging
#         if volume is not None:
#             logger.info(f"volume type: {type(volume)}, shape: {volume.shape}, dtype: {volume.dtype}")
#             volume_cpu = volume.cpu()  # if not already on CPU
#             arr = volume_cpu.numpy()
#             logger.info(f"Numpy array shape: {arr.shape}, dtype: {arr.dtype}")
#         else:
#             logger.info("volume is None!")
        
#         if target is not None:
#             logger.info(f"target type: {type(target)}, shape: {target.shape}, dtype: {target.dtype}")
#             target_cpu = target.cpu()
#             arr = target_cpu.numpy()
#             logger.info(f"Numpy array shape: {arr.shape}, dtype: {arr.dtype}")
#         else:
#             logger.info("target is None!")
        
#         # Create per patient a directory
#         pat_dir = output_directory / pat_id
#         pat_dir.mkdir(exist_ok=True, parents=True)
#         logger.info(f"The patient directory is: {pat_dir}")
        
#         # The output has shape (slice, 1, height, width)
#         if isinstance(filename, pathlib.PosixPath):
#             filename = filename.name
#         logger.info(f"({idx + 1}/{len(output)}): Writing {pat_dir / filename}...")
            
#         reconstruction = volume.cpu().numpy()[:, 0, ...].astype(np.float32)
#         if target is not None:
#             target = target.cpu().numpy()[:, 0, ...].astype(np.float32)
#         if samp_mask is not None:
#             samp_mask = samp_mask.cpu().numpy()[:, 0, ...].astype(np.float32)

#         if volume_processing_func:
#             reconstruction = volume_processing_func(reconstruction)

#         # out_fname = pat_dir / f"{modelname}_R{int(acceleration_factor)}_recon.nii.gz"
#         # logger.info(f"({idx + 1}/{len(output)}): Writing {out_fname}...")

#         if isinstance(acceleration_factor, list):
#             acceleration_factor = acceleration_factor[0]
        
#         if also_write_nifti:
#             # fname_recon_str  = str(out_fname).replace('.h5', '.nii.gz')
#             fname_recon_str = pat_dir / f"{modelname}_R{int(acceleration_factor)}_recon.nii.gz"
#             # fname_target_str = str(out_fname).replace('.h5', '_target.nii.gz')
#             fname_target_str = pat_dir / f"{modelname}_R{int(acceleration_factor)}_target.nii.gz"
#             write_numpy_to_nifti(reconstruction, out_fname=fname_recon_str)
#             if target is not None:
#                 write_numpy_to_nifti(target, out_fname=fname_target_str)
            
#         if samp_mask is not None:
#             # fname_mask_str = str(out_fname).replace('.h5', '_mask.nii.gz')
#             fname_mask_str = pat_dir / f"{modelname}_R{int(acceleration_factor)}_mask.nii.gz"
#             write_numpy_to_nifti(samp_mask, out_fname=fname_mask_str, do_round=False)

#         if fnamepart is not None:
#             # we this fnamepart right before the file extension
#             filename = filename.replace(".h5", f"_{fnamepart}.h5")

#         with h5py.File(pat_dir / filename, "w") as f:
#             f.create_dataset(output_key, data=reconstruction)
#             if target is not None:
#                 f.create_dataset("target", data=target)
#             if samp_mask is not None:
#                 f.create_dataset("mask", data=samp_mask)
#             f.attrs["acceleration_factor"] = acceleration_factor
#             f.attrs["modelname"] = modelname


def round_volume(volume: np.ndarray, decimals: int = 3) -> np.ndarray:
    """
    Round the volume to a specified number of decimals.
    """
    return np.round(volume, decimals=decimals)


def get_gaussian_id(pat_id: str, db_path: str, tablename: str, filename: pathlib.Path, debug: bool = False) -> int:
    """
    Retrieve the gaussian_id for a patient from the database.
    
    Connects to the database, fetches the latest row for the patient,
    and returns the existing gaussian_id if recon_path is NULL,
    or increments it by one otherwise. Also updates the recon_path if it was NULL.
    
    Args:
        pat_id (str): The patient identifier.
        db_path (str): Path to the database.
        tablename (str): Name of the table in the database.
        filename (str): The filename to set as recon_path if it was NULL.
        debug (bool): Flag to enable debug logging.
        
    Returns:
        int: The gaussian_id to use.
    """
    conn = connect(db_path)
    cursor = conn.cursor()
    query = f"""
        SELECT id, seq_id, anon_id, gaussian_id, recon_path
        FROM {tablename}
        WHERE id = ?
        ORDER BY gaussian_id DESC
        LIMIT 1;
    """
    cursor.execute(query, (pat_id,))
    row = cursor.fetchone()
    if row is None:
        conn.close()
        raise ValueError(f"No row found for patient ID: {pat_id} in writers.py")
    
    id, seq_id, anon_id, gaussian_id, recon_path = row
    
    if debug:
        logger.info(f"QVL - Writers.py - Retrieved gaussian_id: {gaussian_id} for patient ID: {pat_id}. recon_path was found to be {recon_path}.")
    
    if recon_path is None:
        # we must add the gaussian_id to the filename before the extension
        new_filename = filename.stem + f"_{gaussian_id}" + filename.suffix
        # Update the recon_path for the current row
        update_query = f"""
            UPDATE {tablename}
            SET recon_path = ?
            WHERE id = ? AND gaussian_id = ?;
        """
        cursor.execute(update_query, (str(filename.with_name(new_filename)), pat_id, gaussian_id))
        conn.commit()
        
        if debug:
            logger.info(f"QVL - Writers.py - Updated recon_path to {filename} for patient ID: {pat_id} with gaussian_id: {gaussian_id}.")
        
        # Insert a new row with incremented gaussian_id and recon_path set to NULL
        insert_query = f"""
            INSERT INTO {tablename} (id, seq_id, anon_id, gaussian_id, recon_path)
            VALUES (?, ?, ?, ?, NULL);
        """
        cursor.execute(insert_query, (pat_id, seq_id, anon_id, gaussian_id + 1))
        conn.commit()
        
        if debug:
            logger.info(f"QVL - Writers.py - Inserted new row for patient ID: {pat_id} with gaussian_id: {gaussian_id + 1} and recon_path set to NULL.")
        
        result = gaussian_id
    else:
        result = gaussian_id + 1
    
    conn.close()
    return result


def process_patient_output(
    patient_data: tuple,
    pat_dir: pathlib.Path,
    modelname: str,
    acceleration_factor: float,
    output_key: str                   = "reconstruction",
    volume_processing_func: callable  = None,
    do_round: bool                    = False,
    decimals: int                     = 3,
    also_write_nifti: bool            = False,
    added_gaussian_noise: bool        = False,
    db_path: str                      = None,
    tablename: str                    = None,
) -> None:
    """
    Process and write output for a single patient.
    
    patient_data is expected to be a tuple:
      (volume, target, _, filename, samp_mask, pat_id)
    where volume is a torch tensor with shape [slices, 1, height, width].
    """
    volume, target, _, filename, samp_mask, pat_id = patient_data

    print(f"Shape of the volume: {volume.shape}")
    print(f"Shape of the target: {target.shape}") if target is not None else None
    print(f"The filename is: {filename}")
    print(f"The patient ID is: {pat_id}")
    print(f"Type patient id: {type(pat_id)}")

    # here we connect with the database and see if what the gaussian noise index should be for this patient and update the database with a new row, so that the next time it will be updated with a new index
    if added_gaussian_noise:
        conn = connect(db_path)
        c = conn.cursor()
        gaussian_id = get_gaussian_id(str(pat_id), db_path, tablename, pat_dir / filename)

        # Modify filename with fnamepart if provided
        if isinstance(filename, pathlib.PosixPath):
            filename = filename.name
        filename = filename.replace(".h5", f"_gaus{gaussian_id}.h5")
        conn.close()
    else:
        gaussian_id = ""

    # Convert torch tensors to numpy arrays
    if volume is not None:
        volume_np = volume.cpu().numpy()[:, 0, ...].astype(np.float32)
    else:
        volume_np = None

    if target is not None:
        target_np = target.cpu().numpy()[:, 0, ...].astype(np.float32)
    else:
        target_np = None

    if samp_mask is not None:
        samp_mask_np = samp_mask.cpu().numpy()[:, 0, ...].astype(np.float32)
    else:
        samp_mask_np = None

    # Optionally process volume
    if volume_processing_func is not None:
        volume_np = volume_processing_func(volume_np)

    # Optionally round volume
    if do_round:
        volume_np = round_volume(volume_np, decimals)
        target_np = round_volume(target_np, decimals) if target_np is not None else None

    # Save as NIfTI if requested
    if also_write_nifti:
        # 1 - Write the reconstruction
        fname_recon = pat_dir / f"{modelname}_R{int(acceleration_factor)}_recon_gaus{gaussian_id}.nii.gz"
        write_numpy_to_nifti(volume_np, fname_recon)
        # 2 - Write the target
        if target_np is not None:
            fname_target = pat_dir / f"{modelname}_R{int(acceleration_factor)}_target.nii.gz"
            write_numpy_to_nifti(target_np, fname_target)
        # 3 - Write the sampling mask
        if samp_mask_np is not None:
            fname_mask = pat_dir / f"{modelname}_R{int(acceleration_factor)}_mask.nii.gz"
            write_numpy_to_nifti(samp_mask_np, fname_mask)
    
    # Write the output to an H5 file
    h5_path = pat_dir / filename
    with h5py.File(h5_path, "w") as f:
        f.create_dataset(output_key, data=volume_np)
        if target_np is not None:
            f.create_dataset("target", data=target_np)
        # if samp_mask_np is not None:
        #     f.create_dataset("mask", data=samp_mask_np)
        f.attrs["acceleration_factor"] = acceleration_factor
        f.attrs["modelname"] = modelname
    logger.info(f"Wrote H5 file: {h5_path}")


def write_output_to_h5(
    output: list,
    output_directory: pathlib.Path,
    volume_processing_func: callable                = None,
    output_key: str                                 = "reconstruction",
    create_dirs_if_needed: bool                     = True,
    modelname: str                                  = None,
    acceleration_factor: Union[float, list, None]   = None,
    also_write_nifti: bool                          = False,
    do_round: bool                                  = False,
    decimals: int                                   = 3,
    added_gaussian_noise: bool                      = False,
    db_path: str                                    = None,
    tablename: str                                  = None,
) -> None:
    """
    Write outputs for multiple patients to H5 files in a modular fashion.
    
    Parameters
    ----------
    output : list
        List of patient data tuples: (volume, target, _, filename, samp_mask, pat_id).
    output_directory : pathlib.Path
        Directory where outputs will be saved.
    volume_processing_func : callable, optional
        Function to postprocess the volume before saving.
    output_key : str, optional
        Key under which to save the reconstruction.
    create_dirs_if_needed : bool, optional
        If True, create the output directory (and parents) if necessary.
    modelname : str, optional
        Model name to include in filenames and attributes.
    acceleration_factor : Union[float, list, None], optional
        Acceleration factor(s) for the reconstruction.
    also_write_nifti : bool, optional
        If True, also write NIfTI files.
    do_round : bool, optional
        If True, round the volumes to a specified number of decimals.
    decimals : int, optional
        Number of decimals for rounding if do_round is True.
    
    Returns
    -------
    None
    """
    if create_dirs_if_needed:
        output_directory.mkdir(exist_ok=True, parents=True)
        logger.info(f"Output directory: {output_directory}")

    logger.info(f"Number of output items: {len(output)}")
    
    # If acceleration_factor is a list, take the first element.
    if isinstance(acceleration_factor, list):
        acceleration_factor = acceleration_factor[0]
    
    # Process each patient's data.
    for idx, patient_data in enumerate(output):
        _, _, _, filename, _, pat_id = patient_data
        pat_dir = output_directory / pat_id
        pat_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Processing patient directory: {pat_dir}")
        logger.info(f"({idx + 1}/{len(output)}): Writing {pat_dir / filename}...")
        
        process_patient_output(
            patient_data,
            pat_dir,
            modelname,
            acceleration_factor,
            output_key              = output_key,
            volume_processing_func  = volume_processing_func,
            do_round                = do_round,
            decimals                = decimals,
            also_write_nifti        = also_write_nifti,
            added_gaussian_noise    = added_gaussian_noise,
            db_path                 = db_path,
            tablename               = tablename,
        )


def write_numpy_to_nifti(img: np.ndarray, out_fname: str) -> None:
    """
    Here we write a numpy array to a NIfTI file.

    Args:
    img: np.ndarray
        The numpy array to write to a NIfTI file.
    out_fname: str
        The output filename.

    Returns:
    None
    """
    assert img.dtype == np.float32, f"Expected float32 array, got {img.dtype}"
    print(f"Incoming write_numpy_to_nifti shape: {img.shape}")

    # if the shape is 4 and the last dim is 1, we remove it
    if img.ndim == 4 and img.shape[-1] == 1:
        img = img.squeeze(-1)

    assert img.ndim == 3, f"Expected 3D array, got {img.ndim}D array"

    if not isinstance(out_fname, str):
        out_fname = str(out_fname)

    img = sitk.GetImageFromArray(img)
    sitk.WriteImage(img, out_fname)
    logger.info(f"Writing {out_fname}...")
