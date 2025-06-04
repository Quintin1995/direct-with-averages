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


def round_volume(volume: np.ndarray, decimals: int = 3) -> np.ndarray:
    """
    Round the volume to a specified number of decimals.
    """
    return np.round(volume, decimals=decimals)


def get_gaussian_id(
        pat_id: str,
        db_path: str,
        tablename: str,
        filename: pathlib.Path,
        avg_acc: float = None,
        debug: bool = False) -> int:
    """
    Retrieve and update the gaussian_id for a patient in the database.
    
    Steps:
    1. Query the latest row for the patient (with matching avg_acceleration).
    2. If the current row's recon_path is NULL:
         - Update the row with a new recon_path (filename with gaussian_id appended).
         - Insert a new row with gaussian_id incremented by 1 (and recon_path as NULL).
       Else (if recon_path is not NULL):
         - Insert a new row with gaussian_id incremented by 1.
    3. Return the new gaussian_id to be used.
    
    This guarantees that each call creates a new row for the next iteration.
    
    Args:
        pat_id (str): The patient identifier.
        db_path (str): Path to the database.
        tablename (str): Name of the table in the database.
        filename (Path): The filename to set as recon_path if it was NULL.
        avg_acc (float): Average acceleration factor (cast to float if stored as int in DB).
        debug (bool): Flag to enable debug logging.
        
    Returns:
        int: The new gaussian_id to use.
    """
    conn = connect(db_path)
    cursor = conn.cursor()
    
    # Query the latest row for the patient with the given acceleration factor.
    query = f"""
        SELECT id, seq_id, anon_id, gaussian_id, recon_path, avg_acceleration
        FROM {tablename}
        WHERE id = ? AND avg_acceleration = ?
        ORDER BY gaussian_id DESC
        LIMIT 1;
    """
    cursor.execute(query, (pat_id, avg_acc))
    row = cursor.fetchone()
    if row is None:
        conn.close()
        message = (f"QVL - Writers.py - No row found for patient ID: {pat_id} "
                   f"and avg_acceleration: {avg_acc}. Query: {query} "
                   f"db_path: {db_path} tablename: {tablename} filename: {filename}")
        raise ValueError(message)
    
    id_val, seq_id, anon_id, gaussian_id, recon_path, avg_acc = row
    
    if debug:
        # (Assuming logger is defined elsewhere, or you can use print statements)
        logger.info(f"QVL - Writers.py - Retrieved gaussian_id: {gaussian_id} for patient ID: {pat_id}. "
                    f"recon_path: {recon_path}.")
    
    new_gaussian_id = gaussian_id + 1
    
    # If recon_path is NULL, update the current row with a new recon_path.
    if recon_path is None:
        # Append the current gaussian_id to the filename before the extension.
        new_filename = filename.stem + f"_{gaussian_id}" + filename.suffix
        update_query = f"""
            UPDATE {tablename}
            SET recon_path = ?
            WHERE id = ? AND gaussian_id = ? AND avg_acceleration = ?;
        """
        cursor.execute(update_query, (str(filename.with_name(new_filename)), pat_id, gaussian_id, avg_acc))
        conn.commit()
        
        if debug:
            logger.info(f"QVL - Writers.py - Updated recon_path to {new_filename} for patient ID: {pat_id} "
                        f"with gaussian_id: {gaussian_id}.")
    
    # In both cases, insert a new row with the incremented gaussian_id and recon_path set to NULL.
    insert_query = f"""
        INSERT INTO {tablename} (id, seq_id, anon_id, gaussian_id, recon_path, avg_acceleration)
        VALUES (?, ?, ?, ?, NULL, ?);
    """
    cursor.execute(insert_query, (pat_id, seq_id, anon_id, new_gaussian_id, avg_acc))
    conn.commit()
    
    if debug:
        logger.info(f"QVL - Writers.py - Inserted new row for patient ID: {pat_id} with gaussian_id: {new_gaussian_id} "
                    f"and recon_path set to NULL. avg_acceleration: {avg_acc}")
    
    conn.close()
    return new_gaussian_id


# def process_patient_output(
#     patient_data: tuple,
#     pat_dir: pathlib.Path,
#     modelname: str,
#     avg_acc: float    = None,
#     output_key: str                   = "reconstruction",
#     volume_processing_func: callable  = None,
#     do_round: bool                    = False,
#     decimals: int                     = 3,
#     also_write_nifti: bool            = False,
#     added_gaussian_noise: bool        = False,
#     db_path: str                      = None,
#     tablename: str                    = None,
#     do_lxo_for_uq: bool               = None,
#     echo_train_fold_idx: int          = None,
#     echo_train_acceleration: int      = None,
# ) -> None:
#     """
#     Process and write output for a single patient.
    
#     patient_data is expected to be a tuple:
#       (volume, target, _, filename, samp_mask, pat_id)
#     where volume is a torch tensor with shape [slices, 1, height, width].
#     """
#     assert avg_acc is not None, "avg_acceleration_factor must be provided."
#     assert modelname is not None, "modelname must be provided."
#     assert db_path is not None, "db_path must be provided."
#     assert tablename is not None, "tablename must be provided."

#     volume, target, _, filename, samp_mask, pat_id = patient_data

#     if True:
#         print(f"Writer - Shape of the volume: {volume.shape}")
#         print(f"Writer - Shape of the target: {target.shape}") if target is not None else None
#         print(f"Writer - The filename is: {filename}")
#         print(f"Writer - The patient ID is: {pat_id}")
#         print(f"Writer - Type patient id: {type(pat_id)}")
#         print(f"Writer - The avg_acceleration_factor is: {avg_acc}")
#         print(f"Writer - The modelname is: {modelname}")
#         print(f"Writer - The output_key is: {output_key}")
#         print(f"Writer - The volume_processing_func is: {volume_processing_func}")
#         print(f"Writer - The do_round is: {do_round}")
#         print(f"Writer - The decimals is: {decimals}")
#         print(f"Writer - The also_write_nifti is: {also_write_nifti}")
#         print(f"Writer - The added_gaussian_noise is: {added_gaussian_noise}")
#         print(f"Writer - The db_path is: {db_path}")
#         print(f"Writer - The tablename is: {tablename}")
#         print(f"Writer - The do_lxo_for_uq is: {do_lxo_for_uq}")
#         print(f"Writer - The echo_train_fold_idx is: {echo_train_fold_idx}")
#         print(f"Writer - The echo_train_acceleration is: {echo_train_acceleration}")

#     # here we connect with the database and see if what the gaussian noise index should be for this patient and update the database with a new row, so that the next time it will be updated with a new index
#     if added_gaussian_noise:
#         gaussian_id = get_gaussian_id(
#             pat_id    = str(pat_id),
#             db_path   = db_path,
#             tablename = tablename,
#             filename  = pat_dir / filename,
#             avg_acc   = avg_acc,
#             debug     = True)
#         # Modify filename with fnamepart if provided
#         if isinstance(filename, pathlib.PosixPath):
#             filename = filename.name
#         filename = filename.replace(".h5", f"_gaus{gaussian_id}.h5")
#     else:
#         gaussian_id = ""

#     # Convert torch tensors to numpy arrays
#     if volume is not None:
#         volume_np = volume.cpu().numpy()[:, 0, ...].astype(np.float32)
#     else:
#         volume_np = None

#     if target is not None:
#         target_np = target.cpu().numpy()[:, 0, ...].astype(np.float32)
#     else:
#         target_np = None

#     if samp_mask is not None:
#         samp_mask_np = samp_mask.cpu().numpy()[:, 0, ...].astype(np.float32)
#     else:
#         samp_mask_np = None

#     # Optionally process volume
#     if volume_processing_func is not None:
#         volume_np = volume_processing_func(volume_np)

#     # Optionally round volume
#     if do_round:
#         volume_np = round_volume(volume_np, decimals)
#         target_np = round_volume(target_np, decimals) if target_np is not None else None

#     if also_write_nifti:
#         # 1 - Write the reconstruction
#         fname_recon = pat_dir / f"{modelname}_R{int(avg_acc)}_recon_gaus{gaussian_id}.nii.gz"
#         write_numpy_to_nifti(volume_np, fname_recon)
#         # 2 - Write the target
#         if target_np is not None:
#             fname_target = pat_dir / f"{modelname}_R{int(avg_acc)}_target.nii.gz"
#             write_numpy_to_nifti(target_np, fname_target)
#         # 3 - Write the sampling mask
#         if samp_mask_np is not None:
#             fname_mask = pat_dir / f"{modelname}_R{int(avg_acc)}_mask.nii.gz"
#             write_numpy_to_nifti(samp_mask_np, fname_mask)
    
#     # H5 Writing
#     h5_path = pat_dir / filename
#     with h5py.File(h5_path, "w") as f:
#         f.create_dataset(output_key, data=volume_np)
#         f.create_dataset("target", data=target_np) if target_np is not None else None
#         f.create_dataset("mask", data=samp_mask_np) if samp_mask_np is not None else None
#         f.attrs["acceleration_factor"] = avg_acc
#         f.attrs["modelname"] = modelname
        
#         if do_lxo_for_uq:
#             f.attrs["do_lxo_for_uq"] = True
#             f.attrs["echo_train_fold_idx"] = echo_train_fold_idx
#             f.attrs["echo_train_acceleration"] = echo_train_acceleration

#     logger.info(f"Wrote H5 file: {h5_path}")


def process_patient_output(
    patient_data: tuple,
    pat_dir: pathlib.Path,
    modelname: str,
    avg_acc: float = None,
    output_key: str = "reconstruction",
    volume_processing_func: callable = None,
    do_round: bool = False,
    decimals: int = 3,
    also_write_nifti: bool = False,
    added_gaussian_noise: bool = False,
    db_path: str = None,
    tablename: str = None,
    do_lxo_for_uq: bool = None,
    echo_train_fold_idx: int = None,
    echo_train_acceleration: int = None,
) -> None:
    """
    Process and write output for a single patient.
    patient_data is expected to be a tuple:
    (volume, target, _, filename, samp_mask, pat_id)
    """
    assert avg_acc is not None, "avg_acceleration_factor must be provided."
    assert modelname is not None, "modelname must be provided."
    assert db_path is not None, "db_path must be provided."
    assert tablename is not None, "tablename must be provided."
    assert not (do_lxo_for_uq and added_gaussian_noise), \
        "Cannot apply both echo-train dropout and Gaussian noise simultaneously."
    
    if True:
        logger.info(f"Modelname: {modelname}")
        logger.info(f"Avg Acceleration: {avg_acc}")
        logger.info(f"Output Key: {output_key}")
        logger.info(f"Also Write NIfTI: {also_write_nifti}")
        logger.info(f"Added Gaussian Noise: {added_gaussian_noise}")
        logger.info(f"DB Path: {db_path}")
        logger.info(f"Tablename: {tablename}")
        logger.info(f"Do LXO for UQ: {do_lxo_for_uq}")
        logger.info(f"Echo Train Fold Index: {echo_train_fold_idx}")
        logger.info(f"Echo Train Acceleration: {echo_train_acceleration}")

    volume, target, _, filename, samp_mask, pat_id = patient_data

    if isinstance(filename, pathlib.Path):
        filename = filename.name
    base_filename = filename.replace(".h5", "")

    if do_lxo_for_uq:
        filename = f"{base_filename}_R{avg_acc}_lxofold{echo_train_fold_idx}.h5"
        gaussian_id = ""
    elif added_gaussian_noise:
        gaussian_id = get_gaussian_id(
            pat_id=str(pat_id),
            db_path=db_path,
            tablename=tablename,
            filename=pat_dir / filename,
            avg_acc=avg_acc,
            debug=True
        )
        filename = f"{base_filename}_gaus{gaussian_id}.h5"
    else:
        filename = f"{base_filename}.h5"
        gaussian_id = ""

    # Convert torch tensors to numpy arrays
    volume_np = volume.cpu().numpy()[:, 0, ...].astype(np.float32) if volume is not None else None
    target_np = target.cpu().numpy()[:, 0, ...].astype(np.float32) if target is not None else None
    samp_mask_np = samp_mask.cpu().numpy()[:, 0, ...].astype(np.float32) if samp_mask is not None else None

    if volume_processing_func is not None:
        volume_np = volume_processing_func(volume_np)

    if do_round:
        volume_np = round_volume(volume_np, decimals)
        if target_np is not None:
            target_np = round_volume(target_np, decimals)

    if also_write_nifti:
        suffix = f"_R{int(avg_acc)}"
        if do_lxo_for_uq:
            suffix += f"_lxofold{echo_train_fold_idx}"
        elif added_gaussian_noise:
            suffix += f"_gaus{gaussian_id}"

        write_numpy_to_nifti(volume_np, pat_dir / f"{modelname}{suffix}_recon.nii.gz")
        if target_np is not None:
            write_numpy_to_nifti(target_np, pat_dir / f"{modelname}{suffix}_target.nii.gz")
        if samp_mask_np is not None:
            write_numpy_to_nifti(samp_mask_np, pat_dir / f"{modelname}{suffix}_mask.nii.gz")

    h5_path = pat_dir / filename
    with h5py.File(h5_path, "w") as f:
        f.create_dataset(output_key, data=volume_np)
        if target_np is not None:
            f.create_dataset("target", data=target_np)
        if samp_mask_np is not None:
            f.create_dataset("mask", data=samp_mask_np)
        f.attrs["acceleration_factor"] = avg_acc
        f.attrs["modelname"] = modelname

        if do_lxo_for_uq:
            f.attrs["do_lxo_for_uq"] = True
            f.attrs["echo_train_fold_idx"] = echo_train_fold_idx
            f.attrs["echo_train_acceleration"] = echo_train_acceleration

    logger.info(f"Wrote H5 file: {h5_path}")


def write_output_to_h5(
    output: list,
    output_directory: pathlib.Path,
    volume_processing_func: callable   = None,
    output_key: str                    = "reconstruction",
    create_dirs_if_needed: bool        = True,
    modelname: str                     = None,
    avg_acc: Union[float, list, None]  = None,
    also_write_nifti: bool             = False,
    do_round: bool                     = False,
    decimals: int                      = 3,
    added_gaussian_noise: bool         = False,
    db_path: str                       = None,
    tablename: str                     = None,
    do_lxo_for_uq: bool                = None,
    echo_train_fold_idx: int           = None,
    echo_train_acceleration: int       = None,
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
    avg_acceleration_factor : Union[float, list, None], optional
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
    if isinstance(avg_acc, list):
        avg_acc = avg_acc[0]
    
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
            avg_acc                 = avg_acc,
            output_key              = output_key,
            volume_processing_func  = volume_processing_func,
            do_round                = do_round,
            decimals                = decimals,
            also_write_nifti        = also_write_nifti,
            added_gaussian_noise    = added_gaussian_noise,
            db_path                 = db_path,
            tablename               = tablename,
            do_lxo_for_uq           = do_lxo_for_uq,
            echo_train_fold_idx     = echo_train_fold_idx,
            echo_train_acceleration = echo_train_acceleration,
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
