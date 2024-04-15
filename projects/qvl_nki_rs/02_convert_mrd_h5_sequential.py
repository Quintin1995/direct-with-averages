import os
import h5py
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import argparse
import pydicom
from datetime import datetime
from projects.qvl_nyu_prostate_rim.helper_functions import *

    
def reorder_k_space_even_odd(ksp: np.ndarray, verbose=False) -> np.ndarray:
    """
    Rearranges the k-space data by interleaving even and odd indexed slices.

    Parameters:
    - ksp (numpy.ndarray): The input k-space data to be rearranged.

    Returns:
    - interleaved_ksp (numpy.ndarray): The k-space data with interleaved slices.
    """
    # assert that the kspace is 5D
    assert len(ksp.shape) == 5, f"kspace is not 5D. Shape: {ksp.shape}, shape should be (navgs, nslices, ncoils, n_freq, n_phase)"

    # Initialize a new complex array to store the interleaved k-space data
    interleaved_ksp = np.zeros_like(ksp, dtype=np.complex64)

    # Calculate the middle index for slicing the array into two halves
    num_slices = ksp.shape[1]
    middle_idx = (num_slices + 1) // 2  # Handles both odd and even cases

    if verbose:
        print(f"\tReordering k-space by interleaving even and odd indexed slices, num_slices: {num_slices}, middle_idx: {middle_idx} is_even: {num_slices % 2 == 0}")

    # Interleave even and odd indexed slices, for some reason it depends on being even or odd.
    if num_slices % 2 == 0: # Even number of slices
        ksp = np.flip(ksp, axis=1)
        interleaved_ksp[:, ::2] = ksp[:, middle_idx:]  # Place the second half at even indices
        interleaved_ksp[:, 1::2] = ksp[:, :middle_idx]  # Place the first half at odd indices
    else: # Odd number of slices
        interleaved_ksp[:, ::2]  = ksp[:, :middle_idx]
        interleaved_ksp[:, 1::2] = ksp[:, middle_idx:]
        interleaved_ksp = np.flip(interleaved_ksp, axis=1)

    return interleaved_ksp


def build_kspace_array_from_mrd_umcg(fpath_mrd: str, verbose=True):
    '''
    Arguments:
        - fpath: path to the .mrd file
    
    Returns:
        - kspace: numpy array of kspace data in shape (navgs, nslices, ncoils, rNx, eNy + 1) complex
    '''
    if verbose:
        print(f"\tBuilding kspace array from .mrd file")

    # Read the header and get encoding information
    dset   = ismrmrd.Dataset(fpath_mrd, create_if_needed=False)
    header = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
    enc    = header.encoding[0]

    # Determine some parameters of the acquisition
    ncoils     = header.acquisitionSystemInformation.receiverChannels
    nslices    = enc.encodingLimits.slice.maximum + 1 if enc.encodingLimits.slice is not None else 1
    eNy        = enc.encodedSpace.matrixSize.y
    rNx        = enc.reconSpace.matrixSize.x
    # eTL        = 25 if DEBUG else echo_train_length(dset)
    # eTC        = 11 if DEBUG else echo_train_count(dset, echo_train_len=eTL)
    firstacq   = get_first_acquisition(dset)
    navgs      = 3 #if DEBUG else get_num_averages(firstacq=firstacq, dset=dset)

    # Loop through the rest of the acquisitions and fill the data array with the kspace data
    kspace = np.zeros((navgs, nslices, ncoils, rNx, eNy + 1), dtype=np.complex64)
    print(f"\tFilling kspace array from mrd object to shape {kspace.shape}...\n\tNum Acquisitions: {dset.number_of_acquisitions()}", end=" \n\t\tLoading... ", flush=True)

    for acqnum in range(firstacq, dset.number_of_acquisitions()):
        
        if acqnum % 1000 == 0:
            print(f"{acqnum/dset.number_of_acquisitions() * 100:.0f}%", end=" ", flush=True)
        
        acq    = dset.read_acquisition(acqnum)
        slice1 = acq.idx.slice
        y      = acq.idx.kspace_encode_step_1
        avg    = acq._head.idx.average

        # Each acquisition is a 2D array of shape (coil, rNx) complex
        kspace[avg, slice1, :, :, y] = acq.data

    print()
    return kspace


def change_headers_based_on_phase_cropping(fpath_mrd: str, max_phase_int: int) -> bytes:
    """
    Adjust the headers of an .mrd file to match the NYU format based on phase cropping.

    Parameters:
    - fpath_mrd: Path to the .mrd file.
    - max_phase_int: Maximum phase integer value for cropping.

    Returns:
    - header_bytes: Transformed headers in byte format.
    """

    # Namespace string for ISMRMRD XML
    ns = "{http://www.ismrm.org/ISMRMRD}"

    # Retrieve the headers from the .mrd file
    umcg_headers_mrd = get_headers_from_ismrmrd(fpath_mrd, verbose=False)

    # Convert the headers to a dictionary
    umcg_headers_dict = convert_ismrmrd_headers_to_dict(umcg_headers_mrd)

    # Update headers with the correct matrix size and encoding limits for NYU data format
    umcg_headers_dict[f"{ns}ismrmrdHeader"][f"{ns}encoding"][f"{ns}encodedSpace"][f"{ns}matrixSize"][f"{ns}y"] = str(max_phase_int)
    umcg_headers_dict[f"{ns}ismrmrdHeader"][f"{ns}encoding"][f"{ns}encodingLimits"][f"{ns}kspace_encoding_step_1"][f"{ns}maximum"] = str(max_phase_int)
    umcg_headers_dict[f"{ns}ismrmrdHeader"][f"{ns}encoding"][f"{ns}encodingLimits"][f"{ns}kspace_encoding_step_1"][f"{ns}center"] = str(max_phase_int//2)

    header_bytes = encode_umcg_header_to_bytes(umcg_to_nyu_dict=umcg_headers_dict)
    return header_bytes


def crop_kspace_in_phase_direction(
        kspace: np.ndarray,
        max_phase_crop: int,
        fpath_mrd: str,
        verbose: bool = False
) -> np.ndarray:
    """
    Crop the k-space in the phase direction to achieve the desired target shape.
    
    Arguments:
    - ksp: 5D numpy array of shape (navgs, nslices, ncoils, read, phase) complex.
    - max_phase: Maximum phase integer value for cropping. in the phase direction
    - fpath_mrd: Path to the .mrd file. This is needed to change the headers.
    - verbose: Print the cropping details.

    Returns:
    - Cropped k-space numpy array.
    """
    
    # Check input shape validity
    if len(kspace.shape) != 5:
        raise ValueError("ksp must be 5D (navgs, nslices, ncoils, read, phase) complex.")
    if kspace.shape[-1] < max_phase_crop:
        raise ValueError("The k-space phase dimension should be smaller than the desired phase crop.")
    
    # the headers of the kspace must be changed if you do a phase cropping. This is read from the MRD file.
    new_hdrs = change_headers_based_on_phase_cropping(fpath_mrd, max_phase_int=max_phase_crop)
    
    if kspace.shape[-1] == max_phase_crop:
        if verbose:
            print("Kspace and desired phase shape are equal so return kspace as is.")
        return kspace, new_hdrs

    # Calculate the cropping size in the phase direction
    phase_crop_size = kspace.shape[-1] - max_phase_crop
    left_crop       = phase_crop_size // 2
    right_crop      = phase_crop_size - left_crop

    if verbose:
        print(f"Original k-space shape: {kspace.shape}")
        cur_shape = list(kspace.shape)
        cur_shape[-1] -= phase_crop_size
        print(f"Cropped kspace will be: {tuple(cur_shape)}")

    # Return the cropped k-space and the new headers
    return kspace[..., left_crop:-right_crop], new_hdrs


def convert_mrd_to_array(
    fpath_mrd: str,
    pat_rec_dir: str,
    max_mag_ref: float,
    do_rm_zero_pad: bool,
    do_norm_to_ref: bool,
    max_phase_crop: int = None,
    verbose: bool = True,
) -> None:
    '''
    Description:
        This function converts a .mrd file to a numpy array.
        The kspace is cropped in the phase direction to the shape of the NYU dataset.
        The kspace is normalized to the reference magnitude of the NYU dataset.
    Args:
        fpath (str): The path to the .mrd file.
        phase_crop_shape (tuple): The shape to crop the kspace to.
        max_mag_ref (float): The reference magnitude.
        do_rm_zero_pad (bool): If True, the zero padding is removed.
        do_norm_to_ref (bool): If True, the magnitude is normalized to the reference magnitude.
    Returns:
        kspace (np.ndarray): The kspace array.
        trans_hdrs (dict): The transformed headers.
    '''
    
    # Construct the kspace array from the sequentail MRD object.
    kspace = build_kspace_array_from_mrd_umcg(fpath_mrd, verbose=True)

    # Reorder the slices of the kspace based on even and odd number of slices
    kspace = reorder_k_space_even_odd(kspace)

    # Remove the zero padding from the kspace.
    if do_rm_zero_pad:
        kspace = remove_zero_padding(kspace, verbose=True)

    if max_phase_crop == None:
        max_phase_crop = kspace.shape[-1]

    # Crop the kspace in the phase dir and obtain the transformed headers. Simply extracts the headers as is, if the crop shape is equal to the kspace shape.
    kspace, trans_hdrs = crop_kspace_in_phase_direction(kspace, max_phase_crop=max_phase_crop, fpath_mrd=fpath_mrd)

    if do_norm_to_ref:
        kspace = normalize_to_reference(kspace, max_magni_ref=max_mag_ref, verbose=True)

    if verbose:
        safe_rss_to_nifti_file(kspace, fname_part="pre_processed_ksp", do_round=True, dir=pat_rec_dir)

    return kspace, trans_hdrs


def remove_zero_padding(kspace: np.ndarray, verbose=False) -> np.ndarray:
    """
    Description:
        Remove the zero padding in the phase encoding direction from the given k-space.
    Args:
        kspace (np.ndarray): The k-space data. Should have 5 dimensions: (n_avg, n_slices, n_coils, n_freq, n_phase)
    Returns:
        np.ndarray: The k-space data without zero padding in the phase encoding direction.
    """
    assert kspace.ndim == 5, "image should have 5 dimensions: (n_avg, n_slices, n_coils, n_freq, n_phase)"

    if verbose:
        print(f"\tkspace shape before zero-padding removal: {kspace.shape}")

    n_avg, n_slices, n_coils, n_freq, n_phase = kspace.shape

    # Remove the zero padding in the phase encoding direction
    zero_padding, idxs = calculate_zero_padding_PE(kspace)

    if verbose:
        print(f"\tFound zero padding of {zero_padding} in phase encoding direction.")
        # print(f"\tFound zero padding indices: {idxs}")

    return kspace[:, :, :, :, 0:n_phase - zero_padding]


def create_h5_for_patient(
    pat_dir: Path,
    patients_dir: Path,
    add_kspace: bool     = True,
    add_attributes: bool = True,
) -> None:

    # Get the patient id and the anonymized patient id.
    pat_seq_id = pat_dir.name.split('_')[0]
    pat_anon_id = pat_dir.name.split('_')[1]

    # Create the h5 file if it does not exist.
    fpath_hf = Path(patients_dir, f"{pat_dir}_pst_T2.h5")
    create_h5_if_not_exists(fpath_hf)

    # Get the dcm headers partially
    pat_dcm_dir = Path(patients_dir, pat_dir, 't2w_tra_dicom')
    pat_rec_dir = Path(patients_dir, pat_dir, 'recons')

    # read the first file in the pat_dcm_dir to get the dcm headers.
    dcm_hdrs = get_meta_data_from_dcm_headers(next(pat_dcm_dir.iterdir()))

    do_rm_zero_pad = True
    do_norm_to_ref = True
    max_phase_crop = None

    # Add the K-space to the H5 file.
    if add_kspace:
        with h5py.File(fpath_hf, 'r+') as hf:
            if not has_key_in_h5(fpath_hf, 'kspace'):
                kspace, headers = convert_mrd_to_array(
                    fpath_mrd        = get_t2_tra_mrd_fname(Path(patients_dir, pat_dir), verbose=True),
                    pat_rec_dir      = pat_rec_dir,
                    max_mag_ref      = 0.010586672,  # Entire NYU test and validation dataset # one patient: 0.006096669
                    do_rm_zero_pad   = do_rm_zero_pad, 
                    do_norm_to_ref   = do_norm_to_ref,
                    max_phase_crop   = max_phase_crop,         # None means that the kspace is not cropped in the phase dir.
                )   
                hf.create_dataset('ismrmrd_header', data=headers)
                hf.create_dataset('kspace', data=kspace)
                print(f"Created 'kspace' dataset and 'ismrmrd_header' in {fpath_hf}")

            if not has_correct_shape(fpath_hf):
                raise Exception(f"kspace shape is not correct. Shape: {hf['kspace'].shape}")

    # Add the attributes to the H5 file.
    if add_attributes:
        max_for_now  = 0.0004   # change in the future to something that makes sense
        norm_for_now = 0.12     # change in the future to something that makes sense
        with h5py.File(fpath_hf, 'r+') as hf:
            if len(dict(hf.attrs)) == 0:
                hf.attrs['acquisition'] = 'AXT2'
                hf.attrs['max'] = max_for_now
                hf.attrs['norm'] = norm_for_now
                hf.attrs['patient_id'] = pat_anon_id
                hf.attrs['patient_id_seq'] = pat_seq_id
                hf.attrs['do_rm_zero_pad'] = do_rm_zero_pad
                hf.attrs['do_norm_to_ref'] = do_norm_to_ref
                hf.attrs['max_phase_crop'] = 'None' if max_phase_crop is None else str(max_phase_crop)
                for key in dcm_hdrs.keys():
                    hf.attrs[key + "_dcm_hdr"] = dcm_hdrs[key]
        print(f"\tAdded attributes to h5")
        with h5py.File(fpath_hf, 'r') as hf:
            for key in dict(hf.attrs).keys():
                print(f"\t\t{key}: {hf.attrs[key]}")


def get_only_patient_directories(patients_dir: Path, pat_exclusion_list = None) -> list:
    '''
    Description:
        This function returns a list of patient directories. 
        Only retain the directories with the pattern: 0003_ANON5046358
        This is check by checking if the length of the split is equal to 2.
    Args:
        patients_dir (str): The path to the patients directory.
    Returns:
        pat_dirs (list): A list of patient directories.
    '''

    pat_dirs = os.listdir(patients_dir)
    pat_dirs = [pat_dir for pat_dir in pat_dirs if len(pat_dir.split('_')) == 2]

    # sort the list based on the first part of the split - not important, just chronological
    pat_dirs = sorted(pat_dirs, key=lambda x: int(x.split('_')[0]))

    if pat_exclusion_list is not None:
        print(f"Excluding patients: {pat_exclusion_list}")
        pat_dirs = [pat_dir for pat_dir in pat_dirs if pat_dir.split('_')[0] not in pat_exclusion_list]

    return pat_dirs


def create_h5s_for_all_patients(patients_dir: Path, pat_exclusion_list = None):

    pat_dirs = get_only_patient_directories(patients_dir, pat_exclusion_list)
    print(f"Found {len(pat_dirs)} patient dirs in {patients_dir}")

    # Create the h5 files for all patients.
    for f_idx, pat_dir in enumerate(pat_dirs):
        print(f"\nProcessing patient {f_idx+1}/{len(pat_dirs)} with dir {pat_dir}")
        
        create_h5_for_patient(
            pat_dir        = Path(pat_dir),
            patients_dir   = patients_dir,
            add_kspace     = True,
            add_attributes = True,
        )
        if DEBUG:
            input(f"This was patient {pat_dir}\nPress enter to continue...\n")


def get_first_acquisition(dset) -> int:
    '''
    Arguments:
        - dset: ismrmrd.Dataset object
    Returns:
        - firstacq: index of the first acquisition
    '''
    firstacq = 0
    for acqnum in range(dset.number_of_acquisitions()):
        acq = dset.read_acquisition(acqnum)
        if acq.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
            continue
        else:
            firstacq = acqnum
            print("\tImaging acquisition starts at acq: ", acqnum)
            break
    return firstacq


def parse_args():

    parser = argparse.ArgumentParser(description='Convert .mrd files to .h5 files.')

    parser.add_argument('--debug', action='store_true', help='Debug mode')

    # add dataset path dir argument
    parser.add_argument('--dataset_dir', type=str, default='/scratch/p290820/datasets/003_umcg_pst_ksps', help='The path to the dataset directory. This is UMCG prostate k-space data')

    args = parser.parse_args()

    # print the args in for loop
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    return args



##########################################################################################
if __name__ == "__main__":

    # I think this file is OLD.
    # The conversion of raw data to .mrd format is done on the UMCG Desktop instead of here.

    args = parse_args()

    DEBUG        = args.debug
    DATASET_DIR  = Path(args.dataset_dir)
    PATIENTS_DIR = Path(DATASET_DIR, "pat_data")

    # H5 files
    # FPATH_UMCG_H5_SAMPLE = 
    FPATH_NYU_H5_SAMPLE  = Path("/scratch/p290820/datasets/prostate_nyu/training_T2_1/file_prostate_AXT2_0001.h5")
    FPATH_RUMC_H5_SAMPLE = Path("/scratch/p290820/datasets/prostate_rumc/workspace/output/h5s/10621_pst_T2.h5")

    # MRD files
    FPATH_UMCG_MRD_SAMPLE = Path('/scratch/p290820/datasets/prostate_ksp_umcg/workspace/output/anon_kspaces/0001_patient_umcg_done/meas_MID00202_FID688156_T2_TSE_tra_obl-out_2.mrd')
    FPATH_RUMC_MRD_SAMPLE = Path("/scratch/p290820/datasets/prostate_rumc/pst_example/raw/10621/2022-11-15/130632826694966761971590153404660617811#t2_tse_tra_snel.mrd")

    # Exclusion list for patients. Sequential IDs
    # PAT_MAP_DF = pd.read_csv(Path(DATASET_DIR, 'mapping.csv'), dtype=str)
    PAT_EXCLUSION_LIST = ['0001', '0011', '0012']
    create_h5s_for_all_patients(patients_dir=PATIENTS_DIR, pat_exclusion_list=PAT_EXCLUSION_LIST)