from pathlib import Path
import SimpleITK as sitk
import pydicom
import logging
from typing import List
from datetime import datetime


def read_patient_ids_from_log(log_file_path):
    if log_file_path.exists():
        with open(log_file_path, 'r') as file:
            return set(file.read().splitlines())
    return set()


def log_processed_patient(log_file_path, pat_id):
    with open(log_file_path, 'a') as file:
        file.write(f"{pat_id}\n")


def convert_nifti_to_mha(source_path: Path, target_path: Path, logger: logging.Logger) -> sitk.Image:
    try:
        img = sitk.ReadImage(str(source_path))
        sitk.WriteImage(img, str(target_path))
        logger.info(f"\tConverted and wrote: {target_path}")
        return img
    except RuntimeError as e:
        logger.error(f"\tError processing {source_path}: {e}")
        return None


def create_and_save_black_image_like(reference_image: sitk.Image, target_path: Path, logger: logging.Logger = None) -> None:
    black_image = sitk.Image(reference_image.GetSize(), reference_image.GetPixelIDValue())
    black_image.SetOrigin(reference_image.GetOrigin())
    black_image.SetSpacing(reference_image.GetSpacing())
    black_image.SetDirection(reference_image.GetDirection())
    
    sitk.WriteImage(black_image, str(target_path))
    logger.info(f"\tBlack reference image saved at: {target_path}")


def copy_inferences_to_target_dir(
    root_dir: Path,
    file_mapping: tuple,
    target_base_dir: Path,
    acc: str,
    do_save_empty_ref: bool = False,
    logger: logging.Logger = None,
) -> None:
    """	
    Process the root directory and convert the specified files to mha format.
    
    Parameters:
    - root_dir (Path): Root directory containing the patient directories.
    - file_mapping (tuple): Tuple containing the source and target file names.
    - target_base_dir (Path): Base directory where the converted files will be stored.
    - acc (str): Acceleration factor.
    - do_save_empty_ref (bool): Flag to save a black reference image.
    - logger (logging.Logger): Logger instance for logging messages.
    """
    for idx, patient_dir in enumerate(root_dir.iterdir()):
        logger.info(f"\tProcessing {idx + 1}/{len(list(root_dir.iterdir()))}: {patient_dir}")
        if not patient_dir.is_dir():
            continue

        id = patient_dir.name
        target_dir = target_base_dir / id
        target_dir.mkdir(exist_ok=True)
        
        source_file, target_file_name = file_mapping
        source_path = patient_dir / source_file

        if source_path.exists():
            target_path = target_dir / f"{id}_{target_file_name}"
            if target_path.exists() and (target_dir / f"{id}_black_ref.mha").exists():
                logger.info(f"\tFile {target_path} already exists.")
                continue
            converted_image = convert_nifti_to_mha(source_path, target_path, logger)
            
            # Save a black reference image if needed 
            if do_save_empty_ref and acc == "3x" and converted_image:
                black_image_path = target_dir / f"{id}_black_ref.mha"
                create_and_save_black_image_like(converted_image, black_image_path, logger)
        else:
            logger.warning(f"\tFile {source_path} not found.")


def setup_logger(log_dir: Path, use_time: bool = True, part_fname: str = None) -> logging.Logger:
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
    elif part_fname is not None and use_time: 
        log_file = log_dir / f"log_{part_fname}_{current_time}.log"
    elif part_fname is not None and not use_time:
        log_file = log_dir / f"log_{part_fname}.log"
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


def is_date(string):
    try:
        datetime.strptime(string, "%Y-%m-%d")
        return True
    except ValueError:
        return False
    

def get_study_dir(pat_dcm_dir: Path, pat_id: str) -> Path:
    study_dirs = []
    for d in pat_dcm_dir.iterdir():
        if d.is_dir() and d.name != "archive" and is_date(d.name):
            study_dirs.append(d)
            
    assert len(study_dirs) == 1, f"Error: {pat_id} has {len(study_dirs)} study directories. We didnt find exactly 1 study dir for patient {pat_id}."
    return study_dirs[0]


def sort_t2w_tra_directories_by_creation_time(t2w_tra_directories, study_directory, logger):
    creation_times = []

    for directory_name in t2w_tra_directories:
        series_dir_path = study_directory / directory_name
        first_file_path = next(series_dir_path.iterdir(), None)

        if first_file_path:
            try:
                dicom_header = pydicom.dcmread(first_file_path, stop_before_pixels=True)
                # Convert the creation time string to a float for correct sorting
                creation_time = float(dicom_header.InstanceCreationTime)
                creation_times.append((directory_name, creation_time))
                
                # Log the datatype and value for manual checking
                logger.info(f"Directory: {directory_name}, Converted InstanceCreationTime: {creation_time} (Type: {type(creation_time)})")
                
            except AttributeError as e:
                logger.error(f"InstanceCreationTime attribute missing in {first_file_path}. Error: {e}")
            except ValueError as e:
                # This catches cases where the conversion to float fails
                logger.error(f"Error converting InstanceCreationTime to float in {first_file_path}. Value: '{dicom_header.InstanceCreationTime}', Error: {e}")
            except Exception as e:
                logger.error(f"Error reading {first_file_path}. Error: {e}")
        else:
            logger.warning(f"No DICOM files found in {series_dir_path}")

    # Sort directories by converted creation time (now as floats)
    sorted_directories = sorted(creation_times, key=lambda x: x[1], reverse=True)

    # Extract sorted directory names
    sorted_t2w_tra_directories = [item[0] for item in sorted_directories]
    
    ans = input(f"Creation time: {creation_times}\nSorted directories: {sorted_directories}\n Continue? (y/n): ")
    if ans.lower() != 'y':
        raise Exception("User aborted sorting of T2W TRA directories.")
    
    return sorted_t2w_tra_directories


def find_sequence_directories_dcm(study_dir: Path, patient_id: str = None, logger: logging.Logger = None) -> tuple:
    series_dirs = list(study_dir.iterdir())
    dwi_dirs, adc_dirs, t2w_tra_dirs = [], [], []

    for series_dir in series_dirs:       
        if 'ep_calc' in series_dir.name:  # DWI series
            dwi_dirs.append(series_dir.name)        
        elif 'ep_b50_1000' in series_dir.name:  # ADC series
            adc_dirs.append(series_dir.name)        
        elif 'tse2d1' in series_dir.name:  # T2W series
            first_file = next(series_dir.iterdir(), None)
            if first_file:
                try:
                    description = pydicom.dcmread(first_file, stop_before_pixels=True).SeriesDescription.lower()
                    if 't2_tse_tra' in description:
                        t2w_tra_dirs.append(series_dir.name)
                except Exception as e:
                    logger.error(f"Error reading DICOM file {first_file}: {e}")
        else:
            logger.warning(f"Skipping unknown series directory: {series_dir.name}")

    if len(t2w_tra_dirs) > 1:
        t2w_tra_dirs = sort_t2w_tra_directories_by_creation_time(t2w_tra_dirs, study_dir, logger)
        logger.info(f"Multiple T2W directories found for patient {patient_id}. Selected {t2w_tra_dirs[0]} based on latest acquisition time.")

    logger.info(f"Selected sequences for patient {patient_id}:\n\tDWI: {dwi_dirs[0]},\n\tADC: {adc_dirs[0]},\n\tT2W TRA: {t2w_tra_dirs[0]}") 
    return dwi_dirs, adc_dirs, t2w_tra_dirs


def get_lesions_fnames_if_exists(pat_nif_study_dir: Path) -> list:
    """
    Here we will get the lesion segmentations if they exist.
    
    """
    lesion_segs = list(pat_nif_study_dir.glob('*.nii.gz')) + list(pat_nif_study_dir.glob('*.mha'))
    lesion_segs = [seg for seg in lesion_segs if 'roi' in seg.name.lower()]
    return lesion_segs


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


def resample_and_save_roi_images(study_dir: Path, source_path: Path, target_dir: Path, pat_id: str, logger: logging.Logger):
    """
    Here we will resample the lesion segmentations to the reference image and save them.
    
    Parameters:
    - study_dir (Path): Study directory containing the lesion segmentations.
    - source_path (Path): Source path for the reference image.
    - target_dir (Path): Target directory where the resampled images will be stored.
    - pat_id (str): Patient ID.
    """
    roi_fpaths = get_lesions_fnames_if_exists(study_dir)
    for idx, roi_f in enumerate(roi_fpaths):
        # read the image with sitk and resample it to the last source path
        roi = sitk.ReadImage(str(roi_f))
        ref_img = sitk.ReadImage(str(source_path))
        resampled_roi = resample_to_reference(roi, ref_img)
        # write the image to target and called it roi1.mha, roi2.mha, etc.
        roiname = roi_f.stem.split('.')
        target_path = target_dir / pat_id / f"{pat_id}_{roiname[0].lower()}_{idx+1}.mha"
        sitk.WriteImage(resampled_roi, str(target_path))
        logger.info(f"Resampled and saved {roi_f} to {target_path}")


def convert_niftis_to_mha(
    dirs: List[Path],
    dir_type: str,
    pat_nif_dir: Path,
    study_dir: Path,
    target_dir: Path,
    pat_id: str,
    logger: logging.Logger
):
    for idx, dir in enumerate(dirs):
        source_path = pat_nif_dir / study_dir / f"{dir}.nii.gz"
        target_path = target_dir / pat_id / f"{pat_id}_{dir_type}_dcm_{idx+1}.mha"
        convert_nifti_to_mha(source_path, target_path, logger)


def copy_reader_set_to_target_dir(
    log_dir: Path,
    root_dirs: dict,
    inference_base_dir: Path,
    target_dir: Path,
    dataset_dir: Path,
    do_copy_dicoms_to_target_dir: bool,
    do_copy_inferences_to_target_dir: bool,
    do_copy_lesion_segs: bool,
    logger: logging.Logger = None,
) -> None:
    """
    Here we will copy the dicoms and the inferences to the target directory.
    
    Parameters:
    - log_dir (Path): Directory where the log file will be stored.
    - root_dirs (dict): Dictionary containing the root directories for the different acceleration factors.
    - inference_base_dir (Path): Base directory containing the inference directories.
    - target_dir (Path): Base directory where the converted files will be stored.
    - dataset_dir (Path): Base directory containing the dataset directories.
    - do_copy_dicoms_to_target_dir (bool): Flag to copy the dicoms to the target directory.
    - do_copy_inferences_to_target_dir (bool): Flag to copy the inferences to the target directory.
    - do_copy_lesion_segs (bool): Flag to copy the lesion segmentations to the target directory.
    - logger (logging.Logger): Logger instance for logging messages.
    """
    processed_patients_log = log_dir / 'copied_patients.log'
    processed_patients = read_patient_ids_from_log(processed_patients_log)
    
    # Phase 1 - Copy Inferences
    if do_copy_inferences_to_target_dir:
        for acc, file_mapping in root_dirs.items():
            root_dir = inference_base_dir / acc
            copy_inferences_to_target_dir(root_dir, file_mapping, target_dir, acc, do_save_empty_ref=True, logger=logger)

    # Phase 2 - Copy Dicoms and Lesion Segs
    if do_copy_dicoms_to_target_dir:
        # we will find the corresponding nifti files in the dataset directory, copy the DWI, ADC and the T2W tra as nifti to the source dir as mha.
        inf_dir_1x = inference_base_dir / "1x"
        patient_dirs = sorted(inf_dir_1x.iterdir(), key=lambda x: x.name)
        
        for idx, pat_dir in enumerate(patient_dirs):
            if not pat_dir.is_dir():
                continue
            pat_id = pat_dir.name
            logger.info(f"\n\n\tProcessing patient {pat_id}.\t {idx + 1}/{len(list(inf_dir_1x.iterdir()))}")
            if pat_id in processed_patients:
                logger.info(f"\tSkipping already processed patient {pat_id}")
                continue
            
            try:
                pat_dcm_dir     = dataset_dir / pat_id / 'dicoms'                                               # 1. go into patient dicoms dir
                study_dir_dcm   = get_study_dir(pat_dcm_dir, pat_id)                                            # 2. get the study dcm directories
                pat_nif_dir     = dataset_dir / pat_id / 'niftis'                                               # 4. go into patient niftis dir
                study_dir_nif   = get_study_dir(pat_nif_dir, pat_id)                                            # 5. get the study nif directories
                dwi_dirs, adc_dirs, t2_tra_dirs = find_sequence_directories_dcm(study_dir_dcm, pat_id, logger)  # 3. find the sequence directories
                
                assert len(dwi_dirs) == 1, f"Error: {pat_id} has {len(dwi_dirs)} DWI directories. We didnt find exactly 1 DWI dir for patient {pat_id}."
                assert len(adc_dirs) == 1, f"Error: {pat_id} has {len(adc_dirs)} ADC directories. We didnt find exactly 1 ADC dir for patient {pat_id}."
                assert len(t2_tra_dirs) == 1, f"Error: {pat_id} has {len(t2_tra_dirs)} T2W TRA directories. We didnt find exactly 1 T2W TRA dir for patient {pat_id}."
                
                convert_niftis_to_mha(dwi_dirs,    'dwi',    pat_nif_dir, study_dir_nif, target_dir, pat_id, logger)
                convert_niftis_to_mha(adc_dirs,    'adc',    pat_nif_dir, study_dir_nif, target_dir, pat_id, logger)
                convert_niftis_to_mha(t2_tra_dirs, 't2_tra', pat_nif_dir, study_dir_nif, target_dir, pat_id, logger)
                
                if do_copy_lesion_segs:
                    source_path = pat_nif_dir / study_dir_nif / f"{t2_tra_dirs[0]}.nii.gz"
                    resample_and_save_roi_images(study_dir_nif, source_path, target_dir, pat_id, logger)
                        
                log_processed_patient(processed_patients_log, pat_id)
            except Exception as e:
                logger.error(f"\tError processing {pat_id}: {e}")
                raise Exception(f"Error processing {pat_id}: {e}")


def get_configurations() -> dict:
    return {
        "root_dirs": {
            "1x": ("rss_target_dcml.nii.gz", "rss_target_dcml.mha"),
            "3x": ("VSharpNet_R3_recon_dcml.nii.gz", "VSharp_R3_recon_dcml.mha"),
            "6x": ("VSharpNet_R6_recon_dcml.nii.gz", "VSharp_R6_recon_dcml.mha")
        },
        "inference_base_dir": Path('/scratch/hb-pca-rad/projects/03_nki_reader_study/output/umcg'),
        "target_dir":         Path('/scratch/hb-pca-rad/projects/03_reader_set_v2'),
        "dataset_dir":        Path('/scratch/p290820/datasets/003_umcg_pst_ksps/data/'),
        "log_dir":            Path('/scratch/hb-pca-rad/projects/03_nki_reader_study/logs'),
        "do_copy_inferences_to_target_dir": True,
        "do_copy_dicoms_to_target_dir":     True,
        "do_copy_lesion_segs":              True,
    }
    
if __name__ == "__main__":
    cfg = get_configurations()
    logger = setup_logger(cfg['log_dir'], use_time=False, part_fname='copy_dicoms_and_lesions')
    copy_reader_set_to_target_dir(logger=logger, **cfg)
