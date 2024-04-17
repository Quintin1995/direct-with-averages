import h5py
import xml.etree.ElementTree as ET
import logging
import pydicom
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from scipy.stats import shapiro
from typing import List, Tuple
from datetime import datetime

def get_configurations():
    return {
            'force_new_file':        False,
            'dataset_root':          Path('/scratch/p290820/datasets/003_umcg_pst_ksps/data'),
            'project_root':          Path('/scratch/p290820/projects/03_nki_reader_study'),
            "log_dir":               Path('/scratch/hb-pca-rad/qvl/logs'),
            'stats_root':            Path('/scratch/p290820/projects/03_nki_reader_study/stats'),
            'h5s_params_out_fpath':  Path('/scratch/p290820/projects/03_nki_reader_study/stats/results/acquisition_parameters_h5s_umcg.csv'),
            'dcms_params_out_fpath': Path('/scratch/p290820/projects/03_nki_reader_study/stats/results/acquisition_parameters_dcms_umcg.csv'),
            'inclusion_anon_ids':    ['ANON5046358','ANON9616598','ANON2379607','ANON8290811','ANON1586301','ANON8890538','ANON7748752','ANON1102778','ANON4982869','ANON7362087','ANON3951049','ANON9844606','ANON9843837','ANON7657657','ANON1562419','ANON4277586','ANON6964611','ANON7992094','ANON3394777','ANON3620419','ANON9724912','ANON3397001','ANON7189994','ANON9141039','ANON7649583','ANON9728185','ANON3474225','ANON0282755','ANON0369080','ANON0604912','ANON2361146','ANON9423619','ANON7041133','ANON8232550','ANON2563804','ANON3613611','ANON6365688','ANON9783006','ANON1327674','ANON9710044','ANON5517301','ANON2124757','ANON3357872','ANON1070291','ANON9719981','ANON7955208','ANON7642254','ANON0319974','ANON9972960','ANON0282398','ANON0913099','ANON7978458','ANON9840567','ANON5223499','ANON9806291','ANON5954143','ANON5895496','ANON3983890','ANON8634437','ANON6883869','ANON8828023','ANON4499321','ANON9763928','ANON9898497','ANON6073234','ANON4535412','ANON6141178','ANON8511628','ANON9534873','ANON9892116','ANON0891692','ANON9786899','ANON9941969','ANON8024204','ANON9728761','ANON4189062','ANON5642073','ANON8583296','ANON4035085','ANON7748630','ANON9883201','ANON0424679','ANON9816976','ANON8266491','ANON9310466','ANON3210850','ANON9665113','ANON0400743','ANON9223478','ANON3865800','ANON7141024','ANON7275574','ANON9629161','ANON7265874','ANON8610762','ANON0272089','ANON4747182','ANON8023509','ANON8627051','ANON5344332','ANON9879440','ANON8096961','ANON8035619','ANON1747790','ANON2666319','ANON0899488','ANON8018038','ANON7090827','ANON9752849','ANON2255419','ANON0335209','ANON7414571','ANON9604223','ANON4712664','ANON5824292','ANON2411221','ANON5958718','ANON7828652','ANON9873056','ANON3504149']
        }


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


def extract_acquisition_parameters(h5_file_path):
    """
    Reads the ISMRMRD headers from an HDF5 file and extracts various acquisition parameters.
    
    Parameters:
    h5_file_path (Path): Path to the HDF5 file.
    
    Returns:
    dict: Dictionary of extracted parameters.
    """
    acq_params = {}
    acq_params['id'] = h5_file_path.parts[-3]
    
    with h5py.File(h5_file_path, 'r') as hf:
        header_bytes = hf['ismrmrd_header'][()]
        # IN-PLANE RESOLUTION: Add the pixel spacing to the acquisition parameters from the h5 attributes
        acq_params['in_plane_resolution'] = tuple(list(hf.attrs['pixel_spacing_dcm_hdr']*2))
    
    # parse headers
    header_str = header_bytes.decode('utf-8')
    root = ET.fromstring(header_str)
    
    # COILS: Extract the number of receiver channels
    rc_elem = root.find('.//{http://www.ismrm.org/ISMRMRD}receiverChannels')
    if rc_elem is not None:
        acq_params['receiver_channels'] = int(rc_elem.text)
    
    # SLICES: Extract the maximum number of slices
    s_elem  = root.find('.//{http://www.ismrm.org/ISMRMRD}slice')
    sm_elem = s_elem.find('.//{http://www.ismrm.org/ISMRMRD}maximum')
    if sm_elem is not None:
        acq_params['max_slices'] = int(sm_elem.text)
    
    # MATRIX SIZE: Extract the matrix size reconspace
    rs_elem = root.find('.//{http://www.ismrm.org/ISMRMRD}reconSpace')
    ms_elem = rs_elem.find('.//{http://www.ismrm.org/ISMRMRD}matrixSize')
    x       = ms_elem.find('.//{http://www.ismrm.org/ISMRMRD}x')
    y       = ms_elem.find('.//{http://www.ismrm.org/ISMRMRD}y')
    acq_params['matrix_size_reconspace'] = (int(x.text), int(y.text))
    
    # MATRIX SIZE: Extract the matrix size
    rs_elem = root.find('.//{http://www.ismrm.org/ISMRMRD}encodedSpace')
    ms_elem = rs_elem.find('.//{http://www.ismrm.org/ISMRMRD}matrixSize')
    x       = ms_elem.find('.//{http://www.ismrm.org/ISMRMRD}x')
    y       = ms_elem.find('.//{http://www.ismrm.org/ISMRMRD}y')
    acq_params['matrix_size_encodedspace'] = (int(x.text), int(y.text))
    
    # fov encoded space
    es_elem  = root.find('.//{http://www.ismrm.org/ISMRMRD}encodedSpace')
    fov_elem = es_elem.find('.//{http://www.ismrm.org/ISMRMRD}fieldOfView_mm')
    x        = fov_elem.find('.//{http://www.ismrm.org/ISMRMRD}x')
    y        = fov_elem.find('.//{http://www.ismrm.org/ISMRMRD}y')
    z        = fov_elem.find('.//{http://www.ismrm.org/ISMRMRD}z')
    acq_params['fov_encodedspace'] = (float(x.text), float(y.text), float(z.text))
    
    # FOV mm and SLICE THICKNESS: Extract the field of view and slice thickness
    # in the UMCG data, the field of view is hardcoded to 180x180mm and the slice thickness to 3mm, this data was not set into the ISMRMRD header
    fov_elem = es_elem.find('.//{http://www.ismrm.org/ISMRMRD}fieldOfView_mm')
    # x        = 180
    # y        = 180
    # z        = 3
    # slice_thinkness        = fov_elem.find('.//{http://www.ismrm.org/ISMRMRD}z')
    # acq_params['fov_mm'] = (x, y)
    # acq_params['slice_thickness'] = z
    
    acq_params = {
        'id_seq': acq_params['id'].split('_')[0],
        'id_anon': acq_params['id'].split('_')[1],
        'receiver_channels': acq_params['receiver_channels'],
        'max_slices': acq_params['max_slices'],
        'matrix_size_reconspace_x': acq_params['matrix_size_reconspace'][0],
        'matrix_size_reconspace_y': acq_params['matrix_size_reconspace'][1],
        'matrix_size_encodedspace_x': acq_params['matrix_size_encodedspace'][0],
        'matrix_size_encodedspace_y': acq_params['matrix_size_encodedspace'][1],
        'fov_encodedspace_x': acq_params['fov_encodedspace'][0],
        'fov_encodedspace_y': acq_params['fov_encodedspace'][1],
        'fov_encodedspace_z': acq_params['fov_encodedspace'][2],
        'slice_thickness': acq_params['fov_encodedspace'][2],
        'in_plane_resolution_x': acq_params['in_plane_resolution'][0],
        'in_plane_resolution_y': acq_params['in_plane_resolution'][1],
    }
    
    return acq_params


def print_column_stats(df: pd.DataFrame, column_name: str, logger: logging.Logger = None) -> None:
    """Prints statistics for a specified column in a DataFrame and decides whether to use STD or IQR based on normality test."""
    
    col_series = df[column_name].dropna()  # Remove NaNs as they can't be processed by shapiro
    col_mu = round(col_series.mean(), 2)
    col_min = round(col_series.min(), 2)
    col_max = round(col_series.max(), 2)
    col_median = round(col_series.median(), 2)
    col_std = round(col_series.std(), 2)
    
    # how much percentage of the data is equal to the median?
    median_count = col_series[col_series == col_median].count()
    median_percentage = round((median_count / len(col_series)) * 100, 2)
    
    # Shapiro-Wilk test for normality
    stat, p_value = shapiro(col_series)
    is_normal = p_value > 0.05  # Assuming alpha=0.05
    
    # Depending on normality, use STD or IQR
    if is_normal:
        col_spread = round(col_series.std(), 2)
        spread_type = "std"
    else:
        col_spread = round(col_series.quantile(0.75) - col_series.quantile(0.25), 2)
        spread_type = "iqr"
    
    logger.info(f"{column_name}")
    logger.info(f"\tMedian: {col_median} ({median_percentage}%)")
    logger.info(f"\tMean: {col_mu}")
    logger.info(f"\tRange: [{col_min}-{col_max}]")
    logger.info(f"\t{spread_type}: {col_spread}")
    logger.info(f"\tIs normally distributed: {is_normal} with p-value: {p_value}")
    logger.info(f"\tStandard deviation: {col_std} BUT using {spread_type} {col_spread}")
    
    # when looking at matrix_size_encodedsapcre_y i want to save a barplot of all values in the range to a tempory matplotlib png file /scratch/p290820/projects/03_nki_reader_study/stats/results/distribution_phase_encoding.png
    if column_name == 'matrix_size_encodedspace_y':
        plot_distribution_histogram(df, column_name, logger)
    
    
def plot_distribution_histogram(df: pd.DataFrame, column_name: str, logger: logging.Logger = None) -> None:
    """Plots a histogram of the distribution of values in the specified column."""
    fig, ax = plt.subplots()
    ax.hist(df[column_name], bins=20)
    ax.set_title(f'Distribution of {column_name}')
    ax.set_xlabel(column_name)
    ax.set_ylabel('Count')
    plt.show()
    plt.close()
    logger.info(f"plotted distribution of {column_name}")


def print_summarized_stats_h5s(df: pd.DataFrame, logger: logging.Logger = None) -> None:
    """Prints statistics for specified columns in the DataFrame."""
    columns_to_analyze = ['max_slices', 'receiver_channels', 'matrix_size_encodedspace_x', 'matrix_size_encodedspace_y']
    
    for column in columns_to_analyze:
        print_column_stats(df, column, logger)


def filter_inclusion_list_h5s(dataset_root: Path, incl_ids: List[str], logger: logging.Logger) -> List[Path]:
    """
    Filters the list of h5 files to only include those that are in the inclusion list.
    
    Parameters:
    dataset_root (Path): Path to the dataset root directory.
    incl_ids (list): List of anonymized IDs to include.
    
    Returns:
    list: List of Path objects to the h5 files that are in the inclusion list.
    """
    h5_files = []
    for pat_dir in dataset_root.iterdir():
        if pat_dir.name.split("_")[1] in incl_ids:
            logger.info(f"processing {pat_dir.name}")
            files = list(pat_dir.glob('h5s/*.h5'))
            assert len(files) == 1, f"expected 1 h5 file, got {len(files)}"
            h5_files.append(files[0])
        else:
            logger.warning(f"skipping {pat_dir.name} as it is not in the inclusion list")
            continue
    return h5_files


def find_sequence_directories(study_dir: Path, patient_id: str = None, logger: logging.Logger = None, return_first=False) -> Tuple[List[str], List[str], List[str]]:
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

    # Sort and select the T2W directory with the latest acquisition time if there are multiple
    if len(t2w_tra_dirs) > 1:
        t2w_tra_dirs = sort_t2w_tra_directories_by_creation_time(t2w_tra_dirs, study_dir, logger)
        logger.info(f"Multiple T2W directories found for patient {patient_id}. Selected {t2w_tra_dirs[0]} based on latest acquisition time.")

    logger.info(f"Selected sequences for patient {patient_id}:\n\tDWI: {dwi_dirs[0]},\n\tADC: {adc_dirs[0]},\n\tT2W TRA: {t2w_tra_dirs[0]}") 
    
    if return_first:
        return [dwi_dirs[0]], [adc_dirs[0]], [t2w_tra_dirs[0]]
    else:
        return dwi_dirs, adc_dirs, t2w_tra_dirs


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
    
    # ans = input(f"Creation time: {creation_times}\nSorted directories: {sorted_directories}\n Continue? (y/n): ")
    # if ans.lower() != 'y':
    #     raise Exception("User aborted sorting of T2W TRA directories.")
    
    return sorted_t2w_tra_directories


def extract_scan_params_from_h5s(dataset_root: Path, inclusion_ids: List[str], logger: logging.Logger) -> pd.DataFrame:
    """
    Extracts acquisition parameters from the ISMRMRD headers in the HDF5 files.
    
    Parameters:
    dataset_root (Path): Path to the dataset root directory.
    inclusion_anon_ids (list): List of anonymized IDs to include.
    
    Returns:
    pd.DataFrame: DataFrame containing the acquisition parameters.
    """
    
    h5_files = filter_inclusion_list_h5s(dataset_root, inclusion_ids, logger)
    acq_params_list = []
    for idx, h5_file in enumerate(h5_files):
        logger.info(f"processing {idx+1}/{len(h5_files)}. fpath: {h5_file}")
        acq_params = extract_acquisition_parameters(h5_file)
        acq_params_list.append(acq_params)
    df = pd.DataFrame(acq_params_list)
    df = df.sort_values(by=['id_seq'])
    return df


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


def extract_sequence_params(sequence_dirs: List[Path], sequence_type: str, patient_dir: Path, logger: logging.Logger) -> List[dict]:
    """Extract DICOM parameters from all files in each sequence directory."""
    sequence_params_list = []
    for seq_dir in sequence_dirs:
        seq_files = list(seq_dir.iterdir())
        if seq_files:    # Check if directory is not empty
            header = pydicom.dcmread(seq_files[0], stop_before_pixels=True)
            private_tag_value = header.get((0x0051, 0x100A))  # Accessing the value of the private tag (example: scan duration)
            scan_duration = private_tag_value.value.strip() if private_tag_value is not None else "NA"
            scan_duration = scan_duration.replace("TA ", "")
            params = {
                'id_seq': patient_dir.name.strip(),
                'id_anon': patient_dir.name.split("_")[1].strip(),
                'seq_type': sequence_type.strip(),
                'recon_space': (header.Rows, header.Columns),
                'fov': (float(header.PixelSpacing[0]) * header.Rows, float(header.PixelSpacing[1]) * header.Columns),
                'slice_thickness': header.SliceThickness,
                'in_plane_resolution': [float(sp) for sp in header.PixelSpacing],  # Ensure numeric values
                'tr': header.RepetitionTime,
                'te': header.EchoTime,
                'scan_duration': scan_duration,  # Use stripped value
                'series_description': header.SeriesDescription.strip(),
                'averages': header.get('NumberOfAverages', 'NA'),
            }
            sequence_params_list.append(params)
    return sequence_params_list


def extract_scan_params_from_dicoms(dataset_root: Path, inclusion_ids: List[str], logger: logging.Logger) -> pd.DataFrame:
    """
    """
    acq_param_list = []
    patient_dirs = sorted(dataset_root.iterdir(), key=lambda x: x.name)
    
    for idx, pat_dir in enumerate(patient_dirs):
        anon_id = pat_dir.name.split("_")[1]
        
        if anon_id not in inclusion_ids:
            logger.info(f"Skipping {pat_dir.name} as it is not in the inclusion list.")
            continue
        
        pat_dcm_dir = dataset_root / pat_dir.name / 'dicoms'                                                             # 1. go into patient dicoms dir
        study_dir   = get_study_dir(pat_dcm_dir, pat_dir.name)                                                           # 2. get the study directories
        dwi_dirs, adc_dirs, t2_tra_dirs = find_sequence_directories(study_dir, pat_dir.name, logger, return_first=True)  # 3. find the sequence directories
        
        acq_param_list.extend(extract_sequence_params([study_dir / d for d in dwi_dirs], 'dwi', pat_dir, logger))
        acq_param_list.extend(extract_sequence_params([study_dir / d for d in adc_dirs], 'adc', pat_dir, logger))
        acq_param_list.extend(extract_sequence_params([study_dir / d for d in t2_tra_dirs], 't2w', pat_dir, logger))

    return pd.DataFrame(acq_param_list)


def main(
    dataset_root: Path,
    inclusion_anon_ids: List[str],
    h5s_params_out_fpath: Path,
    dcms_params_out_fpath: Path,
    force_new_file: bool,
    logger: logging.Logger,
    **kwargs,
):  
    # PART 1
    # First we need to read the HDF5 files that are in the inclusion list from the dataset_root, than we find the HDF5 files and extract the receiver channels, max slices, matrix size reconspace, matrix size encodedspace, fov encodedspace, fov mm and slice thickness.
    if not h5s_params_out_fpath.exists() or force_new_file:
        logger.info(f"creating new file at {h5s_params_out_fpath}")
        df_h5s = extract_scan_params_from_h5s(dataset_root, inclusion_anon_ids, logger)
        df_h5s.to_csv(h5s_params_out_fpath, index=False, sep=';')
        logger.info(f"saved to {h5s_params_out_fpath}")
    else:
        logger.info(f"loading from {h5s_params_out_fpath}")
        df_h5s = pd.read_csv(h5s_params_out_fpath, sep=';')
    print(df_h5s.head(10))
    print_summarized_stats_h5s(df_h5s, logger)
    
    # PART 2
    # we need to read the dicom files that are in the inclusion list from the dataset_root, than we find the dicom files and extract the voxel size, FOV, TR, TE and scan time.
    if not dcms_params_out_fpath.exists() or force_new_file:
        logger.info(f"creating new file at {dcms_params_out_fpath}")
        df_dcms = extract_scan_params_from_dicoms(dataset_root, inclusion_anon_ids, logger)
        df_dcms.to_csv(dcms_params_out_fpath, index=False, sep=';')
        logger.info(f"saved to {dcms_params_out_fpath}")
    else:
        logger.info(f"loading from {dcms_params_out_fpath}")
        df_dcms = pd.read_csv(dcms_params_out_fpath, sep=';')
        
    print(df_dcms.head(10))
    X=3
#    print_summarized_stats_dcms(df_dcms, logger)
    
    
if __name__ == "__main__":
    cfg = get_configurations()
    logger = setup_logger(cfg['log_dir'], use_time=False, part_fname='extr_prms_h5s_dcms')
    main(logger=logger, **cfg)