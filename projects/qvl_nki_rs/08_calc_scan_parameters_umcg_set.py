from pathlib import Path
import h5py
import xml.etree.ElementTree as ET
import pandas as pd
from scipy.stats import shapiro


def get_configurations():
    return {
            'dataset_root': Path('/scratch/p290820/datasets/003_umcg_pst_ksps/data'),
            'project_root': Path('/scratch/p290820/projects/03_nki_reader_study'),
            'stats_root':   Path('/scratch/p290820/projects/03_nki_reader_study/stats'),
            'inclusion_anon_ids': [
                'ANON5046358',
                'ANON9616598',
                'ANON2379607',
                'ANON8290811',
                'ANON1586301',
                'ANON8890538',
                'ANON7748752',
                'ANON1102778',
                'ANON4982869',
                'ANON7362087',
                'ANON3951049',
                'ANON9844606',
                'ANON9843837',
                'ANON7657657',
                'ANON1562419',
                'ANON4277586',
                'ANON6964611',
                'ANON7992094',
                'ANON3394777',
                'ANON3620419',
                'ANON9724912',
                'ANON3397001',
                'ANON7189994',
                'ANON9141039',
                'ANON7649583',
                'ANON9728185',
                'ANON3474225',
                'ANON0282755',
                'ANON0369080',
                'ANON0604912',
                'ANON2361146',
                'ANON9423619',
                'ANON7041133',
                'ANON8232550',
                'ANON2563804',
                'ANON3613611',
                'ANON6365688',
                'ANON9783006',
                'ANON1327674',
                'ANON9710044',
                'ANON5517301',
                'ANON2124757',
                'ANON3357872',
                'ANON1070291',
                'ANON9719981',
                'ANON7955208',
                'ANON7642254',
                'ANON0319974',
                'ANON9972960',
                'ANON0282398',
                'ANON0913099',
                'ANON7978458',
                'ANON9840567',
                'ANON5223499',
                'ANON9806291',
                'ANON5954143',
                'ANON5895496',
                'ANON3983890',
                'ANON8634437',
                'ANON6883869',
                'ANON8828023',
                'ANON4499321',
                'ANON9763928',
                'ANON9898497',
                'ANON6073234',
                'ANON4535412',
                'ANON6141178',
                'ANON8511628',
                'ANON9534873',
                'ANON9892116',
                'ANON0891692',
                'ANON9786899',
                'ANON9941969',
                'ANON8024204',
                'ANON9728761',
                'ANON4189062',
                'ANON5642073',
                'ANON8583296',
                'ANON4035085',
                'ANON7748630',
                'ANON9883201',
                'ANON0424679',
                'ANON9816976',
                'ANON8266491',
                'ANON9310466',
                'ANON3210850',
                'ANON9665113',
                'ANON0400743',
                'ANON9223478',
                'ANON3865800',
                'ANON7141024',
                'ANON7275574',
                'ANON9629161',
                'ANON7265874',
                'ANON8610762',
                'ANON0272089',
                'ANON4747182',
                'ANON8023509',
                'ANON8627051',
                'ANON5344332',
                'ANON9879440',
                'ANON8096961',
                'ANON8035619',
                'ANON1747790',
                'ANON2666319',
                'ANON0899488',
                'ANON8018038',
                'ANON7090827',
                'ANON9752849',
                'ANON2255419',
                'ANON0335209',
                'ANON7414571',
                'ANON9604223',
                'ANON4712664',
                'ANON5824292',
                'ANON2411221',
                'ANON5958718',
                'ANON7828652',
                'ANON9873056',
                'ANON3504149',
            ]
        }


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


def print_column_stats(df: pd.DataFrame, column_name: str) -> None:
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
    
    print(f"{column_name}")
    print(f"\tMedian: {col_median} ({median_percentage}%)")
    print(f"\tMean: {col_mu}")
    print(f"\tRange: [{col_min}-{col_max}]")
    print(f"\t{spread_type}: {col_spread}")
    print(f"\tIs normally distributed: {is_normal} with p-value: {p_value}")
    print(f"\tStandard deviation: {col_std} BUT using {spread_type} {col_spread}")
    print()
    
    # when looking at matrix_size_encodedsapcre_y i want to save a barplot of all values in the range to a tempory matplotlib png file /scratch/p290820/projects/03_nki_reader_study/stats/results/distribution_phase_encoding.png
    if column_name == 'matrix_size_encodedspace_y':
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.hist(col_series, bins=20)
        ax.set_title('Distribution of Phase Encoding Step')
        ax.set_xlabel('matrix_size_encodedspace_y')
        ax.set_ylabel('Count')
        fpath = '/scratch/p290820/projects/03_nki_reader_study/stats/results/distribution_matrix_size_encodedspace_y.png'
        plt.savefig(fpath)
        plt.close()
        print(f"saved distribution to {fpath}")
        print()
    

def print_stats_for_table_to_console(df: pd.DataFrame) -> None:
    """Prints statistics for specified columns in the DataFrame."""
    columns_to_analyze = ['max_slices', 'receiver_channels', 'matrix_size_encodedspace_x', 'matrix_size_encodedspace_y']
    
    for column in columns_to_analyze:
        print_column_stats(df, column)


def filter_inclusion_list(cfg):
    """
    Filters the list of h5 files to only include those that are in the inclusion list.
    
    Parameters:
    cfg (dict): Configuration dictionary.
    
    Returns:
    list: List of Path objects to the h5 files that are in the inclusion list.
    """
    
    h5_files = []
    for pat_dir in cfg['dataset_root'].iterdir():
        if pat_dir.name.split("_")[1] in cfg['inclusion_anon_ids']:
            print(f"processing {pat_dir.name}")
            files = list(pat_dir.glob('h5s/*.h5'))
            assert len(files) == 1, f"expected 1 h5 file, got {len(files)}"
            h5_files.append(files[0])
        else:
            print(f"skipping {pat_dir.name} as it is not in the inclusion list")
            continue
    return h5_files

def main():
    cfg = get_configurations()
    fpath = cfg['stats_root'] / 'results' / 'acquisition_parameters_umcg.csv'
    
    h5_files = filter_inclusion_list(cfg)
    
    if fpath.exists():
        df = pd.read_csv(fpath, sep=';')
    else:
        acq_params_list = []
        for idx, h5_file in enumerate(h5_files):
            print(f"processing {idx+1}/{len(h5_files)}. fpath: {h5_file}")
            acq_params = extract_acquisition_parameters(h5_file)
            acq_params_list.append(acq_params)
        df = pd.DataFrame(acq_params_list)
        df = df.sort_values(by=['id_seq'])
        df.to_csv(fpath, index=False, sep=';')
        print(f"saved to {fpath}")
    
    print(df.head(10))
    print_stats_for_table_to_console(df)
    
        
if __name__ == "__main__":
    main()