import os
import glob
import pandas as pd
import pydicom
from pathlib import Path


if __name__ == "__main__":
    DATASET_DIR    = '/scratch/p290820/datasets/003_umcg_pst_ksps'
    PAT_MAP_FPATH  = os.path.join(DATASET_DIR, 'patient_mapping_excel.xlsx')

    # read the patient mapping excel file
    patient_map_df = pd.read_excel(PAT_MAP_FPATH, sheet_name='Sheet1')

    # loop over the rows in the patient mapping excel file
    for index, row in patient_map_df.iterrows():

        # break if the patient doesnt exist yet in the patient mapping excel file
        if pd.isna(row['anon_id']):
            break

        seq_id  = str(row['pat_seq_id']).zfill(4)
        print(f"\nProcessing Patient {seq_id} with anon_id {row['anon_id']}")

        # create the patient directory
        pat_dir = os.path.join(DATASET_DIR, 'pat_data', f"{seq_id}_{row['anon_id']}")
        Path(pat_dir).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(pat_dir, 'recons')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(pat_dir, 't2w_tra_dicom')).mkdir(parents=True, exist_ok=True)

        print(f"\tCreated directory {pat_dir}")
        print(f"\tCreated directory {os.path.join(pat_dir, 'recons')}")
        print(f"\tCreated directory {os.path.join(pat_dir, 't2w_tra_dicom')}")