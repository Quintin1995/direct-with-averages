from pathlib import Path
from typing import List


def create_umcg_lst_file(
    rootdir: Path = None,
    excl_list: List[str] = None,
    institute = 'umcg',
    splitsize: int = None,
    path_lst_rootdir: Path = Path('/scratch/p290820/projects/03_nki_reader_study/lists/split_by_15')
):
    pats_dir = rootdir / 'data'
    
    # Sorting: Sort on first number of pat_ids: 0001_ANONxxxxxxx
    pat_dirs = sorted([d for d in pats_dir.iterdir() if d.is_dir()], key=lambda x: int(x.name.split('_')[0]))
    
    # create output filename
    lowest = int(pat_dirs[0].name.split('_')[0])
    highest = int(pat_dirs[-1].name.split('_')[0])
    fname = f"{institute}_{lowest:04d}_{highest:04d}"
    
    # Exclusion: exlude patients from the exclusion list
    print(f"Number of patients before exclusion: {len(pat_dirs)}")
    if excl_list is not None:
        pat_dirs = [d for d in pat_dirs if d.name.split('_')[0] not in excl_list]
    print(f"Number of patients after exclusion: {len(pat_dirs)}")
    
    # Create a list of .h5 files
    h5_paths = []
    for patient_dir in pat_dirs:
        if patient_dir.is_dir():
            patient_id = patient_dir.name
            h5_dir = patient_dir / 'h5s'
            h5_files = list(h5_dir.glob('*.h5'))
            if len(h5_files) == 1:
                h5_paths.append(f"{patient_id}/h5s/{h5_files[0].name}")
            else:
                raise ValueError(f'Expected only one .h5 file in {h5_dir}')
            
    
    # split the list in smaller lists of size splitsize and write to file
    if splitsize is not None:
        print(f"Number of files to be created (each containing {splitsize} files): {len(h5_paths) // splitsize}")
        print(f"Last file will have {len(h5_paths) % splitsize} files.")
        for i, split in enumerate(range(0, len(h5_paths), splitsize)):
            this_split_paths = h5_paths[split:split+splitsize]
            with open(f"{path_lst_rootdir}/{fname}_{i+1}.lst", 'w') as f:
                for h5_path in this_split_paths:
                    f.write(f"{h5_path}\n")
                print(f"List file created: {fname}_{i+1}.lst")
    else: # write all to one file
        with open(f"{path_lst_rootdir}/{fname}.lst", 'w') as f:
            for h5_path in h5_paths:
                f.write(f"{h5_path}\n")
        print(f"List file created: {fname}.lst")


def get_seq_id_of_anonids(anonids: List[str], rootdir: Path = None):
    pats_dir = rootdir / 'data'
    
    # Sorting: Sort on first number of pat_ids: 0001_ANON38643
    pat_dirs = sorted([d for d in pats_dir.iterdir() if d.is_dir()], key=lambda x: int(x.name.split('_')[0]))
    
    prior_treatment_seq_ids = []
    for anonid in anonids:
        for pat_dir in pat_dirs:
            full_id = pat_dir.name
            if anonid in full_id:
                seq_id = full_id.split('_')[0]
                prior_treatment_seq_ids.append(seq_id)
                
    print(f"Patients with prior treatment found in dataset: {prior_treatment_seq_ids}")
    return prior_treatment_seq_ids


############################################################################################################################################
if __name__ == "__main__":
    
    # OLD These patients have prior treatment and should be excluded
    # prior_treatment_OLD = ['ANON2411870','ANON2784451','ANON3296151','ANON4085509','ANON5743124','ANON6219624','ANON6994400','ANON7067833','ANON7446652','ANON7467725','ANON7890966','ANON7954361','ANON8283670','ANON8802409','ANON8824369','ANON8989730','ANON9608148','ANON9657060','ANON9661741','ANON9692714','ANON9696728','ANON9827881','ANON9872795','ANON9933589','ANON9947254']
    prior_treatment = [
        'ANON2784451',
        'ANON9844057',
        'ANON8989730',
        'ANON9657060',
        'ANON9947254',
        'ANON6994400',
        'ANON7890966',
        'ANON5314379',
        'ANON8283670',
        'ANON9872795',
        'ANON8824369',
        'ANON7954361',
        'ANON7941661',
        'ANON7467725',
        'ANON9696728',
        'ANON5743124',
        'ANON9608148',
        'ANON4085509',
        'ANON9124571',
        'ANON7067833',
        'ANON0348720',
        'ANON8802409',
        'ANON3296151',
        'ANON6219624',
        'ANON9692714',
        'ANON2411870',
        'ANON9827881',
        'ANON8989730',
        'ANON9933589',
        'ANON7446652',
        'ANON9661741',
        'ANON9190011',
        'ANON1693100',
        'ANON8435394',
        'ANON9190011',
        'ANON1305730',
        'ANON9998086',
        'ANON3513877',
    ]
    print(f"Found {len(prior_treatment)} prior treatment patients")
    
    excl1 = ['0050', '0065', '0117', '0130']    # K-space could not be procecced successfully
    excl2 = get_seq_id_of_anonids(prior_treatment, rootdir=Path('/scratch/p290820/datasets/003_umcg_pst_ksps'))
    
    umcg_rootdir = Path('/scratch/p290820/datasets/003_umcg_pst_ksps')
    create_umcg_lst_file(rootdir=umcg_rootdir, excl_list=excl1+excl2, splitsize=15)
    
    