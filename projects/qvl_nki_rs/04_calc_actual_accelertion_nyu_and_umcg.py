from pathlib import Path
import h5py
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
import SimpleITK as sitk


if __name__ == "__main__":
    # description:
    # Calculate the actual acceleration factor for the NYU and UMC datasets of the sampling mask
    # the NYU subsampling is not going fine it seems
    
    project_root    = Path('/home1/p290820/projects/03_nki_reader_study/output')
    nyu_dir         = project_root / '01_nyu_partial_test_set'
    umcg_dir        = project_root / '02_umcg_test_set_small'
    mask_nyu        = nyu_dir / 'VSharpNet_R4_recon_file_prostate_AXT2_0004_mask.nii.gz'
    mask_umcg       = umcg_dir / 'VSharpNet_R4_recon_meas_MID00385_FID387222_t2_tse_traobl_p2_384-out_2_mask.nii.gz'
    
    tempdir     = Path('/home1/p290820/tmp')
    
    
    # read the images with simpleitk
    mask_nyu = sitk.ReadImage(str(mask_nyu))
    mask_umcg = sitk.ReadImage(str(mask_umcg))
    
    # conver to numpy
    mask_nyu = sitk.GetArrayFromImage(mask_nyu)
    mask_umcg = sitk.GetArrayFromImage(mask_umcg)
    
    # print the shape   
    print(mask_nyu.shape)
    print(mask_umcg.shape)
    
    # take from both the first slice
    mask_nyu = mask_nyu[0]
    mask_umcg = mask_umcg[0]
    
    # print the shape   
    print(mask_nyu.shape)
    print(mask_umcg.shape)
    
    # calculate the acceleration factor by counting the number of zeros versus the number of ones
    acc_nyu = mask_nyu.sum() / mask_nyu.size
    acc_umcg = mask_umcg.sum() / mask_umcg.size
    
    print(f"NYU acceleration factor: {acc_nyu}")
    print(f"UMCG acceleration factor: {acc_umcg}")