import argparse
import h5py
import glob
import SimpleITK as sitk


def get_args():
    parser = argparse.ArgumentParser(description='Convert .h5 files to nifti files with SimpleITK')
    parser.add_argument(
        '-rd',
        '--rootdir',
        type = str,
        help = 'Path to h5 files directory')
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = get_args()
    files = glob.glob(args.rootdir + '/*.h5')
    for file in files:
        h5 = h5py.File(file, 'r')
        recon = h5['reconstruction'][()]
        print(type(recon))
        print(recon.shape)
        
        # convert to nifti with SimpleITK
        recon = sitk.GetImageFromArray(recon)
        sitk.WriteImage(recon, file.replace('.h5', '.nii.gz'))
        print('File converted: ' + file.replace('.h5', '.nii.gz'))
        h5.close()
        