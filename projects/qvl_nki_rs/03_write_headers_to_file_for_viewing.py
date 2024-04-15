from pathlib import Path
import h5py
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString


def write_ismrmrd_headers_to_xml(h5_filename, xml_output_filename, key='ismrmrd_header'):
    """Read ISMRMRD headers from an HDF5 file and write them to a readable XML file.

    Parameters:
    h5_filename (str): Path to the HDF5 file.
    xml_output_filename (str): Output XML file path.
    key (str): Key under which the header is stored in the HDF5 file.
    """
    with h5py.File(h5_filename, 'r') as h5_file:
        print(h5_file.keys())
        # Read header stored under specified key, assuming it's a byte array
        header_bytes = h5_file[key][()]

    # Decode byte array to string
    header_str = header_bytes.decode('utf-8')

    # Parse the XML string
    root = ET.fromstring(header_str)
    tree = ET.ElementTree(root)

    # Use minidom for pretty printing
    xml_str = parseString(ET.tostring(root)).toprettyxml(indent="   ")

    with open(xml_output_filename, 'w', encoding='utf-8') as xml_file:
        xml_file.write(xml_str)
        
    print(f"File saved: {xml_output_filename}")


if __name__ == "__main__":
    # Script description and other details
    nyu_root    = Path('/scratch/p290820/datasets/002_nyu_pst_ksps/test_T2')
    nyu_h5_file = nyu_root / 'file_prostate_AXT2_0004.h5'
    tempdir     = Path('/home1/p290820/tmp/007_write_headers_to_file')
    
    # umcg example too
    umcg_root = Path('/scratch/p290820/datasets/003_umcg_pst_ksps/data/0001_ANON2784451/h5s')
    umcg_h5_file = umcg_root / 'meas_MID00202_FID688156_T2_TSE_tra_obl-out_2.h5'
    
    write_ismrmrd_headers_to_xml(nyu_h5_file, tempdir / 'nyu_headers.xml')
    write_ismrmrd_headers_to_xml(umcg_h5_file, tempdir / 'umcg_headers.xml')
