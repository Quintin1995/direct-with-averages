import h5py
from xml.etree import ElementTree as ET
from pathlib import Path

def print_ismrmrd_header(path):
    with h5py.File(path, 'r') as df:
        # Assuming 'ismrmrd_header' is stored in XML format within the HDF5 file
        xml_header = df['ismrmrd_header'][()]
        
        # Decode bytes to string if necessary
        if isinstance(xml_header, bytes):
            xml_header = xml_header.decode('utf-8')
        
        # Parse the XML
        root = ET.fromstring(xml_header)
        
        # Pretty print the XML. Requires defusedxml if security is a concern.
        pretty_xml_as_string = ET.tostring(root, encoding='utf-8', method='xml').decode('utf-8')
        print(pretty_xml_as_string)
        x=4

if __name__ == "__main__":
    path = Path('/scratch/p290820/datasets/003_umcg_pst_ksps/data/0008_ANON8890538/h5s/meas_MID00224_FID710582_T2_TSE_tra_obl-out_2.h5')
    print_ismrmrd_header(path)
