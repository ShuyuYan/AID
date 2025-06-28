import os
from tqdm import tqdm
import pydicom
import pandas as pd
import dicom2nii


def extract_info(file_path):
    ds = pydicom.read_file(file_path)
    info = {
        'FilePath': file_path,
        'PatientName': ds.get('PatientName', ''),
        'PatientID': ds.get('PatientID', ''),
    }
    return info


def find_ta(i, file_list):
    global zs2name, dicom
    sum = 0
    for file in tqdm(file_list):
        sum += 1
        if sum % 10 == 0:
            info = extract_info(file)
            if info['PatientID'] not in zs2name:
                zs2name[info['PatientID']] = ''.join(c for c in info['PatientName'] if c.isupper())
                dicom.append(i)



if __name__ == "__main__":
    root_path = '/home/yanshuyu/Data/AID/TAK/DICOM_data/'
    ans = {}
    zs2name = {}
    dicom = []
    err_list = [264, 265, 270, 274, 275, 276, 277, 279, 281, 282, 284, 286, 287, 288]
    for i in range(1, 329):
        if i in err_list:
            continue
        dicom_path = root_path + str(i) + ' DICOM'
        if os.path.exists(dicom_path):
            file_list = dicom2nii.list_files(dicom_path)
            find_ta(i, file_list)
        if i % 5 == 0:
            df = pd.DataFrame({
                'DICOM': dicom,
                'PatientID': list(zs2name.keys()),
                'PatientName': list(zs2name.values())
            })
            excel_filename = "/home/yanshuyu/Data/AID/TAK/patients2.xlsx"
            df.to_excel(excel_filename, index=False)

