import os
import pydicom
from tqdm import tqdm
import SimpleITK as sitk
"""
列出文件夹中的所有DICOM文件，按照患者、序列名、图像大小分类并排序，转化为nii格式保存到患者文件夹
"""


def list_files(folder_path):
    if not os.path.exists(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在")
        return

    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.startswith('.'):
                continue
            file_path = os.path.join(root, file)
            file_list.append(file_path)

    print(f"{folder_path.split('/')[-1]} 文件总数: {len(file_list)}")
    return file_list


def extract_info(file_path):
    ds = pydicom.read_file(file_path)
    info = {
        'FilePath': file_path,
        'PatientName': ds.get('PatientName', ''),
        'PatientID': ds.get('PatientID', ''),
        'SeriesDesc': ds.get('SeriesDescription', ''),
        'SeriesNumber': ds.get('SeriesNumber', ''),
        'Modality': ds.get('Modality', ''),
        'ImageSize': f"{ds.get('Rows', '')}x{ds.get('Columns', '')}",
        'Pixel Spacing': ds.get('PixelSpacing', ''),
        'SOP Class UID': ds.get('SOPClassUID', ''),
        'InstanceNumber': ds.get('InstanceNumber', ''),
        'StudyDate': ds.get('StudyDate', ''),
        'SeriesDate': ds.get('SeriesDate', ''),
        'AcquisitionDate': ds.get('AcquisitionDate', ''),
        'ContentDate': ds.get('ContentDate', ''),
    }
    return info


def classify(file_list):
    patient_list = []
    grouped = {}
    for file in tqdm(file_list):
        info = extract_info(file)
        patient = info['PatientID']
        if patient not in grouped:
            grouped[patient] = []
        if info['Modality'] == 'MR':
            info['SeriesDesc'] = info['SeriesDesc'].replace('/', '')
            grouped[patient].append(info)

        patient_list = [(pid, files) for pid, files in grouped.items()]
        patient_list.sort(key=lambda x: x[0])
        for pid, files in patient_list:
            files.sort(key=lambda x: (x['SeriesDesc'], int(x['InstanceNumber']), x['ImageSize'], int(x['SeriesNumber'])))

    # for patient_id, files in patient_list:
    #     print(f"PatientID: {patient_id}")
    #     for file in files:
    #         print(f"  SeriesDesc: {file['SeriesDesc']}, InstanceNumber: {file['InstanceNumber']},"
    #               f" Size: {file['ImageSize']}, FilePath: {file['FilePath']}")
    return patient_list


def convert(series, output_path, save_name):
    if len(series) != 0:
        series_reader = sitk.ImageSeriesReader()
        series_reader.SetFileNames(series)
        image3D = series_reader.Execute()
        sitk.WriteImage(image3D, os.path.join(output_path, save_name))


def dicom2nii(patient_list, root_path):
    cur_series = []
    for patient, files in patient_list:
        output_path = root_path + '/' + patient
        os.makedirs(output_path, exist_ok=True)
        pre = ''
        for file in files:
            if file['SeriesDesc'] + file['ImageSize'] != pre:
                # print(pre, cur_series)
                save_name = pre + '.nii'
                pre = file['SeriesDesc'] + file['ImageSize']
                if len(cur_series) != 0:
                    convert(cur_series, output_path, save_name)
                cur_series = []
            cur_series.append(file['FilePath'])
        save_name = pre + '.nii'
        if len(cur_series) != 0:
            convert(cur_series, output_path, save_name)
        print(f'PatientID: {patient} converted.')


if __name__ == "__main__":
    root_path = '~/Desktop/新建文件夹/'
    output_path = '~/Data/AID/TAK/0'
    if os.path.exists(root_path):
        file_list = list_files(root_path)
        patient_list = classify(file_list)
        dicom2nii(patient_list, output_path)
        exit()


