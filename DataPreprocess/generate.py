import os
import pandas as pd
from dicom2nii import list_files, classify
import SimpleITK as sitk


def convert(series, output_path, save_name):
    if len(series) != 0:
        series_reader = sitk.ImageSeriesReader()
        series_reader.SetFileNames(series)
        image3D = series_reader.Execute()
        sitk.WriteImage(image3D, os.path.join(output_path, save_name))


def dicom2nii(patient_list, output_path):
    cur_series = []
    for patient, files in patient_list:
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


if __name__ == '__main__':
    root_path = os.path.expanduser('~/Data/AID/')
    mra_path = os.path.expanduser('~/Data/AID/428mra.xlsx')
    df = pd.read_excel(mra_path, sheet_name='Report')
    data = []
    n = len(df)
    for i in range(n):
        ta, date, dcm = df['serial_number'][i], df['mra_examination_date'][i], df['DICOM'][i]
        date = str(date).split()[0].replace('-', '')
        py = ''.join(c for c in df['pinyin'][i] if c.isupper())
        data.append([ta, date, dcm, py])
        print(ta, date, dcm, py)
        if '+' in str(dcm):
            continue

        dcm_folder = root_path + 'DICOM_data/' + str(dcm) + ' DICOM'
        file_list = list_files(dcm_folder)
        patient_list = classify(file_list)
        flag = 0
        for y in patient_list:
            for file in y[1]:
                name = ''.join(c for c in file['PatientName'] if c.isupper())
                if name == py:
                    flag = 1
                    new_date = [file['StudyDate'], file['SeriesDate'], file['AcquisitionDate'], file['ContentDate']]
                    output_path = root_path + '428MRA/' + str(ta)
                    if date in new_date:
                        flag = 2
                        dicom2nii([y], output_path)
                break
        if flag == 0:
            data[i].append('No name')
            data[i] += [0, 0, 0, 0]
        elif flag == 1:
            data[i].append('Wrong date')
            data[i] += new_date
        else:
            data[i].append('Successful')
            data[i] += [0, 0, 0, 0]

        # opt = input()
        # if opt == '1':
        #     break
    save = pd.DataFrame(data=data, columns=['TA', 'date', 'dcm', 'pinyin', 'status',
                                            'StudyDate', 'SeriesDate', 'AcquisitionDate', 'ContentDate'])
    save.to_excel(root_path + '/428MRA/data.xlsx', sheet_name='data')



