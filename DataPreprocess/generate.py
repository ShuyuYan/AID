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
    err_list = [264, 265, 270, 274, 275, 276, 277, 279, 281, 282, 284, 286, 287, 288]
    df = pd.read_excel(mra_path, sheet_name='Report')
    data = []
    n = len(df)
    tot = -1
    for i in range(n):
        ta, date, dcm = df['serial_number'][i], df['mra_examination_date'][i], str(df['DICOM'][i])
        date = str(date).split()[0].replace('-', '')
        py = ''.join(c for c in df['pinyin'][i] if c.isupper())

        dicom = [int(d) for d in dcm.split('+')]
        for d in dicom:
            data.append([ta, date, d, py])
            tot += 1
            print(ta, date, d, py)
            if d in err_list:
                data[tot].append('DICOM error')
                data[tot] += [0, 0, 0, 0]
                continue

            dcm_folder = root_path + 'DICOM_data/' + str(d) + ' DICOM'
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
                data[tot].append('No name')
                data[tot] += [0, 0, 0, 0]
            elif flag == 1:
                data[tot].append('Wrong date')
                data[tot] += new_date
            else:
                data[tot].append('Successful')
                data[tot] += [0, 0, 0, 0]

            save = pd.DataFrame(data=data, columns=['TA', 'date', 'dcm', 'pinyin', 'status',
                                                    'StudyDate', 'SeriesDate', 'AcquisitionDate', 'ContentDate'])
            save.to_excel(root_path + '/428MRA/data.xlsx', sheet_name='data')

