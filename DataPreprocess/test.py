import pandas as pd
import os


def convert_folder_name():
    file_path = "/home/yanshuyu/Data/AID/TAK/patients.xlsx"
    df = pd.read_excel(file_path, sheet_name="zhengli")
    f = {}
    id = df['PatientID']
    ta = df['TA']
    for i in range(len(ta)):
        if ta[i] != 'XXXXXXXX':
            f[id[i]] = ta[i]

    print(f)
    folder_path = '/home/yanshuyu/Data/AID/TAK/nii_data'
    for root, dirs, files in os.walk(folder_path, topdown=False):
        old = root
        if old.split('/')[-1] in f.keys():
            new = old.replace(old.split('/')[-1], f[old.split('/')[-1]])
            os.rename(old, new)


if __name__ == "__main__":
    ta = []
    folder_path = '/home/yanshuyu/Data/AID/TAK/nii_data'
    for root, dirs, files in os.walk(folder_path, topdown=False):
        last = root.split('/')[-1]
        if last[:2]=='TA':
            ta.append(last)
    print(len(ta), ta)

    file_path = '/home/yanshuyu/Data/AID/TAK/0616_436_mra.xlsx'
    df = pd.read_excel(file_path, sheet_name="report")
    patients = df['serial_number']
    for patient in patients:
        if patient not in ta:
            print(patient)
