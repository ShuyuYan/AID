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
    baseline_path = '/home/yanshuyu/Data/AID/Baseline.xlsx'
    df = pd.read_excel(baseline_path, sheet_name="Baseline", skiprows=1)
    bl = []
    for i in df['serial_number']:
        if str(i)[:2] == 'TA' and len(str(i)) <= 6:
            bl.append(i)

    mra = []
    folder_path = '/home/yanshuyu/Data/AID/nii_data'
    for root, dirs, files in os.walk(folder_path, topdown=False):
        last = root.split('/')[-1]
        if last[:2] == 'TA':
            mra.append(last)

    all = []
    all_path = '/home/yanshuyu/Data/AID/428mra.xlsx'
    df = pd.read_excel(all_path, sheet_name="Report")
    for i in df['serial_number']:
        if str(i)[:2] == 'TA' and len(str(i)) <= 6:
            all.append(i)
    print(len(bl), len(mra), len(all))

    common_elements = list(set(bl) & set(mra))
    marked_data = [
        (element, 1 if element in all else 0)
        for element in common_elements
    ]
    df = pd.DataFrame(marked_data, columns=['重合元素', '在all中出现'])

    df.to_excel('~/Data/AID/1.xlsx', index=False)

