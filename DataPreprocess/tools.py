import pandas as pd
import tempfile
import os
from pathlib import Path


def convert_date(date_str):
    try:
        # 解析2011-05-19 00:00:00格式
        dt = pd.to_datetime(date_str)
        return dt.strftime('%Y-%m-%d')
    except ValueError:
        try:
            # 解析19/05/2011格式
            dt = pd.to_datetime(date_str, format='%d/%m/%Y')
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            return date_str


def harmonize_date_format(file_path, sheet_name):
    """
    统一两种时间格式
    """
    date = ['mra_examination_date','date_enrolled', 'dob',	'origin_symptom_date', 'diagnosis_date']
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    for id in date:
        df[id] = df[id].apply(convert_date)
    df.to_excel(file_path, index=False, sheet_name=sheet_name)


def remove_angle_brackets_in_filenames(root_dir):
    """
    遍历root_dir目录及其子目录，将文件名中的'<>'去掉
    """
    root_path = Path(root_dir)
    if not root_path.is_dir():
        print(f"错误：目录 {root_dir} 不存在或不是一个文件夹。")
        return

    for path in root_path.rglob('*.nii'):
        old_name = path.name
        new_name = old_name.replace('<', '').replace('>', '')
        if new_name != old_name:
            new_path = path.with_name(new_name)
            try:
                path.rename(new_path)
                print(f'已重命名：\n  {path}\n→ {new_path}\n')
            except Exception as e:
                print(f'重命名失败：{path}\n原因：{e}\n')


if __name__ == '__main__':
    harmonize_date_format('/home/yanshuyu/Data/AID/all.xlsx', 'data')
    # remove_angle_brackets_in_filenames(os.path.expanduser('~/Data/AID/MRA'))
