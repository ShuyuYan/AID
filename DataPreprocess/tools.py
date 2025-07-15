import pandas as pd
import tempfile
import os


def convert_date(date_str):
    try:
        # 尝试解析 2011-05-19 00:00:00 格式
        dt = pd.to_datetime(date_str)
        return dt.strftime('%Y-%m-%d')
    except ValueError:
        try:
            # 尝试解析 19/05/2011 格式
            dt = pd.to_datetime(date_str, format='%d/%m/%Y')
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            return date_str


def harmonize_date_format(file_path):
    """
    统一两种时间格式
    """
    date = ['mra_examination_date', 'dob',	'origin_symptom_date', 'diagnosis_date']
    df = pd.read_excel(file_path, sheet_name='data')
    for id in date:
        df[id] = df[id].apply(convert_date)

    df.to_excel(file_path, index=False, sheet_name='data')


if __name__ == '__main__':
    harmonize_date_format('/home/yanshuyu/Data/AID/all.xlsx')
