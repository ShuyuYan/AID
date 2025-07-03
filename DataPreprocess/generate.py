import os
import pandas as pd
import dicom2nii

if __name__ == '__main__':
    mra_path = '~/Data/AID/428mra.xlsx'
    df = pd.read_excel(mra_path, sheet_name='Report')
    print(df['serial_number'])