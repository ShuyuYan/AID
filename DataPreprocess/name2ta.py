import pandas as pd


def merge(file_path):
    df1 = pd.read_excel(file_path, sheet_name='Sheet1')
    df2 = pd.read_excel(file_path, sheet_name='Sheet2')
    ta = []
    name = df1['PatientName']
    for i in range(0, len(name)):
        flag = 0
        for j in range(0, 1476):
            if name[i] == df2['name'][j]:
                ta.append(df2['TA'][j])
                flag = 1
                break
        if flag == 0:
            ta.append('XXXXXXXX')

    print(len(name), len(ta))
    df1.insert(2, 'TA', ta)
    df1.to_excel('~/Data/AID/TAK/patients1.xlsx', index=False)  # index=False表示不保存行索引




if __name__ == "__main__":
    file_path = "/home/yanshuyu/Data/AID/TAK/patients.xlsx"
    merge(file_path)
