import nibabel as nib
import numpy as np
import os

# 假设所有的nii文件都在一个目录下，这里替换为你的实际目录
nii_dir = "./12/0001/T1"
nii_file_names = [f for f in os.listdir(nii_dir) if f.endswith(".nii.gz")]
nii_file_names.sort()  # 对文件名进行排序，确保拼接顺序正确
# nii_file_names = nii_file_names[3:]
nii_file_names = nii_file_names[::2]
print(nii_file_names)

# 读取第一个文件的头信息，用于后续创建新的nii文件
first_nii_path = os.path.join(nii_dir, nii_file_names[0])
first_nii_img = nib.load(first_nii_path)
affine_matrix = first_nii_img.affine
header = first_nii_img.header

# 初始化拼接后的数据数组
data_list = []
for nii_file_name in nii_file_names:
    nii_path = os.path.join(nii_dir, nii_file_name)
    nii_img = nib.load(nii_path)
    data = nii_img.get_fdata()
    data = np.array(data)
    data = data[:, :, None]
    data_list.append(data)

# 沿着轴向（这里假设是z轴，索引为2）拼接数据
concatenated_data = np.concatenate(data_list, axis=2)
new_nii_img = nib.Nifti1Image(concatenated_data, affine_matrix, header)
# 保存新的拼接后的nii文件，替换为你想要保存的路径和文件名
nib.save(new_nii_img, "./12/0001/concat_wall.nii")