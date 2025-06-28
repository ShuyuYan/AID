import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

def show_3d_nii_image(nii_file_path):
    nii_img = nib.load(nii_file_path)
    img_data = nii_img.get_fdata()
    img_data = np.transpose(img_data, (2,1,0))
    print(img_data.shape)
    # 获取图像的维度信息
    num_slices, height, width = img_data.shape

    # 创建subplot
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # 展示冠状面（yz平面，切片方向为x轴）
    axs[0].imshow(np.rot90(img_data[:, :, width // 2], k=-1), cmap='gray')
    axs[0].set_title('Coronal View')
    axs[0].axis('off')

    # 展示矢状面（xz平面，切片方向为y轴）
    axs[1].imshow(np.rot90(img_data[:, height // 2, :]), cmap='gray')
    axs[1].set_title('Sagittal View')
    axs[1].axis('off')

    # 展示轴面（xy平面，切片方向为z轴）
    axs[2].imshow(img_data[num_slices // 2, :, :], cmap='gray')
    axs[2].set_title('Axial View')
    axs[2].axis('off')

    plt.show()

if __name__ == "__main__":
    nii_file_path = "/home/yanshuyu/Data/AID/pre post (T2)/TA1042ZCH/I_fl3d-cor_head_post/post.nii"
    show_3d_nii_image(nii_file_path)