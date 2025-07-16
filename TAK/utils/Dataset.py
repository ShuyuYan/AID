import torchvision.transforms.functional as F_tv
import nibabel as nib
import torch
from torch.utils.data import Dataset


class VesselDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = nib.load(self.image_paths[idx]).get_fdata()
        image = (image - image.min()) / (image.max() - image.min()) * 255
        image = torch.from_numpy(image).int()

        if image.dim() > 3:
            image = image.narrow(4, 0, 1).squeeze(-1).squeeze(-1)

        if image.shape[0] != 256 or image.shape[1] != 208:
            image = image.permute(2, 0, 1).unsqueeze(0)
            image = F_tv.center_crop(image, (256, 208))
            image = image.squeeze(0).permute(1, 2, 0)

        return image
