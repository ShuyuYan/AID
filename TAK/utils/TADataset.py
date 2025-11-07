import torchvision.transforms.functional as F_tv
import nibabel as nib
import torch
from torch.utils.data import Dataset


def load_nii(path, target_size=(256, 256), num_slices=3):
    if path == '0':
        return torch.zeros((num_slices, *target_size), dtype=torch.float32)

    try:
        img = nib.load(path).get_fdata()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
        img = torch.from_numpy(img).float()
        # (H, W) → (H, W, 1)
        if img.ndim == 2:
            img = img.unsqueeze(-1)
        # 选取/补齐到 num_slices 张
        D = img.shape[-1]
        if D >= num_slices:
            mid = D // 2
            start = max(0, mid - num_slices // 2)
            img = img[:, :, start:start + num_slices]
        else:
            last = img[:, :, -1:]
            repeat = last.repeat(1, 1, num_slices - D)
            img = torch.cat([img, repeat], dim=-1)

        # (H, W, num_slices) → (num_slices, H, W)
        img = img.permute(2, 0, 1)
        img = F_tv.resize(img, target_size)
        return img

    except Exception as e:
        # print(f"[Warning] Load failed: {path}")
        return torch.zeros((num_slices, *target_size), dtype=torch.float32)


class TADataset(Dataset):
    def __init__(self, df, report, tabular, label, tokenizer, max_length):
        self.df = df.reset_index(drop=True)
        self.report = report
        self.tabular = tabular
        self.label = label
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        report = self.report[idx]
        tabular = torch.tensor(self.tabular[idx], dtype=torch.float32)
        label = torch.tensor(self.label[idx], dtype=torch.long)
        encoding = self.tokenizer(
            report,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        row = self.df.iloc[idx]
        TA = str(row['TA'])
        head = load_nii(row['head'], (224, 224), 3)
        thorax = load_nii(row['thorax'], (224, 224), 3)

        return {
            "TA": TA,
            "head": head,
            "thorax": thorax,
            "report": report,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "tabular": tabular,
            "label": label
        }
