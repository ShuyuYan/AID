import torchvision.transforms.functional as F_tv
import nibabel as nib
import torch
from torch.utils.data import Dataset


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
        head_path = str(row["head"])
        if head_path == '0':
            head = torch.zeros(1, dtype=torch.float32)
        else:
            head = nib.load(head_path).get_fdata()
            head = (head - head.min()) / (head.max() - head.min()) * 255
            head = torch.from_numpy(head).float()

        return {
            "image": head,
            "report": report,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "tabular": tabular,
            "label": label
        }
