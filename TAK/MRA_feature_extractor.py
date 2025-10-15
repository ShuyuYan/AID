import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from tqdm import tqdm
from torchvision.models import ResNet50_Weights
import os
from utils.Dataset import MRADataset


def load_model():
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    model.to(device)
    return model


def extract_features(img, model, batch_size):
    slices = np.transpose(img, (2, 0, 1))  # (N, 256, 208)
    slices = slices.float().unsqueeze(1)  # (N, 1, 256, 208)
    slices = slices.repeat(1, 3, 1, 1)  # (N, 3, 256, 208)

    slices = slices / slices.max()  # 归一化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    slices = (slices - mean) / std

    features = []
    with torch.no_grad():
        for i in range(0, slices.size(0), batch_size):
            batch = slices[i:i + batch_size].to(device)  # (batch_size, 3, 256, 208)
            batch_features = model(batch)  # (batch_size, 2048, 1, 1)
            batch_features = batch_features.view(batch.size(0), -1)  # (batch_size, 2048)
            features.append(batch_features.cpu().numpy())

    features = np.concatenate(features, axis=0)  # (N, 2048)
    return np.mean(features, axis=0)  # (2048, )


image_folder = os.path.expanduser('~/Data/AID/filtered_nii_data/head')
image_paths = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith(".nii")]
dataset = MRADataset(image_paths)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model()

all_features = []
for img in tqdm(dataset, desc="Extracting"):
    features = extract_features(img, model, 32)
    all_features.append(features)
all_features = np.array(all_features)
print(all_features)
print(all_features.shape)

