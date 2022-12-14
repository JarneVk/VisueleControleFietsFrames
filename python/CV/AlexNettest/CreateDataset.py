import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
import pandas as pd
from torchvision.io import read_image
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.utils.data 
import joblib

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

dataset = CustomImageDataset("test_data_cat_dogs/classes.csv","test_data_cat_dogs/Img_dataset",transform=transforms.ToTensor())
print("dataset heeft een grootte van : "+str(len(dataset)))
test_aant = int(len(dataset)*0.75)
train_aant = len(dataset)-test_aant

train_x, test_y = torch.utils.data.random_split(dataset,[train_aant,test_aant])

train_dataloader = DataLoader(train_x, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_y, batch_size=64, shuffle=True)

#save the data
joblib.dump(train_dataloader,"python/CV/train_x.joblib")
joblib.dump(test_dataloader,"python/CV/test_y.joblib")
