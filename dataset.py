import os
import pandas as pd
import random
import torch
from torch.utils.data import Dataset
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import requests
from torchvision.datasets.utils import download_url, extract_archive

celeba_dataset_url = "https://cseweb.ucsd.edu/~weijian/static/datasets/celeba/img_align_celeba.zip"
face_attr_url = "https://raw.githubusercontent.com/taki0112/StarGAN-Tensorflow/master/dataset/celebA/list_attr_celeba.txt"


# Download the dataset
def download_dataset():
    if not os.path.exists("img_align_celeba"):
        if not os.path.exists("img_align_celeba.zip"):
            download_url(celeba_dataset_url, "./", "img_align_celeba.zip")
        extract_archive("img_align_celeba.zip", "./")
    if not os.path.exists("list_attr_celeba.txt"):
        download_url(face_attr_url, "./", "list_attr_celeba.txt")


class CelebADataset(Dataset):
    def __init__(self, image_dir, attr_file, transform=None):
        self.image_dir = image_dir
        self.attr_file = attr_file
        self.transform = transform
        self.attribute = pd.read_csv(attr_file, delim_whitespace=True, skiprows=1)

    def __len__(self):
        return len(self.attribute)

    def show(self, idx):
        image, attributes = self[idx]
        label = random.choice(attributes)

        # Inverse normalization if necessary
        if self.transform is not None:
            if isinstance(self.transform, torchvision.transforms.Compose):
                for tr in self.transform.transforms:
                    if isinstance(tr, torchvision.transforms.Normalize):
                        mean = torch.tensor(tr.mean)
                        std = torch.tensor(tr.std)
                        image = image * std[:, None, None] + mean[:, None, None]
            elif isinstance(self.transform, torchvision.transforms.Normalize):
                mean = torch.tensor(self.transform.mean)
                std = torch.tensor(self.transform.std)
                image = image * std[:, None, None] + mean[:, None, None]

        image = image.permute(1, 2, 0).numpy()
        plt.imshow(image)
        plt.axis("off")
        plt.title(label)
        plt.show()

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            images, attributes_list = [], []
            indices = range(*idx.indices(len(self)))
            for i in indices:
                image, attribute = self[i]
                images.append(image)
                attributes_list.append(attribute)
            return images, attributes_list

        else:
            img_name = self.attribute.index[idx]
            image = Image.open(os.path.join(self.image_dir, str(img_name)))
            if self.transform:
                image = self.transform(image)
            observation = self.attribute.iloc[idx]
            attributes = observation[observation == 1].index.tolist()
            label = random.choice(attributes)

            return image, label


image_size = 112
batch_size = 128

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((image_size, image_size)),
    torchvision.transforms.ToTensor()
])

download_dataset()
image_dir = 'img_align_celeba'
attr_file = 'list_attr_celeba.txt'
dataset = CelebADataset(image_dir, attr_file, transform=transform)
