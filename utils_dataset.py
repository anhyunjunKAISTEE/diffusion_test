import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
import random
from datetime import datetime

# HJZDataset class definition to load images and embeddings from filenames
class HJZDataset(Dataset):
    def __init__(self, root_dir, shuffle=True):
        """
        Args:
            root_dir (string): save_dir with all the images.
        """
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.png')]
        if shuffle:
            random.shuffle(self.image_files)
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((201, 201)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name)
        image = self.transform(image)

        # Extract probing_port and decap from file name
        probing_port, decap, pb_x, pb_y, dp_x, dp_y = self.extract_embeddings(img_name)
        
        # Combine embeddings into a single tensor for the model

        # hj_modified (flag)
        embeddings = torch.tensor([pb_x, pb_y, dp_x, dp_y], dtype=torch.float) #, dtype=torch.long)
        return image, embeddings

    def extract_embeddings(self, img_name):
        basename = os.path.basename(img_name)
        parts = basename.split('_')  # Split by '_'
        probing_port = int(parts[1])  # Extract probing port value
        pb_y = ((probing_port // 10) + 1)/10
        pb_x = ((probing_port % 10) + 1)/10

        decap = int(parts[2].split('.')[0])  # Extract decap value, removing the '.png' part

        dp_x = ((decap %10) +1)/10
        dp_y = ((decap //10) +1)/10

        return probing_port, decap, pb_x, pb_y, dp_x, dp_y

def test_HJZDataset(root_dir):
    
    dataset = HJZDataset(root_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    print(f"데이터셋 크기: {len(dataset)}")
    
    images, embeddings = next(iter(dataloader))

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for i in range(4):
        img = images[i].squeeze().numpy()
        ax = axs[i // 2, i % 2]
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Probing Port: {embeddings[i][0]}, Decap: {embeddings[i][1]}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 임베딩 분포 확인
    all_embeddings = []
    for _, emb in dataset:
        all_embeddings.append(emb)
        print("_", _)
        print("len(_)", len(_))
        print("emb", emb)
        break
    all_embeddings = torch.stack(all_embeddings)
    
    print("임베딩 통계:")
    print(f"Probing Port - Min: {all_embeddings[:, 0].min()}, Max: {all_embeddings[:, 0].max()}")
    print(f"Decap - Min: {all_embeddings[:, 1].min()}, Max: {all_embeddings[:, 1].max()}")



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

