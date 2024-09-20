# dimension을 4로 바꾸기. 

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

from utils_dataset import *
from utils_diff import *
import argparse

###############################
## Architectures ##


# Training function
def train_z(dataset_dir):

    dataset_dir = dataset_dir
   
    n_epoch = 10
    batch_size = 48
    n_T = 400

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= "0,1,2,3" # hj
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_classes = 4 # Updated for 100x100 combinations
    n_feat = 128
    lrate = 1e-4

    save_model = True
    timestamp = datetime.now().strftime('%m%d%H%M')
    save_dir = './data/models/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    loss_save_dir = './data/loss/'
    if not os.path.exists(loss_save_dir):
        os.makedirs(loss_save_dir)

    ws_test = [0.0, 0.5, 2.0]

    ddpm = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes),
                betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm = nn.DataParallel(ddpm) #hj_modified
    ddpm.to(device)
    print("model parameter # [x10^6]:", count_parameters(ddpm)/1e6)

    dataset = HJZDataset(root_dir=dataset_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)

    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)
    losses = []
    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()
        optim.param_groups[0]['lr'] = lrate * (1 - ep / n_epoch)
        pbar = tqdm(dataloader)
        loss_ema = None
        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            loss = ddpm(x, c)
            loss = loss.mean() #hj_modified
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            losses.append(loss.item())
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()


        if save_model and ep == n_epoch-1: 
            # ddpm.module.state_dict() 으로 하면 ㄱㅊ?
            torch.save(ddpm.state_dict(), save_dir +f"step_{n_T}_feat_{n_feat}_model_{n_epoch - 1}_{timestamp}.pth")
            print('saved model at ' + save_dir + f"step_{n_T}_feat_{n_feat}_model_{n_epoch - 1}_{timestamp}.pth")


    
    # 손실 그래프를 파일로 저장
    plt.figure(figsize=(10, 5))  # 그래프 크기 고정
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')  # 로그 스케일 설정
    plt.ylim(0.05, 2)  # y축 범위 고정: log(0.001)에서 log(10)까지
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.grid(True)
    loss_plot_path = loss_save_dir + f"step_{n_T}_feat_{n_feat}_model_{n_epoch - 1}_{timestamp}.png"
    plt.savefig(loss_plot_path)
    plt.close()

    # 손실 값 저장 경로 및 이름 설정
    loss_data_path = loss_save_dir + f"step_{n_T}_feat_{n_feat}_model_{n_epoch - 1}_{timestamp}.txt"
    np.savetxt(loss_data_path, losses, fmt='%.6f')

def val_z(model_path):
    model_path = model_path

    n_T = 400

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= "2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_classes = 4  # hj_modified
    n_feat = 128
    
    timestamp = datetime.now().strftime('%m%d%H%M')
    model_dir = model_path
    save_dir = './data/models/'
    img_save_dir = './data/images/'
    ws_test = [0.0, 0.5, 2.0]

    ddpm = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm = nn.DataParallel(ddpm)  # hj_modified DataParallel을 ㅗ감싸기
    ddpm.load_state_dict(torch.load(save_dir + model_path, map_location=device))  # Load your trained model

    ddpm.eval()


    with torch.no_grad():
        n_sample = 1 # hj_modifid

        for w in ws_test:
            # Generate images based on selected probing_port and decap positions

            # Create one-hot encoded context
            context_index = [0.1000, 0.1000, 0.8000, 0.3000] # hj_modified
            context = torch.tensor(context_index, dtype=torch.float).to(device)  # Make context a 1D tensor
            
            # Generate images #hj_modified
            # x_gen, x_gen_store = ddpm.sample(n_sample, (201, 201), device,  context, guide_w=w)
            x_gen, x_gen_store = ddpm.module.sample(n_sample, (201, 201), device,  context, guide_w=w)
            # Save generated images
            save_image(x_gen * -1 + 1, img_save_dir+ f"ctx{context_index}_w_{w}_{timestamp}.png")
            print(f'saved image for ctx_{context_index}_w_{w}_{timestamp}')

            # # Save evolution of generated images over time if needed
            # fig, axs = plt.subplots(1, len(x_gen_store), figsize=(12, 3))
            # for i, img in enumerate(x_gen_store):
            #     axs[i].imshow(-img[0, 0], cmap='gray')
            #     axs[i].axis('off')
            # plt.tight_layout()
            # plt.savefig(save_dir + f"P{probing_port}_D{decap}_w{w}.png")
            # plt.close()




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train or validate Z model")
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--val', action='store_true', help='Run validation')
    # parser.add_argument('--dataset_dir', type=str, default="./data/hj_z_data_3_240912", help='Dataset directory for training')
    # parser.add_argument('--model_path', type=str, default="step_400_feat_128_model_9_09131341.pth", help='Model path for validation')
    
    args = parser.parse_args()

    dataset_dir = "./data/hj_z_data_3_240912"

    if args.train:
        train_z(dataset_dir=dataset_dir)
    elif args.val:
        val_z(model_path="step_400_feat_128_model_9_09131341.pth")
    else:
        # test_HJZDataset(dataset_dir)
        train_z(dataset_dir=dataset_dir)
        # val_z(model_path="step_400_feat_128_model_9_09131341.pth")