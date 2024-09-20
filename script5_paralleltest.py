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

###############################
## Architectures ##

class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2

class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels, output_padding=0):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2,0, output_padding=output_padding), #hj_modified
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        # Ensure skip connection size matches x
        # modified
        # if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
        #     skip = F.interpolate(skip, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        
        x = torch.cat((x, skip), 1)
        x = self.model(x)

        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat = 256, n_classes=10):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2*n_feat)
        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2*n_feat)
        self.contextembed2 = EmbedFC(n_classes, 1*n_feat)

        self.up0 = nn.Sequential(
            # nn.ConvTranspose2d(6 * n_feat, 2 * n_feat, 7, 7), # when concat temb and cemb end up w 6*n_feat
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7, 0, output_padding=1), # hj_modified # otherwise just have 2*n_feat
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat, output_padding=1) #hj_modified
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, t, context_mask):
        # x is (noisy) image, c is context label, t is timestep, 
        # context_mask says which samples to block the context on
        # print(f"Input size: {x.size()}")
        x = self.init_conv(x)
        # print(f"Input size: {x.size()}")
        down1 = self.down1(x)
        # print(f"Input size: {down1.size()}")
        down2 = self.down2(down1)
        # print(f"After down2: {down2.size()}")

        hiddenvec = self.to_vec(down2)
        # print(f"After to_vec: {hiddenvec.size()}")
        
        # convert context to one hot embedding
        # c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)
        
        # (masking)
        # mask out context if context_mask == 1

        # context_mask = context_mask[:, None]
        # context_mask = context_mask.repeat(1,self.n_classes)
        context_mask = (1*(1-context_mask)) # need to flip 0 <-> 1
        c = c * context_mask

 

        # embed context, time step
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        # could concatenate the context embedding here instead of adaGN
        # hiddenvec = torch.cat((hiddenvec, temb1, cemb1), 1)

        
        up1 = self.up0(hiddenvec)
        # print(f"After up0: {up1.size()}")


        #up2 = self.up1(up1, down2) # if want to avoid add and multiply embeddings
        up2 = self.up1(cemb1*up1+ temb1, down2)  # add and multiply embeddings
        # print(f"After up2: {up2.size()}")
        
        up3 = self.up2(cemb2*up2+ temb2, down1)
        # print(f"After up3: {up3.size()}")
        
        if up3.size(2) != x.size(2) or up3.size(3) != x.size(3):
            x = F.interpolate(x, size=(up3.size(2), up3.size(3)), mode='bilinear', align_corners=False)
        
        out = self.out(torch.cat((up3, x), 1))
        # print(f"After out: {out.size()}")
        # quit()
        return out

class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """

        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros_like(c)+self.drop_prob).to(self.device)
        
        # return MSE between added noise, and our predicted noise
        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))

    def sample(self, n_sample, size, device,  context, guide_w = 0.0):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance
        c_i = context
        c_i = context.to(device)
        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise

        # c_i = c_i.repeat(int(n_sample/c_i.shape[0]))
        
        # don't drop context at test time
        context_mask = torch.zeros_like(c_i).to(device)
        
        # double the batch
        #hj_modified
        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1. # makes second half of batch context free

        x_i_store = [] # keep track of generated steps in case want to plot something 
    
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample,1,1,1)

            # double batch
            x_i = x_i.repeat(2,1,1,1)
            t_is = t_is.repeat(2,1,1,1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(x_i, c_i, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1+guide_w)*eps1 - guide_w*eps2
            x_i = x_i[:n_sample]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            if i%20==0 or i==self.n_T or i<8:
                x_i_store.append(x_i.detach().cpu().numpy())
        
        x_i_store = np.array(x_i_store)
        return x_i, x_i_store

def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


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
    dataset_dir = "./data/hj_z_data_3_240912"

    # test_HJZDataset(dataset_dir)
    train_z(dataset_dir=dataset_dir)
    # val_z(model_path="step_400_feat_128_model_9_09131341.pth")

