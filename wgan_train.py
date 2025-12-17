import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
LR = 0.00005        # WGAN often uses lower learning rate
CLIP_VALUE = 0.01   # Weight clipping value
N_CRITIC = 5        # Train critic 5 times for every 1 generator step
EPOCHS = 30
LATENT_DIM = 100 

os.makedirs("results/wgan_results", exist_ok=True)
print(f"Using device: {DEVICE}")

# 1. Dataset Loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]) 
])

train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 2. Generator Network (Same as Standard GAN)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM, 128, 7, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, x):
        return self.main(x)

# 3. Critic Network (Modified Discriminator)
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, 7, 1, 0, bias=False),
            # NO SIGMOID HERE for WGAN!
        )

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)

# Initialize models
netG = Generator().to(DEVICE)
netD = Critic().to(DEVICE) # "Discriminator" is now called "Critic"

# Optimizers (Use RMSprop for WGAN with clipping)
optimizerD = optim.RMSprop(netD.parameters(), lr=LR)
optimizerG = optim.RMSprop(netG.parameters(), lr=LR)

print("Starting WGAN Training...")
fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=DEVICE)

for epoch in range(EPOCHS):
    for i, (data, _) in enumerate(train_loader):
        
        real_data = data.to(DEVICE)
        b_size = real_data.size(0)

        # ---------------------
        #  1. Train Critic (N_CRITIC times)
        # ---------------------
        # WGAN Logic: Maximize (E[real] - E[fake])
        netD.zero_grad()
        
        # Real images
        output_real = netD(real_data)
        # Loss is negative because we want to maximize the output for real images
        loss_d_real = -torch.mean(output_real) 
        
        # Fake images
        noise = torch.randn(b_size, LATENT_DIM, 1, 1, device=DEVICE)
        fake_data = netG(noise).detach()
        output_fake = netD(fake_data)
        loss_d_fake = torch.mean(output_fake)
        
        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        optimizerD.step()

        # Weight Clipping (The constraint for WGAN)
        for p in netD.parameters():
            p.data.clamp_(-CLIP_VALUE, CLIP_VALUE)

        # ---------------------
        #  2. Train Generator (Every N_CRITIC steps)
        # ---------------------
        if i % N_CRITIC == 0:
            netG.zero_grad()
            noise = torch.randn(b_size, LATENT_DIM, 1, 1, device=DEVICE)
            fake_data = netG(noise)
            output = netD(fake_data)
            
            # Generator wants to maximize Critic's output for fakes
            loss_g = -torch.mean(output)
            loss_g.backward()
            optimizerG.step()
            
            if i % 100 == 0:
                 print(f'[{epoch+1}/{EPOCHS}][{i}/{len(train_loader)}] '
                       f'Loss_D: {loss_d.item():.4f} Loss_G: {loss_g.item():.4f}')

    # Save images
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
        save_image(fake * 0.5 + 0.5, f'results/wgan_results/wgan_epoch_{epoch+1}.png')

print("WGAN Training Completed.")