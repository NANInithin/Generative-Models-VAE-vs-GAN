import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from scipy.stats import norm

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
LR = 0.0002
BETA1 = 0.5 # Momentum for Adam (standard for DCGAN)
EPOCHS = 30 # GANs often need more epochs than VAEs
LATENT_DIM = 100 
IMG_SIZE = 28 # Fashion-MNIST size

# Create results directory
os.makedirs("GAN/results/gan_results", exist_ok=True)
print(f"Using device: {DEVICE}")

# 1. Dataset Loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]) # Normalize to [-1, 1] for Tanh
])

train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 2. Generator Network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input is Z (Latent Vector)
            # Project and reshape
            nn.ConvTranspose2d(LATENT_DIM, 128, 7, 1, 0, bias=False), # 1 -> 7x7
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), # 7 -> 14
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False), # 14 -> 28
            nn.Tanh() # Output: [-1, 1]
        )

    def forward(self, x):
        return self.main(x)

# 3. Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input is (1, 28, 28)
            nn.Conv2d(1, 64, 4, 2, 1, bias=False), # 28 -> 14
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), # 14 -> 7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 1, 7, 1, 0, bias=False), # 7 -> 1 (Scalar output)
            nn.Sigmoid() # Output: [0, 1] probability
        )

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)

# Initialize models
netG = Generator().to(DEVICE)
netD = Discriminator().to(DEVICE)

# Custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

netG.apply(weights_init)
netD.apply(weights_init)

# Optimizers
optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, 0.999))

criterion = nn.BCELoss()


print("Starting GAN Training...")
fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=DEVICE) # For consistent visualization

for epoch in range(EPOCHS):
    for i, (data, _) in enumerate(train_loader):
        # ---------------------
        #  1. Train Discriminator
        # ---------------------
        netD.zero_grad()
        
        # Train on REAL images
        real_data = data.to(DEVICE)
        b_size = real_data.size(0)
        label_real = torch.full((b_size,), 1.0, device=DEVICE) # Real label = 1
        
        output = netD(real_data)
        errD_real = criterion(output, label_real)
        errD_real.backward()
        
        # Train on FAKE images
        noise = torch.randn(b_size, LATENT_DIM, 1, 1, device=DEVICE)
        fake_data = netG(noise)
        label_fake = torch.full((b_size,), 0.0, device=DEVICE) # Fake label = 0
        
        output = netD(fake_data.detach()) # Detach to avoid training G here
        errD_fake = criterion(output, label_fake)
        errD_fake.backward()
        
        errD = errD_real + errD_fake
        optimizerD.step()

        # -----------------
        #  2. Train Generator
        # -----------------
        netG.zero_grad()
        
        # We want the discriminator to think these fakes are real (label=1)
        label_flipped = torch.full((b_size,), 1.0, device=DEVICE)
        
        output = netD(fake_data)
        errG = criterion(output, label_flipped)
        errG.backward()
        optimizerG.step()
        
        if i % 100 == 0:
            print(f'[{epoch+1}/{EPOCHS}][{i}/{len(train_loader)}] '
                  f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}')

    # Save images every epoch
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
        # Denormalize from [-1, 1] to [0, 1] for saving
        save_image(fake * 0.5 + 0.5, f'GAN/results/gan_results/gan_epoch_{epoch+1}.png')

print("GAN Training Completed.")