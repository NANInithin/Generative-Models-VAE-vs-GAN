import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from scipy.stats import norm
import numpy as np
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
print (f"Using device: {device}")

Batch_Size = 128
Learning_Rate = 1e-3
Epochs = 20
Latent_Dim = 20

os.makedirs('results/vae_results', exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),  
])

train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=Batch_Size, shuffle=True)
test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=Batch_Size, shuffle=False)

class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 28 -> 14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 14 -> 7
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Latent space stats
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)
        
        # Decoder Bridge (Linear expansion)
        self.fc_decode = nn.Linear(latent_dim, 64 * 7 * 7)
        
        # Decoder CNN (Note: fc_decode is NOT in here)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 7, 7)), # Expects input size 3136 (64*7*7)
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # 7 -> 14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14 -> 28
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar) # z shape is [Batch, 20]
        
        # Decode
        z_expanded = self.fc_decode(z)      # z_expanded shape is [Batch, 3136]
        reconstruction = self.decoder(z_expanded) # Decoder consumes [Batch, 3136]
        
        return reconstruction, mu, logvar

model = VAE(latent_dim=Latent_Dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=Learning_Rate)

def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
print("Starting VAE training...")
model.train()
for epoch in range(Epochs):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
    print(f'Epoch {epoch+1}, Loss: {train_loss / len(train_loader.dataset):.4f}')
    
print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")
print(f"GPU Memory Cached: {torch.cuda.memory_reserved(device)/1024**2:.2f} MB")

with torch.no_grad():
    sample = torch.randn(64, Latent_Dim).to(device)
    
    # 1. Expand Latent Vector
    sample_expanded = model.fc_decode(sample)
    
    # 2. Decode Expanded Vector
    sample = model.decoder(sample_expanded).view(-1, 1, 28, 28)
    
    save_image(sample.cpu(), f'results/vae_results/sample_epoch_{epoch+1}.png')

    if (epoch + 1) % 5 == 0:
        with torch.no_grad():
            n = 8
            comparision = torch.cat([data[:n], recon_batch[:n]])
            save_image(comparision.cpu(), f'results/vae_results/reconstruction_epoch_{epoch+1}.png', nrow=n)

            z = torch.randn(64, Latent_Dim).to(device)
            
            # FIX HERE AS WELL
            z_expanded = model.fc_decode(z)
            sample = model.decoder(z_expanded).cpu()
            
            save_image(sample, f'results/vae_results/sample_epoch_{epoch+1}.png')
print("VAE training completed.")

def interpolate_points(model, p1, p2, n_steps=10):
    model.eval()
    with torch.no_grad():
        # Create n_steps between p1 and p2
        ratios = torch.linspace(0, 1, n_steps).to(device)
        vectors = []
        for ratio in ratios:
            v = (1 - ratio) * p1 + ratio * p2
            vectors.append(v)
        
        z = torch.stack(vectors)
        z_expanded = model.fc_decode(z)
        generated = model.decoder(z_expanded)
        return generated

# Get two real images to encode and interpolate between
data_iter = iter(train_loader)
images, _ = next(data_iter)
images = images.to(device)

# Encode two specific images (e.g., a shoe and a shirt)
with torch.no_grad():
    _, mu, _ = model(images)
    p1, p2 = mu[0], mu[1] # Take first two images from batch

# Generate interpolation
interpolation = interpolate_points(model, p1, p2)
save_image(interpolation, 'results/vae_results/interpolation.png', nrow=10)
print("Interpolation saved to results/vae_results/interpolation.png")

# --- Correct Manifold Function ---
def plot_latent_manifold(model, n=20, digit_size=28):
    # Only run this if latent_dim was set to 2
    if model.fc_mu.out_features != 2:
        print("Skipping 2D manifold plot (requires latent_dim=2)")
        return

    model.eval()
    figure = np.zeros((digit_size * n, digit_size * n))
    
    # Grid of values
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = torch.tensor([[xi, yi]], dtype=torch.float).to(device)
            
            with torch.no_grad():
                # Corrected Decoding Logic:
                z_expanded = model.fc_decode(z_sample)
                x_decoded = model.decoder(z_expanded)
                
            digit = x_decoded[0].reshape(digit_size, digit_size).cpu().numpy()
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.axis('off')
    plt.savefig('results/manifold_2d.png')
    plt.show()
    print("2D Manifold saved to 'results/manifold_2d.png'")

# Execute Manifold Plot (Will skip if latent_dim != 2)
plot_latent_manifold(model)