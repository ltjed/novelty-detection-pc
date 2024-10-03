import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import random
import os
from tqdm import tqdm
from torch.distributions.multivariate_normal import MultivariateNormal
from src.get_data import *
from src.models import *
from src.utils import *
from src.nd import *
from datetime import datetime

save_dir = os.path.join('results', 'autoencoders')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Define Autoencoder and Variational Autoencoder with one hidden layer
class Autoencoder(nn.Module):
    def __init__(self, image_size, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(image_size, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, image_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_recon = self.decoder(z)
        # x_recon = x_recon.view(x.size(0), 1, int(np.sqrt(image_size)), int(np.sqrt(image_size)))
        return x_recon

class VariationalAutoencoder(nn.Module):
    def __init__(self, image_size, latent_dim):
        super(VariationalAutoencoder, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(image_size, latent_dim)
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)
        # Decoder
        self.fc3 = nn.Linear(latent_dim, image_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc_mu(h1), self.fc_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.sigmoid(self.fc3(z))
        return h3

    def forward(self, x, add_noise=True):
        x = x.view(x.size(0), -1)
        if add_noise:
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            x_recon = self.decode(z)
            # x_recon = x_recon.view(x.size(0), 1, int(np.sqrt(image_size)), int(np.sqrt(image_size)))
            return x_recon, mu, logvar
        else:
            z, _ = self.encode(x)
            x_recon = self.decode(z)
            return x_recon

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Generate a training set of N images
def get_datasets(N, seed, dataset='gaussian', b=0, batch_size=64, device='cpu'):
    dimension = image_size
    if dataset == "gaussian":
            cov = np.ones((dimension, dimension)) * b
            np.fill_diagonal(cov, 1)
            L = np.linalg.cholesky(cov)
            fam = np.random.randn(N, dimension) @ L
            nov = np.random.randn(N, dimension) @ L
            X = torch.from_numpy(fam).float().to(device)
            X_test = torch.from_numpy(nov).float().to(device)
    elif dataset == "tinyimagenet":
        (X, _), (X_test, _) = get_tiny_imagenet(
            "./data",
            sample_size=N,
            sample_size_test=N,
            batch_size=batch_size,
            seed=seed,
            device=device,
        )
        X = X.reshape((X.shape[0], -1)).float()
        X_test = X_test.reshape((X_test.shape[0], -1)).float()
    return X, X_test
    

# Step 3: Train both models on the training set
def train_autoencoder(model, dataloader, num_epochs=100, learning_rate=1e-3, device='cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    loss_fn = nn.MSELoss()
    model.train()
    losses = []
    for epoch in range(num_epochs):
        total_loss = 0
        # for img, _ in dataloader:
        for img in dataloader:
            img = img.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = loss_fn(output, img)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        lr_scheduler.step()
        print(f"[AE] Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    # # Plot the loss curve
    # plt.figure(figsize=(10, 5))
    # plt.plot(range(1, num_epochs + 1), losses, label='Autoencoder Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Autoencoder Training Loss')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

def train_vae(model, dataloader, num_epochs=5, learning_rate=1e-3, device='cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    model.train()
    losses = []
    for epoch in range(num_epochs):
        total_loss = 0
        for img in dataloader:
            img = img.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(img)
            # print(f"recon_batch is on {recon_batch.device}")
            # print(f"img is on {img.device}")
            # print(f"recon_batch shape: {recon_batch.shape}")
            # print(f"img shape: {img.shape}")
            assert torch.isfinite(recon_batch).all(), "recon_batch contains NaNs or Infs"
            assert torch.isfinite(img).all(), "img contains NaNs or Infs"
            loss = vae_loss_function(recon_batch, img, mu, logvar, lmbda=lmbda)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        lr_scheduler.step()
        print(f"[VAE] Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    # # Plot the loss curve
    # plt.figure(figsize=(10, 5))
    # plt.plot(range(1, num_epochs + 1), losses, label='VAE Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Variational Autoencoder Training Loss')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

def vae_loss_function(recon_x, x, mu, logvar, lmbda=1):
    recon_x = recon_x.view(x.size(0), -1)
    x = x.view(x.size(0), -1)
    error = F.mse_loss(recon_x, x, reduction='mean') # MSE loss since could have negative pixel values
    # if dataset == 'gaussian':
    #     error = F.mse_loss(recon_x, x, reduction='mean') # MSE loss since could have negative pixel values
    # else:
    #     error = F.binary_cross_entropy(recon_x, x, reduction='mean')
    # print(f"mu is on {mu.device}")
    # print(f"logvar is on {logvar.device}")
    # print(f"logvar: {logvar}")
    # assert torch.isfinite(mu).all(), "mu contains NaNs or Infs"
    # assert torch.isfinite(logvar).all(), "logvar contains NaNs or Infs"
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * lmbda
    return error + KLD


# Run a forward pass on both sets of images, collect errors
def get_reconstruction_errors(model, dataloader, device='cpu', model_type='ae'):
    model.eval()
    errors = []
    with torch.no_grad():
        for img in dataloader:
            img = img.to(device)
            if model_type == 'ae':
                output = model(img)
            elif model_type == 'vae':
                output = model(img, add_noise=False) # No added noise when sampling from VAE; improves accuracy for novelty detection
            else:
                raise ValueError("model_type must be 'ae' or 'vae'")
            # Compute reconstruction error per sample
            recon_error = F.mse_loss(output, img, reduction='none')
            recon_error = recon_error.view(recon_error.size(0), -1).mean(dim=1)
            errors.extend(recon_error.cpu().numpy())
    return np.array(errors)

# Step 6: Compare the two arrays using the provided function
def compare_novelty(energy_nov, energy_fam):
    # Compare the judgments for each of the two
    energy_comparison = energy_nov - energy_fam
    # If an entry is positive, then the subject has correctly identified the novel picture from that pair
    return np.where(energy_comparison >= 0, 1, 0)

# Step 7: Repeat the experiment for different N and plot the accuracy for each
def run_experiment(N_values, num_epochs=100, seeds=5):
    accuracies_ae = []
    accuracies_vae = []
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    for N in N_values:
        print(f"\nRunning experiment with N = {N}")
        acc_ae_seeds = []
        acc_vae_seeds = []
        for seed in range(seeds):
            print(f"  Seed {seed+1}/{seeds}")
            # Prepare datasets
            train_dataset, novel_dataset = get_datasets(N, seed, dataset, b, batch_size=64, device=device)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            novel_loader = DataLoader(novel_dataset, batch_size=64, shuffle=False)
            fam_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
            
            # Initialize models
            latent_dim = d_latent
            
            print(f"Device: {device}")
            ae_model = Autoencoder(image_size=image_size, latent_dim=latent_dim).to(device)
            vae_model = VariationalAutoencoder(image_size=image_size, latent_dim=latent_dim).to(device)
            
            # Train models
            train_autoencoder(ae_model, train_loader, num_epochs=num_epochs, device=device)
            train_vae(vae_model, train_loader, num_epochs=num_epochs, device=device)
            
            # Get reconstruction errors
            errors_fam_ae = get_reconstruction_errors(ae_model, fam_loader, device=device, model_type='ae')
            errors_nov_ae = get_reconstruction_errors(ae_model, novel_loader, device=device, model_type='ae')
            errors_fam_vae = get_reconstruction_errors(vae_model, fam_loader, device=device, model_type='vae')
            errors_nov_vae = get_reconstruction_errors(vae_model, novel_loader, device=device, model_type='vae')
            
            # Compare errors and compute accuracy
            comparisons_ae = compare_novelty(errors_nov_ae, errors_fam_ae)
            accuracy_ae = comparisons_ae.mean()
            acc_ae_seeds.append(accuracy_ae)
            print(f"  [AE] Accuracy: {accuracy_ae:.4f}")
            
            comparisons_vae = compare_novelty(errors_nov_vae, errors_fam_vae)
            accuracy_vae = comparisons_vae.mean()
            acc_vae_seeds.append(accuracy_vae)
            print(f"  [VAE] Accuracy: {accuracy_vae:.4f}")
        
        accuracies_ae.append((np.mean(acc_ae_seeds), np.std(acc_ae_seeds)))
        accuracies_vae.append((np.mean(acc_vae_seeds), np.std(acc_vae_seeds)))
    
    # Plot the accuracies with error bars
    ae_means, ae_stds = zip(*accuracies_ae)
    vae_means, vae_stds = zip(*accuracies_vae)
    print(f"Autoencoder: {ae_means}, {ae_stds}")
    print(f"Variational Autoencoder: {vae_means}, {vae_stds}")



    plt.figure(figsize=(10, 5))
    plt.errorbar(N_values, ae_means, yerr=ae_stds, label='Autoencoder', fmt='-o')
    plt.errorbar(N_values, vae_means, yerr=vae_stds, label='Variational Autoencoder', fmt='-o')
    plt.xlabel('Number of Samples (N)')
    plt.ylabel('Accuracy')
    plt.title('Novelty Detection Accuracy vs Number of Samples')
    plt.legend()
    plt.grid(True)
    plt.show()
    # plt.savefig(os.path.join(save_dir, 'autoencoders_accuracy_vs_N.pdf'))

    ae_model = Autoencoder(image_size=image_size, latent_dim=latent_dim)
    vae_model = VariationalAutoencoder(image_size=image_size, latent_dim=latent_dim)

    # Count parameters
    ae_params = count_parameters(ae_model)
    vae_params = count_parameters(vae_model)

    print(f"Number of trainable parameters in Autoencoder: {ae_params}")
    print(f"Number of trainable parameters in Variational Autoencoder: {vae_params}")

    if dataset == "gaussian":
        np.savez(
            os.path.join(save_dir, f'means_{dataset}_b_{str(b).replace(".", "")}.npz'), 
            ae_means=ae_means,
            vae_means=vae_means,
        )
        np.savez(
            os.path.join(save_dir, f'stds_{dataset}_b_{str(b).replace(".", "")}.npz'),
            ae_stds=ae_stds,
            vae_stds=vae_stds, 
        )
    else:
        np.savez(
            os.path.join(save_dir, f'means_{dataset}.npz'), 
            ae_means=ae_means,
            vae_means=vae_means,
        )
        np.savez(
            os.path.join(save_dir, f'stds_{dataset}.npz'),
            ae_stds=ae_stds,
            vae_stds=vae_stds, 
        )


N_values = [20, 40, 100, 200, 400, 1000, 4000, 10000]
# N_values = [4000]

num_seeds = 5

# dataset = 'gaussian' # 'gaussian' or 'tinyimagenet'
dataset = 'tinyimagenet'
size_dict = {"gaussian": 500, "mnist": 784, "tinyimagenet": 4096}
image_size = size_dict[dataset]
d_latent = image_size // 2  # latent dimension
b = 0 # set to 0 for uncorrelated gaussian data, 0.4 for correlated gaussian data
lmbda = 1e-5  # lambda for VAE loss

# Step size and gamma for learning rate scheduler
step_size = 25
gamma = 0.9

# Run the experiment for different N values
run_experiment(N_values=N_values, num_epochs=400, seeds=num_seeds)
