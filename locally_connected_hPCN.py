import os
import torch
from torchvision import datasets, transforms
from torchvision.transforms import transforms
import torch.optim as optim
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
from src.get_data import *
from src.models import *
from src.utils import *
from src.nd import *
from src.visualize import *
from datetime import datetime

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

save_dir = os.path.join('results', 'hierarchical_models')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

nonlin = 'Tanh' # 'Tanh' to replicate Fig.7
dataset = 'mnist'; dimension = 784 # set to 'mnist' to replicate Fig.7

base_class = [4]; test_class = [5] # set to [4] and [5] to replicate Fig.7, can be set to other classes, too
# Create a unique folder name with base_class and test_class
folder_name = f"base_{'_'.join(map(str, base_class))}_test_{'_'.join(map(str, test_class))}"
save_dir = os.path.join(save_dir, folder_name)
# Create the directory if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

n_layers = 2 # adjust the number of hidden layers; 2 for hPCN


hidden_sizes = [200, 400] # adjust the number of neurons in each hidden layer
kernel_sizes = [0, 9]
strides = [0, 1]
padding = 0
paddings = [padding] * n_layers

# in the paper we the sparsity constraint we imposed is hardcoded, i.e., w_sparsity_penalty = 0. 
# As an alternative implementation of sparsity, can set w_sparsity_penalty to a positive value to add a l1 penalty to the weights during optimization
w_sparsity_penalty = 0 

# vary the number of patterns shown to the model to memorize
sample_size = 100
batch_size = 50
seeds = np.arange(5) # number of seeds to average over
inference_lr = 0.1 # learning rate of inference;
inference_iters = 50 # number of inference iterations
learning_lr = 2e-4 # learning rate of PCN; set to 8e-5 to replicate Fig.7
lamb = 0
epochs = 2000 # number of learning iterations; set to 4000 to replicate Fig.7
decay_step_size = 10
decay_gamma = 1

# separability measures
separability_12 = np.zeros((len(seeds), n_layers+1)) # sensory (pixel) novelty
separability_23 = np.zeros((len(seeds), n_layers+1)) # semantic (digit for mnist) novelty

for seed in seeds:
    # X: train 4, X_test: test 4, Y_test: test 5
    (X, _), (X_test, _) = get_mnist(
        './data', 
        sample_size=sample_size, 
        sample_size_test=sample_size,
        batch_size=batch_size, 
        seed=seed, 
        device=device,
        binary=False,
        classes=base_class
    )
    (_, _), (Y_test, _) = get_mnist(
        './data', 
        sample_size=sample_size, 
        sample_size_test=sample_size,
        batch_size=batch_size, 
        seed=seed, 
        device=device,
        binary=False,
        classes=test_class
    )

    X = X.reshape((X.shape[0], -1)).float()
    X_test = X_test.reshape((X_test.shape[0], -1)).float()
    Y_test = Y_test.reshape((Y_test.shape[0], -1)).float()

    nodes = hidden_sizes + [784]
    pcn = ConvolutionalPCN(nodes, nonlin, inference_lr, kernel_sizes, strides, paddings, lamb).to(device)
    optimizer = torch.optim.Adam(pcn.parameters(), lr=learning_lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_step_size, gamma=decay_gamma)

    train_energy = []
    energy_fam = np.empty((sample_size, n_layers+1))
    energy_nov = np.empty((sample_size, n_layers+1))
    energy_test_nov = np.empty((sample_size, n_layers+1))

    update_mask = torch.ones_like(X).to(device)

    # train model
    for i in range(epochs):
        running_loss = 0.0
        pbar = tqdm(range(0, sample_size, batch_size), desc=f"Epoch {i+1}/{epochs}")
        for batch_idx in pbar:
            data = X[batch_idx:batch_idx+batch_size].to(device)
            optimizer.zero_grad()
            pcn.train_pc_generative(data, inference_iters, update_mask, w_sparsity_penalty)
            optimizer.step()
            loss = pcn.energy().item()
            running_loss += loss
            pbar.set_postfix({'loss': loss})

        train_energy.append(running_loss / (sample_size // batch_size))
        lr_scheduler.step()

    plt.figure()
    plt.plot(train_energy)
    plt.savefig(os.path.join(save_dir, f"train_energy.png"))
    plt.close()

    # evaluate energy/novelty
    test_inf_iters = 50
    pcn.Dt = 0.1
    test_energy = pcn.test_pc_generative(X.to(device), test_inf_iters, update_mask)
    energy_fam = pcn.layered_energy()
    pcn.test_pc_generative(X_test.to(device), test_inf_iters, update_mask)
    energy_nov = pcn.layered_energy()
    pcn.test_pc_generative(Y_test.to(device), test_inf_iters, update_mask)
    energy_test_nov = pcn.layered_energy()

    plt.figure()
    plt.plot(test_energy)
    plt.savefig(os.path.join(save_dir, f"test_energy.png"))
    plt.close()

    # d prime
    for l in range(n_layers+1):
        separability_12[seed, l] = separability(energy_nov[:, l], energy_fam[:, l])
        separability_23[seed, l] = separability(energy_test_nov[:, l], energy_fam[:, l])

# save energy and separability
np.savez(
    os.path.join(save_dir, f'energy.npz'), 
    energy_fam=energy_fam, 
    energy_nov=energy_nov, 
    energy_test_nov=energy_test_nov,
)
np.savez(
    os.path.join(save_dir, f'separability.npz'), 
    separability_12=separability_12, 
    separability_23=separability_23,
)


