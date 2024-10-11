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

device = "cuda" if torch.cuda.is_available() else "cpu"
# plt.style.use('seaborn')

save_dir = os.path.join("results", "vanilla_h_models")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# hyperparameters
learning_iters = 100
learning_lr = 3e-4
inference_iters = 50
inference_lr = 0.05
noise_var = 0.05
divisor = 2
image_size = 64 * 64
sample_size = 1
sample_size_test = 1
batch_size = 1
seeds = range(5)
n_layer = 2
model_path = "./models/"
result_path = "./results/"
dendritic = False
save_every = learning_iters // 2
latent_size = 256

# training
energies = np.zeros((len(seeds), learning_iters))
for seed in seeds:
    error_neurons = np.zeros(
        (learning_iters // save_every + 1, latent_size * n_layer + image_size)
    )

    print(f"sample size {sample_size}, seed {seed}, layers {n_layer}")

    (X, _), (_, _) = get_tiny_imagenet(
        "./data",
        sample_size=sample_size,
        sample_size_test=sample_size,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )
    X = X.reshape((X.shape[0], -1)).float()

    nodes = [latent_size] * n_layer + [image_size]
    pcn_h = HierarchicalPCN(nodes, "Tanh", inference_lr).to(device)

    optimizer = torch.optim.Adam(pcn_h.parameters(), lr=learning_lr)

    ########################################################################
    # train model
    train_mses = []
    for i in range(learning_iters):
        with tqdm(total=sample_size) as progress_bar:
            for batch_idx in range(0, sample_size, batch_size):
                data = X[batch_idx : batch_idx + batch_size].to(device)
                optimizer.zero_grad()
                pcn_h.train_pc_generative(
                    data, inference_iters, torch.ones_like(data).to(device)
                )
                optimizer.step()
                progress_bar.update(batch_size)
        train_mse = pcn_h.energy().cpu().detach().numpy()
        print("Epoch", i, "Energy", train_mse)
        train_mses.append(train_mse)

        if i == 0 or (i + 1) % save_every == 0:
            # pcn_h.test_pc_generative(data, 500, torch.ones_like(data).to(device))
            error_neurons[(i + 1) // save_every] = (
                pcn_h.error_neurons().detach().cpu().numpy()
            )

    # plot
    plt.figure()
    plt.plot(train_mses)
    plt.savefig(os.path.join(save_dir, f"train_energy.png"))

    energies[seed] = train_mses

# save results
np.save(os.path.join(save_dir, f"energies_evolution.npy"), energies)
np.save(os.path.join(save_dir, f"error_neurons.npy"), error_neurons)