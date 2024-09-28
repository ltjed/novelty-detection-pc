import os
import torch
from torchvision import datasets, transforms
from torchvision.transforms import transforms
import torch.optim as optim
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.distributions.multivariate_normal import MultivariateNormal
from src.get_data import *
from src.models import *
from src.utils import *
from src.nd import *
from datetime import datetime
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

save_dir = os.path.join("results", "recurrent_models")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

parser = argparse.ArgumentParser(description="rPCN for novelty detection")
parser.add_argument("--dataset", type=str, default="gaussian", help="Dataset to use")
parser.add_argument("--learning_lr", type=float, default=8e-4, help="Learning rate")
parser.add_argument("--epochs", type=int, default=400, help="Number of epochs")
parser.add_argument("--num_seeds", type=int, default=5, help="Number of seeds")
parser.add_argument(
    "--b", type=float, default=0.4, help="Covariance between any two distinct pixels"
)
args = parser.parse_args()


sample_sizes = np.logspace(4, 15, 45, base=2).astype(int)

dimensions = np.logspace(6, 10, 9, base=2).astype(int)

# records the capacity of the PCN for each dimension/network size 
# a value of -1 indicating that the PCN has not yet reached a capacity for largest value in sample_sizes
capacities = np.zeros(len(dimensions))
capacities.fill(-1)

prob = 0.5 + args.b / 2
save_every = args.epochs // 2

PCN_error_probs = np.zeros((len(sample_sizes), args.num_seeds))
for j, dimension in enumerate(dimensions):
    for k, sample_size in enumerate(sample_sizes):
        batch_size = sample_size
        energies = np.zeros((args.num_seeds, args.epochs))
        for seed in range(args.num_seeds):
            print(f"Sample size: {sample_size}, Seed: {seed}")
            err_neurons = np.zeros((args.epochs // save_every + 1, dimension))

            if args.dataset == "gaussian":
                cov = np.ones((dimension, dimension)) * args.b
                np.fill_diagonal(cov, 1)
                L = np.linalg.cholesky(cov)
                fam = np.random.randn(sample_size, dimension) @ L
                nov = np.random.randn(sample_size, dimension) @ L
                X = torch.from_numpy(fam).float()
                X_test = torch.from_numpy(nov).float()
            elif args.dataset == "tinyimagenet":
                (X, _), (X_test, _) = get_tiny_imagenet(
                    "./data",
                    sample_size=sample_size,
                    sample_size_test=sample_size,
                    batch_size=batch_size,
                    seed=seed,
                    device=device,
                )
                X = X.reshape((X.shape[0], -1)).float()
                X_test = X_test.reshape((X_test.shape[0], -1)).float()

            # train model
            pcn = RecPCN(dimension, dendrite=False, mode="linear").to(device)
            optimizer = optim.Adam(pcn.parameters(), lr=args.learning_lr)
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

            train_mses = []
            for i in range(args.epochs):
                running_loss = 0.0
                pbar = tqdm(range(0, sample_size, batch_size))
                for batch_idx in pbar:
                    data = X[batch_idx : batch_idx + batch_size].to(device)
                    optimizer.zero_grad()
                    pcn.learning(data)
                    optimizer.step()
                    loss = pcn.train_mse.item()
                    running_loss += loss
                    pbar.set_postfix({"loss": loss})

                train_mses.append(running_loss / (sample_size // batch_size))
                lr_scheduler.step()

                if i == 0 or (i + 1) % save_every == 0:
                    err_neurons[(i + 1) // save_every] = (
                        pcn.error_neurons(data).detach().cpu().numpy()
                    )

            plt.figure()
            plt.plot(train_mses)
            plt.savefig(
                os.path.join(
                    save_dir,
                    f'train_mses_{args.dataset}_b_{str(args.b).replace(".", "")}_{sample_size}.png',
                )
            )
            plt.close()
            energies[seed] = train_mses

            # after training, evaluate energy
            energy_fam = pcn.energy(X.to(device)).detach().cpu().numpy()
            energy_nov = pcn.energy(X_test.to(device)).detach().cpu().numpy()
            comparison = compare_novelty(energy_nov, energy_fam)
            PCN_error_probs[k, seed] = 1 - np.mean(comparison)
        # finish capacity search if the error probability exceeds 0.05
        if PCN_error_probs[k].mean() > 0.05:
            # if the capacity is lower than the smallest entry in sample_sizes, set it to 0; otherwise set it to the previous value
            if k == 0:
                capacities[j] = 0
            else:
                capacities[j] = sample_sizes[k-1]
            break
print(f"dimensions: {dimensions}")
print(f"capacities: {capacities}")
print(f"PCN_error_probs: {PCN_error_probs}")

# plot dimensions vs. capacities
plt.figure()
plt.plot(dimensions, capacities)
plt.xlabel("Dimension")
plt.ylabel("Capacity")
plt.xscale("log")
plt.yscale("log")

plt.savefig(
    os.path.join(
        save_dir,
        f'capacity_vs_dimension_{args.dataset}_b_{str(args.b).replace(".", "")}.png',
    )
)

# save results
# np.save(
#     os.path.join(
#         save_dir,
#         f'PCN_error_probs_{args.dataset}_b_{str(args.b).replace(".", "")}.npy',
#     ),
#     PCN_error_probs,
# )

# Save dimensions and capacities
np.save(
    os.path.join(
        save_dir,
        f'dimensions_{args.dataset}_b_{str(args.b).replace(".", "")}.npy',
    ),
    dimensions,
)
np.save(
    os.path.join(
        save_dir,
        f'capacities_{args.dataset}_b_{str(args.b).replace(".", "")}.npy',
    ),
    capacities,
)

# np.save(os.path.join(save_dir, f"energies_evolution.npy"), energies)
# np.save(os.path.join(save_dir, f"error_neurons.npy"), err_neurons)
