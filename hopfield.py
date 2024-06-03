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

save_dir = os.path.join("results", "hopfield")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

parser = argparse.ArgumentParser(description="HN for novelty detection")
parser.add_argument("--dataset", type=str, default="gaussian", help="Dataset to use")
parser.add_argument("--num_seeds", type=int, default=5, help="Number of seeds")
parser.add_argument(
    "--b", type=float, default=0, help="Covariance between any two distinct pixels"
)
args = parser.parse_args()

sample_sizes = [20, 40, 100, 200, 400, 1000, 4000, 10000]
# sample_sizes = [40]
size_dict = {"gaussian": 500, "mnist": 784, "tinyimagenet": 4096}
dimension = size_dict[args.dataset]
prob = 0.5 + args.b / 2

HN_error_probs = np.zeros((len(sample_sizes), args.num_seeds))
MCHN_error_probs = np.zeros((len(sample_sizes), args.num_seeds))
for k, sample_size in enumerate(sample_sizes):
    batch_size = sample_size // 10
    PCN_error_probs = np.zeros((len(sample_sizes), args.num_seeds))
    print(f"Sample size: {sample_size}")
    for seed in tqdm(range(args.num_seeds)):
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

        # evaluate HN energy
        energy_HN_fam = (
            hopfield_energy(X.to(device), X.to(device)).detach().cpu().numpy()
        )
        energy_HN_nov = (
            hopfield_energy(X_test.to(device), X.to(device)).detach().cpu().numpy()
        )
        HN_error_probs[k, seed] = 1 - np.mean(
            compare_novelty(energy_HN_nov, energy_HN_fam)
        )

        # evaluate MCHN energy
        energy_MCHN_fam = (
            dense_associative_energy(X.to(device), X.to(device)).detach().cpu().numpy()
        )
        energy_MCHN_nov = (
            dense_associative_energy(X_test.to(device), X.to(device))
            .detach()
            .cpu()
            .numpy()
        )
        MCHN_error_probs[k, seed] = 1 - np.mean(
            compare_novelty(energy_MCHN_nov, energy_MCHN_fam)
        )

# save results
np.save(
    os.path.join(
        save_dir, f'HN_error_probs_{args.dataset}_b_{str(args.b).replace(".", "")}.npy'
    ),
    HN_error_probs,
)

np.save(
    os.path.join(
        save_dir,
        f'MCHN_error_probs_{args.dataset}_b_{str(args.b).replace(".", "")}.npy',
    ),
    MCHN_error_probs,
)
