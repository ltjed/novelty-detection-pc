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
parser.add_argument("--dataset", type=str, default="tinyimagenet", help="Dataset to use")
parser.add_argument("--learning_lr", type=float, default=3e-4, help="Learning rate")
parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
parser.add_argument("--num_seeds", type=int, default=5, help="Number of seeds")
parser.add_argument(
    "--b", type=float, default=0, help="Covariance between any two distinct pixels"
)
args = parser.parse_args()

# sample_sizes = [4000, 10000]
# sample_sizes = [1]
sample_sizes = [20, 40, 100, 200, 400, 1000, 4000, 10000]

size_dict = {"gaussian": 500, "mnist": 784, "tinyimagenet": 4096}
dimension = size_dict[args.dataset]
prob = 0.5 + args.b / 2
save_every = args.epochs // 2

PCN_error_probs = np.zeros((len(sample_sizes), args.num_seeds))
for k, sample_size in enumerate(sample_sizes):
    batch_size = sample_size
    energies = np.zeros((args.num_seeds, args.epochs))
    for seed in range(args.num_seeds):
        print(f"Sample size: {sample_size}, Seed: {seed}")
        err_neurons = np.zeros((args.epochs // save_every + 1, size_dict[args.dataset]))

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

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Number of parameters: {count_parameters(pcn)}")

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

# save results
np.save(
    os.path.join(
        save_dir,
        f'PCN_error_probs_{args.dataset}_b_{str(args.b).replace(".", "")}.npy',
    ),
    PCN_error_probs,
)
np.save(os.path.join(save_dir, f"energies_evolution.npy"), energies)
np.save(os.path.join(save_dir, f"error_neurons.npy"), err_neurons)
