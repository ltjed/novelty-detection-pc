import numpy as np
import torch
import matplotlib.pyplot as plt

def binarize(tensor):
    device = tensor.device  # Get the device of the input tensor
    return torch.where(tensor >= 0.5, torch.tensor([1], device=device), torch.tensor([-1], device=device))

# pattern is 1*N vector
def hopfield_energy(pattern, X):
    energy = 0
    W = torch.matmul(X.t(), X)
    # W.fill_diagonal_(0)
    energy = - torch.matmul(torch.matmul(pattern, W), pattern.t())
    return energy

# an implementation of MCHN's energy function
def dense_associative_energy(pattern, X, a=3):
    energies = torch.matmul(X, pattern.t())
    energy = - torch.logsumexp(energies, 0) + torch.matmul(pattern, pattern.t())*0.5
    return energy

# The following method is used to replicate Standing (1973)'s experiment results (i.e., the power-law relationship).
# It takes in two vectors of energies (presumably one is 'in truth' 'familiar' while the other is randomly generated--i.e., 'novel')
# Then output a binary vector of the same dimension as either argument with 1 signalling that, to the model (used to compute the energy),
# the pattern in the first vector is identified as more 'novel'
# WARNING: The argument order should be reversed for HN/MCHN due to assumptions in those model for ND!
def compare_novelty(energy_nov, energy_fam):
    # Compare the judgments for each of the two
    energy_comparison = energy_nov - energy_fam
    # If an entry is positive, then the subject has correctly identified the novel picture from that pair
    return np.where(energy_comparison >= 0, 1, 0)

# calculates the d' between two distributions
def separability(signal, noise):
    # Calculate means
    mu_signal = np.mean(signal)
    mu_noise = np.mean(noise)
    
    # Calculate standard deviations
    s_signal = np.std(signal, ddof=1)
    s_noise = np.std(noise, ddof=1)
    
    # Calculate pooled standard deviation
    n_signal = len(signal)
    n_noise = len(noise)
    
    pooled_std = np.sqrt(((n_signal - 1) * s_signal**2 + (n_noise - 1) * s_noise**2) / (n_signal + n_noise - 2))
    
    # Calculate d'
    d_prime = (mu_signal - mu_noise) / pooled_std
    
    return d_prime