import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import time

# return a 1D tensor of size (input_size*input_size) for the neuron in the output layer at position (x,y)
def return_mask(input_size, x, y, kernel_size, stride=1, padding=0):
    # Calculate the region where the kernel would be applied
    start_x = x * stride
    start_y = y * stride
    end_x = start_x + kernel_size
    end_y = start_y + kernel_size
    padded_size = input_size + 2 * padding
    if end_x > padded_size or end_y > padded_size:
        raise ValueError("Kernel end position exceeds input size")
    
    mask = torch.zeros(input_size, input_size)
    mask[start_x:end_x, start_y:end_y] = 1
    flattened_mask = mask.view(-1)
    return flattened_mask

# return a 2D tensor of size (output_size_x*output_size_y, input_size*input_size)
# the first dimension is the position of the neuron in the output layer, the second dimension is the flattened mask (of size input_size*input_size)
def create_all_masks(input_size, kernel_size, stride=1, padding=0):
    # Calculate the output dimensions
    print("parameters for create_all_masks are:",input_size, kernel_size, stride, padding)
    output_size_x = (input_size - kernel_size + 2 * padding) // stride + 1
    
    output_size_y = (input_size - kernel_size + 2 * padding) // stride + 1
    print("output sizes are:", output_size_x, output_size_y)
    flat_output_size = output_size_x * output_size_y
    # Initialize the 3D tensor for all masks
    all_masks = torch.zeros(output_size_x, output_size_y, input_size*input_size)
    # Generate masks for each position and store them
    for x in range(output_size_x):
        for y in range(output_size_y):
            try:
                mask = return_mask(input_size, x, y, kernel_size, stride, padding)
                all_masks[x, y] = mask
            except ValueError as e:
                print(f"Error at position ({x}, {y}):", e)
    print("at this layer, size of all masks before flatten is:", all_masks.shape)
    flattened_all = all_masks.view(-1, all_masks.size(2)).T
    print("at this layer, size of all masks after flatten and transpose is:", flattened_all.shape)
    return flattened_all

class Tanh(nn.Module):
    def forward(self, inp):
        return torch.tanh(inp)

    def deriv(self, inp):
        return 1.0 - torch.tanh(inp) ** 2.0

    # run the following if this class inherits object.
    # def __call__(self, inp):
    #     return self.forward(inp)

class ReLU(nn.Module):
    def forward(self, inp):
        return torch.relu(inp)

    def deriv(self, inp):
        out = self(inp)
        out[out > 0] = 1.0
        return out

class Sigmoid(nn.Module):
    def forward(self, inp):
        return torch.sigmoid(inp)

    def deriv(self, inp):
        out = self(inp)
        return out * (1 - out)

class Binary(nn.Module):
    def forward(self, inp, threshold=0.):
        return torch.where(inp > threshold, 1., 0.)

    def deriv(self, inp):
        return torch.zeros((1,))

class Linear(nn.Module):
    def forward(self, inp):
        return inp

    def deriv(self, inp):
        return torch.ones((1,)).to(inp.device)


def cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
    """Estimates covariance matrix like numpy.cov"""
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w/w_sum)[:,None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact

    return c.squeeze()