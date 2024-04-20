import torch
from torchvision import datasets, transforms
from torchvision.transforms import transforms
from torch.distributions.multivariate_normal import MultivariateNormal
import random
import numpy as np
import os
import urllib.request
import zipfile
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class TinyImageNet(Dataset):
    def __init__(self, root_dir='../data', is_train=True, transform=None, download=False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train

        # Check if dataset exists, if not, download it
        if download and not os.path.exists(os.path.join(self.root_dir, 'tiny-imagenet-200')):
            self.download()

        self.train_dir = os.path.join(root_dir, 'tiny-imagenet-200', 'train')
        # Create a mapping from string class IDs to integer labels
        self.class_to_label = {class_name: class_id for class_id, class_name in enumerate(os.listdir(self.train_dir))}

        if self.is_train:
            self.data_dir = os.path.join(root_dir, 'tiny-imagenet-200', 'train')
            self.image_paths = []
            self.labels = []
    
            for class_id, class_name in enumerate(os.listdir(self.data_dir)):
                class_dir = os.path.join(self.data_dir, class_name, 'images')
                for image_name in os.listdir(class_dir):
                    self.image_paths.append(os.path.join(class_dir, image_name))
                    self.labels.append(class_id)
        else:
            self.data_dir = os.path.join(root_dir, 'tiny-imagenet-200', 'val', 'images')
            with open(os.path.join(root_dir, 'tiny-imagenet-200', 'val', 'val_annotations.txt'), 'r') as f:
                lines = f.readlines()
                self.image_paths = [os.path.join(self.data_dir, line.split('\t')[0]) for line in lines]
                # Use the mapping to convert string class IDs to integer labels
                self.labels = [self.class_to_label[line.split('\t')[1]] for line in lines]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

    def download(self):
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        filename = os.path.join(self.root_dir, "tiny-imagenet-200.zip")

        # Download the dataset
        urllib.request.urlretrieve(url, filename)

        # Extract the dataset
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(self.root_dir)

        # Remove the downloaded zip file after extraction
        os.remove(filename)

# Wrap up the data loading process into a function
def get_tiny_imagenet(datapath, sample_size, sample_size_test, batch_size, seed, device):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), # convert to grayscale for simplicity
        transforms.ToTensor(),
    ])

    train = TinyImageNet(root_dir=datapath, is_train=True, transform=transform, download=True)
    test = TinyImageNet(root_dir=datapath, is_train=False, transform=transform, download=True)

    # randomly sample a subset of the dataset
    random.seed(seed)
    # print(len(train), len(test))
    train = torch.utils.data.Subset(train, random.sample(range(len(train)), sample_size))
    test = torch.utils.data.Subset(test, random.sample(range(len(test)), sample_size_test))
    
    # create dataloaders
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    X, y = [], []
    for batch_idx, (data, targ) in enumerate(train_loader):
        X.append(data)
        y.append(targ)
    X = torch.cat(X, dim=0).to(device) # size, c, 64, 64
    y = torch.cat(y, dim=0).to(device)

    X_test, y_test = [], []
    for batch_idx, (data, targ) in enumerate(test_loader):
        X_test.append(data)
        y_test.append(targ)
    X_test = torch.cat(X_test, dim=0).to(device) # size, c, 64, 64
    y_test = torch.cat(y_test, dim=0).to(device)

    return (X, y), (X_test, y_test)

def cover_bottom(X, divisor, device):
    size = X.shape # 10, 3, 32, 32
    mask = torch.ones_like(X).to(device)
    mask[:, :, size[-2]//divisor+1:, :] -= 1
    update_mask = torch.zeros_like(X).to(device)
    update_mask[:, :, size[-2]//divisor+1:, :] += 1
    update_mask = update_mask.reshape(-1, size[-1]*size[-2]*size[-3])
    X_c = (X * mask).to(device)
    X_c = X_c.reshape(-1, size[-1]*size[-2]*size[-3])

    return X_c, update_mask

def cover_center(X, cover_size, device):
    size = X.shape
    mask = torch.ones_like(X).to(device)
    mask[:, (size[1]-cover_size)//2:(size[1]+cover_size)//2, (size[2]-cover_size)//2:(size[2]+cover_size)//2] -= 1
    update_mask = torch.zeros_like(X).to(device)
    update_mask[:, (size[1]-cover_size)//2:(size[1]+cover_size)//2, (size[2]-cover_size)//2:(size[2]+cover_size)//2] += 1
    update_mask = update_mask.reshape(-1, size[1]*size[2])
    X_c = (X * mask).to(device)
    X_c = X_c.reshape(-1, size[1]*size[2])

    return X_c, update_mask

def add_gaussian_noise(X, var, device):
    size = X.shape
    mask = (torch.randn(size) * np.sqrt(var)).to(device)
    update_mask = torch.ones_like(X).to(device)
    update_mask = update_mask.reshape(-1, size[-1]*size[-2]*size[-3])
    X_c = (X + mask).to(device)
    X_c = X_c.reshape(-1, size[-1]*size[-2]*size[-3])

    return X_c, update_mask


def get_2d_gaussian(sample_size, device, seed=10):
    dim = 2
    # simulation data
    mean = np.array([0,0])
    cov = np.array([[2,1],
                    [1,2]])
    np.random.seed(seed)
    X = np.random.multivariate_normal(mean, cov, size=sample_size)
    X_c = X * np.concatenate([np.ones((sample_size,1))]+[np.zeros((sample_size,1))]*(dim-1), axis=1)
    update_mask = np.concatenate([np.zeros((sample_size,1))]+[np.ones((sample_size,1))]*(dim-1), axis=1)

    X = torch.tensor(X).float().to(device)
    X_c = torch.tensor(X_c).float().to(device)
    update_mask = torch.tensor(update_mask).float().to(device)

    return X, X_c, update_mask


def create_gaussian(savepath, seed=10):
    # create random covariance matrix
    torch.manual_seed(seed)
    G = torch.randn(25, 25)
    S = torch.matmul(G, G.t()) / 10.
    L = torch.linalg.cholesky(S)

    # sample from this normal
    m = MultivariateNormal(torch.zeros(25), scale_tril=L)
    X = m.sample(sample_shape=(5000,))
    X = X.reshape(-1, 5, 5)
    torch.save(X, savepath+'/gaussian_data.pt')


def get_gaussian(datapath, sample_size, batch_size, seed, device):
    train = torch.load(datapath+'/gaussian_data.pt')
    if sample_size != len(train):
        random.seed(seed)
        train = train[random.sample(range(len(train)), sample_size)] # size, 5, 5
    
    X = train.clone().detach().to(device)

    return X


# def get_mnist(datapath, sample_size, sample_size_test, batch_size, seed, device, binary=False, classes=None):
#     # classes: a list of specific class to sample from
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#     ])
#     train = datasets.MNIST(datapath, train=True, transform=transform, download=True)
#     test = datasets.MNIST(datapath, train=False, transform=transform, download=True)

#     # subsetting data based on sample size and number of classes
#     idx = sum(train.targets == c for c in classes).bool() if classes else range(len(train))
#     train.targets = train.targets[idx]
#     train.data = train.data[idx]
#     if sample_size != len(train):
#         random.seed(seed)
#         train = torch.utils.data.Subset(train, random.sample(range(len(train)), sample_size))
#     random.seed(seed)
#     test = torch.utils.data.Subset(test, random.sample(range(len(test)), sample_size_test))

#     train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
#     test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

#     X, y = [], []
#     for batch_idx, (data, targ) in enumerate(train_loader):
#         X.append(data)
#         y.append(targ)
#     X = torch.cat(X, dim=0).to(device) # size, 28, 28
#     y = torch.cat(y, dim=0).to(device)

#     X_test, y_test = [], []
#     for batch_idx, (data, targ) in enumerate(test_loader):
#         X_test.append(data)
#         y_test.append(targ)
#     X_test = torch.cat(X_test, dim=0).to(device) # size, 28, 28
#     y_test = torch.cat(y_test, dim=0).to(device)

#     if binary:
#         X[X > 0.5] = 1
#         X[X < 0.5] = 0
#         X_test[X_test > 0.5] = 1
#         X_test[X_test < 0.5] = 0

#     print(X.shape)
#     return (X, y), (X_test, y_test)

def get_mnist(datapath, sample_size, sample_size_test, batch_size, seed, device, binary=False, classes=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train = datasets.MNIST(datapath, train=True, transform=transform, download=True)
    test = datasets.MNIST(datapath, train=False, transform=transform, download=True)

    # If specific classes are given, filter the data
    if classes is not None:
        idx_train = [i for i, label in enumerate(train.targets) if label in classes]
        idx_test = [i for i, label in enumerate(test.targets) if label in classes]
        
        train.targets = train.targets[idx_train]
        train.data = train.data[idx_train]

        test.targets = test.targets[idx_test]
        test.data = test.data[idx_test]

    # If sampling is needed
    if sample_size != len(train):
        random.seed(seed)
        train = torch.utils.data.Subset(train, random.sample(range(len(train)), sample_size))
    
    random.seed(seed)
    test = torch.utils.data.Subset(test, random.sample(range(len(test)), sample_size_test))

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    X, y = [], []
    for batch_idx, (data, targ) in enumerate(train_loader):
        X.append(data)
        y.append(targ)
    X = torch.cat(X, dim=0).to(device)
    y = torch.cat(y, dim=0).to(device)

    X_test, y_test = [], []
    for batch_idx, (data, targ) in enumerate(test_loader):
        X_test.append(data)
        y_test.append(targ)
    X_test = torch.cat(X_test, dim=0).to(device)
    y_test = torch.cat(y_test, dim=0).to(device)

    if binary:
        X[X > 0.5] = 1
        X[X < 0.5] = 0
        X_test[X_test > 0.5] = 1
        X_test[X_test < 0.5] = 0

    print(X.shape)
    return (X, y), (X_test, y_test)


def get_cifar10(datapath, sample_size, sample_size_test, batch_size, seed, device, binary=False, classes=None):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    train = datasets.CIFAR10(datapath, train=True, transform=transform, download=True)
    test = datasets.CIFAR10(datapath, train=False, transform=transform, download=True)
    
     # subsetting data based on sample size and number of classes
    idx = sum(torch.tensor(train.targets) == c for c in classes).bool() if classes else range(len(train))
    train.targets = torch.tensor(train.targets)[idx]
    train.data = train.data[idx]
    if sample_size != len(train):
        random.seed(seed)
        train = torch.utils.data.Subset(train, random.sample(range(len(train)), sample_size))
    random.seed(seed)
    test = torch.utils.data.Subset(test, random.sample(range(len(test)), sample_size_test))

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    X, y = [], []
    for batch_idx, (data, targ) in enumerate(train_loader):
        X.append(data)
        y.append(targ)
    X = torch.cat(X, dim=0).to(device) # size, c, 32, 32
    y = torch.cat(y, dim=0).to(device)

    X_test, y_test = [], []
    for batch_idx, (data, targ) in enumerate(test_loader):
        X_test.append(data)
        y_test.append(targ)
    X_test = torch.cat(X_test, dim=0).to(device) # size, 32, 32
    y_test = torch.cat(y_test, dim=0).to(device)

    if binary:
        X[X > 0.5] = 1
        X[X < 0.5] = 0
        X_test[X_test > 0.5] = 1
        X_test[X_test < 0.5] = 0

    return (X, y), (X_test, y_test)


def get_fashionMNIST(datapath, sample_size, batch_size, seed, device, classes=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train = datasets.FashionMNIST(datapath, train=True, transform=transform, download=True)
    test = datasets.FashionMNIST(datapath, train=False, transform=transform, download=True)
    
    # subsetting data based on sample size and number of classes
    idx = sum(train.targets == c for c in classes).bool() if classes else range(len(train))
    train.targets = train.targets[idx]
    train.data = train.data[idx]
    if sample_size != len(train):
        random.seed(seed)
        train = torch.utils.data.Subset(train, random.sample(range(len(train)), sample_size))

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    X = []
    for batch_idx, (data, targ) in enumerate(train_loader):
        X.append(data)
    X = torch.cat(X, dim=0).squeeze().to(device) # size, 28, 28

    return X