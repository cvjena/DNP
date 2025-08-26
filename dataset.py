import glob
import numpy as np
import torch
from math import pi
from PIL import Image
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets, transforms
import sklearn
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from helpers import DIR_DATA, NotLoadedError, load_chunk, save_chunk
import logging
import random
#import GPy

logging.basicConfig(level=logging.INFO)
    

    

class SineData(Dataset):
    """
    Dataset of functions f(x) = a * sin(x - b) where a and b are randomly
    sampled. The function is evaluated from -pi to pi.

    Parameters
    ----------
    amplitude_range : tuple of float
        Defines the range from which the amplitude (i.e. a) of the sine function
        is sampled.

    shift_range : tuple of float
        Defines the range from which the shift (i.e. b) of the sine function is
        sampled.

    num_samples : int
        Number of samples of the function contained in dataset.

    num_points : int
        Number of points at which to evaluate f(x) for x in [-pi, pi].
    """
    def __init__(self, amplitude_range=(-1., 1.), shift_range=(-.5, .5),
                 num_samples=1000, num_points=100):
        self.amplitude_range = amplitude_range
        self.shift_range = shift_range
        self.num_samples = num_samples
        self.num_points = num_points
        self.x_dim = 1  # x and y dim are fixed for this dataset.
        self.y_dim = 1

        # Generate data
        self.data = []

        all_X = []
        all_y = []

        a_min, a_max = amplitude_range
        b_min, b_max = shift_range
        for i in range(num_samples):
            # Sample random amplitude
            a = (a_max - a_min) * np.random.rand() + a_min
            # Sample random shift
            b = (b_max - b_min) * np.random.rand() + b_min
            # Shape (num_points, x_dim)
            x = torch.linspace(-pi, pi, num_points).unsqueeze(1)
            # Shape (num_points, y_dim)
            y = a * torch.sin(x - b)
            self.data.append((x, y))
            all_X.append(x)
            all_y.append(y)

        # Stack all data to compute mean and std
        all_X = torch.cat(all_X, dim=0)
        all_y = torch.cat(all_y, dim=0)
        


    def __getitem__(self, index):
        return self.data[index]

    def normalize(self, data, mean, std):
        return (data - mean) / std

    def __len__(self):
        return self.num_samples
    



def mnist(batch_size=16, size=28, path_to_data='../../mnist_data'):
    """MNIST dataloader.

    Parameters
    ----------
    batch_size : int

    size : int
        Size (height and width) of each image. Default is 28 for no resizing.

    path_to_data : string
        Path to MNIST data files.
    """
    all_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    train_data = datasets.MNIST(path_to_data, train=True, download=True,
                                transform=all_transforms)
    test_data = datasets.MNIST(path_to_data, train=False,
                               transform=all_transforms)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_data, train_loader, test_data, test_loader

def fmnist(batch_size=16, size=28, path_to_data='../../fmnist_data'):
    """Fashion MNIST dataloader.

    Parameters
    ----------
    batch_size : int

    size : int
        Size (height and width) of each image. Default is 28 for no resizing.

    path_to_data : string
        Path to MNIST data files.
    """
    all_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    train_data = datasets.FashionMNIST(path_to_data, train=True, download=True,
                                transform=all_transforms)
    test_data = datasets.FashionMNIST(path_to_data, train=False,
                               transform=all_transforms)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_data, train_loader, test_data, test_loader

def kmnist(batch_size=16, size=28, path_to_data='../../kmnist_data'):
    """Fashion MNIST dataloader.

    Parameters
    ----------
    batch_size : int

    size : int
        Size (height and width) of each image. Default is 28 for no resizing.

    path_to_data : string
        Path to MNIST data files.
    """
    all_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    train_data = datasets.KMNIST(path_to_data, train=True, download=True,
                                transform=all_transforms)
    test_data = datasets.KMNIST(path_to_data, train=False,
                               transform=all_transforms)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_data, train_loader, test_data, test_loader


def cifar10(batch_size=16, size=32, path_to_data='../../cifar10'):
    """Fashion MNIST dataloader.

    Parameters
    ----------
    batch_size : int

    size : int
        Size (height and width) of each image. Default is 28 for no resizing.

    path_to_data : string
        Path to MNIST data files.
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    trainset = datasets.CIFAR10(root=path_to_data, train=True, download=True, transform=transform_train)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    testset = datasets.CIFAR10(root=path_to_data, train=False, download=True, transform=transform_test)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    return trainset, train_loader, testset, test_loader

def cifar100(batch_size=16, size=32, path_to_data='../../cifar100'):
    """Fashion MNIST dataloader.

    Parameters
    ----------
    batch_size : int

    size : int
        Size (height and width) of each image. Default is 28 for no resizing.

    path_to_data : string
        Path to MNIST data files.
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    trainset = datasets.CIFAR100(root=path_to_data, train=True, download=True, transform=transform_train)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    testset = datasets.CIFAR100(root=path_to_data, train=False, download=True, transform=transform_test)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    return trainset, train_loader, testset, test_loader


def svhn(batch_size=16, size=32, path_to_data='../../svhn'):
    """Fashion MNIST dataloader.

    Parameters
    ----------
    batch_size : int

    size : int
        Size (height and width) of each image. Default is 28 for no resizing.

    path_to_data : string
        Path to MNIST data files.
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    trainset = datasets.SVHN(root=path_to_data, split='train', download=True, transform=transform_train)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    testset = datasets.SVHN(root=path_to_data, split='test', download=True, transform=transform_test)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    return trainset, train_loader, testset, test_loader

def lsun(batch_size=16, size=32, path_to_data='../../lsun'):
    """Fashion MNIST dataloader.

    Parameters
    ----------
    batch_size : int

    size : int
        Size (height and width) of each image. Default is 28 for no resizing.

    path_to_data : string
        Path to MNIST data files.
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    trainset = datasets.LSUN(root=path_to_data, classes='train', transform=transform_train)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    testset = datasets.LSUN(root=path_to_data, classes='test', transform=transform_test)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    return trainset, train_loader, testset, test_loader

def tiny(batch_size=16, size=32, path_to_data='./data/tiny-imagenet-200/test/images/'):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    tiny_data = TinyImageNetDataset(path_to_data,
                                transform=transform_test)
    tiny_loader = DataLoader(tiny_data, batch_size=batch_size,
                               shuffle=True)
    return tiny_data, tiny_loader


def celeba(batch_size=16, size=32, crop=89, path_to_data='./celeba/img_align_celeba/img_align_celeba',
           shuffle=True):
    """CelebA dataloader.

    Parameters
    ----------
    batch_size : int

    size : int
        Size (height and width) of each image.

    crop : int
        Size of center crop. This crop happens *before* the resizing.

    path_to_data : string
        Path to CelebA data files.
    """
    transform = transforms.Compose([
        transforms.CenterCrop(crop),
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    celeba_data = CelebADataset(path_to_data,
                                transform=transform)
    celeba_loader = DataLoader(celeba_data, batch_size=batch_size,
                               shuffle=shuffle)
    return celeba_loader, celeba_data


class TinyImageNetDataset(Dataset):
    def __init__(self, path_to_data, subsample=1, transform=None):
        """
        Parameters
        ----------
        path_to_data : string
            Path to TinyImageNet data files.

        subsample : int
            Only load every |subsample| number of images.

        transform : torchvision.transforms
            Torchvision transforms to be applied to each image.
        """
        self.img_paths = glob.glob(path_to_data + '/*.JPEG')[::subsample]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        sample_path = self.img_paths[idx]
        sample = Image.open(sample_path).convert("RGB")

        if self.transform:
            sample = self.transform(sample)
        # Since there are no labels, we just return 0 for the "label" here
        return sample, 0 
      


def rbf_kernel_single(x, jitter=1e-8):
    """Computes the RBF (Squared Exponential) kernel covariance matrix."""
    num_points = x.shape[0]
    length_scale = 0.3  # Adjust for different smoothness
    output_scale = 1.0   # Overall variance scaling
    
    # Compute pairwise squared Euclidean distances
    x1 = x[:, np.newaxis, :]
    x2 = x[np.newaxis, :, :]
    sq_distance = np.sum((x1 - x2) ** 2, axis=-1)

    # Compute the RBF kernel
    covariance = output_scale**2 * np.exp(-sq_distance / (2 * length_scale**2))
    
    # Add jitter for numerical stability
    covariance += jitter * np.eye(num_points)
    
    return covariance


def toy_regression_dataset():
    N, num_extra = 20, 500
    np.random.seed(1)
    
    # Define training and test inputs
    x = np.random.uniform(low=-2, high=2, size=(N, 1))
    dx = np.linspace(-4, 4, num_extra)[:, np.newaxis]

    # Stack all points together for consistent function realization
    X_full = np.vstack((x, dx))
    
    # Compute kernel covariance matrix over combined set
    K_full = rbf_kernel_single(X_full)

    # Sample function values from the joint Gaussian process
    y_full = np.random.multivariate_normal(mean=np.zeros(len(X_full)), cov=K_full)[:, np.newaxis]

    # Extract corresponding function values
    y = y_full[:N]    # Function values for training points
    dy = y_full[N:]   # Function values for test points

    return x, y, dx, dy