
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
    

      


def rbf_kernel_single(x, jitter=1e-8):
    """Computes the RBF (Squared Exponential) kernel covariance matrix."""
    num_points = x.shape[0]
    length_scale = 0.4  # Adjust for different smoothness
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
    N, num_extra = 15, 500
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

