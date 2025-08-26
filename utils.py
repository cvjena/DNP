import numpy as np
import torch
from torch.distributions.kl import kl_divergence
import model
from random import randint
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler


def context_target_split(x, y, num_context, num_extra_target):
    """Given inputs x and their value y, return random subsets of points for
    context and target. Note that following conventions from "Empirical
    Evaluation of Neural Process Objectives" the context points are chosen as a
    subset of the target points.

    Parameters
    ----------
    x : torch.Tensor
        Shape (batch_size, num_points, x_dim)

    y : torch.Tensor
        Shape (batch_size, num_points, y_dim)

    num_context : int
        Number of context points.

    num_extra_target : int
        Number of additional target points.
    """
    num_points = x.shape[1]
    # Sample locations of context and target points
    locations = np.random.choice(num_points,
                                 size=num_context + num_extra_target,
                                 replace=False)
    x_context = x[:, locations[:num_context], :]
    y_context = y[:, locations[:num_context], :]
    x_target = x[:, locations, :]
    y_target = y[:, locations, :]
    return x_context, y_context, x_target, y_target

def context_target_split_real(x, y, num_context, num_extra_target):
    """Given inputs x and their value y, return random subsets of points for
    context and target. Note that following conventions from "Empirical
    Evaluation of Neural Process Objectives" the context points are chosen as a
    subset of the target points.

    Parameters
    ----------
    x : torch.Tensor
        Shape (batch_size, num_points, x_dim)

    y : torch.Tensor
        Shape (batch_size, num_points, y_dim)

    num_context : int
        Number of context points.

    num_extra_target : int
        Number of additional target points.
    """
    num_points = x.shape[0]
    num_context = int(0.4 * num_points)
    num_extra_target = num_points - num_context
    # Sample locations of context and target points
    locations = np.random.choice(num_points,
                                 size=num_context + num_extra_target,
                                 replace=False)
    x_context = x[locations[:num_context], :, :]
    y_context = y[locations[:num_context], :, :]
    #x_target = x[:, locations, :]
    #y_target = y[:, locations, :]
    x_target = x
    y_target = y
    return x_context, y_context, x_target, y_target

def context_target_split_img(x, y, num_context, num_extra_target):
    """Given inputs x and their value y, return random subsets of points for
    context and target. Note that following conventions from "Empirical
    Evaluation of Neural Process Objectives" the context points are chosen as a
    subset of the target points.

    Parameters
    ----------
    x : torch.Tensor
        Shape (batch_size, num_points, x_dim)

    y : torch.Tensor
        Shape (batch_size, num_points, y_dim)

    num_context : int
        Number of context points.

    num_extra_target : int
        Number of additional target points.
    """
    num_points = x.shape[0]
    # Sample locations of context and target points
    locations = np.random.choice(num_points,
                                 size=num_context + num_extra_target,
                                 replace=False)
    x_context = x[locations[:num_context], :, :, :]
    y_context = y[locations[:num_context]]
    x_target = x[locations]
    y_target = y[locations]
    return x_context, y_context, x_target, y_target

def context_target_split_fnp(x, y, num_context, num_extra_target):
    """Given inputs x and their value y, return random subsets of points for
    context and target. Note that following conventions from "Empirical
    Evaluation of Neural Process Objectives" the context points are chosen as a
    subset of the target points.

    Parameters
    ----------
    x : torch.Tensor
        Shape (batch_size, num_points, x_dim)

    y : torch.Tensor
        Shape (batch_size, num_points, y_dim)

    num_context : int
        Number of context points.

    num_extra_target : int
        Number of additional target points.
    """
    num_points = x.shape[1]
    # Sample locations of context and target points
    locations = np.random.choice(num_points,
                                 size=num_context + num_extra_target,
                                 replace=False)
    x_context = x[:, locations[:num_context], :]
    y_context = y[:, locations[:num_context], :]
    x_target = x[:, locations[num_context:], :]
    y_target = y[:, locations[num_context:], :]

    #print(locations[:num_context], locations[num_context:])

    return x_context, y_context, x_target, y_target


def img_mask_to_np_input(img, mask, normalize=True):
    """
    Given an image and a mask, return x and y tensors expected by Neural
    Process. Specifically, x will contain indices of unmasked points, e.g.
    [[1, 0], [23, 14], [24, 19]] and y will contain the corresponding pixel
    intensities, e.g. [[0.2], [0.73], [0.12]] for grayscale or
    [[0.82, 0.71, 0.5], [0.42, 0.33, 0.81], [0.21, 0.23, 0.32]] for RGB.

    Parameters
    ----------
    img : torch.Tensor
        Shape (N, C, H, W). Pixel intensities should be in [0, 1]

    mask : torch.ByteTensor
        Binary matrix where 0 corresponds to masked pixel and 1 to a visible
        pixel. Shape (N, H, W). Note the number of unmasked pixels must be the
        SAME for every mask in batch.

    normalize : bool
        If true normalizes pixel locations x to [-1, 1] and pixel intensities to
        [-0.5, 0.5]
    """
    batch_size, num_channels, height, width = img.size()
    # Create a mask which matches exactly with image size which will be used to
    # extract pixel intensities
    mask_img_size = mask.unsqueeze(1).repeat(1, num_channels, 1, 1)
    # Number of points corresponds to number of visible pixels in mask, i.e. sum
    # of non zero indices in a mask (here we assume every mask has same number
    # of visible pixels)
    num_points = mask[0].nonzero().size(0)
    # Compute non zero indices
    # Shape (num_nonzeros, 3), where each row contains index of batch, height and width of nonzero
    nonzero_idx = mask.nonzero()
    # The x tensor for Neural Processes contains (height, width) indices, i.e.
    # 1st and 2nd indices of nonzero_idx (in zero based indexing)
    x = nonzero_idx[:, 1:].view(batch_size, num_points, 2).float()
    # The y tensor for Neural Processes contains the values of non zero pixels
    y = img[mask_img_size].view(batch_size, num_channels, num_points)
    # Ensure correct shape, i.e. (batch_size, num_points, num_channels)
    y = y.permute(0, 2, 1)

    if normalize:
        # TODO: make this separate for height and width for non square image
        # Normalize x to [-1, 1]
        x = (x - float(height) / 2) / (float(height) / 2)
        # Normalize y's to [-0.5, 0.5]
        y -= 0.5

    return x, y


def img_mask_to_np_input_predict(img, mask, normalize=True):
    """
    Given an image and a mask, return x and y tensors expected by Neural
    Process. Specifically, x will contain indices of unmasked points, e.g.
    [[1, 0], [23, 14], [24, 19]] and y will contain the corresponding pixel
    intensities, e.g. [[0.2], [0.73], [0.12]] for grayscale or
    [[0.82, 0.71, 0.5], [0.42, 0.33, 0.81], [0.21, 0.23, 0.32]] for RGB.

    Parameters
    ----------
    img : torch.Tensor
        Shape (N, C, H, W). Pixel intensities should be in [0, 1]

    mask : torch.ByteTensor
        Binary matrix where 0 corresponds to masked pixel and 1 to a visible
        pixel. Shape (N, H, W). Note the number of unmasked pixels must be the
        SAME for every mask in batch.

    normalize : bool
        If true normalizes pixel locations x to [-1, 1] and pixel intensities to
        [-0.5, 0.5]
    """
    batch_size, num_channels, height, width = img.size()
    # Create a mask which matches exactly with image size which will be used to
    # extract pixel intensities
    mask_img_size = mask.unsqueeze(1).repeat(1, num_channels, 1, 1)
    # Number of points corresponds to number of visible pixels in mask, i.e. sum
    # of non zero indices in a mask (here we assume every mask has same number
    # of visible pixels)
    num_points = mask[0].nonzero().size(0)
    # Compute non zero indices
    # Shape (num_nonzeros, 3), where each row contains index of batch, height and width of nonzero
    nonzero_idx = mask.nonzero()
    # The x tensor for Neural Processes contains (height, width) indices, i.e.
    # 1st and 2nd indices of nonzero_idx (in zero based indexing)
    x = nonzero_idx[:, 1:].view(batch_size, num_points, 2).float()
    # The y tensor for Neural Processes contains the values of non zero pixels
    y = img[mask_img_size].view(batch_size, num_channels, num_points)
    # Ensure correct shape, i.e. (batch_size, num_points, num_channels)
    y = y.permute(0, 2, 1)

    if normalize:
        # TODO: make this separate for height and width for non square image
        # Normalize x to [-1, 1]
        x = (x - float(height) / 2) / (float(height) / 2)
        # Normalize y's to [-0.5, 0.5]
        y -= 0.5

    return x, y

def random_context_target_mask(img_size, num_context, num_extra_target):
    """Returns random context and target masks where 0 corresponds to a hidden
    value and 1 to a visible value. The visible pixels in the context mask are
    a subset of the ones in the target mask.

    Parameters
    ----------
    img_size : tuple of ints
        E.g. (1, 32, 32) for grayscale or (3, 64, 64) for RGB.

    num_context : int
        Number of context points.

    num_extra_target : int
        Number of additional target points.
    """
    _, height, width = img_size
    # Sample integers without replacement between 0 and the total number of
    # pixels. The measurements array will then contain pixel indices
    # corresponding to locations where pixels will be visible.
    measurements = np.random.choice(range(height * width),
                                    size=num_context + num_extra_target,
                                    replace=False)
    # Create empty masks
    context_mask = torch.zeros(width, height).byte()
    target_mask = torch.zeros(width, height).byte()
    # Update mask with measurements
    for i, m in enumerate(measurements):
        row = int(m / width)
        col = m % width
        target_mask[row, col] = 1
        if i < num_context:
            context_mask[row, col] = 1
    return context_mask, target_mask


def batch_context_target_mask(img_size, num_context, num_extra_target,
                              batch_size, repeat=False):
    """Returns bacth of context and target masks, where the visible pixels in
    the context mask are a subset of those in the target mask.

    Parameters
    ----------
    img_size : see random_context_target_mask

    num_context : see random_context_target_mask

    num_extra_target : see random_context_target_mask

    batch_size : int
        Number of masks to create.

    repeat : bool
        If True, repeats one mask across batch.
    """
    context_mask_batch = torch.zeros(batch_size, *img_size[1:]).byte()
    target_mask_batch = torch.zeros(batch_size, *img_size[1:]).byte()
    if repeat:
        context_mask, target_mask = random_context_target_mask(img_size,
                                                               num_context,
                                                               num_extra_target)
        for i in range(batch_size):
            context_mask_batch[i] = context_mask
            target_mask_batch[i] = target_mask
    else:
        for i in range(batch_size):
            context_mask, target_mask = random_context_target_mask(img_size,
                                                                   num_context,
                                                                   num_extra_target)
            context_mask_batch[i] = context_mask
            target_mask_batch[i] = target_mask
    return context_mask_batch, target_mask_batch


def xy_to_img(x, y, img_size):
    """Given an x and y returned by a Neural Process, reconstruct image.
    Missing pixels will have a value of 0.

    Parameters
    ----------
    x : torch.Tensor
        Shape (batch_size, num_points, 2) containing normalized indices.

    y : torch.Tensor
        Shape (batch_size, num_points, num_channels) where num_channels = 1 for
        grayscale and 3 for RGB, containing normalized pixel intensities.

    img_size : tuple of ints
        E.g. (1, 32, 32) for grayscale or (3, 64, 64) for RGB.
    """
    _, height, width = img_size
    batch_size, _, _ = x.size()
    # Unnormalize x and y
    x = x * float(height / 2) + float(height / 2)
    x = x.long()
    y += 0.5
    # Permute y so it matches order expected by image
    # (batch_size, num_points, num_channels) -> (batch_size, num_channels, num_points)
    y = y.permute(0, 2, 1)
    # Initialize empty image
    img = torch.zeros((batch_size,) + img_size)
    for i in range(batch_size):
        img[i, :, x[i, :, 0], x[i, :, 1]] = y[i, :, :]
    return img

def inpaint_fnp(np_model, img, context_mask, device):
    """
    Given an image and a set of context points, the model samples pixel
    intensities for the remaining pixels in the image.

    Parameters
    ----------
    model : models.NeuralProcessImg instance

    img : torch.Tensor
        Shape (channels, height, width)

    context_mask : torch.Tensor
        Binary tensor where 1 corresponds to a visible pixel and 0 to an
        occluded pixel. Shape (height, width). Must have dtype=torch.uint8
        or similar. 

    device : torch.device
    """
    #is_training = np_model.neural_process.training
    # For inpainting, use Neural Process in prediction mode
    #np_model.neural_process.training = False
    target_mask = 1 - context_mask  # All pixels which are not in context
    # Add a batch dimension to tensors and move to GPU
    img_batch = img.unsqueeze(0).to(device)
    context_batch = context_mask.unsqueeze(0).to(device)
    target_batch = target_mask.unsqueeze(0).to(device)
    p_y_pred = np_model(img_batch, context_batch, target_batch, train=False, cov=True)
    # Transform Neural Process output back to image
    x_target, _ = img_mask_to_np_input(img_batch, target_batch)
    # Use the mean (i.e. loc) parameter of normal distribution as predictions
    # for y_target
    img_rec = xy_to_img(x_target.cpu(), p_y_pred.loc.detach().cpu(), img.size())
    img_rec = img_rec[0]  # Remove batch dimension
    # Add context points back to image
    context_mask_img = context_mask.unsqueeze(0).repeat(3, 1, 1)
    img_rec[context_mask_img] = img[context_mask_img]
    # Reset model to mode it was in before inpainting
    #np_model.neural_process.training = is_training
    return img_rec

def inpaint(np_model, img, context_mask, device):
    """
    Given an image and a set of context points, the model samples pixel
    intensities for the remaining pixels in the image.

    Parameters
    ----------
    model : models.NeuralProcessImg instance

    img : torch.Tensor
        Shape (channels, height, width)

    context_mask : torch.Tensor
        Binary tensor where 1 corresponds to a visible pixel and 0 to an
        occluded pixel. Shape (height, width). Must have dtype=torch.uint8
        or similar. 

    device : torch.device
    """
    is_training = np_model.neural_process.training
    # For inpainting, use Neural Process in prediction mode
    np_model.neural_process.training = False
    target_mask = 1 - context_mask  # All pixels which are not in context
    # Add a batch dimension to tensors and move to GPU
    img_batch = img.unsqueeze(0).to(device)
    context_batch = context_mask.unsqueeze(0).to(device)
    target_batch = target_mask.unsqueeze(0).to(device)
    p_y_pred = np_model(img_batch, context_batch, target_batch)
    # Transform Neural Process output back to image
    x_target, _ = img_mask_to_np_input(img_batch, target_batch)
    # Use the mean (i.e. loc) parameter of normal distribution as predictions
    # for y_target
    img_rec = xy_to_img(x_target.cpu(), p_y_pred.loc.detach().cpu(), img.size())
    img_rec = img_rec[0]  # Remove batch dimension
    # Add context points back to image
    context_mask_img = context_mask.unsqueeze(0).repeat(3, 1, 1)
    img_rec[context_mask_img] = img[context_mask_img]
    # Reset model to mode it was in before inpainting
    np_model.neural_process.training = is_training
    return img_rec

class NeuralProcessTrainer():
    """
    Class to handle training of Neural Processes for functions and images.

    Parameters
    ----------
    device : torch.device

    neural_process : neural_process.NeuralProcess or NeuralProcessImg instance

    optimizer : one of torch.optim optimizers

    num_context_range : tuple of ints
        Number of context points will be sampled uniformly in the range given
        by num_context_range.

    num_extra_target_range : tuple of ints
        Number of extra target points (as we always include context points in
        target points, i.e. context points are a subset of target points) will
        be sampled uniformly in the range given by num_extra_target_range.

    print_freq : int
        Frequency with which to print loss information during training.
    """
    def __init__(self, device, neural_process, optimizer, num_context_range,
                 num_extra_target_range, is_img=False, print_freq=100):
        self.device = device
        self.neural_process = neural_process
        self.optimizer = optimizer
        self.num_context_range = num_context_range
        self.num_extra_target_range = num_extra_target_range
        self.print_freq = print_freq

        # Check if neural process is for images
        self.is_img = is_img #isinstance(self.neural_process, model.NeuralProcessImg)
        self.steps = 0
        self.epoch_loss_history = []

    def train(self, data_loader, dataset, epochs):
        """
        Trains Neural Process.

        Parameters
        ----------
        dataloader : torch.utils.DataLoader instance

        epochs : int
            Number of epochs to train for.
        """
        for epoch in range(epochs):
            epoch_loss = 0.
            for i, data in enumerate(data_loader):
                self.optimizer.zero_grad()

                # Sample number of context and target points
                num_context = randint(*self.num_context_range)
                num_extra_target = randint(*self.num_extra_target_range)

                # Create context and target points and apply neural process
                if self.is_img:
                    img, _ = data  # data is a tuple (img, label)
                    batch_size = img.size(0)
                    context_mask, target_mask = \
                        batch_context_target_mask(self.neural_process.img_size,
                                                  num_context, num_extra_target,
                                                  batch_size)

                    img = img.to(self.device)
                    context_mask = context_mask.to(self.device)
                    target_mask = target_mask.to(self.device)

                    p_y_pred, q_target, q_context = \
                        self.neural_process(img, context_mask, target_mask)

                    # Calculate y_target as this will be required for loss
                    _, y_target = img_mask_to_np_input(img, target_mask)
                else:
                    x, y = data
                    x, y = x.to(self.device), y.to(self.device)
                    x_context, y_context, x_target, y_target = \
                        context_target_split(x, y, num_context, num_extra_target)
                    p_y_pred, q_target, q_context = \
                        self.neural_process(x_context, y_context, x_target, y_target)
                #visualize(self.neural_process, data_loader, dataset, cond_x=x_context, cond_y=y_context, all_x=x_target, all_y=y_target, range_y=(-1., 1.), num_context=10, num_target=10, samples=100)
                loss = self._loss(p_y_pred, y_target, q_target, q_context)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                self.steps += 1

                if self.steps % self.print_freq == 0:
                    print("iteration {}, loss {:.3f}".format(self.steps, loss.item()))

            print("Epoch: {}, Avg_loss: {}".format(epoch, epoch_loss / len(data_loader)))
            self.epoch_loss_history.append(epoch_loss / len(data_loader))
            #if epoch % 10 == 0:

            #    visualize(self.neural_process, data_loader, dataset, cond_x=x_context, cond_y=y_context, all_x=x_target, all_y=y_target, range_y=(-2., 2.), num_context=10, num_target=10, samples=64)

    def _loss(self, p_y_pred, y_target, q_target, q_context):
        """
        Computes Neural Process loss.

        Parameters
        ----------
        p_y_pred : one of torch.distributions.Distribution
            Distribution over y output by Neural Process.

        y_target : torch.Tensor
            Shape (batch_size, num_target, y_dim)

        q_target : one of torch.distributions.Distribution
            Latent distribution for target points.

        q_context : one of torch.distributions.Distribution
            Latent distribution for context points.
        """
        # Log likelihood has shape (batch_size, num_target, y_dim). Take mean
        # over batch and sum over number of targets and dimensions of y
        log_likelihood = p_y_pred.log_prob(y_target).mean(dim=0).sum()
        # KL has shape (batch_size, r_dim). Take mean over batch and sum over
        # r_dim (since r_dim is dimension of normal distribution)
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return -log_likelihood + kl
    

class RealNeuralProcessTrainer():
    """
    Class to handle training of Neural Processes for functions and images.

    Parameters
    ----------
    device : torch.device

    neural_process : neural_process.NeuralProcess or NeuralProcessImg instance

    optimizer : one of torch.optim optimizers

    num_context_range : tuple of ints
        Number of context points will be sampled uniformly in the range given
        by num_context_range.

    num_extra_target_range : tuple of ints
        Number of extra target points (as we always include context points in
        target points, i.e. context points are a subset of target points) will
        be sampled uniformly in the range given by num_extra_target_range.

    print_freq : int
        Frequency with which to print loss information during training.
    """
    def __init__(self, device, neural_process, optimizer, num_context_range,
                 num_extra_target_range, is_img=False, print_freq=100):
        self.device = device
        self.neural_process = neural_process
        self.optimizer = optimizer
        self.num_context_range = num_context_range
        self.num_extra_target_range = num_extra_target_range
        self.print_freq = print_freq

        # Check if neural process is for images
        self.is_img = is_img #isinstance(self.neural_process, model.NeuralProcessImg)
        self.steps = 0
        self.epoch_loss_history = []

    def train(self, data_loader, dataset, epochs):
        """
        Trains Neural Process.

        Parameters
        ----------
        dataloader : torch.utils.DataLoader instance

        epochs : int
            Number of epochs to train for.
        """
        for epoch in range(epochs):
            epoch_loss = 0.
            for i, data in enumerate(data_loader):
                self.optimizer.zero_grad()

                # Sample number of context and target points
                num_context = randint(*self.num_context_range)
                num_extra_target = randint(*self.num_extra_target_range)

                # Create context and target points and apply neural process
                if self.is_img:
                    img, _ = data  # data is a tuple (img, label)
                    batch_size = img.size(0)
                    context_mask, target_mask = \
                        batch_context_target_mask(self.neural_process.img_size,
                                                  num_context, num_extra_target,
                                                  batch_size)

                    img = img.to(self.device)
                    context_mask = context_mask.to(self.device)
                    target_mask = target_mask.to(self.device)

                    p_y_pred, q_target, q_context = \
                        self.neural_process(img, context_mask, target_mask)

                    # Calculate y_target as this will be required for loss
                    _, y_target = img_mask_to_np_input(img, target_mask)
                else:
                    x, y = data
                    x, y = x.to(self.device), y.to(self.device)
                    x_context, y_context, x_target, y_target = \
                        context_target_split_real(x, y, num_context, num_extra_target)
                    p_y_pred, q_target, q_context = \
                        self.neural_process(x_context, y_context, x_target, y_target)
                #visualize(self.neural_process, data_loader, dataset, cond_x=x_context, cond_y=y_context, all_x=x_target, all_y=y_target, range_y=(-1., 1.), num_context=10, num_target=10, samples=100)
                loss = self._loss(p_y_pred, y_target, q_target, q_context)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                self.steps += 1

                if self.steps % self.print_freq == 0:
                    print("iteration {}, loss {:.3f}".format(self.steps, loss.item()))

            print("Epoch: {}, Avg_loss: {}".format(epoch, epoch_loss / len(data_loader)))
            self.epoch_loss_history.append(epoch_loss / len(data_loader))
            #if epoch % 10 == 0:

            #    visualize(self.neural_process, data_loader, dataset, cond_x=x_context, cond_y=y_context, all_x=x_target, all_y=y_target, range_y=(-2., 2.), num_context=10, num_target=10, samples=64)

    def _loss(self, p_y_pred, y_target, q_target, q_context):
        """
        Computes Neural Process loss.

        Parameters
        ----------
        p_y_pred : one of torch.distributions.Distribution
            Distribution over y output by Neural Process.

        y_target : torch.Tensor
            Shape (batch_size, num_target, y_dim)

        q_target : one of torch.distributions.Distribution
            Latent distribution for target points.

        q_context : one of torch.distributions.Distribution
            Latent distribution for context points.
        """
        # Log likelihood has shape (batch_size, num_target, y_dim). Take mean
        # over batch and sum over number of targets and dimensions of y
        log_likelihood = p_y_pred.log_prob(y_target).mean(dim=0).sum()
        # KL has shape (batch_size, r_dim). Take mean over batch and sum over
        # r_dim (since r_dim is dimension of normal distribution)
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return -log_likelihood + kl
    
class FNPTrainer():
    """
    Class to handle training of Neural Processes for functions and images.

    Parameters
    ----------
    device : torch.device

    neural_process : neural_process.NeuralProcess or NeuralProcessImg instance

    optimizer : one of torch.optim optimizers

    num_context_range : tuple of ints
        Number of context points will be sampled uniformly in the range given
        by num_context_range.

    num_extra_target_range : tuple of ints
        Number of extra target points (as we always include context points in
        target points, i.e. context points are a subset of target points) will
        be sampled uniformly in the range given by num_extra_target_range.

    print_freq : int
        Frequency with which to print loss information during training.
    """
    def __init__(self, device, neural_process, optimizer, num_context_range,
                 num_extra_target_range, is_img=False, print_freq=100):
        self.device = device
        self.neural_process = neural_process
        self.optimizer = optimizer
        self.num_context_range = num_context_range
        self.num_extra_target_range = num_extra_target_range
        self.print_freq = print_freq

        # Check if neural process is for images
        self.is_img = is_img #isinstance(self.neural_process, model.NeuralProcessImg)
        self.steps = 0
        self.epoch_loss_history = []

    def train(self, data_loader, dataset, epochs):
        """
        Trains Neural Process.

        Parameters
        ----------
        dataloader : torch.utils.DataLoader instance

        epochs : int
            Number of epochs to train for.
        """
        for epoch in range(epochs):
            epoch_loss = 0.
            for i, data in enumerate(data_loader):
                self.optimizer.zero_grad()

                # Sample number of context and target points
                num_context = randint(*self.num_context_range)
                num_extra_target = randint(*self.num_extra_target_range)

                # Create context and target points and apply neural process
                if self.is_img:
                    img, label = data  # data is a tuple (img, label)
                    batch_size = img.size(0)
                    num_context = batch_size // 2
                    num_extra_target = batch_size - num_context
                    context_mask, target_mask = \
                        batch_context_target_mask(self.neural_process.img_size,
                                                  num_context, num_extra_target,
                                                  batch_size)

                    img = img.to(self.device)
                    context_mask = context_mask.to(self.device)
                    target_mask = target_mask.to(self.device)

                    loss = \
                        self.neural_process(img, context_mask, target_mask)

                    # Calculate y_target as this will be required for loss
                    _, y_target = img_mask_to_np_input(img, target_mask)
                    
                    '''
                    img, label = img.to(self.device), label.to(self.device)
                    x_context, y_context, x_target, y_target = \
                        context_target_split_img(img, label, num_context, num_extra_target)
                    loss = \
                        self.neural_process(x_context, y_context, x_target, y_target)
                    '''
                else:
                    #print(data[0].shape, data[1].shape, data[2].shape, data[3].shape)
                    x, y = data
                    x, y = x.to(self.device), y.to(self.device)
                    x_context, y_context, x_target, y_target = \
                        context_target_split(x, y, num_context, num_extra_target)
                    loss = \
                        self.neural_process(x_context, y_context, x_target, y_target)
                #visualize_my_fnp(self.neural_process, data_loader, dataset, cond_x=x_context, cond_y=y_context, all_x=x_target, all_y=y_target, range_y=(-1., 1.), num_context=10, num_target=10, samples=100, device=self.device)
                #loss = self._loss(p_y_pred, y_target, q_target, q_context)
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.neural_process.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()

                self.steps += 1

                if self.steps % self.print_freq == 0:
                    print("Iteration {}, loss {:.3f}".format(self.steps, loss.item()))

            print("Epoch: {}, Avg_loss: {}".format(epoch, epoch_loss / len(data_loader)))
            self.epoch_loss_history.append(epoch_loss / len(data_loader))
            #if epoch % 100 == 0:

            #    visualize_my_fnp(self.neural_process, data_loader, dataset, cond_x=x_context, cond_y=y_context, all_x=x_target, all_y=y_target, range_y=(-2., 2.), num_context=10, num_target=10, samples=64, device=self.device)



class ConvCNPTrainer():
    """
    Class to handle training of Neural Processes for functions and images.

    Parameters
    ----------
    device : torch.device

    neural_process : neural_process.NeuralProcess or NeuralProcessImg instance

    optimizer : one of torch.optim optimizers

    num_context_range : tuple of ints
        Number of context points will be sampled uniformly in the range given
        by num_context_range.

    num_extra_target_range : tuple of ints
        Number of extra target points (as we always include context points in
        target points, i.e. context points are a subset of target points) will
        be sampled uniformly in the range given by num_extra_target_range.

    print_freq : int
        Frequency with which to print loss information during training.
    """
    def __init__(self, device, neural_process, optimizer, num_context_range,
                 num_extra_target_range, is_img=False, print_freq=100):
        self.device = device
        self.neural_process = neural_process
        self.optimizer = optimizer
        self.num_context_range = num_context_range
        self.num_extra_target_range = num_extra_target_range
        self.print_freq = print_freq

        # Check if neural process is for images
        self.is_img = is_img #isinstance(self.neural_process, model.NeuralProcessImg)
        self.steps = 0
        self.epoch_loss_history = []

    def train(self, data_loader, dataset, epochs):
        """
        Trains Neural Process.

        Parameters
        ----------
        dataloader : torch.utils.DataLoader instance

        epochs : int
            Number of epochs to train for.
        """
        for epoch in range(epochs):
            epoch_loss = 0.
            for i, data in enumerate(data_loader):
                self.optimizer.zero_grad()

                # Sample number of context and target points
                num_context = randint(*self.num_context_range)
                num_extra_target = randint(*self.num_extra_target_range)

                # Create context and target points and apply neural process
                if self.is_img:
                    img, label = data  # data is a tuple (img, label)
                    batch_size = img.size(0)
                    num_context = batch_size // 2
                    num_extra_target = batch_size - num_context
                    context_mask, target_mask = \
                        batch_context_target_mask(self.neural_process.img_size,
                                                  num_context, num_extra_target,
                                                  batch_size)

                    img = img.to(self.device)
                    context_mask = context_mask.to(self.device)
                    target_mask = target_mask.to(self.device)

                    loss = \
                        self.neural_process(img, context_mask, target_mask)

                    # Calculate y_target as this will be required for loss
                    #_, y_target = img_mask_to_np_input(img, target_mask)
                    
                    '''
                    img, label = img.to(self.device), label.to(self.device)
                    x_context, y_context, x_target, y_target = \
                        context_target_split_img(img, label, num_context, num_extra_target)
                    loss = \
                        self.neural_process(x_context, y_context, x_target, y_target)
                    '''
                else:
                    #print(data[0].shape, data[1].shape, data[2].shape, data[3].shape)
                    x, y = data
                    x, y = x.to(self.device), y.to(self.device)
                    x_context, y_context, x_target, y_target = \
                        context_target_split(x, y, num_context, num_extra_target)
                    loss = \
                        self.neural_process(x_context, y_context, x_target, y_target)
                #visualize_my_fnp(self.neural_process, data_loader, dataset, cond_x=x_context, cond_y=y_context, all_x=x_target, all_y=y_target, range_y=(-1., 1.), num_context=10, num_target=10, samples=100, device=self.device)
                #loss = self._loss(p_y_pred, y_target, q_target, q_context)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                self.steps += 1

                if self.steps % self.print_freq == 0:
                    print("iteration {}, loss {:.3f}".format(self.steps, loss.item()))

            print("Epoch: {}, Avg_loss: {}".format(epoch, epoch_loss / len(data_loader)))
            self.epoch_loss_history.append(epoch_loss / len(data_loader))
            #if epoch % 100 == 0:

            #    visualize_my_fnp(self.neural_process, data_loader, dataset, cond_x=x_context, cond_y=y_context, all_x=x_target, all_y=y_target, range_y=(-2., 2.), num_context=10, num_target=10, samples=64, device=self.device)

class RealConvCNPTrainer():
    """
    Class to handle training of Neural Processes for functions and images.

    Parameters
    ----------
    device : torch.device

    neural_process : neural_process.NeuralProcess or NeuralProcessImg instance

    optimizer : one of torch.optim optimizers

    num_context_range : tuple of ints
        Number of context points will be sampled uniformly in the range given
        by num_context_range.

    num_extra_target_range : tuple of ints
        Number of extra target points (as we always include context points in
        target points, i.e. context points are a subset of target points) will
        be sampled uniformly in the range given by num_extra_target_range.

    print_freq : int
        Frequency with which to print loss information during training.
    """
    def __init__(self, device, neural_process, optimizer, num_context_range,
                 num_extra_target_range, is_img=False, print_freq=100):
        self.device = device
        self.neural_process = neural_process
        self.optimizer = optimizer
        self.num_context_range = num_context_range
        self.num_extra_target_range = num_extra_target_range
        self.print_freq = print_freq

        # Check if neural process is for images
        self.is_img = is_img #isinstance(self.neural_process, model.NeuralProcessImg)
        self.steps = 0
        self.epoch_loss_history = []

    def train(self, data_loader, dataset, epochs):
        """
        Trains Neural Process.

        Parameters
        ----------
        dataloader : torch.utils.DataLoader instance

        epochs : int
            Number of epochs to train for.
        """
        for epoch in range(epochs):
            epoch_loss = 0.
            for i, data in enumerate(data_loader):
                self.optimizer.zero_grad()

                # Sample number of context and target points
                num_context = randint(*self.num_context_range)
                num_extra_target = randint(*self.num_extra_target_range)

                # Create context and target points and apply neural process
                if self.is_img:
                    img, label = data  # data is a tuple (img, label)
                    batch_size = img.size(0)
                    num_context = batch_size // 2
                    num_extra_target = batch_size - num_context
                    context_mask, target_mask = \
                        batch_context_target_mask(self.neural_process.img_size,
                                                  num_context, num_extra_target,
                                                  batch_size)

                    img = img.to(self.device)
                    context_mask = context_mask.to(self.device)
                    target_mask = target_mask.to(self.device)

                    loss = \
                        self.neural_process(img, context_mask, target_mask)

                    # Calculate y_target as this will be required for loss
                    #_, y_target = img_mask_to_np_input(img, target_mask)
                    
                    '''
                    img, label = img.to(self.device), label.to(self.device)
                    x_context, y_context, x_target, y_target = \
                        context_target_split_img(img, label, num_context, num_extra_target)
                    loss = \
                        self.neural_process(x_context, y_context, x_target, y_target)
                    '''
                else:
                    #print(data[0].shape, data[1].shape, data[2].shape, data[3].shape)
                    x, y = data
                    x, y = x.to(self.device), y.to(self.device)
                    x_context, y_context, x_target, y_target = \
                        context_target_split_real(x, y, num_context, num_extra_target)
                    loss = \
                        self.neural_process(x_context, y_context, x_target, y_target)
                #visualize_my_fnp(self.neural_process, data_loader, dataset, cond_x=x_context, cond_y=y_context, all_x=x_target, all_y=y_target, range_y=(-1., 1.), num_context=10, num_target=10, samples=100, device=self.device)
                #loss = self._loss(p_y_pred, y_target, q_target, q_context)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                self.steps += 1

                if self.steps % self.print_freq == 0:
                    print("iteration {}, loss {:.3f}".format(self.steps, loss.item()))

            print("Epoch: {}, Avg_loss: {}".format(epoch, epoch_loss / len(data_loader)))
            self.epoch_loss_history.append(epoch_loss / len(data_loader))
            #if epoch % 100 == 0:

            #    visualize_my_fnp(self.neural_process, data_loader, dataset, cond_x=x_context, cond_y=y_context, all_x=x_target, all_y=y_target, range_y=(-2., 2.), num_context=10, num_target=10, samples=64, device=self.device)


class FNPOriginalTrainer():
    """
    Class to handle training of Neural Processes for functions and images.

    Parameters
    ----------
    device : torch.device

    neural_process : neural_process.NeuralProcess or NeuralProcessImg instance

    optimizer : one of torch.optim optimizers

    num_context_range : tuple of ints
        Number of context points will be sampled uniformly in the range given
        by num_context_range.

    num_extra_target_range : tuple of ints
        Number of extra target points (as we always include context points in
        target points, i.e. context points are a subset of target points) will
        be sampled uniformly in the range given by num_extra_target_range.

    print_freq : int
        Frequency with which to print loss information during training.
    """
    def __init__(self, device, neural_process, optimizer, num_context_range,
                 num_extra_target_range, is_img=False, print_freq=100):
        self.device = device
        self.neural_process = neural_process
        self.optimizer = optimizer
        self.num_context_range = num_context_range
        self.num_extra_target_range = num_extra_target_range
        self.print_freq = print_freq

        # Check if neural process is for images
        self.is_img = is_img #isinstance(self.neural_process, model.NeuralProcessImg)
        self.steps = 0
        self.epoch_loss_history = []

    def train(self, XR, yR, XM, yM, X, y, dx, stdx, stdy, epochs):
        """
        Trains Neural Process.

        Parameters
        ----------
        dataloader : torch.utils.DataLoader instance

        epochs : int
            Number of epochs to train for.
        """
        for i in range(epochs):
            self.optimizer.zero_grad()
    
            loss = self.neural_process(XR, yR, XM, yM)
            loss.backward()
            self.optimizer.step()
        
            if i % int(epochs / 5) == 0:
                print('Epoch {}/{}, loss: {:.3f}'.format(i, epochs, loss.item()))
                fnp_visualize(self.neural_process, dx, stdx, stdy, cond_x=XR, cond_y=yR, all_x=X, all_y=y, range_y=(-2., 3.), samples=100)
        fnp_visualize(self.neural_process, dx, stdx, stdy, cond_x=XR, cond_y=yR, all_x=X, all_y=y, range_y=(-2., 3.), samples=100)
        print('Done.')


def fnp_visualize(model, dx, stdx, stdy, cond_x=None, cond_y=None, all_x=None, all_y=None, samples=30, 
              range_y=(-100., 100.), title='', train=False):
    '''
    Visualizes the predictive distribution
    '''
    dxy = np.zeros((dx.shape[0], samples))
    if not train:
        model.eval()
    with torch.no_grad():
        dxi = torch.from_numpy(stdx.transform(dx).astype(np.float32))
        if torch.cuda.is_available():
            dxi = dxi.cuda()
        for j in range(samples):
            dxy[:, j] = model.predict(dxi, cond_x, cond_y).cpu().ravel()
    print()

    plt.figure()
    mean_dxy, std_dxy = dxy.mean(axis=1), dxy.std(axis=1)
    # smooth it in order to avoid the sampling jitter
    mean_dxys = savgol_filter(mean_dxy, 61, 3)
    std_dxys = savgol_filter(std_dxy, 61, 3)
    
    if torch.cuda.is_available():
        all_x, all_y, cond_x, cond_y = all_x.cpu(), all_y.cpu(), cond_x.cpu(), cond_y.cpu()

    plt.plot(dx.ravel(), mean_dxys, label='Mean function')
    plt.plot(stdx.inverse_transform(all_x.data.numpy()).ravel(), stdy.inverse_transform(all_y.data.numpy()).ravel(), 'o',
             label='Observations')
    if cond_x is not None:
        plt.plot(stdx.inverse_transform(cond_x.data.numpy()).ravel(), stdy.inverse_transform(cond_y.data.numpy()).ravel(), 'o',
             label='Reference')
    plt.fill_between(dx.ravel(), mean_dxys-1.*std_dxys, mean_dxys+1.*std_dxys, alpha=.1)
    plt.fill_between(dx.ravel(), mean_dxys-2.*std_dxys, mean_dxys+2.*std_dxys, alpha=.1)
    plt.fill_between(dx.ravel(), mean_dxys-3.*std_dxys, mean_dxys+3.*std_dxys, alpha=.1)

    plt.xlim([np.min(dx), np.max(dx)])
    plt.ylim(range_y)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=3, fancybox=False, shadow=False)
    plt.title(title)
    model.train()
    plt.show()

def visualize_my_fnp(model, data_loader, dataset, cond_x=None, cond_y=None, all_x=None, all_y=None, samples=30, 
              range_y=(-100., 100.), num_context=10, num_target=10, title='', device='cpu', train=False):
    '''
    Visualizes the predictive distribution
    '''

    # Extract a batch from data_loader
    for batch in data_loader:
        break

    # Use batch to create random set of context points
    x, y = batch
    #std_x = StandardScaler().fit(x[:,:,0]) 
    #std_y = StandardScaler().fit(y[:,:,0])
    #std_x.fit(x[:,:,0])
    #std_y.fit(y[:,:,0])
    #x, y = std_x.transform(x), std_y.transform(y)
    mean_x, std_x, mean_y, std_y = dataset.get_mean_std()
    #print(mean_x, std_x, mean_y, std_y)
    x_context, y_context, _, _ = context_target_split(x[0:1], y[0:1], 
                                                  num_context, 
                                             num_target)
    #x_context = torch.tensor(std_x.transform(x_context[:,:,0]))
    #x_context = x_context.unsqueeze(2).to(torch.float32)
    #y_context = torch.tensor(std_y.transform(y_context[:,:,0]))
    #y_context = y_context.unsqueeze(2).to(torch.float32)
    # Create a set of target points corresponding to entire [-pi, pi] range
    x_context = x_context.to(device)
    y_context = y_context.to(device)
    x_target = torch.Tensor(np.linspace(-1, 1, 100)).to(device)
    #x_target = (x_target - mean_x)/std_x
    #std_x.fit(x_target.unsqueeze(0))
    #x_target = torch.tensor(std_x.transform(x_target[:,:,0]))


    x_target = x_target.unsqueeze(1).unsqueeze(0)

    dxy = np.zeros((x_target.shape[1], samples))
    model.training = False
    for i in range(samples):
        # Neural process returns distribution over y_target
        p_y_pred = model.predict(x_target, x_context, y_context)
        # Extract mean of distribution
        dxy[:, i] = p_y_pred.detach().cpu().ravel()
        #plt.plot(x_target.numpy()[0], mu.numpy()[0], 
        #     alpha=0.05, c='b')
        #plt.fill_between(mu.numpy()[0], mu.numpy()[0]-p_y_pred.scale.detach().numpy()[0], mu.numpy()[0]+p_y_pred.scale.detach().numpy()[0], color='yellow', alpha=0.5)

    #plt.scatter(cond_x[0].numpy(), cond_y[0].numpy(), c='k')
    #plt.show()
            

    plt.figure()
    mean_dxy, std_dxy = dxy.mean(axis=1), dxy.std(axis=1)
    # smooth it in order to avoid the sampling jitter
    mean_dxys = savgol_filter(mean_dxy, 61, 3)
    std_dxys = savgol_filter(std_dxy, 61, 3)

    #print(p_y_pred.loc.detach().shape, p_y_pred.scale.detach().shape)
    #mean_dxys = p_y_pred.loc.detach()[0,:,0]
    #std_dxys = p_y_pred.scale.detach()[0,:,0]
    #print(std_dxys)
    
    if torch.cuda.is_available():
        all_x, all_y, cond_x, cond_y = all_x.cpu(), all_y.cpu(), cond_x.cpu(), cond_y.cpu()

    plt.plot(x_target.detach().cpu().ravel(), mean_dxys, label='Mean function')
    #plt.plot(all_x.data.numpy().ravel(), all_y.data.numpy().ravel(), 'o',
    #         label='Observations')
    if cond_x is not None:
        #print(x_context.data.numpy()[0,:,:].shape, y_context.data.numpy()[0,:,:].shape)
        plt.plot(x_context.data.cpu().numpy()[0,:,:].ravel(), y_context.data.cpu().numpy()[0,:,:].ravel(), 'o',
             label='Reference')
    plt.fill_between(x_target.cpu().ravel(), mean_dxys-1.*std_dxys, mean_dxys+1.*std_dxys, alpha=.1)
    plt.fill_between(x_target.cpu().ravel(), mean_dxys-2.*std_dxys, mean_dxys+2.*std_dxys, alpha=.1)
    plt.fill_between(x_target.cpu().ravel(), mean_dxys-3.*std_dxys, mean_dxys+3.*std_dxys, alpha=.1)

    #plt.xlim([np.min(x_target), np.max(x_target)])
    plt.ylim(range_y)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=3, fancybox=False, shadow=False)
    plt.title(title)
    model.train()
    plt.show()


def visualize(model, data_loader, dataset, cond_x=None, cond_y=None, all_x=None, all_y=None, samples=30, 
              range_y=(-100., 100.), num_context=10, num_target=10, title='', train=False):
    '''
    Visualizes the predictive distribution
    '''

    # Extract a batch from data_loader
    for batch in data_loader:
        break

    # Use batch to create random set of context points
    x, y = batch
    #std_x = StandardScaler().fit(x[:,:,0]) 
    #std_y = StandardScaler().fit(y[:,:,0])
    #std_x.fit(x[:,:,0])
    #std_y.fit(y[:,:,0])
    #x, y = std_x.transform(x), std_y.transform(y)
    mean_x, std_x, mean_y, std_y = dataset.get_mean_std()
    #print(mean_x, std_x, mean_y, std_y)
    x_context, y_context, _, _ = context_target_split(x[0:1], y[0:1], 
                                                  num_context, 
                                             num_target)
    #x_context = torch.tensor(std_x.transform(x_context[:,:,0]))
    #x_context = x_context.unsqueeze(2).to(torch.float32)
    #y_context = torch.tensor(std_y.transform(y_context[:,:,0]))
    #y_context = y_context.unsqueeze(2).to(torch.float32)
    # Create a set of target points corresponding to entire [-pi, pi] range
    x_target = torch.Tensor(np.linspace(-5, 5, 100))
    #x_target = (x_target - mean_x)/std_x
    #std_x.fit(x_target.unsqueeze(0))
    #x_target = torch.tensor(std_x.transform(x_target[:,:,0]))


    x_target = x_target.unsqueeze(1).unsqueeze(0)

    dxy = np.zeros((x_target.shape[1], samples))
    model.training = False
    for i in range(samples):
        # Neural process returns distribution over y_target
        p_y_pred = model(x_context, y_context, x_target)
        # Extract mean of distribution
        mu = p_y_pred.loc.detach()[0,:,0]
        dxy[:, i] = mu
        #plt.plot(x_target.numpy()[0], mu.numpy()[0], 
        #     alpha=0.05, c='b')
        #plt.fill_between(mu.numpy()[0], mu.numpy()[0]-p_y_pred.scale.detach().numpy()[0], mu.numpy()[0]+p_y_pred.scale.detach().numpy()[0], color='yellow', alpha=0.5)

    #plt.scatter(cond_x[0].numpy(), cond_y[0].numpy(), c='k')
    #plt.show()
            

    plt.figure()
    mean_dxy, std_dxy = dxy.mean(axis=1), dxy.std(axis=1)
    # smooth it in order to avoid the sampling jitter
    mean_dxys = savgol_filter(mean_dxy, 61, 3)
    std_dxys = savgol_filter(std_dxy, 61, 3)

    #print(p_y_pred.loc.detach().shape, p_y_pred.scale.detach().shape)
    mean_dxys = p_y_pred.loc.detach()[0,:,0]
    std_dxys = p_y_pred.scale.detach()[0,:,0]
    #print(std_dxys)
    
    if torch.cuda.is_available():
        all_x, all_y, cond_x, cond_y = all_x.cpu(), all_y.cpu(), cond_x.cpu(), cond_y.cpu()

    plt.plot(x_target.ravel(), mean_dxys, label='Mean function')
    #plt.plot(all_x.data.numpy().ravel(), all_y.data.numpy().ravel(), 'o',
    #         label='Observations')
    if cond_x is not None:
        #print(x_context.data.numpy()[0,:,:].shape, y_context.data.numpy()[0,:,:].shape)
        plt.plot(x_context.data.numpy()[0,:,:].ravel(), y_context.data.numpy()[0,:,:].ravel(), 'o',
             label='Reference')
    plt.fill_between(x_target.ravel(), mean_dxys-1.*std_dxys, mean_dxys+1.*std_dxys, alpha=.1)
    plt.fill_between(x_target.ravel(), mean_dxys-2.*std_dxys, mean_dxys+2.*std_dxys, alpha=.1)
    plt.fill_between(x_target.ravel(), mean_dxys-3.*std_dxys, mean_dxys+3.*std_dxys, alpha=.1)

    #plt.xlim([np.min(x_target), np.max(x_target)])
    plt.ylim(range_y)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=3, fancybox=False, shadow=False)
    plt.title(title)
    model.train()
    plt.show()



class CNPTrainer():
    """
    Class to handle training of Neural Processes for functions and images.

    Parameters
    ----------
    device : torch.device

    neural_process : neural_process.NeuralProcess or NeuralProcessImg instance

    optimizer : one of torch.optim optimizers

    num_context_range : tuple of ints
        Number of context points will be sampled uniformly in the range given
        by num_context_range.

    num_extra_target_range : tuple of ints
        Number of extra target points (as we always include context points in
        target points, i.e. context points are a subset of target points) will
        be sampled uniformly in the range given by num_extra_target_range.

    print_freq : int
        Frequency with which to print loss information during training.
    """
    def __init__(self, device, neural_process, optimizer, num_context_range,
                 num_extra_target_range, is_img=False, print_freq=100):
        self.device = device
        self.neural_process = neural_process
        self.optimizer = optimizer
        self.num_context_range = num_context_range
        self.num_extra_target_range = num_extra_target_range
        self.print_freq = print_freq

        # Check if neural process is for images
        self.is_img = is_img #isinstance(self.neural_process, model.NeuralProcessImg)
        self.steps = 0
        self.epoch_loss_history = []

    def train(self, data_loader, dataset, epochs):
        """
        Trains Neural Process.

        Parameters
        ----------
        dataloader : torch.utils.DataLoader instance

        epochs : int
            Number of epochs to train for.
        """
        for epoch in range(epochs):
            epoch_loss = 0.
            for i, data in enumerate(data_loader):
                self.optimizer.zero_grad()

                # Sample number of context and target points
                num_context = randint(*self.num_context_range)
                num_extra_target = randint(*self.num_extra_target_range)

                # Create context and target points and apply neural process
                if self.is_img:
                    img, _ = data  # data is a tuple (img, label)
                    batch_size = img.size(0)
                    context_mask, target_mask = \
                        batch_context_target_mask(self.neural_process.img_size,
                                                  num_context, num_extra_target,
                                                  batch_size)

                    img = img.to(self.device)
                    context_mask = context_mask.to(self.device)
                    target_mask = target_mask.to(self.device)

                    p_y_pred = \
                        self.neural_process(img, context_mask, target_mask)

                    # Calculate y_target as this will be required for loss
                    _, y_target = img_mask_to_np_input(img, target_mask)
                else:
                    x, y = data
                    x, y = x.to(self.device), y.to(self.device)
                    x_context, y_context, x_target, y_target = \
                        context_target_split(x, y, num_context, num_extra_target)
                    p_y_pred = \
                        self.neural_process(x_context, y_context, x_target, y_target)

                loss = self._loss(p_y_pred, y_target)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                self.steps += 1

                if self.steps % self.print_freq == 0:
                    print("iteration {}, loss {:.3f}".format(self.steps, loss.item()))

            #visualize(fnp, dx, stdx, stdy, cond_x=XR, cond_y=yR, all_x=X, all_y=y, range_y=(-2., 3.), samples=100)

            print("Epoch: {}, Avg_loss: {}".format(epoch, epoch_loss / len(data_loader)))
            self.epoch_loss_history.append(epoch_loss / len(data_loader))

            #if epoch % 5 == 0:

            #    visualize(self.neural_process, data_loader, dataset, cond_x=x_context, cond_y=y_context, all_x=x_target, all_y=y_target, range_y=(-5., 5.), num_context=10, num_target=10, samples=100)


    def _loss(self, p_y_pred, y_target):
        """
        Computes Neural Process loss.

        Parameters
        ----------
        p_y_pred : one of torch.distributions.Distribution
            Distribution over y output by Neural Process.

        y_target : torch.Tensor
            Shape (batch_size, num_target, y_dim)

        """
        # Log likelihood has shape (batch_size, num_target, y_dim). Take mean
        # over batch and sum over number of targets and dimensions of y
        log_likelihood = p_y_pred.log_prob(y_target).mean(dim=0).sum()
        return -log_likelihood 
    
class RealCNPTrainer():
    """
    Class to handle training of Neural Processes for functions and images.

    Parameters
    ----------
    device : torch.device

    neural_process : neural_process.NeuralProcess or NeuralProcessImg instance

    optimizer : one of torch.optim optimizers

    num_context_range : tuple of ints
        Number of context points will be sampled uniformly in the range given
        by num_context_range.

    num_extra_target_range : tuple of ints
        Number of extra target points (as we always include context points in
        target points, i.e. context points are a subset of target points) will
        be sampled uniformly in the range given by num_extra_target_range.

    print_freq : int
        Frequency with which to print loss information during training.
    """
    def __init__(self, device, neural_process, optimizer, num_context_range,
                 num_extra_target_range, is_img=False, print_freq=100):
        self.device = device
        self.neural_process = neural_process
        self.optimizer = optimizer
        self.num_context_range = num_context_range
        self.num_extra_target_range = num_extra_target_range
        self.print_freq = print_freq

        # Check if neural process is for images
        self.is_img = is_img #isinstance(self.neural_process, model.NeuralProcessImg)
        self.steps = 0
        self.epoch_loss_history = []

    def train(self, data_loader, dataset, epochs):
        """
        Trains Neural Process.

        Parameters
        ----------
        dataloader : torch.utils.DataLoader instance

        epochs : int
            Number of epochs to train for.
        """
        for epoch in range(epochs):
            epoch_loss = 0.
            for i, data in enumerate(data_loader):
                self.optimizer.zero_grad()

                # Sample number of context and target points
                num_context = randint(*self.num_context_range)
                num_extra_target = randint(*self.num_extra_target_range)

                # Create context and target points and apply neural process
                if self.is_img:
                    img, _ = data  # data is a tuple (img, label)
                    batch_size = img.size(0)
                    context_mask, target_mask = \
                        batch_context_target_mask(self.neural_process.img_size,
                                                  num_context, num_extra_target,
                                                  batch_size)

                    img = img.to(self.device)
                    context_mask = context_mask.to(self.device)
                    target_mask = target_mask.to(self.device)

                    p_y_pred = \
                        self.neural_process(img, context_mask, target_mask)

                    # Calculate y_target as this will be required for loss
                    _, y_target = img_mask_to_np_input(img, target_mask)
                else:
                    x, y = data
                    x, y = x.to(self.device), y.to(self.device)
                    x_context, y_context, x_target, y_target = \
                        context_target_split_real(x, y, num_context, num_extra_target)
                    p_y_pred = \
                        self.neural_process(x_context, y_context, x_target, y_target)

                loss = self._loss(p_y_pred, y_target)
                #loss = torch.nn.MSELoss()(p_y_pred.loc, y_target)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                self.steps += 1

                if self.steps % self.print_freq == 0:
                    print("iteration {}, loss {:.3f}".format(self.steps, loss.item()))

            #visualize(fnp, dx, stdx, stdy, cond_x=XR, cond_y=yR, all_x=X, all_y=y, range_y=(-2., 3.), samples=100)

            print("Epoch: {}, Avg_loss: {}".format(epoch, epoch_loss / len(data_loader)))
            self.epoch_loss_history.append(epoch_loss / len(data_loader))

            #if epoch % 5 == 0:

            #    visualize(self.neural_process, data_loader, dataset, cond_x=x_context, cond_y=y_context, all_x=x_target, all_y=y_target, range_y=(-5., 5.), num_context=10, num_target=10, samples=100)


    def _loss(self, p_y_pred, y_target):
        """
        Computes Neural Process loss.

        Parameters
        ----------
        p_y_pred : one of torch.distributions.Distribution
            Distribution over y output by Neural Process.

        y_target : torch.Tensor
            Shape (batch_size, num_target, y_dim)

        """
        # Log likelihood has shape (batch_size, num_target, y_dim). Take mean
        # over batch and sum over number of targets and dimensions of y
        log_likelihood = p_y_pred.log_prob(y_target).mean(dim=0).sum()
        return -log_likelihood 