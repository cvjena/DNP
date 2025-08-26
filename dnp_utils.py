import torch
import math
import torch.nn.functional as F
import numpy as np
from torch.distributions import Bernoulli
from itertools import product

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
float_tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def logitexp(logp):
    # https: // github.com / pytorch / pytorch / issues / 4007
    pos = torch.clamp(logp, min=-0.69314718056)
    neg = torch.clamp(logp, max=-0.69314718056)
    neg_val = neg - torch.log(1 - torch.exp(neg))
    pos_val = -torch.log(torch.clamp(torch.expm1(-pos), min=1e-20))
    return pos_val + neg_val


def one_hot(x, n_classes=10):
    x_onehot = float_tensor(x.size(0), n_classes).zero_()
    x_onehot.scatter_(1, x[:, None], 1)

    return x_onehot



class Normal(object):
    def __init__(self, means, logscales, **kwargs):
        self.means = means
        self.logscales = logscales

    def log_prob(self, value):
        log_prob = torch.pow(value - self.means, 2)
        log_prob *= - (1 / (2. * self.logscales.mul(2.).exp()))
        log_prob -= self.logscales + .5 * math.log(2. * math.pi)
        return log_prob

    def sample(self, **kwargs):
        eps = torch.normal(float_tensor(self.means.size()).zero_(), float_tensor(self.means.size()).fill_(1))
        return self.means + self.logscales.exp() * eps

    def rsample(self, **kwargs):
        return self.sample(**kwargs)



class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        assert len(x.shape) > 1

        return x.view(x.shape[0], -1)

