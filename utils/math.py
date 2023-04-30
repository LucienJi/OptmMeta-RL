import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def dump_info(D,info):
  for k,v in info.items():
    l = D.get(k,list())
    l.append(v) 
    D[k] = l
  return D

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [layer_init(nn.Linear(sizes[j], sizes[j+1])), act()]
    return nn.Sequential(*layers)

def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)
    return entropy.sum(-1, keepdim=True)


def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(-1, keepdim=True)

def to_device(data,device):
    if type(data) is not np.ndarray:
        data = np.array(data,dtype=np.float32)
    return torch.from_numpy(data).float().to(device)


def from_queue(data,device = 'cpu'):
    return torch.tensor(np.array(data)).float().to(device)

def product_of_gaussians3D(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=1)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=1)
    return mu, sigma_squared

def generate_gaussian(mu_sigma, latent_dim, sigma_ops="softplus", mode=None):
    """
    Generate a Gaussian distribution given a selected parametrization.
    """
    mus, sigmas = torch.split(mu_sigma, split_size_or_sections=latent_dim, dim=-1)

    if sigma_ops == 'softplus':
        # Softplus, s.t. sigma is always positive
        # sigma is assumed to be st. dev. not variance
        sigmas = F.softplus(sigmas)
    if mode == 'multiplication':
        mu, sigma = product_of_gaussians3D(mus, sigmas)
    else:
        mu = mus
        sigma = sigmas
    return torch.distributions.normal.Normal(mu, sigma)