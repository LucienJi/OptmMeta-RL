import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F
from utils.math import mlp,generate_gaussian



class PriorPz(nn.Module):
    def __init__(self,
                 num_classes,
                 latent_dim
                 ):
        super(PriorPz, self).__init__()
        self.latent_dim = latent_dim
        # feed cluster number y as one-hot, get mu_sigma out
        self.linear = nn.Linear(num_classes, self.latent_dim * 2)

    def forward(self, m):
        return self.linear(m)

class ClassEncoder(nn.Module):
    def __init__(self,
                 num_classes,
                 shared_dim
    ):
        super(ClassEncoder, self).__init__()

        self.shared_dim = shared_dim
        self.num_classes = num_classes
        self.linear = nn.Linear(self.shared_dim, self.num_classes)

    def forward(self, m):
        return F.softmax(self.linear(m), dim=-1)

class EncoderMixtureModelTransitionSharedY(nn.Module):
    '''
    Overall encoder network, putting a shared_encoder, class_encoder and gauss_encoder together.
    '''
    def __init__(self,
                obs_dim,act_dim,
                 shared_dim,
                 emb_dim,
                 num_classes,
                 merge_mode,
                 device,
    ):
        super(EncoderMixtureModelTransitionSharedY, self).__init__()
        self.shared_dim = shared_dim
        self.encoder_input_dim = obs_dim + act_dim + obs_dim + 1
        self.emb_dim = emb_dim


        self.num_classes_init = num_classes
        self.num_classes = num_classes

        self.merge_mode = merge_mode
        self.shared_encoder = mlp([self.encoder_input_dim,]+[self.shared_dim,self.shared_dim] + [self.shared_dim,],nn.Tanh)
        self.class_encoder = ClassEncoder(self.num_classes, self.shared_dim)
        self.gauss_encoder_list = nn.ModuleList([nn.Linear(self.shared_dim, self.emb_dim * 2) for _ in range(self.num_classes)])

        self.device = torch.device('cpu')
    def to(self, device):
        if not device == self.device:
            self.device = device
            super().to(device)
    def _default_forward(self,obs,act,obs2,reward):
        delta_obs = obs2 - obs 
        x = torch.cat((obs,delta_obs,act,reward),dim = -1)
        m = self.shared_encoder(x)
        return m 

    def forward(self, x):
        y_distribution, z_distributions = self.encode(x)
        # TODO: could be more efficient if only compute the Gaussian layer of the y that we pick later
        return self.sample_z(y_distribution, z_distributions, y_usage="most_likely", sampler="mean")

    def encode(self, obs,act,obs2,reward):
        m = self._default_forward(obs,act,obs2,reward) # shape = (bz,n_support,shared_dim)
        if self.merge_mode == "add":
            y = self.class_encoder(m) # shape = (bz,n_support,n_class)
            y = y.sum(dim=-2) / y.shape[1]  # add the outcome of individual samples, scale down
        elif self.merge_mode == "add_softmax":
            y = self.class_encoder(m)
            y = F.softmax(y.sum(dim=-2), dim=-1)  # add the outcome of individual samples, softmax
        elif self.merge_mode == "multiply":
            y = self.class_encoder(m)
            y = F.softmax(y.prod(dim=-2), dim=-1)  # multiply the outcome of individual samples
        y_distribution = torch.distributions.categorical.Categorical(probs=y)
        all_mu_sigma = []
        for net in self.gauss_encoder_list:
            all_mu_sigma.append(net(m))
        z_distributions = [generate_gaussian(mu_sigma, self.latent_dim, mode='multiplication') for mu_sigma in all_mu_sigma]

        return y_distribution, z_distributions

    def sample_z(self, y_distribution, z_distributions, y_usage="specific", y=None, sampler="random"):
        # Select from which Gaussian to sample

        # Used for individual sampling when computing ELBO
        if y_usage == "specific":
            y = torch.ones(self.batch_size, dtype=torch.long,device=self.device) * y
        # Used while inference
        elif y_usage == "most_likely":
            y = torch.argmax(y_distribution.probs, dim=1)
        else:
            raise RuntimeError("Sampling strategy not specified correctly")

        mask = y.view(-1, 1).unsqueeze(2).repeat(1, 1, self.latent_dim)

        if sampler == "random":
            # Sample from specified Gaussian using reparametrization trick
            # this operation samples from each Gaussian for every class first
            # (diag_embed not possible for distributions), put it back to tensor with shape [class, batch, latent]
            sampled = torch.cat([torch.unsqueeze(z_distributions[i].rsample(), 0) for i in range(self.num_classes)], dim=0)

        elif sampler == "mean":
            sampled = torch.cat([torch.unsqueeze(z_distributions[i].mean, 0) for i in range(self.num_classes)], dim=0)

        # tensor with shape [batch, class, latent]
        permute = sampled.permute(1, 0, 2)
        z = torch.squeeze(torch.gather(permute, 1, mask), 1)
        return z, y

