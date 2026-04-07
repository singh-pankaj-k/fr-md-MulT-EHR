"""
Bayesian Graph Convolutional Layer
"""

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch_geometric.nn import GCNConv

from losses import calculate_kl
from utils import get_device


class BBBGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, allow_zero_in_degree=False, activation=None, priors=None):

        super(BBBGraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = 1
        self.use_bias = bias
        self.allow_zero_in_degree = allow_zero_in_degree
        self.activation = activation
        self.device = get_device()

        if priors is None:
            priors = {
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (0, 0.1),
                'posterior_rho_initial': (-3, 0.1),
            }
        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        self.W_mu = Parameter(torch.empty((in_channels, out_channels), device=self.device))
        self.W_rho = Parameter(torch.empty((in_channels, out_channels), device=self.device))

        if self.use_bias:
            self.bias_mu = Parameter(torch.empty((out_channels), device=self.device))
            self.bias_rho = Parameter(torch.empty((out_channels), device=self.device))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        # Define frequentist graph convolutional layer
        self.graph_conv = GCNConv(
            in_channels=in_channels,
            out_channels=out_channels,
            normalize=True,
            add_self_loops=True,
            bias=False  # Use external bias
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)

        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)

    def forward(self, edge_index, feat, sample=True):
        if self.training or sample:
            W_eps = torch.empty(self.W_mu.size()).normal_(0, 1).to(self.device)
            self.W_sigma = torch.log1p(torch.exp(self.W_rho))
            weight = self.W_mu + W_eps * self.W_sigma

            if self.use_bias:
                bias_eps = torch.empty(self.bias_mu.size()).normal_(0, 1).to(self.device)
                self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
                bias = self.bias_mu + bias_eps * self.bias_sigma
            else:
                bias = None
        else:
            weight = self.W_mu
            bias = self.bias_mu if self.use_bias else None

        # Apply weight manually for GCNConv if needed, or use a custom message passing
        # PyG's GCNConv usually applies weight internally. To use our custom weight:
        # We can use message passing directly or multiply feat by weight first
        
        x = torch.matmul(feat, weight)
        out = self.graph_conv(x, edge_index)

        if bias is not None:
            out += bias

        if self.activation is not None:
            out = self.activation(out)

        return out

    def kl_loss(self):
        """
        KL loss of the prior distribution and the current weights
        :return: The KL loss
        """
        kl = calculate_kl(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.use_bias:
            kl += calculate_kl(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl