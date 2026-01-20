"""
Bayesian priors for ABMLL algorithm.
""" 
import torch
from torch.distributions import Normal

def phi_prior(theta):
    mu, logvar = theta
    return Normal(mu, torch.exp(0.5 * logvar))

def theta_prior(dim):
    return Normal(torch.zeros(dim), torch.ones(dim))

