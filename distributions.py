from collections import namedtuple
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

SphericalGaussianParams = namedtuple('SphericalGaussianParams', ('mu', 'logstd'))

class SphericalMultivariateNormal(MultivariateNormal):
    def __init__(self, mu, logstd, min_std_dev=0):
        self.mu = mu
        self.logstd = logstd
        self.min_std_dev = min_std_dev
        self.std = F.softplus(logstd) + self.min_std_dev
        MultivariateNormal.__init__(self, loc=self.mu, scale_tril=torch.diag_embed(self.std))

def standard_normal_like(dist, device):
    return SphericalMultivariateNormal(torch.zeros_like(dist.mu).to(device), torch.zeros_like(dist.logstd).to(device))

class GaussianHead(nn.Module):
    def __init__(self, hdim, zdim):
        super(GaussianHead, self).__init__()
        self.mu = nn.Linear(hdim, zdim)
        self.logstd = nn.Linear(hdim, zdim)

    def forward(self, x):
        return SphericalMultivariateNormal(mu=self.mu(x), logstd=self.logstd(x))

class GaussianHeadGlobalVar(nn.Module):
    def __init__(self, hdim, zdim, logstd=None):
        super(GaussianHeadGlobalVar, self).__init__()
        self.mu = nn.Linear(hdim, zdim)
        if logstd is not None:
            assert logstd.shape == (zdim,)
            self.logstd = logstd  # (zdim)
        else:
            self.logstd = nn.Parameter(torch.randn(zdim))

    def forward(self, x):
        # probably need to do some expand_as
        mu = self.mu(x)
        logstd = self.logstd.expand_as(mu)
        # return SphericalGaussianParams(mu=mu, logstd=logstd)
        return SphericalMultivariateNormal(mu=mu, logstd=logstd)
