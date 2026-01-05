import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

class VariationalBayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, prior_log_sig2=0, log_sig2_init=-4.6):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        self.weight_mu = nn.Parameter(torch.empty((out_features, in_features)))
        self.weight_log_sig2 = nn.Parameter(torch.empty((out_features, in_features)))
        self.weight_mu_prior = nn.Parameter(torch.zeros((out_features, in_features)), requires_grad=False)
        self.weight_log_sig2_prior = nn.Parameter(prior_log_sig2 * torch.ones((out_features, in_features)), requires_grad=False)
        
        if self.has_bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias_mu", None)
        self.reset_parameters(log_sig2_init)

    def reset_parameters(self, log_sig2_init):
        init.kaiming_uniform_(self.weight_mu, a=math.sqrt(self.weight_mu.shape[1]))
        init.constant_(self.weight_log_sig2, log_sig2_init)
        if self.has_bias:
            init.zeros_(self.bias_mu)

    def forward(self, input):
        output_mu = F.linear(input, self.weight_mu, self.bias_mu)
        output_sig2 = F.linear(input.pow(2), self.weight_log_sig2.exp(), bias=None)
        return output_mu + output_sig2.sqrt() * torch.randn_like(output_sig2)

    def get_mean_var(self, input):
        mu = F.linear(input, self.weight_mu, self.bias_mu)
        sig2 = F.linear(input**2, self.weight_log_sig2.exp(), bias=None)
        return mu, sig2

    def kl_loss_informative(self, prior_layer):
        weight_mu_prior = prior_layer.weight_mu
        weight_log_sig2_prior = prior_layer.weight_log_sig2
        kl_weight = 0.5 * (weight_log_sig2_prior - self.weight_log_sig2 + 
                          (self.weight_log_sig2.exp() + (weight_mu_prior - self.weight_mu) ** 2) / 
                          weight_log_sig2_prior.exp() - 1.0)
        return kl_weight.sum(), len(self.weight_mu.view(-1))

def calculate_kl_terms_informative(model, priornet):
    kl, n = 0, 0
    for m, p in zip(model.modules(), priornet.modules()):
        if m.__class__.__name__.startswith("Variational"):
            kl_, n_ = m.kl_loss_informative(p)
            kl += kl_
            n += n_
    return kl, n

class ValueNetVBfull(nn.Module):
    def __init__(self, dim_obs, local_reparam=True, n_hidden=256):
        super().__init__()
        self.arch = nn.Sequential(
                VariationalBayesianLinear(dim_obs[0], n_hidden),
                nn.ReLU(inplace=True),
                VariationalBayesianLinear(n_hidden, n_hidden),
                nn.ReLU(inplace=True),
        )
        self.head = VariationalBayesianLinear(n_hidden, 1)

    def forward(self, x):
        h = self.arch(x)
        return self.head(h), self.head.get_mean_var(h)
    
class valueNetlastVB(nn.Module):
    def __init__(self, dim_obs, local_reparam=True, n_hidden=256):
        super().__init__()
        self.arch = nn.Sequential(
                nn.Linear(dim_obs[0], n_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(inplace=True),
        )
        self.head = VariationalBayesianLinear(n_hidden, 1)

    def forward(self, x):
        h = self.arch(x)
        return self.head(h), self.head.get_mean_var(h)