import torch
from torch.distributions import Normal

from .priors import phi_prior, theta_prior
from .losses import inner_elbo, outer_elbo
from .likelihood import log_likelihood


class ABMLL:
    def __init__(
        self,
        inference_net,
        likelihood,
        phi_dim,
        beta=1.0,
        lr_inner=0.1,
        inner_steps=5,
    ):
        self.inference_net = inference_net
        self.likelihood = likelihood
        self.beta = beta
        self.lr_inner = lr_inner
        self.inner_steps = inner_steps

        # theta = {A, B}
        self.theta_mu = torch.zeros(phi_dim, requires_grad=True)
        self.theta_logvar = torch.zeros(phi_dim, requires_grad=True)

    @property
    def theta(self):
        return self.theta_mu, self.theta_logvar

    def train_epoch(self, tasks, optimizer):
        meta_loss = 0.0

        for D in tasks:
            q_phi = self.inference_net(D[0])
            phi = q_phi.rsample()

            # Inner loop
            for _ in range(self.inner_steps):
                loglik = log_likelihood(self.likelihood, D, phi)
                loss = inner_elbo(
                    loglik,
                    q_phi,
                    phi_prior(self.theta),
                    self.beta,
                )
                grad = torch.autograd.grad(loss, phi, create_graph=True)[0]
                phi = phi - self.lr_inner * grad

            # Outer loop 
            loglik = log_likelihood(self.likelihood, D, phi)
            q_theta = Normal(
                self.theta_mu,
                torch.exp(0.5 * self.theta_logvar),
            )

            loss = outer_elbo(
                loglik,
                self.inference_net(D[0]),
                phi_prior(self.theta),
                q_theta,
                theta_prior(phi.shape[-1]),
                self.beta,
            )
            meta_loss += loss

        meta_loss /= len(tasks)
        optimizer.zero_grad()
        meta_loss.backward()
        optimizer.step()

        return meta_loss.item()

    @torch.no_grad()
    def adapt(self, D):
        q_phi = self.inference_net(D[0])
        phi = q_phi.rsample()

        for _ in range(self.inner_steps):
            loglik = log_likelihood(self.likelihood, D, phi)
            loss = inner_elbo(
                loglik,
                q_phi,
                phi_prior(self.theta),
                self.beta,
            )
            phi = phi - self.lr_inner * torch.autograd.grad(loss, phi)[0]

        return phi
