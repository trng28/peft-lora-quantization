from torch.distributions import kl_divergence

def inner_elbo(loglik, q_phi, p_phi, beta):
    kl = kl_divergence(q_phi, p_phi).mean()
    return -loglik + beta * kl

def outer_elbo(loglik, q_phi, p_phi, q_theta, p_theta, beta):
    kl_phi = kl_divergence(q_phi, p_phi).mean()
    kl_theta = kl_divergence(q_theta, p_theta).mean()
    return -loglik + beta * (kl_phi + kl_theta)
