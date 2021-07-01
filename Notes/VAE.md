# Variational Auto-Encoder
![](../Resources/VAE.png)

## Overview
Given a data point $x\in\mathcal{X}$, the encoder is a probabilistic distribution $p(z\mid x)$ where $z\in \mathcal{Z}$ is the latent variable drawn from the prior $p(z)$, encoding $x$ into a hidden representation $z$. The decoder is also a probabilistic distribution $p(x\mid z)$, which reconstructs $\hat{x}$ from the given $z$ so that $\hat{x}$ is maximally similar to $x$. The distribution $p(z)$ is natually regularized i.e. sum to one, so every value in the latent variable space $\mathcal{Z}$ is promising to be decoded to a reasonable output $\hat{x}$.
## Remark
- The original paper make assumptions that the **decoder** and the **prior** both belongs to the Gaussian Family, but in neural architechtures it is not made as is. We can use *Bert* for prior and *RNN* for devoder, then applying back-propagation to update the parameters. Also, the paper assumes the variational approximated encoder $q_x(z)$ is a Gaussian distribution with a diagnal covariance matrix (the variables are independent within features).
- **Reparameterization Trick**: cannot be implemented for discrete distributions
