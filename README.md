# bayesian_neural_network

prml 3.55

we can get log posterior distribution by adding log prior and log likelihood

$ lnp(w|t) = -{\beta \over 2} \sum_{n=1}^N {t_n - w^T \phi (x_n)}^2 - {\alpha \over 2} w^2 w + const $

and we can regress posterior distribution by subtracting derivative of log posterior with respect to "w"
then w in likelihood term $w^T \phi (x_n)$ and prior term ${\alpha \over 2} w^2 w$ have to be derivated

in dropout as a bayesian approximation appendix, we have to maximize elbo term

$$ \int q(w) logp(Y|X,w)dw - KL(q(w)||p(w)) $$


