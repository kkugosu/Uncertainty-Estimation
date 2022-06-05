# bayesian_neural_network

prml 3.55

we can get log posterior distribution by adding log prior and log likelihood

$$ lnp(w|t) = -{\beta \over 2} \sum^N_{n=1} {t_n - w^T \phi (x_n)}^2 - {\alpha \over 2} w^2 w + const $$

and we can regress posterior distribution by subtracting derivative of log posterior with respect to "w"

then w in likelihood term $w^T \phi (x_n)$ and prior term ${\alpha \over 2} w^2 w$ have to be derivated

in dropout as a bayesian approximation appendix, we have to maximize elbo term

$$ \int q(w) logp(Y|X,w)dw - KL(q(w)||p(w)) $$

in here the w in likelihood term $ logp(Y|X,w) $ and w in prior term $ p(w) $ behave differentely.

in 3.55 that formual is like $ w_d = w_d + \alpha {d \over dw_d} f(x) $

in elbo, formula is like find $ \Delta w $  to maximize $ f(w + \Delta w) $ 

we already have the form of auxiliary variable $ w + \Delta w $ and we don't have to get derivative form of elbo.

so we make w in $ logp(Y|X,w) $ follows $ w + \Delta w $ and w in $ KL(q(w)||"p(w)") $ follows w

