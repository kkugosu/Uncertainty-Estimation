# bayesian_neural_network

in fitting problem, we don't have to use noise assumption, but in regression using noise assumption make problem tractable. so in bayesian neural network, we assume there is noise between output F and target Y.

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

by clt

<img width="559" alt="스크린샷 2022-06-05 오후 10 49 02" src="https://user-images.githubusercontent.com/24292848/172053919-81ed5d46-58ac-4c32-bbea-f8f7d90a0384.png">

no matter how $W^2_j$ $W^1_i$ behaves, if $ W^1_j $ are iid distributed and n goes infinitely, output follows gaussian distribution.

but, in that situation, that model only can learn linear relation between input and output data
because of iid assumption.

so we have to add one more linear layer or one more bayesian linear layer to learn nonlinear relationship.

but if we use lindeberg clt, we don't have to care about how $ W^1_j $ behave too. so we can impose non linear learning ability to this model without using additional layer
