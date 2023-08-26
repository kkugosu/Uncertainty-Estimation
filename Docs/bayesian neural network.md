# Bayesian Neural Network

## Contents

- [Noise Assumption](#noise-assumption)
- [Update Parameter W](#update-parameter-w)
- [Is It a Gaussian Process?](#is-it-a-gaussian-process)
- [Output Assumption](#do-we-need-to-assume-output-as-expected-value-of-hidden-units)
- [Other Changes and Assumptions](#some-other-change-other-assumption)
- [Conclusion](#conclusion)

---

## Noise Assumption

In the context of fitting problems, there's no need for a noise assumption. However, in regression, models are trained progressively to reduce error. This noise assumption enables a gradual model update. In Bayesian neural networks, it's assumed there's noise between the output `F` and the target `Y`.

## Update Parameter W

Referencing PRML 3.3 Bayesian Linear Regression formula (3.55):

We can express the log posterior distribution as the sum of log prior and log likelihood:

$$ ln(p(w|t)) = -{\beta \over 2} \sum^N_{n=1} {t_n - w^T \phi (x_n)}^2 - {\alpha \over 2} w^2 w + const $$

We derive the posterior distribution by differentiating the log posterior with respect to Parameter W. Here, \( w^T \phi (x_n) \) in the likelihood term and \( {\alpha \over 2} w^2 w \) in the prior term must be differentiated.

In the "Dropout as a Bayesian Approximation" appendix, the objective is to maximize the ELBO term:

$$ \int q(w) logp(Y|X,w)dw - KL(q(w)||p(w)) $$

Here, `W` in the likelihood term \( logp(Y|X,w) \) and `W` in the prior term \( p(w) \) have different behaviors.

## Is It a Gaussian Process?

This model's output isn't Gaussian. However, through the Central Limit Theorem (CLT), it can approximate a Gaussian process.

![Model](https://user-images.githubusercontent.com/24292848/172053919-81ed5d46-58ac-4c32-bbea-f8f7d90a0384.png)

Regardless of how \(W^2_j\) and \(W^1_i\) behave, if \(W^1_j\) are IID and `n` is large, the output approaches a Gaussian distribution. Yet, in this scenario, the model only captures the linear relationship between input and output due to the IID assumption. To capture non-linear relationships, additional linear layers or Bayesian linear layers are needed. With Lindeberg's CLT, \(W^1_j\) behaviors don't need adjustment, thus adding a layer isn't mandatory.

## Do We Need to Assume Output as Expected Value of Hidden Units?

In the dropout paper, the author defined the output as the mean value of the hidden layer units, a concept rooted in the CLT. But actually, instead of averaging, we can sum the values.

Kernel representations:

$$ \widehat{K} (x,y) = {1 \over K} \sum^K_{k=1} \sigma (w^T_k x + b_k) \sigma (w^T_k y + b_k)$$

And a potentially better representation:

$$ \widehat{K} (x_1,x_2)= \sum^K_{k=1} \sigma (w^T_k x_1 + b_k) \sigma (w^T_k x_2 + b_k)$$

Here, the notation `y` is swapped with `x` for clarity.

## Some Other Change, Other Assumption

This work doesn't employ dropout but rather uses "dropconnect", which fits better theoretically. Interestingly, dropconnect is often referred to as dropout. 

A significant statement is depicted in the following illustration:

![kld](https://user-images.githubusercontent.com/24292848/172194266-970c554a-c9fb-49aa-9f40-631a9e7ce684.jpeg)

To compute the above, the sigma(var) must be very small, as in the value 10^-33 used in this paper. If a model trained with dropout is also used with dropout, it's equivalent to learning a Gaussian process. For this work, a single Gaussian assumption is adopted for experimentation.

## Conclusion

Trimming the ELBO and removing constant terms results in:

$$ - \sum^N_{n=1} \tau || y_n - \widehat{y_n} ||^2_2 + \sum^Q \sum^K (\mu_{1,q,k} - \mu_{1,q,k} ')^2 + \sum^K \sum^D (\mu_{2,k,d} - \mu_{2,k,d} ')^2 + \sum^K (\mu_{b,k} - \mu_{b,k} ')^2$$
