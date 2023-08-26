
# Bayesian Neural Network (BNN)

Bayesian Neural Networks offer a probabilistic take on traditional neural network structures, integrating prior beliefs and uncertainties about model parameters directly into the network. This document touches on some critical aspects of Bayesian Neural Networks, including the noise assumption and parameter updating.

## Table of Contents
1. [Noise Assumption](#noise-assumption)
2. [Update Parameter W](#update-parameter-w)

---

## Noise Assumption

In most fitting problems, a noise assumption might not be necessary. However, regression problems demand a different approach. As we train the model, the objective is to reduce the error progressively.

The noise assumption in a BNN makes this gradual update feasible. Specifically, it posits that there exists some noise between the output \( F \) and the target \( Y \).

---

## Update Parameter W

Drawing from the Bayesian Linear Regression formula (3.55) in PRML 3.3:

The log posterior distribution can be described as the sum of the log prior and the log likelihood:

\[ ln(p(w|t)) = -{\beta \over 2} \sum^N_{n=1} {t_n - w^T \phi (x_n)}^2 - {\alpha \over 2} w^2 w + const \]

By taking the derivative of the log posterior concerning Parameter \( W \), we can determine the posterior distribution.

Here, the \( W \) in the likelihood term \( w^T \phi (x_n) \) and the prior term \( {\alpha \over 2} w^2 w \) should be derived.

When examining the "dropout as a Bayesian approximation" appendix, our aim is to maximize the ELBO term:

\[ \int q(w) logp(Y|X,w)dw - KL(q(w)||p(w)) \]

In this context, the \( W \) in the likelihood term \( logp(Y|X,w) \) and \( W \) in the prior term \( p(w) \) have distinct behaviors:

- In formula (3.55): \( w_d = w_d + \alpha {d \over dw_d} f(x) \)
  
- For the ELBO, the formula seems to identify a \( \Delta w \) to maximize \( f(w + \Delta w) \). Here, we have the auxiliary variable form \( w + \Delta w \) already, negating the need to derive the ELBO form.

Thus, while \( W \) in \( logp(Y|X,w) \) follows the \( w + \Delta w \) form, \( W \) in \( p(w) \) simply follows \( w \).


# Is that can be gaussian process?

The output never become gaussian so this model can't be gaussian process. But this model can approximate gaussian process by clt.

<img width="559" alt="스크린샷 2022-06-05 오후 10 49 02" src="https://user-images.githubusercontent.com/24292848/172053919-81ed5d46-58ac-4c32-bbea-f8f7d90a0384.png">

No matter how $W^2_j$ $W^1_i$ behaves, if $W^1_j $ are iid distributed and n goes infinitely, output follows gaussian distribution.

But, in that situation, that model only learn linear relation between input and output data
because of iid assumption.

So we have to add one more linear layer or one more bayesian linear layer to learn nonlinear relationship.

but if we use "lindeberg clt", we don't have to care about how $W^1_j $ behave too. so we can impose non linear learning ability to this model without adding layer

# Do we need to assume output as expected value of hidden units?

In the dropout paper, auther make output to be mean value of hidden layer units which is concept of clt. 

But actually we don't need to get mean value, we can just sum of it. 

this paper make kernel 

$$ \widehat{K} (x,y) = {1 \over K} \sum^K_{k=1} \sigma (w^T_k x + b_k) \sigma (w^T_k y + b_k)$$

But using below one becomes better

$$ \widehat{K} (x_1,x_2)= \sum^K_{k=1} \sigma (w^T_k x_1 + b_k) \sigma (w^T_k x_2 + b_k)$$

I just subtract K,

Change notation y as x to avoid confusing.

# Some other change, other assumption

We don't use dropout. We use dropconnect which is more theoretically fitted.

Wut we call dropconnect as dropout lol.

The entropy of a mixture of Gaussians with a large enough dimensionality and randomly distributed means approximate to the sum of the Gaussians’ volumes

this statements is this
![kld](https://user-images.githubusercontent.com/24292848/172194266-970c554a-c9fb-49aa-9f40-631a9e7ce684.jpeg)

To calculate this, we need to put sigma(var) as really small. This paper use 10^-33.

So if we train model with dropout and use model with dropout, then it is same as learning gaussian process.

But in this time, i will use single gaussian assumption to practice.

# Conclusion

If we trim the ELBO and subtract constant ($\sigma = 1, \tau$...) term,

Then we just have to maximize below

$$ - \sum^N_{n=1} \tau || y_n - \widehat{y_n} ||^2_2 + \sum^Q \sum^K (\mu_{1,q,k} - \mu_{1,q,k} ')^2 + \sum^K \sum^D (\mu_{2,k,d} - \mu_{2,k,d} ')^2 + \sum^K (\mu_{b,k} - \mu_{b,k} ')^2$$

In this project, i set sigma as 0.1
