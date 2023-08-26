
# Bayesian Neural Network (BNN)

Bayesian Neural Networks offer a probabilistic take on traditional neural network structures, integrating prior beliefs and uncertainties about model parameters directly into the network. This document touches on some critical aspects of Bayesian Neural Networks, including the noise assumption and parameter updating.

## Table of Contents
1. [Noise Assumption](#noise-assumption)
2. [Update Parameter W](#update-parameter-w)
3. [Gaussian Process Approximation](#Gaussian Process Approximation)
4. [Hidden Units Output Assumption](#hidden-units-output-assumption)
5. [Model Adjustments and Assumptions](#model-adjustments-and-assumptions)
6. [Conclusion](#conclusion)
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

Certainly, I've reformatted the content while ensuring the image file links remain unchanged:

---


## Gaussian Process Approximation

While this model's output doesn't conform to a Gaussian distribution, it can approximate a Gaussian process through the Central Limit Theorem (CLT).

![Model Visualization](https://user-images.githubusercontent.com/24292848/172053919-81ed5d46-58ac-4c32-bbea-f8f7d90a0384.png)

Regardless of the behavior of \(W^2_j\) \(W^1_i\), if \(W^1_j\) are identically and independently distributed and as \(n\) approaches infinity, the output adopts a Gaussian distribution. However, under these conditions, the model can only capture linear relations between inputs and outputs due to the IID assumption. To cater for non-linearity, we could introduce an additional linear or Bayesian linear layer. But with the Lindeberg CLT, we're free from the specifics of \(W^1_j\)'s behavior, enabling non-linear learning without adding extra layers.

## Hidden Units Output Assumption

The original dropout paper proposed modeling the output as the expected value of the hidden layer units, reflecting the CLT concept.

Contrary to this, acquiring the mean value isn't always necessary; a simple summation suffices. For instance, while the paper presents the kernel as:
\[ \widehat{K} (x,y) = {1 \over K} \sum^K_{k=1} \sigma (w^T_k x + b_k) \sigma (w^T_k y + b_k) \]
a more effective approach might be:
\[ \widehat{K} (x_1,x_2) = \sum^K_{k=1} \sigma (w^T_k x_1 + b_k) \sigma (w^T_k x_2 + b_k) \]
This is achieved simply by omitting \(K\), and changing the notation from \(y\) to \(x\) for clarity.

## Model Adjustments and Assumptions

Instead of dropout, we employ dropconnect for a better theoretical fit, though, ironically, we still term it as 'dropout'. A pertinent observation: the entropy of a mixture of Gaussians, with sufficiently large dimensionality and random mean distributions, approaches the collective volume of these Gaussians.

![Entropy Illustration](https://user-images.githubusercontent.com/24292848/172194266-970c554a-c9fb-49aa-9f40-631a9e7ce684.jpeg)

For effective calculation, the paper uses a significantly small sigma value (\(10^{-33}\)). Therefore, training with dropout and subsequently using the dropout model equates to learning via a Gaussian process. However, for this exploration, a single Gaussian assumption is preferred.

## Conclusion

By trimming the ELBO and excluding constant terms (like \(\sigma = 1, \tau...\)), our objective becomes the maximization of:

\[ - \sum^N_{n=1} \tau || y_n - \widehat{y_n} ||^2_2 + \sum^Q \sum^K (\mu_{1,q,k} - \mu_{1,q,k} ')^2 + \sum^K \sum^D (\mu_{2,k,d} - \mu_{2,k,d} ')^2 + \sum^K (\mu_{b,k} - \mu_{b,k} ')^2 \]

For this project, the set value for sigma is 0.1.

