# Uncertainty Estimation

you can see some implementations of uncertainty estimation

Gaussian process 
> [GP](https://github.com/kkugosu/Uncertainty-Estimation/blob/master/Docs/gaussian%20process.md)
The performance of which is best of all but estimation time infinitely incereases as data increase

Bayesian Neural Network
I implemented naive bnn and convolutional bnn
the performance of which is not like gaussian process but reduced estimation time

Mixture Density Network
Doesn't have noise assumption so we need more data but The most practical model so far

# Bayesian neural network

There are a few things we need to mention

# Noise assumption

In fitting problem, we don't need noise assumption, but when it comes to regression, we gradually train the model to make less error. 

Noise assumption make it possible to update gradually. so in bayesian neural network, we assume there is noise between output F and target Y.

# Update Parameter W

PRML 3.3 Bayesian Linear Regression formula (3.55)

We can express log posterior distribution as addition of log prior and log likelihood

$$ ln(p(w|t)) = -{\beta \over 2} \sum^N_{n=1} {t_n - w^T \phi (x_n)}^2 - {\alpha \over 2} w^2 w + const $$

We can get posterior distribution by subtracting derivative of log posterior with respect to Parameter W

Then W in likelihood term $w^T \phi (x_n)$ and prior term 
${\alpha \over 2} w^2 w$ should be derivated

In dropout as a bayesian approximation appendix, we have to maximize ELBO term.

$$ \int q(w) logp(Y|X,w)dw - KL(q(w)||p(w)) $$

In here, the W in likelihood term $logp(Y|X,w) $ and W in prior term $p(w) $ behave differentely.

W in fomula (3.55) is like $w_d = w_d + \alpha {d \over dw_d} f(x) $

In ELBO, formula is like finding $\Delta w$ to maximize $f(w + \Delta w)$ 

we already have the form of auxiliary variable $w + \Delta w $ and we don't have to get derivative form of ELBO.

so we make W in logp(Y|X,w) follows $w + \Delta w $, W in p(w) follows w

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
