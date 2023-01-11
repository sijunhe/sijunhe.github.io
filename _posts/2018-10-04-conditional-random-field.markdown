---
layout: article
title: "Conditional Random Field"
subtitle: "Probabilistic Models for Segmenting and Labeling Sequence Data"
tags: nlp ml-basics
---

Conditional Random Field (CRF) is a [probabilistic graphical model](https://en.wikipedia.org/wiki/Graphical_model) that excels at modeling and labeling sequence data with wide applications in NLP, Computer Vision or even biological sequence modeling. In ICML 2011, it received "Test-of-Time" award for best 10-year paper, as time and hindsight proved it to be a seminal machine learning model. It is a shame that I didn't know much about CRF till now but better late than never!

Reading summaries of the following paper:

- Original paper: [Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data](https://dl.acm.org/citation.cfm?id=655813)
- Tutorial from original author of CRF: [Intro to Conditional Random Fields](https://people.cs.umass.edu/~mccallum/papers/crf-tutorial.pdf)
- Technique for confidence estimation for entities: [Confidence estimation for information extraction](https://dl.acm.org/citation.cfm?id=1614012)

![CRF](https://s3-us-west-1.amazonaws.com/sijunhe-blog/plots/post19/crf.png)

<!--more-->

# 1. Hidden Markov Model (HMM)

Hidden Markov Model (HMM) models a sequence of observations $X = \\{x\_t \\}\_{t=1}\^T$ by assuming that there is an underlying sequence of states (also called **hidden** states) $Y = \\{y\_t \\}\_{t=1}\^T$ drawn from a finite state $S$. HMM is powerful because it models many variables that are interdependent sequentially. Some typical tasks for HMM is modeling time-series data where observations close in time are related, or modeling natural languages where words close together are interdependent. 

In order to model the joint distribution $p(Y, X)$ tractably, HMM makes two strong independence assumptions:

- **Markov property**: each state $y\_t$ depends only on its immediate predecessor $y\_{t-1}$ and independent of all its ancestors $y\_{t-2}, \cdots y\_{1}$. 
- **Output independence**: each observation $x\_t$ depends only on the current state $y\_t$

With these assumptions, wen can model the joint probability of a state sequence $Y$ and an observation sequence $X$ as

$$p(Y, X) = \prod\_{t=1}\^T p(y\_t|y\_{t-1}) p(x\_t|y\_t)\tag{1}$$

where the initial state distribution $p(y\_1)$ is written as $p(y\_1|y\_0)$.


# 2. Generative vs Discriminative Models

**Generative models** learn a model of the joint probability $p(y,x)$ of the inputs $x$ and labels $y$. HMM is an generative model. Modeling the joint distribution is often difficult since it requires modeling the distribution $p(x)$, which can include complex dependencies. A solution is to use **discriminative models** to directly model the conditional distribution $p(y|x)$. With this approach, dependencies among the input variables $x$ do not need to be explicitly represented, affording the use of rich, global features of the input.

An interesting read about this topic is [On Discriminative & Generative Classifiers: A comparison of logistic regression and naive Bayes](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf) from the famous Prof. Andrew Ng back when he was a graduate student. A generative model and a discriminative model can form a **Generative-Discriminative pair** if they are in the same hypothesis space. For example, 

- if $p(x|y)$ is Gaussian and $p(y)$ is multinomial, then [Linear Discriminant Analysis](https://en.wikipedia.org/wiki/Linear_discriminant_analysis) and Logistic Regression models the same hypothesis space
- if $p(x|y)$ is Gaussian and $p(y)$ is binary, then [Gaussian Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Gaussian_naive_Bayes) has the same model form as Logistic Regression
- There is a discriminative analog to HMM, and it's the **linear-chain Conditional Random Field (CRF)**.

# 3. Linear-Chain Conditional Random Field

## From HMM to CRF

To motivate the comparison between HMM and CRF, we can re-write the Eq. (1) in a different form

$$p(Y, X) = \frac{1}{Z} \prod\_{t=1}\^T \text{exp} \left\\{ \sum\_{k = 1}\^K \lambda\_k f\_k\left( y\_t, y\_{t-1}, x\_t\right) \right\\}\tag{2}$$

The $K$ feature function $f\_k\left( y\_t, y\_{t-1}, x\_t\right)$ are a general form that takes into account of all state transitions probabilities and state-observation probabilities. There is one feature function $f\_{ij}( y, y', x) = \boldsymbol{1}\_{y =i} \boldsymbol{1}\_{y' =j}$ for each state transition pair $(i,j)$ and one feature function $f\_{io}( y, y', x) = \boldsymbol{1}\_{y=i} \boldsymbol{1}\_{x=0}$ for each state-observation pair $(i,o)$. Z is a normalization constant for the probability to sum to 1.

To turn the above into a linear-chain CRF, we need to write the conditional distribution 

$$
\begin{align}
p(Y|X) &= \frac{p(Y, X)}{\sum\_Y p(Y, X)} \\\\
&= \frac{\prod\_{t=1}\^T \text{exp} \left\\{ \sum\_{k = 1}\^K \lambda\_k f\_k\left( y\_t, y\_{t-1}, x\_t\right) \right\\} }{\sum\_{y'} \prod\_{t=1}\^T \text{exp} \left\\{ \sum\_{k = 1}\^K \lambda\_k f\_k\left( y'\_t, y'\_{t-1}, x\_t\right) \right\\}} \\\\
&= \frac{1}{Z(X)} \prod\_{t=1}\^T \text{exp} \left\\{ \sum\_{k = 1}\^K \lambda\_k f\_k\left( y\_t, y\_{t-1}, x\_t\right) \right\\}
\end{align} \tag{3}
$$

## Parameter Estimation 

Just like most other machine learning models, the parameter is estimated via Maximum Likelihood Estimation (MLE). The objective is to find the parameter that maximize the **conditional log likelihood** $l(\theta)$

$$
\begin{align}
l(\theta) &= \sum\_{i=1}\^N \ \text{log} p(y\^{(i)} | x\^{(i)}) \\\\
&= \sum\_{i=1}\^N \sum\_{t=1}\^T \sum\_{k=1}\^K \lambda\_k f\_k\left( y\_t\^{(i)}, y\_{t-1}\^{(i)}, x\_t\^{(i)}\right) - \sum\_{i=1}\^N  \text{log} Z(x\^{(i)})
\end{align} \tag{4}
$$

The objective function $\(\theta)$ cannot be maximized in closed form, so numerical optimization is needed. The partial derivative of Eq. (4) is

$$\frac{\partial l(\theta)}{\partial \lambda\_k} = \sum\_{i=1}\^N \sum\_{t=1}\^T f\_k\left( y\_t\^{(i)}, y\_{t-1}\^{(i)}, x\_t\^{(i)}\right) - \sum\_{i=1}\^N \sum\_{t=1}\^T \sum\_{y'\_{t-1}, y'\_{t}} f\_k\left( y'\_{t}, y'\_{t-1}, x\_t\^{(i)}\right) p(y'\_{t}, y'\_{t-1}| x\_t\^{(i)}) \tag{5}$$

which has the form of (observed counts of $f\_k$) - (expected counts of $f\_k$). To compute the gradient, inference is required to compute all the marginal edge distributions $p(y'\_{t}, y'\_{t-1}| x\_t\^{(i)})$. Since the quantities depend on $x\^{(i)}$, we need to run inference once for each training instance every time the likelihood is computed.

## Inference

Before we go over the typical inference tasks for CRF, let's define a shorthand for the weight on the transition from state $i$ to state $j$ when the current observation is $x$. 

$$
\begin{align}
\Psi\_t(j,i,x) &= p(y\_{t} = j | y\_{t-1} = i) \cdot p(x\_{t} = x |y\_{t} = j) \\\\
&= \left[ \delta\_{t}(i) \ \text{exp} \left( \sum\_{k = 1}\^K \lambda\_k f\_k\left( j, i, x\_{t+1}\right) \right) \right] 
\end{align} \tag{6}
$$


####  Most probable state sequences

The most needed inference task for CRF is to find the most likely series of states $Y\^{*} = \text{argmax}\_{Y} \ p(Y|X)$, given the observations. This can be computed by the [Viterbi recursion](https://en.wikipedia.org/wiki/Viterbi_algorithm). The Viterbi algorithm stores the probability of the most likely path at time $t$ that accounts for the first $t$ observations and ends in state $j$. 

$$\delta\_{t}(j) = \text{max}\_{i \in S} \ \delta\_{t-1}(i) \cdot \Psi\_t(j,i,x) \tag{7}$$

The recursive formula terminates in $p\^{*} = \text{argmax}\_{i \in S} \ \delta\_{T}(i)$. We can backtrack through the dynamic programming table to find the mostly probably state sequences.


####  Probability of an observed sequence

We can use Eq. (3) to compute the likelihood of an observed sequence $p(Y|X)$. While the numerator is easy to compute, the denominator $Z(X)$ is very difficult to compute since it contains an exponential number of terms. Luckily, there is another dynamic programming algorithms called [forward-backward](https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm) to compute it efficiently. 

The idea behind forward-backward is to compute and store two sets of variables, each of which is a vector with size as the number of states. The forward variables $\alpha\_t(j) = p(x\_1, \cdots, x\_t, y\_t = j)$ stores the probability of all the paths through the first $t$ observations and ends in state $j$. The backward variables $\beta\_t(i) = p(x\_t, \cdots, x\_T, y\_t = i)$ is the exact reverse and stores the probability of all the paths through the last $T-t$ observations with the *t*-th state as $i$

$$\alpha\_t(j) = \sum\_{i \in S} \Psi\_{t}(j, i, x\_t) \alpha\_{t-1}(i)\tag{8}$$
$$\beta\_t(i) = \sum\_{j \in S} \Psi\_{t+1}(j, i, x\_t) \beta\_{t+1}(j)\tag{9}$$

The initialization for the forward-backward is $\alpha\_1{j} = \Psi\_{t}(j, y\_0, x\_1)$ and $\beta\_T(i) = 1$. After the dynamic programming table is filled, we can compute $Z(X)$ as

$$Z(x) = \sum\_{i \in S} \alpha\_T(i)\tag{10}$$

Forward-backward algorithm is also used to compute all the marginal edge distributions $p(y\_{t}, y\_{t-1}| x\_t)$ in Eq. (5) that is needed for computing the gradient.

$$p(y\_{t}, y\_{t-1}| x\_t) = \alpha\_{t-1}(y\_{t-1}) \Psi\_t(y\_{t},y\_{t-1},x\_t) \beta\_t(y\_t)$$


#### Confidence in predicted labeling over a specific segment

Sometimes in task like Named Entity Recognition (NER), we are interested in the model's confidence in its predicted labeling over a segment of input to estimate the probability that a field is extracted correctly. This marginal probability $p(y\_t, y\_{t+1}, \cdots, y\_{t+k}|X)$ can be computed using constrained forward-backward algorithm, introduced by [Culotta and McCallum](https://dl.acm.org/citation.cfm?id=1614012).

The algorithm is an extension to the forward-backward we described above, but with added constraints such that each path must conforms to some sub-path of constraints $C = \\{ y\_t, y\_{t+1}, \cdots\\}$. $y\_t$ can either be a *positive* constraint (sequence must pass through $y\_t$) or a *negative* constraint (sequence must not pass through $y\_t$). In the context of NER, the constraints $C$ corresponds to an extracted field. The positive constraints specify the tokens labeled inside the field, and the negative field specify the field boundary. 

The constraints is a simple trick to shut off the probability of all paths that don't conform to the constraints. The calculation of the forward variables in Eq. (8) can be modified slightly to factor in the constraints

$$\alpha'\_t(j) = 
\begin{cases}
\sum\_{i \in S} \Psi\_{t}(j, i, x\_t) \alpha\_{t-1}(i),  & \text{if} \ j \ \text{conforms to} \ y\_{t} \\\\
0, & \text{otherwise}
\end{cases}$$

For time steps not constrained by $C$, Eq. (8) is used instead. Similar to Eq. (10), we calculate the probability of the set of all paths that conform to $C$ as $Z'(X) = \sum\_{i \in S} \alpha'\_T(i)$. The marginal probability can be computed by replacing $Z(X)$ with $Z'(X)$ in Eq. (3).



