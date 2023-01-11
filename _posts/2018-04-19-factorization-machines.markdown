---
layout: article
title: "On Factorization Models"
subtitle: "Factorization Machines (FM) and Field-aware Factorization Machines (FFM)"
tags: reading-notes ctr
---

Recently, I have been working on the [TalkingData AdTracking Fraud Detection Challenge](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection). This is my first experience with CTR prediction, which is similar to NLP and Recommendation Systems in a way that features are very sparse. [Field-aware Factorization Models](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf) have been dominating the last few CTR prediction competitions on Kaggle so here is my a little write-up for [Field-aware Factorization Models](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf) and the its origin - [Factorization Model](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf).

<!--more-->

## 1. Factorization Machines (FM)

- The most common prediction task is to estimate a function $y: \mathcal{R}\^n \rightarrow T$ from a real valued feature vector $\textbf{x} \in \mathcal{R}\^n$ to a target domain $T$ ($T = \mathcal{R}$ for regression or $T = \\{+, -\\}$ for classification)
- Under sparsity, almost all of the elements in $\textbf{x}$ are zero. Huge sparsity appears in many real-world data like recommender systems or text analysis. One reason for huge sparsity is that the underlying problem deals with large categorical variable domains.
- The paper uses the following example of transaction data of a movie review system, with user $u \in U$ rates a movie $i \in I$ at a time $t \in \mathcal{R}$ with a rating $r = \\{1, 2, 3, 4, 5\\}$
![fm_example](https://s3-us-west-1.amazonaws.com/sijunhe-blog/plots/post15/fm_example.png)
- The above figure shows an example of how feature vectors can be created.
	- $|U|$ and $|I|$ binary indicators variables for the active user and item, respectively
	- Implicit indicators of all other movies the users has ever rated, normalized to sum up to 1
	- Binary indicators of the last movie rated before the current one

#### 1.1 Factorization Machine Model
- The equation for a factorization machine of degree $d = 2$
$$\hat{y}(\textbf{x}) = w\_0 + \sum\_{i=1}^n w\_i x\_i + \sum\_{i=1}^n \sum\_{j=i+1}^n \langle \textbf{v}\_i, \textbf{v}\_j \rangle x\_i x\_j \tag{1}$$
where $w\_0 \in \mathcal{R} \ \  \textbf{w} \in \mathcal{R}^n \ \ \textbf{V} \in \mathcal{R}^{n \times k}$

- A row $\textbf{v}\_i$ describes the *i*-th variable with *k*-factors. $k$ is a hyperparameter that defines the dimensionality of the factorization
- The 2-way FM captures all single and pairwise interactions between variables
	- $w\_0$ is the global bias, $w\_i$ models the strength of the i-th variable
	- $\hat{w}\_{i,j} = \langle \textbf{v}\_i, \textbf{v}\_j \rangle$ models the interaction between i-th and j-th variable. The FM models the interaction by factorizing it, which is the key point which allows high quality parameter estimates of higher-order interactions under sparsity

- **Expressiveness**: For any positive definite matrix $\textbf{W}$, there exists a matrix $\textbf{V}$ such that $W = \textbf{V}\textbf{V}\^T$ if $k$ is large enough. Thus, FM can express any interaction matrix $\textbf{W}$ if $k$ is chosen large enough. However, in sparse settings, typically a small $k$ should be chosen because there is not enough data to estimate complex interactions. **Restricting $k$ and the expressiveness of FM often lead to better generalization under sparsity.**

- **Parameter Estimation Under Sparsity**: FM can estimate interactions even in sparse settings well because they break the independence of the interaction parameters by factorizing them. This means that the data for one interaction helps also to estimate the parameters for related interactions.

- **Computation**: The complexity of straight forward computational of Eq. 1 is in $O\(kn\^2\)$ due to pairwise interactions. With the kernel trick, the model equation can be computed in linear time $O\(kn\)$

$$\begin{align}
\sum\_{i=1}^n \sum\_{j=i+1}^n \langle \textbf{v}\_i, \textbf{v}\_j \rangle x\_i x\_j &= \frac{1}{2} \left[ \sum\_{i=1}^n \sum\_{j=1}^n \langle \textbf{v}\_i, \textbf{v}\_j \rangle x\_i x\_j - \sum\_{i=1}^n \langle \textbf{v}\_i, \textbf{v}\_i \rangle x\_i\^2 \right] \\\\
&= \frac{1}{2} \left[ \sum\_{i=1}^n \sum\_{j=1}^n \sum\_{f=1}^k v\_{i,f} v\_{j,f}  x\_i x\_j - \sum\_{i=1}^n  \sum\_{f=1}^k v\_{i,f} v\_{i,f} x\_i\^2 \right] \\\\
&= \frac{1}{2} \sum\_{f=1}^k \left[ \left( \sum\_{i=1}^n v\_{i,f} x\_i \right)  \left( \sum\_{j=1}^n v\_{j,f} x\_j \right) - \sum\_{i=1}^n v\_{i,f}\^2 x\_i\^2 \right] \\\
&= \frac{1}{2} \sum\_{f=1}^k \left[ \left( \sum\_{i=1}^n v\_{i,f} x\_i \right)\^2 - \sum\_{i=1}^n v\_{i,f}\^2 x\_i\^2 \right]
\end{align}$$

#### 1.2 Loss Function of FM
- **Regression**: $\hat{y}(\textbf{x})$ can be used directly as the predicator and the loss is MSE
- **Binary Classification** the sign of $\hat{y}(\textbf{x})$ is used and the loss is the hinge loss or logit loss
- **Ranking**: the vectorr $\textbf{x}$ are odered by the score of $\hat{y}(\textbf{x})$ and loss is calculated over pairs of vectors with a pairwise classification loss

#### 1.3 Learning Factorization Machines
Since FM has a closed model equation, the model parameters $w\_0, \textbf{w}, \textbf{V}$ can be learned efficiently by gradient descent methods for a varity of loss.

$$\frac{\partial }{\partial \theta}\hat{y}(\textbf{x}) =
\begin{cases}
1,  & \theta = w\_0 \\\\
x\_i, & \theta = w\_i \\\\
x\_i \sum\_{i=1}^n v\_{i,f} x\_i - v\_{i,f} x\_i\^2, & \theta = v\_{i, f}
\end{cases}$$

#### 1.4 d-way FM
The 2-way FM can easily be generalized into a d-way FM and it can still be computed in linear time.

## 2. FM vs SVM
- The model equation of an SVm can be expressed as the dot product between the transformed input $\textbf{x}$ and model parameter $\textbf{w}$, $\hat{y}(\textbf{x}) = \langle \phi(\textbf{x}), \textbf{w} \rangle$, where $\phi$ is a mapping from the feature space $\mathcal{R}\^n$ to a more complex space $\mathcal{F}$. 
- We can define the kernel as $K(\textbf{x}, \textbf{z}) = \langle \phi(\textbf{x}), \phi(\textbf{z}) \rangle$

#### 2.1 SVM Model

##### Linear Kernel
- The linear kernel is $K(\textbf{x}, \textbf{z}) = 1 + \langle \textbf{x}, \textbf{z} \rangle$, which translate to the mapping $\theta(\textbf{x}) = [1, x\_1, \cdots, x\_n]$. 
- The model equation of a linear SVM can also be written as $\hat{y}(\textbf{x}) = w\_0 + \sum\_{i=1}^n w\_i x\_i$, which is identical to FM with degree $d = 1$

##### Polynomial Kernel
- The polynomial kernel  $K(\textbf{x}, \textbf{z}) = (1 + \langle \textbf{x}, \textbf{z} \rangle)\^d$ allow the SVM to model higher interactions between variables. 
- For $d = 2$, the polynomial SVMs can be written as
$$\hat{y}(\textbf{x}) = w\_0 + \sqrt{2}\sum\_{i=1}^n w\_i x\_i + \sum\_{i=1}^n w\_{i,i}\^{(2)} x\_i\^2 + \sqrt{2}\sum\_{i=1}^n \sum\_{j=i+1}^n w\_{i,j}\^{(2)} x\_i \tag{1}$$
- The main difference between a polynomial SVM (eq (2)) and the FM with degree $d = 2$ (eq (1)) is the parameterization: **all interaction parameters $w\_{i,j}$ of SVMs are completely independent**. In contrast to the this, **interaction parameters of FMs are factorized** and $\langle \textbf{v}\_i, \textbf{v}\_j \rangle$ and $\langle \textbf{v}\_i, \textbf{v}\_l \rangle$ dependent on each other

#### 2.2 Parameter Estimation Under Sparsity
- For very sparse problems in collaborative filtering settings, linear and polynomial SVMs fail. 
- This is primarily due to the fact that all interaction parameters of SVMs are independent. For a reliable estimate of the interaction parameter $w\_{i,j}$, there must be enough data points where $x\_i \neq 0$ and $x\_j \neq 0$. As soon as either $x\_i = 0$ or $x\_j$, the case $\textbf{x}$ cannot be used for estimating the parameter $w\_{i,j}$. If the data is too sprase, SVMs are likely to fail due to too few or even no cases for $(i,j)$

## 3. Field-aware Factorization Machine (FFM)
- A variant of FMs, field-aware factorization machiens (FFMs), have been outperforming existing models in a number of CTR-prediction competitions ([Avazu CTR Prediction](https://www.kaggle.com/c/avazu-ctr-prediction) and [Criteo CTR Prediction](https://www.kaggle.com/c/criteo-display-ad-challenge))
- The key idea in FFM is **field** and each feature has several latent vectors, depending on the field of other features.

| Clicked | Publisher (P) | Advertiser (A) | Gender (G) |
|:-------:|:-------------:|:--------------:|:----------:|
| Yes     | ESPN | Nike | Male |

- For the above example, FM will model it as (neglecting the bias terms),
$$\hat{y}\_{FM} = \langle v\_{ESPN}, v\_{Nike} \rangle + \langle v\_{ESPN}, v\_{Male} \rangle + \langle v\_{Male}, v\_{Nike} \rangle$$
- In FM, eveyr feature has only one latent vector to learn the latent effect with any other features. $v\_{ESPN}$ is used to learn the latent effect with Nike $\langle v\_{ESPN}, v\_{Nike} \rangle$ and Male $\langle v\_{ESPN}, v\_{Male} \rangle$. However, Nike and Male belong to different fields so the latent efects of (ESPN, Nike) and (ESPN, Male) may be different.
- In FFM, each feature has several latent vectors. Depending on the field of other features, one of them is used to do the inner product. For the same example, FFM will model it as
$$\hat{y}\_{FFM} = \langle v\_{ESPN,A}, v\_{Nike,P} \rangle + \langle v\_{ESPN,G}, v\_{Male,P} \rangle + \langle v\_{Male,G}, v\_{Nike,A} \rangle$$

- Neglecting the global bias terms, the model equation of degree $d = 2$ FM is defined as
$$\hat{y}(\textbf{x}) = \sum\_{i=1}^n \sum\_{j=i+1}^n \langle \textbf{v}\_i, \textbf{v}\_j \rangle x\_i x\_j \tag{3}$$
- the model equation of degree $d = 2$ FFM is defined as
$$\hat{y}(\textbf{x}) = \sum\_{i=1}^n \sum\_{j=i+1}^n \langle \textbf{v}\_i\^{(J)}, \textbf{v}\_j\^{(I)} \rangle x\_i x\_j \tag{4}$$
where $I$ and $J$ are the fields of $i, j$
- If $m$ is the number of fields, the total parameters of FFM is $mnk$ and the total parameter of FM is $nk$
- The time complexity of FFM is $O\(kn\^2\)$ and the time complexity of FM is $O\(kn\)$


