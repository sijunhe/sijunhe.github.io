---
layout: article
title: "On Normalization Layers"
subtitle: "Batch Normalization, Layer Normalization and Why They Work"
tags: reading-notes deep-learning nlp computer-vision
---

Reading notes / survey of three papers related to Batch Normalization

- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf), the paper that introduced Batch Normalization, one of the breakthroughs in Deep Learning
- [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf) that extended Batch Normalization to RNNs
- [How Does Batch Normalization Help Optimization?(No, It Is Not About Internal Covariate Shift)](https://arxiv.org/pdf/1805.11604.pdf), a paper (barely one week old at the time of writing) that dived into the fundamental factors for Batch Normalization's success empirically and theoretically

<!--more-->

##1. Why Normalization? 

#### Covariate Shift
Covariate Shift refers to the change in the distribution of the input variables $X$ between a source domain $\mathcal{s}$ and a target domain $\mathcal{t}$. We assume $P\_{\mathcal{s}}(Y|X) = P\_{\mathcal{t}}(Y|X)$ but a different marginal distribution $P\_{\mathcal{s}}(X) \neq P\_{\mathcal{t}}(X)$.

We are interested in modeling $P(Y|X)$. However, we can only observe $P\_{\mathcal{s}}(Y|X)$. The optimal model for source domain $\mathcal{s}$ will be different from the optimal model for target domain $\mathcal{t}$. The intuition, as shown in the diagram below, is that the optimal model for $P\_{\mathcal{s}}(X)$ will put more weights and perform better in dense area of $P\_{\mathcal{s}}(X)$, which is different from the dense area of $P\_{\mathcal{t}}(X)$.
![covariate_shift](https://s3-us-west-1.amazonaws.com/sijunhe-blog/plots/post16/covariate_shift.png =400x400)
*Covariate Shift Diagram [Source](http://iwann.ugr.es/2011/pdfInvitedTalk-FHerrera-IWANN11.pdf)*

#### Internal Covariate Shift (ICS)
In Neural Networks (NN), we face a similar situation like Covariate Shift. A layer $l$ in a vanilla feedforward NN can be defined as

$$X\^{l} = f\left(X\^{l-1}W\^{l} + b\^l \right) \ \ \ $$

where $X\^{l-1}$ is $m \times n\_{in}$  and $W\^{l}$ is $n\_{in} \times n\_{out}$. $m$ is the number of samples in the batch. $n\_{in}$ and $n\_{out}$ are the input and output feature dimension of the layer.

The weights $W\^{l}$ is learned to approximate $P\_{\mathcal{s}}(X\^{l}|X\^{l-1})$. However, the input from last layer $X\^{l-1}$ is constantly changing so $W\^{l}$ needs to continuously adapt to the new distribution of $X\^{l-1}$. [Ioffe et al.](https://arxiv.org/pdf/1502.03167.pdf) defined such change in the distributions of internal nodes of a deep network during training as **Internal Covariate Shift**.

##2. Batch Normalization (BN)
An obvious procedure to reduce ICS is to fix the input distribution to each layer. And that is exactly what Ioffe et al. proposed. Batch Normalization (BN) is a layer that normalizes each input feature to have mean of 0 and variance of 1. For a BN layer with $d$-dimensional input $X = (x\^{1}, \cdots, x\^{d})$, each feature is normalized as 

$$\hat{x}\^{(k)} = \frac{x\^{(k)} - \mu\_{x\^{(k)}}}{\sigma\_{x\^{(k)}}}$$

#### Mini Batch Statistics
Computing the mean and standard deviation of each feature requires iterating through the whole dataset, which is impractical. Thus, $\mu\_{x\^{(k)}}$ and $\sigma\_{x\^{(k)}}$ are estimated using the empirical samples from the current batch. 

![bn](https://s3-us-west-1.amazonaws.com/sijunhe-blog/plots/post16/bn.png =300x10)

#### Scale and Shift Parameters

To compensate for the loss of expressiveness due to normalization, a pair of parameters $\gamma\^{(k)}$ and $\beta\^{(k)}$ are trained to scale and shift the normalized value.

$$y\^{(k)} = \gamma\^{(k)} x\^{(k)} + \beta^{(k)}$$

The scale and shift parameters restore the representation power of the network. By setting $\beta^{(k)} = \mu\_{x\^{(k)}}$ and $\gamma\^{(k)} = \sigma\_{x\^{(k)}}$, the original activations could be recovered, if that were the optimal thing to do.

#### PyTorch Implementation

```python
class BatchNorm(nn.Module):
    def __init__(self, num_features, momentum=0.1, eps=1e-6):
        super(BatchNorm, self).__init__()
        self.num_features = num_features
        self.gain = nn.Parameter(torch.ones(num_features), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(num_features), requires_grad=True)
        self.momentum = momentum
        self.eps = eps
        self.initialized = False

    def forward(self, x):
        # x: [batch, num_feature, ?, ...]
        mean = torch.mean(x, dim=0, keepdim=True)  # [1, num_feature, ?, ...]
        var = torch.var(x, dim=0, unbiased=False, keepdim=True)  # [1, num_feature, ?, ...]
        if not self.initialized:
            self.register_buffer('running_mean', torch.zeros_like(mean))
            self.register_buffer('running_var', torch.ones_like(var))
            self.initialized = True
        if self.training:
            bn_init = (x - mean) / torch.sqrt(var + self.eps)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            bn_init = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        return self.gain * bn_init + self.bias
```

#### Batch Normalization in Feed-forward NN
Consider the *l*-th hidden layer in a feed-forward NN. The summed inputs are computed through a linear projection with the weight matrix $W\^l$ and the bottom-up inputs $X\^l$. The summed inputs are passed through a BN layer and then an activation layer (*whether to apply BN before or after the activation layer is a topic of debate*), as following:

$Z\^{l}$ is a 

$$Z\^{l} = X\^{l-1}W\^{l} + b\^l \ \ \ \ \
\hat{Z}\^{l} = \textbf{BN}\_{\gamma, \beta}(Z\^{l}) \ \ \ \ \ 
X\^{l} = f(\hat{Z}\^{l})$$



$Z\^{l}$ is a $m \times n\_{out}$ matrix, whose element $z\_{ij}$ is the summed input to the *j*-th neuron from the *i*-th sample in the mini-batch. 

$$ Z\^{l} = 
\begin{bmatrix}
z\_{11} & \cdots & z\_{1n\_{out}} \\\\ 
\vdots & \ddots & \vdots \\\\
z\_{m1} & \cdots & z\_{mn\_{out}}
\end{bmatrix} = 
\begin{bmatrix}
| & | & \cdots &| \\\\ 
\textbf{z}\_{1} & \textbf{z}\_{2} & \cdots & \textbf{z}\_{n\_{out}} \\\\
| & | & \cdots& | \\\\ 
\end{bmatrix} 
$$

$$\textbf{BN}\_{\gamma, \beta}(Z\^{l}) = 
\begin{bmatrix}
| & | & \cdots &| \\\\ 
\gamma\_1 \hat{\textbf{z}}\_{1} + \beta\_1 & \gamma\_2 \hat{\textbf{z}}\_{2} + \beta\_2 & \cdots & \gamma\_{n\_{out}} \hat{\textbf{z}}\_{n\_{out}} + \beta\_{n\_{out}} \\\\
| & | & \cdots& | \\\\ 
\end{bmatrix} 
$$

Column *j* of $Z\^{l}$ is the summed inputs to the *j*-th neuron from each *m* samples in the mini-batch. The BN layer is a **whitening / column-wise normalization** procedure to normalize $\left[ \textbf{z}\_{1}, \textbf{z}\_{2}, \cdots, \textbf{z}\_{n\_{out}}\right]$ to $\mathcal{N}(0,1)$. Each neuron/column has a pair of scale $\gamma$ and shift parameters $\beta$.




##3. Layer Normalization (LN)
BN has had a lot of success in Deep Learning, especially in Computer Vision due to its effect on CNNs. However, it also has a few shortcomings:

- BN replies on mini-batch statistics and is thus dependent on the mini-batch size. BN cannot be applied to to online learning tasks (batch size of 1) or tasks that require a small batch size.
- There is no elegant way to apply BN to RNNs. Applying BN to RNNs requires computing and storing batch statistics for each time step in a sequence.

To tackle the above issues, [Ba et al.](https://arxiv.org/pdf/1607.06450.pdf) proposed Layer Normalization(LN), a transpose of BN that computes the mean and variance used for normalization from all of the summed inputs to the neurons in a layer on a **single** training sample.

Using the same notation as above, we have $Z\^{l}$ is a $m \times n\_{out}$ matrix, whose element $z\_{ij}$ is the summed input to the *j*-th neuron from the *i*-th sample in the mini-batch. Row *i* of $Z\^{l}$ is the summed inputs to the all neuron in the *l*-th layer from the *i*-th sample in the mini-batch. As a direct transpose of BN, the LN layer is a **row-wise normalization** procedure to normalize $\left[ \textbf{z}\_{1}, \textbf{z}\_{2}, \cdots, \textbf{z}\_{m}\right]$ to have mean zero and standard deviation of one. Same as BN, each neuron is given its own adaptive bias and scale parameters.

$$ Z\^{l} = 
\begin{bmatrix}
z\_{11} & \cdots & z\_{1n\_{out}} \\\\ 
\vdots & \ddots & \vdots \\\\
z\_{m1} & \cdots & z\_{mn\_{out}}
\end{bmatrix} = 
\begin{bmatrix}
- & \textbf{z}\_{1} & - \\\\ 
- & \textbf{z}\_{2} & - \\\\ 
\cdots & \cdots & \cdots \\\\
- & \textbf{z}\_{m} & - \\\\ 
\end{bmatrix} 
$$

$$\textbf{BN}\_{\gamma, \beta}(Z\^{l}) = 
\begin{bmatrix}
- & \hat{\textbf{z}}\_{1} & - \\\\ 
- & \hat{\textbf{z}}\_{2} & - \\\\ 
\cdots & \cdots & \cdots \\\\
- & \hat{\textbf{z}}\_{m} & - \\\\ 
\end{bmatrix}
\circ
\begin{bmatrix}
| & | & \cdots &| \\\\ 
\gamma\_1 & \gamma\_2 & \cdots & \gamma\_{n\_{out}} \\\\
| & | & \cdots& | \\\\ 
\end{bmatrix} 
+ 
\begin{bmatrix}
| & | & \cdots &| \\\\ 
\beta\_1 & \beta\_2 & \cdots & \beta\_{n\_{out}} \\\\
| & | & \cdots& | \\\\ 
\end{bmatrix} 
$$

#### PyTorch Implementation

```python
class LayerNorm(nn.Module):
	def __init__(self, num_features, eps=1e-6):
	    super(LayerNorm, self).__init__()
	    self.gain = nn.Parameter(torch.ones(num_features), requires_grad=True)  # [num_features]
	    self.bias = nn.Parameter(torch.zeros(num_features), requires_grad=True)  # [num_features]
	    self.eps = eps

	def forward(self, x):
	    # [?, ..., num_features]
	    mean = x.mean(-1, keepdim=True)
	    std = x.std(-1, keepdim=True)
	    return self.gain * (x - mean) / (std + self.eps) + self.bias
```
 
#### Layer Normalization on RNN

In RNN, the summed input are computed from the current input $\textbf{x}\^t$ and previous hidden state $\textbf{h}\^{t-1}$ as 
$$\textbf{a}\^{(t)} = W\_{hh}\textbf{h}\^{(t-1)} + W\_{xh}\textbf{x}\^{(t)}$$. 

LN computes the layer-wise mean and standard deviation, then then re-centers and re-scales the activations
$$\boldsymbol{\mu}\^{(t)} =\frac{1}{H} \sum\_{i=1}\^H \textbf{a}\^{(t)} \ \ \ \ \ 
\boldsymbol{\sigma}\^{(t)} = \sqrt{\frac{1}{H} \sum\_{i=1}\^H (\textbf{a}\^{(t)} - \boldsymbol{\mu}\^{(t)})\^2 } \ \ \ \ \ 
\textbf{h}\^{(t)} = f \left( \frac{\boldsymbol{\gamma}}{\boldsymbol{\sigma}\^{(t)}} \circ \left( \textbf{a}\^{(t)} - \boldsymbol{\mu}\^{(t)}\right) + \boldsymbol{\beta} \right)
$$

LN provides the following benefits when applied to RNN:

- No need to compute and store separate running averages for each time step in a sequence because the normalization terms depend on only the current time-step.
- With LN, the normalization makes it invariant to re-scaling all of the summed inputs to a layer, which helps preventing exploding or vanishing gradients and results in much more stable hidden-to-hidden dynamics.

#### Invariance Properties of Normalizations

The below table shows the invariant properties of three different normalization procedures. **These invariance properties make the training of the network more robust**. Invariance to the scaling and shifting of weights means that proper weight initialization is not as important. Invariance to the scaling and shifting of data means that one bad (too big, too small, etc.) batch of input from the previous layer don't ruin the training of next layer.

![covariate_shift](https://s3-us-west-1.amazonaws.com/sijunhe-blog/plots/post16/invariance.png =800x400)

##4. Not ICS, But A Smoother Optimization Landscape?

Despite its pervasiveness, the effectiveness of BN still lacks theoretical proof. [Santurkar and Tsipras et al.](https://arxiv.org/pdf/1805.11604.pdf) recently proposed that **ICS has little to do with the success of BN**. Instead, BN **makes the optimization landscape much smoother**, which induces a more predictive and stable behavior of the gradients.

#### The performance of BN Doesn't Stem From reducing ICS

Santurkar and Tsipras et al. designed a clever experiment, where a network was trained with *random* noise (non-zero mean and non-unit variance distribution, changes at every time step) injected after BN layers, creating an artificial ICS. The performance of the network with "noisy" BN was compared with networks trained with and without BN. “Noisy” BN network has less stable distributions than the standard, no BN network due to the artificial ICS, yet it still performs better.

![BN_ICS](https://s3-us-west-1.amazonaws.com/sijunhe-blog/plots/post16/noisy_bn_ics.png =800x400)

#### BN doesn't even reduce ICS

Previously, ICS is a conception that has no measurement. Santurkar and Tsipras et al. defined a metric for ICS, which is difference ($||G\_{t,i} - G\_{t,i}\prime||\_2$) between the gradient $G\_{t,i}$ of the layer parameters and the same gradient $G\_{t,i}\prime$ **after** all the previous layers have been updated. Experiments showed that models with BN have similar, or even worse, ICS, despite performing better.

#### The Fundamental Phenomenon at Play: the Smoothing Effect

Santurkar and Tsipras et al. argued that the key impact of BN is that it reparametrizes the underlying optimization problem to **make its landscape significantly more smooth**. With BN,

- The loss landscape is smoother and has less discontinuity (i.e. kinks, sharp minima). The loss changes at a smaller rate and the magnitudes of the gradient is smaller too. In other words, the Lipschitzness of the loss function is improved. (a function f is *L*-Lipschitz, $|f(x\_1) - f(x\_2)| \leq L||x\_1 - x\_2||$)
- Improved Lipschitzness of the gradients gives us confidence that
when we take a larger step in a direction of a computed gradient, this gradient direction remains a fairly accurate estimate of the actual gradient direction after taking that step.
- The gradients are more stable and changes more reliably and predictively. In other words, the loss exhibits a significantly better “effective” $\beta$-smoothness. (a function f is $\beta$-smooth if its gradients are $\beta$-Lipschitz, i.e. $||\nabla f(x\_1) - \nabla f(x\_2)| \leq \beta||x\_1 - x\_2||$)
- Improved Lipschitzness of the gradients gives us confidence that
when we take a larger step in a direction of a computed gradient, this gradient direction remains a fairly accurate estimate of the actual gradient direction after taking that step.

![BN_ICS](https://s3-us-west-1.amazonaws.com/sijunhe-blog/plots/post16/bn_smooth.png =800x400)


