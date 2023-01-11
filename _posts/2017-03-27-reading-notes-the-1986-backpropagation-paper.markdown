---
layout: article
title: "The 1986 Backpropagation Paper"
subtitle: "Learning Representations by Back-propagating Errors"
tags: deep-learning reading-notes
---
Deep learning is without doubt the hottest topic in both the academia and the industry at the moment, as it enables machines to recognize objects, translate speech or even play chess at human or super-human level. The workhorse behind the training of every neural network is backpropagation, where the weights of connections between neurons get adjusted to minimize the difference between the output of the neural network and the desired target. The idea of backpropagation came around in 1960 - 1970, but it wasn't until 1986 when it was formally introduced as the learning procedure to train neural networks. This is my reading notes of the famous 1986 paper in Nature [Learning Representations by Back-propagating Errors](https://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf) by Rumelhart, Hinton and Williams. 

<!--more-->

### Intro
- The aim is to find a synaptic modification rule that will allow an arbitrarily connected neural network to develop an internal structure for a particular task domain
- The learning rules are simple if the input units are directly connected to the output units, but becomes more interesting when hidden units whose states are not specificed by the task are introduced
- The learning rule must decide what these hidden units represent in order to achieve the desired input-output behavior

### Feed Forward
- Layered network with a layer of input units at the bottom, any number of hidden layers and a layer of output units at the top
- Connection within a layer or from higher to lower layers are forbidden, but connections may skip hidden layer
- The total input $x_j$ to unit $j$ is a linear function of the outputs $y_i$ of the units connected to $j$ with the weights
$$x_j = \sum_i y_i w_{ji}$$
- The output of each neuron is an non-linear function of its total input, here the paper uses the logistic function
$$y_j = \frac{1}{1 + e^{-x_j}}$$
- The total error in the performance is measured as squared loss summed over all the input-output pairs and output units 
$$E = \frac{1}{2} \sum_c \sum_j (y\_{j,c}-d\_{j,c})\^2$$
### Backpropagation
- To minimize $E$ by gradient descent, we need to compute the partial derivative of $E$ with respect to each weight in the network, which is the sum of the partial derivatives for each of the input-output cases
- The partial derivative is computed with the forward pass (feed forward) and the backward pass (backpropagation), which propagates derivatives from the top layer back to the bottom layer
- Compute $\partial E/\partial y$, the partial derivative of $E$ to the output of the output units
$$\frac{\partial E}{\partial y_j} = y_j - d_j$$
- Apply the chain rule to get $\partial E/\partial x_j$, the partial derivative of $E$ to the input of the output units, which represents **how a change in the total input $x$ will affect the error**
$$\frac{\partial E}{\partial x_j} = \frac{\partial E}{\partial y_j}\frac{\partial y_j}{\partial x_j} = \frac{\partial E}{\partial y_j} y_j(1-y_j)$$
- The total input $x$ is a linear function of the states of the lower level units and the weights of the connection. We can compute **how the error will be affected by a change in the weights**
$$\frac{\partial E}{\partial W\_{ji}} = \frac{\partial E}{\partial x\_j} \frac{\partial x\_j}{\partial W\_{ji}} = \frac{\partial E}{\partial x\_j} y\_i$$
- The contribution of the output of unit $i$ to $\partial E/\partial y_i$, resulting from the effect of $i$ on $j$ is
$$\frac{\partial E}{\partial x_j} \frac{\partial x_j}{\partial y_i} = \frac{\partial E}{\partial x_j} W_{ji}$$
- Taking into account of all the connections from unit $i$, we can get the **contribution of error from unit $i$**
$$\frac{\partial E}{\partial y_i} = \sum_j \frac{\partial E}{\partial x_j} W_{ji}$$

### Gradient Descent
- The paper brieftly stated that the gradient descent is not as efficient as methods using second derivative (**Note**: methods with Jacobian like Newton Method), but is much simpler and parallelizable
- The paper also mentioned the initiation of weights and suggested starting with small random weights to break summary (**Note**: this is still true in 2017, as we use [xavier initiation](http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization))
- The paper stated that the drawback of the learning procedure is that the error-surface may contain local minima so that gradient descend may not find the global minimum (**Note**: For almost 2 decades later, this was thought to be the case but the 2014 paper [Identifying and attacking the saddle point problem in high-dimensional non-convex optimization](https://arxiv.org/abs/1406.2572) proved that the problem is actually saddle points instead of local minima [xavier initiation](http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization))

### Comments
- Although I did not learn anything new with this paper, it was definitely fun seeing how influencial this paper was and how much of it has turned into "deep learning basics" in 2017, while they were cutting edge in 1986
- It is particularly interesting to see the intuitive observations made by the authors like small random initiation and local minima get their theoretical proofs decades later