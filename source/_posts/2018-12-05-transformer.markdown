---
layout: post
title: "The Transformer"
subtitle: "A new paradigm of neural networks based entirely on Attention "
date: 2018-12-05 15:29:22 -0800
comments: true
categories: nlp deep-learning reading-notes
---

RNNs have been the state-of-the-art approach in modeling sequences. They align the symbol positions of the input and output sequences and generate a sequence of hidden states $h\_t$ as a function of previous hidden state $h\_{t-1}$ and the input for position $t$. This inherently sequential nature precludes parallelization . 

In the paper [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf), Google researchers proposed the **Transformer** model architecture that eschews recurrence and instead relies entirely on an attention mechanism to draw global dependencies between input and output. While it achieves state-of-the-art performances on Machine Translation, its application is much broader.

P.S. the [blog post](http://jalammar.github.io/illustrated-transformer/) by Jay Alammar has awesome illustration explaining the Transformer. the [blog post](http://nlp.seas.harvard.edu/2018/04/03/attention.html) on Harvard NLP also provides a working notebook type of explanation with some implementation.

<!--more-->

## Model Architecture

![transformer architecture](https://s3-us-west-1.amazonaws.com/sijunhe-blog/plots/post21/transformer_architecture.png)
*Model Architecture of Transformer [(Credit)](http://jalammar.github.io/images/t/transformer_resideual_layer_norm_3.png)*

The Transformer follows the Encoder-Decoder architecture. The encoder maps an input sequence $(x\_1, \cdots, x\_n)$ to a sequence of continuous representation $\textbf{z} = (z\_1, \cdots, z\_n)$. Given $\textbf{z}$, the decoder generates an output sequence $(y\_1, \cdots, y\_n)$ one step at a time. The model is auto-regressive, as the previous generated symbols are consumed as additional input at every step. The Transformer proposed several new mechanism that enabled abandoning recurrence, namely **multi-head self-attention**, **point-wise feed-forward networks** and **positional encoding**. 

### 1. Multi-Head Self-Attention

![attention](https://s3-us-west-1.amazonaws.com/sijunhe-blog/plots/post21/multi-head-attention.png)
*Scaled Dot-Product Attention and Multi-Head Attention [(Credit)](https://arxiv.org/pdf/1706.03762.pdf)*

#### Scaled Dot-Product Attention
Attention takes a query and a set of key-value pairs and output a weighted sum of the values. The Transformer uses the Scaled Dot-Product Attention, which takes the dot products of the query with all keys, divide each by $\sqrt{d\_k}$ and apply a softmax function to obtain the weights on the value. Dividing the dot products by $\sqrt{d\_k}$ prevents the its magnitude from getting too large and saturate the gradient on the softmax function.

$$\text{Attention}(Q,K,V) = \text{softmax} \( \frac{QK\^T}{\sqrt{d\_k}}\)V$$

#### Multi-Head Attention
The authors found it beneficial to linearly project the queries, keys and values $h$ times with different learned linear projections ($W\_i\^Q, W\_i\^K, W\_i\^V$). Attention is then performed in parallel on each of these projected versions. These are concatenated and once again projected by $W\^O$ and result in the final values. **Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.**

$$\text{MultiHead}(Q,K,V) = \text{Concat} \( \text{head}\_1, \text{head}\_h\)W\^O$$
$$\text{head}\_i = \text{Attention}\(QW\_i\^Q, KW\_i\^K, VW\_i\^V\)$$

#### Three types of Attention Mechanisms

- In the attention layers between an Encoder and a Decoder, the queries come from the decoder and the key-value pairs come from the encoder. This allow every position in the decoder to attention over all the positions in the encoder.
- The attention layers in the Encoders serve as self-attentions, where each position in the later encoder and attend to all positions in the previous layer of encoder.
- The attention layers in the Decoders are also self-attention layers. In order to prevent leftward information flow and preserve the auto-regressive property (*new output consumes previous outputs to the left, but not the other way around*), all values in the input that correspond to illegal connection are masked out as $-\infty$.

### 2. Position-wise Feed-forward Networks

Each of the Feed-Forward (also called Fully-Connected) layers is applied to each position separately and identically. This is similar to the idea of [TimeDistributed](https://www.tensorflow.org/api_docs/python/tf/keras/layers/TimeDistributed) wrapper in Tensorflow/Keras, where the same layer is applied to every temporal slice of an input.

### 3. Positional Encoding

In order for the model to make use of the order of the sequence, the paper introduces **positional encoding**, which encodes the relative or absolute position of the tokens in the sequence. The positional encoding is a vector that is added to each input embedding. They follow a specific pattern that helps the model determine the distance between different words in the sequence. For each dimension of the vector, the position of the token are encoded along with the sine/cosine functions.

$$\text{PE}\_{(pos,2i)} = sin(\frac{pos}{10000\^{2i/d\_{model}}}) \ \ \ \ \ \ \text{PE}\_{(pos,2i+1)} = cos(\frac{pos}{10000\^{2i/d\_{model}}})$$

The intuition is that each dimension corresponds to a sinusoid with wavelengths from $2\pi$ to $10000 \cdots 2\pi$ and it would allow the model to learn to attention by relative positions, since for any fixed offset $k$, $PE\_{pos+k}$ can be represented as a linear function of $PE\_{pos}$. As shown in the figure below, the earlier dimensions have smaller wavelengths and can capture short range offset, while the later dimensions can capture longer distance offset. *While I understand the intuition, I am quite doubtful about whether this really work.*

![sinusoid](https://s3-us-west-1.amazonaws.com/sijunhe-blog/plots/post21/sinusoid_positional_encoding.png)
*Examples of Sinusoid with different wavelengths for different dimensions [(Credit)](http://nlp.seas.harvard.edu/images/the-annotated-transformer_49_0.png)*

## Discussion on Self-Attention

The authors devoted a whole section of the paper to compare various aspects of self-attention to recurrent and convolutional layers on three criteria:

- **Complexity** is the total amount of computation needed per layer. 
- **Sequential Operations** is the minimum number of required sequential operations. These operations cannot be parallelized and thus largely determine the actual complexity of the layer.
- **Maximum Path Length** is the length of paths forward and backward signals have to traverse in the network. The shorter these path, the easier it is to learn long-range dependencies.

![self_attention_complexity](https://s3-us-west-1.amazonaws.com/sijunhe-blog/plots/post21/self-attention-complexity.png)
*Complexity Comparison of Self Attention, Convolutional and Recurrent Layer. [(Credit)](https://arxiv.org/pdf/1706.03762.pdf)*

In the above table, $n$ is the sequence length, $d$ is the representation dimension, $k$ is the kernel size of convolution and $r$ is the size of the neighborhood in restricted self-attention.

#### Recurrent Layer
Computing each recurrent step takes $O(d\^2)$ for matrix multiplication. Stepping through the entire sequences of length $n$ takes a total computation complexity of $O(nd\^2)$. The Sequential Operations and Maximum Path Length are $O(n)$ due to the sequential nature.

#### Convolutional Layer
Assuming the output feature map is $n$ by $d$, each 1D convolution takes $O(k \cdot d)$ operation, making the total complexity $O(k \cdot n \cdot d\^2)$. Since convolution is fully parallelizable, the Sequential Operations is $O(1)$. A single convolutional layer with kernel width $k < n$ does not connect all pairs of input and positions, thus requiring a stack of $O(n/k)$ contiguous kernels, or $O(log\_k(n))$ in case of dilated convolutions.

#### Self-Attention Layer
Computing the dot product between the representations of two positions take $O(d)$. Computing the attention for all pairs of positions takes $O(n\^2d)$. The compute is parallelizable, the Sequential Operations is $O(1)$. The self-attention layer connects all positions with a constant number of operations since there is a direct connection between any two positions in input and output. $O(1)$

#### Self-Attention Layer with Restriction
To improve the computational performance for tasks involving very long sequences, self-attention can be restricted to considering only a neighborhood size of $r$ centered around the respective position. This decreases the total complexity to $O(r\cdot n \cdot d)$, though it takes $O(n/r)$ operations to cover the maximum path length.

## Training

- **Training time**: full model takes 3.5 days to train on 8 NVIDIA P100 GPUs. :0
- **Optimizer**: Adam. Increase learning rate for the first *warmup_steps* training steps and decrease it thereafter. Similar [cyclic learning rate](https://arxiv.org/abs/1506.01186) or the Slanted Triangular LR in [ULMFiT](https://arxiv.org/pdf/1801.06146.pdf)
- **Regularization**: 
	- Residual dropout: apply dropout to output layer before the residual connection and layer normalization $P\_{drop} = 0.1$
	- Embedding dropout: apply dropout to the sums of the embeddings and the positional encodings $P\_{drop} = 0.1$
	- [Label Smoothing](https://arxiv.org/pdf/1512.00567.pdf) of $\epsilon\_{ls} =0.1$. This hurts perplexity but improves BLEU score.