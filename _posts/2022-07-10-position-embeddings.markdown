---
layout: article
title: "Encoding Positional Information"
subtitle: "A short Survey on Position Embeddings in Transformer models"
tags: reading-notes deep-learning nlp
---

A while ago, I [contributed](https://github.com/huggingface/transformers/pull/17776) a pytorch implementation of the [NEZHA](https://arxiv.org/abs/1909.00204) model to [huggingface/transformers](https://github.com/huggingface/transformers). While doing it, I became interested in how position embeddings evolved since the birth of Transformers model. And without further ado, here is a short literature review on position embeddings.
![pe](https://sijunhe-blog.s3.us-west-1.amazonaws.com/plots/post30/pe.png)


<!--more-->

## Intro

### Why position embedding?

Word position and order matter. While some classic models such as Bag of Words do not keep track of word orders, almost all models have a way of encoding order and position information. N-gram can be thought of as a way of keep tracking of local relative word orders. CNNs encode word orders through receptive fields. RNNs encode word orders through recurrence. Instead of recurrence and convolution, Transformer models rely entirely on attention mechanism, which has no inherent way of encoding position information. Hence the need for position embedding.

### Aboslute vs Relative

On a high level, position embedding can be divided into two broad category: **absolute** and **relative**.

**Absolute position embeddings** encode the absolute position of the *i*-th token as $f(i)​$. The intuition behind absolute position embedding it that as long as we featurize the absolute position, attention mechanism can model the relative relationship between tokens accordingly. Since the Absolute position embedding of a token only depends on the token itself, it is easily combined with the token embedding and usually part of the input layer.

**Relative position embeddings** encode the relative relation between the *i*-th and the *j*-th tokens as $f(i, j)​$. Compared with the absolute embedding, this is a more direct way of modeling the positonal relationship between two tokens. An important characteristics of relative positional embedding is that it allows us to extrapolate. If we format $f(i, j)$ as $g(j - i)$, then we have $f(i, j) = f(i + k, j + k)$.
Since relative position embeddings depends on the positions of two tokens, it is usually part of the attention module, which is where the relationships between tokens get computed.

### Functional vs Parametric

Another angle to separate different position embedding approaches is **functional representation** vs **parametric representation**.

The **parametric** approach treats position embeddings as treatable parameters, which means $f$ simply is a embedding lookup in either $f(i)$ or $f(i, j)$. Just like token embeddings, parametric position embeddings are trained with the rest of the model in an end-to-end fashion. One could argue that the parametric approach is expressive and has a high performance ceiling because everything is trained end-to-end. But for the same reason, it also potentially suffers from generalization issues, e.g. when certain positons/positon pairs were not in the training data, the performance would suffer.

Another approach is the **functional** approach, where a certain function is picked to represent the position information. Sinusoidal functions such as sine and cosine are popular choices. Compared with the data-driven approach above, the functional approach is like doing feature engineering, where we influence the model behavior with our own heuristics. Such human interpretation could be sub-optimal and limiting, but (with certain functions) it can allow generalization beyond what can be observed in the training data. Another upside here is the minor reduction of trainable parameters.

## Vanilla Transformers

[Attention is All You Need](https://arxiv.org/abs/1706.03762) by Vasawni et al. is easily one of the most influential ML/NLP papers in the last decade, as it proposed a novel attention-only mechanism for contextual modeling. Since attention is location-independent and has no way of tracking positions explicitly or implictly, the authors proposed using sinusoidal waves to inject position of the tokens in the sequence. 

$$PE\_(pos, 2i) = sin(\frac{pos}{10000\^{2i/d\_{model}}})$$
$$PE\_(pos, 2i+1) = cos(\frac{pos}{10000\^{2i/d\_{model}}})$$

![pe](https://sijunhe-blog.s3.us-west-1.amazonaws.com/plots/post30/pe.png)

This formulation of position embedding is the textbox **absolute functional** appproach. When I first read the paper, I remember I couldn't believe this work but obviously the empirical results suggest otherwise. After years of using Transformer models, I came up with a few hypotesis on why this works:

1. continuity: The sinusoidal functions' continuity allows the model to infer relative positions between two tokens. If two tokens are close in absolute positions (e.g. $PE(i, d)$ and $PE(i+1, d)$), the difference between their position embeedings will be small across all the channels (varies across the channels of course, but small nevertheless). This allows the model to infer proximity between tokens through simply taking the difference between the two position embeddings.
2. periodicity: If continuity is the only trait we are after, we can use something simpler like a linear function. But sinusoidal functions also offer periodicity, which means the position embeddings of two tokens could be close in value (on a channel) if the distance between them is close to the wavelength of the function. The authors designed the sinusodial functions with varying length to utilize the wavelength trait to model short and long relative distances. 

The authors also mentioned that they experimented with using learned positional embeddings (which is the **absolute parametric** approach) and found that the two versions produced nearly identical results.

## Transformer with Relative Position Embedding

Instead of encoding token positions as a function of their absolute positions, Shaw et al. proposed an **relative parametric** approach of encoding positions as a function of the distance between tokens in a subsequent paper [[2018 Shaw et al.]](https://arxiv.org/abs/1803.02155). Since relative attention requires the positions of two tokens *i* and *j*, Shaw et al. modifies the self attention module slightly by adding in two sets of relative position embeddings (colored in blue):

$$ z\_{i} = \sum_{j=1}^{n} \alpha\_{ij} (x\_{i} W\^{V} + \color{blue}{a\_{ij}\^{V}}) $$
$$ e\_{ij} = \frac{x\_{i} W\^{Q} (x\_{j} W\^{K} + \color{blue}{a\_{ij}\^{K}})}{\sqrt{d\_z}} $$

where $a\_{ij}^{K}$ and $ a\_{ij}^{V}$ are trainable embeddings that encodes the relative position between $i$ and $j$ when computing the softmax scores and the final output value respectively. However, buffering embeddings for all potential pairs of token positions would require $L\^2 * d\_{model}$ parameters ($L=512$ and $d\_{model}=768$ for BERT base), which is wasteful and relative position information beyond a certain distance is not useful anyway. Hence we can truncate the pairwise relative position embeddings as $a\_{ij}^{K} = W\_{max(j-i, k)}$ which limits each relative position embeddings matrix to $(2k+1) * d\_{model}$ (ablation study shows $k=64$ is a good number for en-de translation).

[[Huang et al. 2018]](https://arxiv.org/abs/2009.13658) continued this line of work by examining 4 different ways of incorporating relative position embeddings into self attention modules. As an example, one of the methods proposed calls for modeling the dot product of all possible pairs of query, key, and relative position embeddings:

$$ e\_{ij} = \frac{(x\_{i} W\^{Q}) (x\_{j} W\^{K}) + (x\_{i} W\^{Q}) * a\_{ij} + (x\_{j} W\^{K}) *a\_{ij} }{\sqrt{d\_z}} $$

This is a natural (albeit not very innovative) follow-up to Shaw et al.'s work but I think it unintentionally sparked the disentangled attention workstream such as DeBERTa [[2020 He et al.]](https://arxiv.org/abs/2006.03654).

## BERT

![bert](https://sijunhe-blog.s3.us-west-1.amazonaws.com/plots/post30/bert.jpeg)

Since it's publication, BERT [[2019, Devlin et al.]](https://arxiv.org/abs/1810.04805) has fundamentally changed the field of NLP/ML and made the pretraining - finetuning paradigm an industry standard. Yet its positional embeddings take the classic **absolute parametric** approach of training the position embedding end-to-end with the rest of the model, the very same method that the original Transformer paper tested on but didn't go with. Why? Other than empirical evidence (I am sure the team tried quite a few options here), here are my thoughts: 

1. BERT was trained on a large amount (by 2019 standard) of data in its pretraining stage, which means that the parametric approach could have a higher performance ceiling compared with the functional approach. The original Transformer didn't have such pretraining stage.
2. Most of the training in BERT happened in pretraining stage. The added training time due to the parametric approach should be inconsequential in the finetuning stage.
3. Other than token embeddings and position embeddings, BERT also added type embeddings for sentence pair tasks. The parametric approach is an elegant way to unify and combine the three embeddings to form the input layer.

## Nezha

A logical extension to all the previous work is to try the **relative functional** approach. And Wei et al. just did that in [NEZHA](https://arxiv.org/abs/1909.00204). As you can see from below, Nezha simply combines the sinusoidal functions used in the original Transformer paper and the relative attention idea introduced by Shaw et al.

$$a\_{ij}(2k) = sin(\frac{j - i}{10000\^{2k/d\_{z}}})$$
$$a\_{ij}(2k+1) = cos(\frac{j - i}{10000\^{2k/d\_{z}}})$$

Other than the results shown in the paper, I personally have had great experience pretrained Nezha models in Chinese span extraction tasks.

## RoFormer

The current state-of-the-art method is Rotary Position Embedding([RoPE](https://arxiv.org/abs/2104.09864)), introduced by Su et al. in 2021. The intuition is that the inner product between the embeddings of two tokens $p$ and $q$ and positions $m$ and $n$ should only be a function of $p$, $q$ and $m-n$. RoPE achieves that by representing the token embeddings as complex numbers and their positions as rotation angles. This way, changing the absolute positions of two tokens by the same amount doesn't change the angle between them. My Complex Math is quite rusty so I am skipping all the derivations, but the derived RoPE is the following:

$$RoPE(p, i) = pe^{mi\epsilon}$$

Implementation-wise, RoPE creates a block diagonal matrix with *cos* and *sin* functions along the axis to represent the rotation in complex space. In a sense, RoPE is very similar to the sinusoidal functions proposed by the original Transformer paper that except it multiplies instead of adding the position embeddings. It also follows the same principle of *represent relative position in an absolute manner*, which makes this a **absolute function** approach.

## Summary

|                | Absolute    | Relative                            |
|:--------------:|:-----------:|:-----------------------------------:|
| **Functional** | Transformer, RoFormer | Nezha                               |
| **Parametric** | BERT        | Transformer with Relative Attention |

The above table summarizes the post as the two axes of position embeddings (absolute vs relative, functional vs parametric) and the four quadrants of common position embedding approaches, for each we provided with at least one example. This is by no means comprehensive, as hybrid approaches do exist (e.g. [XLNet](https://arxiv.org/abs/1906.08237) uses both the functional and parametric approach).
