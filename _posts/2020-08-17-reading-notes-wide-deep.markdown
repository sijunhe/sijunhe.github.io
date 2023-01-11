---
layout: article
title: "Wide & Deep Learning for Recommendation Systems"
subtitle: "Reading Notes on Classic RecSys Papers I"
tags: deep-learning rec-sys reading-notes
---

Recently I started working on the applications of text representations in Recommendations. Since I don't have much background in RecSys other than the few lectures I took in graduate school, I will read some classic RecSys paper. Starting with [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792), the de facto standard baseline for industry recommender systems!

![wide_and_deep_model](https://sijunhe-blog.s3-us-west-1.amazonaws.com/plots/post25/wide_deep_model.png)


<!--more-->

## Wide & Deep Learning for Recommender Systems

- *Memorization* learns the frequent co-occurence of items and feature and exploits the correlation in the training data
- *Generalization* explores new feature combinations that have never or rarely occured in the past via transitivity of correlation
- Memorization requires in recommendations that are more directly relevant to user history, while generalization tends to improve the diversity of the recommended items <span style="color:blue"> [Sijun] this dichotomy sounds like explore/exploit </span>
- Memorization through cross-product transformation requires manually feature engineering and do not generalize to unseen feature pairs
- Embedding-based models can generalize to previously unseen feature pairs by learning a low-dimensional dense embedding for query and item. However, it is difficult to learn effective representation when the query-item is sparse and high-rank, such as users with specific preferences or niche itmes with a narrow appeal. In this case, linear models with cross-product features are far more efficient. <span style="color:blue"> [Sijun] this is an interesting point-of-view on embeddings in RecSys. I think this is more about matrix factorization than word2vec. Coming from NLP, I rarely think about if niche words are over-generalized. </span>


### Wide & Deep Learning

![wide_and_deep_model](https://sijunhe-blog.s3-us-west-1.amazonaws.com/plots/post25/wide_deep_model.png)

- Wide component is a GLM $y=\textbf{w}^{T}\textbf{x}+b$. One of the most important features is the cross-product transformation, defined as $\theta\_{k}(\textbf{x}) = \prod\_{i=1}^{d} x\_{i}^{c\_{ki}} \ \ \ c\_{ki} \in \{0, 1\}$
- Deep component is a feed-forward NN where sparse, high-dimensional categorical features are first converted into a embedding. The embedding are initialized randomly and trainable
- The wide and deep compoent are combined at the output layer, which is then fed to a logistici loss function for joint training
- The wide component is optimized by FTRL with L1 regularization while the deep part is optimized by AdaGrad. <span style="color:blue"> [Sijun]: The FTRL algorithm was originally designed to train Google-scale GLMs. It induces sparsity, which is needed in the wide part of the model, as most of the cross-product features are not meaningful. The deep part of the model is trained with AdaGrad, which is an regular adapative optimizer for NNs </span>

### System Implementation

#### Data
- Build vocabulary, which maps cateogrical feature to integer IDs during data generation. Categorical features are requried to have a minimal number of occurences to be included.
- Continuous features are normlalized to [0, 1] and divided into *n* quantiles. <span style="color:blue"> [Sijun]: I hypothesize they normalize to ensure equal feature scale and use quantization to reduce the noise in real-value features. They kept the quantized feature as a continuous feature instead of categorical feature to account for the ordinal nature. </span>

#### Model Training
![wide_and_deep_structure](https://sijunhe-blog.s3-us-west-1.amazonaws.com/plots/post25/deep_and_wide_structure.png)
- 32-dimensional vector is learned for each categorical feature. All embeddings along with dense features are concatenated and fed into a MLP with ReLu activation <span style="color:blue"> [Sijun]: I wonder if the dense features are under-utilized compared with the categorical features that are represented by embeddings </span>


