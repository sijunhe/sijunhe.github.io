---
layout: post
title: "Reading: XGBoost"
subtitle: "XGBoost: A Scalable Tree Boosting System"
date: 2017-05-08 00:22:50 -0700
comments: true
published: false
categories: Traditional_ML Reading_Notes BigData
---

XGBoost is the most popular machine learning system on Kaggle and was used in 17 out of the 29 published challenge-winning solutions. This is my reading notes for [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/pdf/1603.02754.pdf) by Chen and Guestrin.

<!--more-->

### Introduction
Gradient Tree Boosting is a very popular technique that shines in many applications. XGBoost is a machine learning system for tree boosting that is faster than any existing boosting system on a single machine and can scale in distributed setting. The scalability of XGBoost is due to:

- A novel tree learning algorithm for handling sparse data
- A theoretically justified weighted quantile sketch procedure for efficient proposal calculation
- An effective cache-aware block structure for out-of-core computation

### Tree Boosting In a Nutshell

#### Regularized Learning Objective

For a given dataset with $n$ examples and $m$ features, a tree ensemble model uses $K$ additive functions to predict the output
$$\hat{y}\_i = \phi(x\_i) = \sum\_{k=1}^K f\_k(x\_i),  \ f\_k \in \mathcal{F}$$
where $\mathcal{F} = \\{ f(x) = w\_{q(x)} \\} (q: \mathcal{R}^m \rightarrow T, w \in \mathcal{R}^T)$ is the space of regression trees. $q$ is the tree structure with $T$ leaves and leaf weights $w$. 

To learn the set of functions used in the model, we minimize the **regularized** objective
$$\mathcal{L}(\phi) = \sum\_i l(\hat{y\_i}, y\_i) + \sum\_k \Omega(f\_k)$$
$$\Omega(f) = \gamma T\_k + \frac{1}{2} \lambda ||w||^2$$
The regularization term $\Omega$ penalizes the complexity of the trees and helps to smooth the final learnt weights. When the regularization parameter is set to 0, the objective becomes the traditional gradient tree boosting. 

#### Gradient Tree Boosting










