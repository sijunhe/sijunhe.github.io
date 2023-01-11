---
layout: article
title: "Deep Neural Networks for YouTube Recommendations"
subtitle: "Reading Notes on Awesome RecSys Papers II"
tags: deep-learning rec-sys reading-notes
---

![youtube_system](https://sijunhe-blog.s3-us-west-1.amazonaws.com/plots/post25/google_system.png)

<!--more-->

## System Overview

- Overall system is comprised of two neural networks:
	- Candidate generation (CG) network retrieves a small subset of videos that are generally applicable to user from a huge corpus. The feature used are coarse features
	- Ranking network distinguish the relative important among the candidates by assigning a score to each video according to a desired objective function. The network uses a rich set of features describing the video and the user
	- <span style="color:blue"> [Sijun]: This is the industry standard production recommendation system steup. It was probably novel when it was introduced in 2016 </span>

## Candidate Generation (CG)

### Problem Formulation
- Post recommendation as a extreme multiclass classification with the goal of accurately classifiying a specific video watch $w_t$ at time $t$ among millions of videos $i$ from a corpus $V$ based on a user $U$ and context $C$, where $u$ represents embedding of user, context pair and $v$ represents represent video embedding of the same dimension. 

$$P(w_t = i | U, C) = \frac{e^{v\_{i}u}}{\sum_{j \in V}e^{v\_{j}u}}$$

- To efficiently train the model with millions of classes, negative classes were sampled from background distribution and then corrected via important weighting. 
- At serving time, approximate nearest neighbor (ANN) in the dot product space were used to retrieve the most likely N classes. This is because the calibrated likelihoods from the softmax output layer is not needed.<span style="color:blue"> [Sijun]: This is a really smart decision  </span>

### Model Architecture
![cg_network](https://sijunhe-blog.s3-us-west-1.amazonaws.com/plots/post25/cg_network.png)

- A user's watch history is represented by a variable-length sequence of sparse video IDs, which is mapped to a dense vector represntationv ia the embeddings
- The embeddings are averaged to produce a fixed-size dense inputs and learned jointly with all other model parameters
- Other features are concaternated with the dense embeddings of watch history and searchi history at the first wide layer

### Labels and Context Selection