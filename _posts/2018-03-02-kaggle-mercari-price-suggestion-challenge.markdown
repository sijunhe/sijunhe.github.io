---
layout: article
title: "Mercari Price Suggestion"
subtitle: "Learning and Reflection from the Mercari Price Suggestion Challenge"
tags: kaggle nlp deep-learning
---

One of my first data science experience was with Kaggle more than two years ago when I played around with the [Titanic competition](https://sijunhe.github.io/blog/2015/10/31/kaggle-titanic-part-i/). While the competition itself was minimal, the experience was magical, intuitive and one of the reasons I got into data science. 

Two years later, I work full-time as a data scientist. One night, I decided to challenge myself on a whim and ended up spending all of my free time on this competition for two weeks.

**TL;DR I had my first serious Kaggle competition and ranked 65th out of 2384 teams (top 3%)**


![mecari](https://s3-us-west-1.amazonaws.com/sijunhe-blog/kaggle/mercari_rank.png)

<!--more-->

## 1. The Competition
The [Mercari Price Suggestion Challenge](https://www.kaggle.com/c/mercari-price-suggestion-challenge) was to build an algorithm that automatically suggests the right product prices with provided user-inputted text descriptions of their products, including details like product category name, brand name, and item condition. 

The most challenging part was that this was a **kernel-only** competition, which meant the training and inference of the model would be in a container environment provided by Kaggle. The script had to finish within 60 minutes and consumed no more than 16 GB of RAM. The test dataset was also quite large so that it couldn't be loaded into memory at once for inference, and batch inference was required. Overall, the idea of submitting a script to train a model and do inference on an unknown dataset was very unsettling, compared with a normal Kaggle competition where models are trained locally.

## 2. My Experience

### 2.1 Entering Late
I decided to seriously compete in this competition about a month before the deadline. One of the benefits of entering the competition late was that I could directly start from some strong public kernels and use them as my baselines. Due to my lack of creativity and my inexperience in Kaggle competitions, all of the models that I used in the final submissions originated from public kernels, including

- [RNN Baseline Model](https://www.kaggle.com/knowledgegrappler/a-simple-nn-solution-with-keras-0-48611-pl) for a RNN model that got me started
- [Ensemble of RNN and Ridge](https://www.kaggle.com/valkling/mercari-rnn-2ridge-models-with-notes-0-42755) for a ridge baseline model and ensembling methods in a kernel environment
- [Wordbatch FTRL and FM_FTRL](https://www.kaggle.com/anttip/wordbatch-ftrl-fm-lgb-lbl-0-42555) for teaching me [FTRL-Proximal](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41159.pdf), a very popular online algorithm

In addition to using public kernels as baseline models, I also benefited tremendously from 

- [EDA and Topic Modeling for Mercari](https://www.kaggle.com/thykhuely/mercari-interactive-eda-topic-modelling)
- [ELI5 for Mercari](https://www.kaggle.com/lopuhin/eli5-for-mercari) that introduced the [eli5](https://pypi.python.org/pypi/eli5), which is a fantastic library that helps explain and inspect black-box models

### 2.2 Base Models

#### 2.2.1 Ridge Regression

Linear model with L2 regularization on TF-IDF features. The main advantage of Ridge Regression is its speed (~ 10 minutes). One notable observation is that Ridge was a lot faster than LASSO since L2 loss is much easier to optimize compared with L1 loss.

#### 2.2.2 RNN

Recurrent Neural Network (RNN) is one of the strongest models in this competition. A lot of teams in the competition included a RNN model in their final Ensemble. The tricky part of RNN is that we need to consider **speed** when designing its architecture and tuning the hyper-parameter (especially batch size, and number of epochs). RNNs are the bread and butter for most Ensembles in this competition and usually takes 25 - 45 minutes. RNN was my main model but I failed to make a significant improvement over the RNNs from public kernels. To my surprise, none of the top teams used RNN in their final ensemble, probably due to slow training time.

#### 2.2.3 FTRL & FM_FTRL
Models using [Follow-the-Regularized-Leader(FTRL)](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41159.pdf) online learning algorithm became very popular for this competition, thanks to the wonderful [kernel](https://www.kaggle.com/anttip/wordbatch-ftrl-fm-lgb-lbl-0-42555) by **attip**, who is also the author of the package [Wordbatch](https://github.com/anttttti/Wordbatch). I have never heard or used FTRL before and was amazed by its speed and performance. Ridge Regression and the Linear model Proximal-FTRL is essentially the same model but optimized by different algorithms. It was fascinating how much of a difference the optimization algorithm made.

The FM_FTRL implementation in Wordbatch of Factorization Machines estimates linear effects with FTRL and factor effects with adaptive SGD. FM_FTRL is a very strong model for this competition since it turned out that capturing the interaction effect between some of the variables (item category, item condition) was key. FM_FTRL was also very fast (~ 10 minutes) which made it a very good base model to ensemble.

#### 2.2.4 MLP on Sparse Feature

MLP is definitely the largest winner of the competition, as most of the top teams chose it as their core base model. It not only ran blazingly fast, but also outperforms every other single model on this dataset. I think the main reason that MLP is so strong is it can capture complex feature interaction. This is something I should have realized when I saw the strong performance of FM_FTRL, which only captures pairwise interaction.

#### 2.2.5 LightGBM

Some [popular pubic kernels](https://www.kaggle.com/anttip/wordbatch-ftrl-fm-lgb-lbl-0-42555) used LightGBM on TF-IDF features as the main base model, which I didn't really understand. Despite I have never used LightGBM before at that time, my reasoning was that TF-IDF features are too high-dimensional and sparse for tree-based models, which lead to slows training and weak performance. Looking back at the published solutions, LightGBM is clearly not a mainstream model . I am glad to see a few teams achieved good results with LightGBM and I am more than happy to learn the their secret sauce from the [kernels](https://www.kaggle.com/peterhurford/lgb-and-fm-18th-place-0-40604/data). 


### 2.3 Ensembling
I have always heard that Kaggle competitions are essentially ensembling competitions. While that might be true for almost all Kaggle competitions, it's less so for the Mercari Price Suggestion Challenge. The time limit of 60 minutes and the memory limit of 16GB adds constraints to ensembling and competitors need to produce an ensemble that is not only accurate, but also fast and memory efficient. **The limits of kernels add a "production" flavor and transform the ensembling process from "shove as many things in it as we can" to "choose a small subset of models that blend well while complying with the constraints".** Due to the time constraints, most teams did not do stacking. Some attempts were made by keeping a small validation set and use it to find the optimal blending weights. 


Ensembling was definitely the biggest learning of the competition for me. It was my first time to seriously blend models together and I was amazed by the amount of performance gains from even a simple weighted average of diverse models. For this competition, I employed a rarely-used trick of ensembling using different checkpoints of a single RNN over time, inspired my [lecture notes](http://cs231n.github.io/neural-networks-3/#ensemble) from CS231 at Stanford. The time cost of this trick is only a few minutes but the performance gain was considerable (from `~0.429` to `~0.423`).

### 2.4 My final submissions
My [conservative solution](https://www.kaggle.com/sijunhe9248/rnn-ensemble-fm-ready-multithread?scriptVersionId=2482847) consists of a RNN trained for 3 epochs and a FM_FTRL, ran multi-threaded in Keras and scikit-learn. The public score is `0.41574` and private score is `0.41678`.

My [aggressive solution](https://www.kaggle.com/sijunhe9248/corrected-final-rnn-ridge) was a last minute Hail Mary, consists of the same RNN as the conservative solution and a Ridge Regression from a [kernel](https://www.kaggle.com/rumbok/ridge-lb-0-41944) that was made public a day before the deadline. The kernel caused major disruption to the leaderboard as many people were trying to incorporate it last minute. I was among them and a trivial bug in my hasty last-minute effort lead to a no score. Without the bug, the public score is `0.41170` and private score would be `0.41111`. **I would have ranked around 35th instead of 65th, though I still wouldn't get a Gold medal.**

## 3. Reflections

### 3.1 What I did well
- Ensembling using different checkpoints of a single RNN over time
- Focus on learning from and improving upon public kernels

### 3.2 Areas of Improvements
- I didn't try MLP. I should have gotten the hint from the strong performance of FM that feature interaction is key in this competition.
- The inability to improve my RNN meaningfully over the public kernel
- I need more discipline in hyper-parameter search (i.e. random grid search, [Bayesian Optimization](https://github.com/fmfn/BayesianOptimization)) since I don't have enough experience for a good intuition for hhyper-parameter tuning yet

### 3.3. Learning From Other Solutions

#### 3.3.1 1st Place Solution - MLP 

Pawel and Konstantin won this competition by a huge margin. Their [solution](https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s) is as graceful as one can be - a standard MLP neural network of 83 lines of code that ran in 1900 seconds (In comparison, my spaghetti-like Ensemble is more than 500 lines of code, took almost an entire hour and performed much worse ). Some of their key ideas are:

- **MLP**: Their model was a standard MLP ran on sparse feature (TF-IDF). Usually MLP overfits like crazy and I am very surprised it worked so well on this dataset. As Konstantin wrote 
> It has huge variance and each single model is tuned to overfit to a different local minima, but when averaged, they give a really good result. It seems to capture feature interactions which look really important here, and it's fast to train.

- **Parallelization**: Their kernel train 4 MLP models in parallel (one model per core) and average them to get the final results. The parallelization squeezed all the juice out of the 4-core Kaggle kernel. I think I need to learn how to wrote multiprocessing code in Python next.

- **Text Concatenation**: The only text preprocessing they did in the kernel was concatenate all text fields (name, item_description, category, brand) together to reduce the dimensionality of the text fields. This didn't make much sense to me theoretically, so I feel this is purely empirical and a testament to the amount of effort they put in this.

- **Doubling the batch size**: Since the training time is a key constraint in this competition, many of the top teams utilized the technique of increasing batch size instead of decaying the learning rate. As described in this [paper](https://arxiv.org/abs/1711.00489) by Google, increasing the batch size as opposed to decaying the learning rate gives equivalent test accuracies, but with fewer parameter updates, leading to greater parallelism and shorter training times.

#### 3.3.2 4th Place Solution - XNN

[Chenglong Chen](https://www.kaggle.com/chenglongchen) not only did extremely well in this competition, but also wrote a detailed [documentation](https://github.com/ChenglongChen/tensorflow-XNN) about his models. I admire the his modular design of the code which followed a [post](https://explosion.ai/blog/deep-learning-formula-nlp) by Matthew Honnibal to break all models into 4 components: **embed, encode, attend and predict**. Chen methodically experimented a wide range of options for each of the components:

- **Embed**: FastText, TextCNN, TextRNN, TextBiRNN, TextRCNN
- **Attend**: Average Pooling, Max Pooling, Context Attention, Self Attention
- **Predict**: NN-based FM, ResNet, MLP

Chen also ended up choosing MLP, like all other top teams. Another highlight was that he used a technique called [Snapshot Ensemble](https://openreview.net/pdf?id=BJYwwY9ll), which is similar to my idea of using different checkpoints of the same models but coupled with cyclic learning schedule. 

## 4. Summary

I had a great experience in my first Kaggle competition and I am quite happy with the result I had. What mattered much more than the result was the learning - I am amazed by how much I learned during the competition, may it be new models, ensembles or best practices. I am humbled by how smart the Kaggle community is and I think I am addicted! Hopefully I will improve in my next competition.



