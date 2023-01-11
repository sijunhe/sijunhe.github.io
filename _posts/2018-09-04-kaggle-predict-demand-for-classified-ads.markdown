---
layout: article
title: "Kaggle: Learning and Reflection from the Avito Demand Prediction Challenge"
tags: kaggle nlp deep-learning
---

After three competitions, I felt I was ready to push for my first Gold Medal. Along with my four teammates, we took on the challenge of predicting demand for classified ads. We fought tooth and nail till the last moment and were in the Gold zone on the public leaderboard. While we ended up losing the Gold Metal narrowly by 1 spot, it was overall a great learning experience. I look forward to making progress toward a real gold medal in the future.

**TL;DR A failed attempt to push for my first Gold Metal on Kaggle. My team ranked top 1% but missed the Gold Metal narrowly by 1 spot!**


![avito](https://s3-us-west-1.amazonaws.com/sijunhe-blog/kaggle/avito_rank.png)

<!--more-->

## 1. The Competition
The [Avito Demand Prediction Challenge](https://www.kaggle.com/c/avito-demand-prediction) is sponsored by [Avito](https://www.avito.ru/rossiya), Russia’s largest classified advertisements website. The task is to predict demand for an online advertisement based on its full description (title, description, images, etc.), its context (geographically where it was posted, similar ads already posted) and historical demand for similar ads in similar contexts. 

## 2. Our Solution 

My teammates and I have put together a [write-up](https://www.kaggle.com/c/avito-demand-prediction/discussion/60059#350998) on the Kaggle discussion forum. 

An interesting characteristics of this competition is the Cross-Validation (CV) performance is very consistent with Leaderboard (LB) Performance. Therefore, all top teams ended up doing crazy stacking "towers" (from 3-level to even 6-level towers!). All of my time in the later stage of the competition is spent on the stacking tower below.

![stacking](https://s3-us-west-1.amazonaws.com/sijunhe-blog/plots/post17/stacking_tower.png)

## 3. Learning From Top Solutions

### [1st place solution](https://www.kaggle.com/c/avito-demand-prediction/discussion/59880)

Below is the architecture of Neural Network from the 1st place winner. The architecture is quite similar to [what we end up using](https://www.kaggle.com/c/avito-demand-prediction/discussion/59917).

![1st-place](https://s3-us-west-1.amazonaws.com/sijunhe-blog/plots/post17/1st_solution.jpg)



**What separates them from the rest of the crowd is that they found a "magic" feature from the unlabeled part of the dataset** (something I have been trying so hard throughout the competition but couldn't accomplish). 

> I’ve designed 3 nn models trained on active data to predict log1p(price), renewed and log1p(total_duration). These models had 2 rnn branches for title and description and also used category embeddings. The difference between actual log1p(price) and predicted one was an extremely important feature.                                   -- [Georgiy Danshchin](https://www.kaggle.com/gdanschin)

A well-deserved win!

### [2nd place solution](https://www.kaggle.com/c/avito-demand-prediction/discussion/59871)

Their best single model was a Neural Network at 0.2163 on public leaderboard (a big better than ours was at 0.2181). Some characteristics as follows:

- target encoded features for categorical features and their second and third order interactions
- cyclic LR, Nadam optimizer
- plenty of BNs, big dropouts

A whopping 6-level stack! And 10-fold Cross-Validation! They must have a lot of compute resource. 

### [3rd place solution](https://www.kaggle.com/c/avito-demand-prediction/discussion/59885)

The 3rd place winner is lead by the famous [KazAnova](https://www.kaggle.com/kazanova). They have a very interesting time-series ensemble in addition to normal ensembles:

> The second ensemble used a time-series schema. The test data begins a number of days past the final day included in the training data. We tried to replicate this in our internal validation. To do so, we trained on the first six days of training and used days ten through thirteen as validation. This meant days six through nine were excluded, mimicking the gap between train and test.Then, to generate predictions for the test data, we trained on the entire training set.

> In order to generate likelihoods of categorical features for this approach, we always applied a gap of 4 days. For example to estimate likelihoods for day four of the training data, we use the average of target for day zero. To estimate likelihoods for day five, we used likelihoods of (day0+day1)/2. We decided on a gap of four days for stability, as it gave similar CV and LB performance.

I remember trying something like this for the [TalkingData AdTracking Fraud Detection Challenge](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection) but it wasn't particularly strong. I am happy that somebody can pull this off.


