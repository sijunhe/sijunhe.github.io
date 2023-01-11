---
layout: article
title: "Kaggle: Learning and Reflection from the Jigsaw Toxic Comment Classification Challenge"
tags: kaggle nlp deep-learning
---

I had lots of fun at my last Kaggle competition [Mercari Price Suggestion Challenge](https://sijunhe.github.io/blog/2018/03/02/kaggle-mercari-price-suggestion-challenge/). Without a second thought, I dived right in the [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) to further practice my NLP skills. 

To get a different experience, I decided to team up instead of going solo. It turned out great, as I learned a ton from my teammates [Thomas](https://www.kaggle.com/learnmower), [Konrad](https://www.kaggle.com/konradb) and [Song](https://www.kaggle.com/newtohere), who have been doing this much longer than I have. Unknownly, I put myself in the best situation for learning - being the least experienced team member. 

**TL;DR The Jigsaw Toxic Comment Classification Challenge is the most nail-biting that I have participated in. I am estatic that my team ranked top 1% out of 4,500+ teams**


![toxic](https://s3-us-west-1.amazonaws.com/sijunhe-blog/kaggle/toxic_comment_rank.png)

<!--more-->

## 1. The Competition
The [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) is sponsored by Google and Jigsaw, with a purpose to improve online conversation. The task is to build a multi-headed model that’s capable of detecting different types of of toxicity like threats, obscenity, insults, and identity-based hate. The dataset consists of comments from Wikipedia’s talk page edits.

The competition isn't smooth sailing by any means. It had several major changes through its course:

- **Dataset Change**: since a portion of the data was previously released (a.k.a data leak), Jigsaw had to collect and label more data for a new test set, which turned out to be on a slightly different distribution from the training set. Therefore one of the biggest challenge of the competition is to find a reliable cross-validation strategy. The competition was also extended for 1 month to accommodate the new dataset. 
- **Metric Change**: the competition started with [log loss](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html) (also known as cross entropy) as the evaluation metric. However, log loss is not [scale-invariant](https://en.wikipedia.org/wiki/Scale_invariance) so teams were focusing on silly post-processing that improves the score. The host made a good call to switch to [ROC AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic), which is scale-invariant and makes more business sense. 

Since the corpus is relatively small and there is a difference in distribution between training and test set, the key to success is to **build a robust model that can generalize to the different distribution of the test set.** In retrospect, we found that all successful techniques in this competition specifically addressed the point above.

## 2. Our Experience

### 2.1 Pre-Processing
Since the corpus consists of online comments that are likely toxic, there is an abundance of intentional misspells (to avoid bad words filtering), typos, emojis and derogatory ascii art. Thomas and I wrote quite an exhaustive script trying to clean and normalize the corpus. Here are some examples:

```python
## cleaning smile emoji
text = re.sub("\(:", " smile ", text) 
## cleaning the penis asci art...
text = re.sub('8=+D', 'dick', text)
```
It turns out that the pre-processing was not impactful at all. All of the top teams have had the same experience as well. Our effort of a 400+ line pre-processing script went nowhere, but I am actually happy since I hate pre-processing/feature engineering and I don't believe they are part of the path to generalizable machine learning /AI.

### 2.2 Embeddings

Word embedding is a technique of representing the meaning of words by mapping them to a continuous high dimensional vector space. The most well-known embedding is [word2vec](https://www.tensorflow.org/tutorials/word2vec) by [Mikolov et al](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf). The embedding we used in the competition were [GloVe](https://nlp.stanford.edu/projects/glove/) (common crawl & twitter), [fastText](https://github.com/facebookresearch/fastText) and [LexVec](http://anthology.aclweb.org/P16-2068), though we were only able to re-train a small portion of models with different embeddings due to time constraint.

**Using a variety of embeddings turned out to be crucial in this competition**. In the context of [bias-variance decomposition](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff), there is an irreducible bias term due to the different distribution of training and test set. Hence, the best way to improve the model performance is to decrease the variance. Ensembling a wide range of diverse models is a great way to decrease the variance of the Ensemble. Alas, it was towards the end of the competition that Thomas found out training the same model with different embedding was a very effective way to produce performant yet diverse base models.

Out of all embeddings, [fastText](https://github.com/facebookresearch/fastText) gave the best performance overall. The capability of [generating word vectors for out-of-vocabulary words using subword information](https://arxiv.org/abs/1607.04606) was tremendously helpful for correcting misspells and typos.

### 2.3 Base Models

#### 2.3.1 NB-SVM
[NB-SVM](https://sijunhe.github.io/blog/2018/04/03/nb-svm/) is a widely-used baseline for text classification. It is based on bag-of-words approach and has a robust performance across tasks and datasets (important for this competition). NB-SVM was popular in this competition as a base model, thanks to this wonderful baseline [kernel](https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline) by [Jeremy Howard](https://www.kaggle.com/jhoward) with a public leaderboard score of `0.9770`. 

After some additional pre-processing and hyperparameter tuning, we were able to improve the model to `0.9813` on public leaderboard. The surprise came after the competition, when we found that our NB-SVM scored `0.9821` on the private leaderboard. This is impressive since

- **Generalization**: all other models of ours *overfitted* to the public leaderboard (higher public score than private score) and NB-SVM was the only exception.
- **Performance**: the performance of `0.9821` was very good for such a minimal linear model. In comparison, our best single model scored `0.9857` (+0.36) and the best single model of the competition was around `0.9869` (+0.48). NB-SVM is also much faster to train and a breeze to deploy compared with the other two so one could argue that the NB-SVM is a more viable solution.

#### 2.3.2 RNN

As expected in the NLP domain, Recurrent Neural Network (RNN) dominates this competition. Our strongest single model was a single layer RNN-Capsule Network with GRU cell at `0.9857` on the private leaderboard. Following closely is a single layer RNN with linear attention and GRU cell at `0.9856`. From what I have read, the best single models of all released top solutions are RNN-based.

Given that the challenge is on generalization, most RNN-based models I have seen lean on the simple side, with a single layer of RNN followed by some variant of an attention layer (i.e. global max pooling, global mean pooling, attention, etc) over the time axis. Also, we have observed that GRU outperforms LSTM in general, probably because GRU has less parameters and thus less prone to overfitting.

#### 2.3.3 CNN

There is a lot of arguments about whether RNN or CNN is better at NLP task. A [comparative study](https://arxiv.org/abs/1702.01923) suggested that CNN is good at extracting position-invariant features and RNN at modeling units in sequence. For me, CNN is a generalized n-gram feature extractor. I thought CNN would be strong at this competition since the task is more similar to key word detection than sentiment analysis. I was wrong.

I didn't have much success on CNN and I didn't see any CNNs that could compete with RNNs on public kernel. My best CNN is a [wide and shallow CNN](https://arxiv.org/abs/1408.5882) with a private leaderboard score of `0.9835`. The CNN got "feature selected" in our ensemble and contributed nothing. Though, the 2nd place team wrote that they had [DPCNN](http://ai.tencent.com/ailab/media/publications/ACL3-Brady.pdf) as one of their base models. I too have experimented with DPCNN during the competition but didn't have any success.

#### 2.3.4 Tree-Based Models

In theory, tree-based models aren't well-suited for NLP task, primarily due to high cardinality categorical features from huge word vocabulary space. Tree-based models sucked in practice as well. We experimented with Random Forest, Extra Tree, LightGBM and XGBoost and all of them severely underperformed other models. Our best attempt was a [Extra Tree model](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html) by Konrad with `0.9792` on private leaderboard. It is worth noting that despite their underwhelming performances, tree-based models still accounted for about 10% of the weight in our linear stacking ensemble.


### 2.4 Ensembling

[Stacking](http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/) (also called meta ensembling) is a model ensembling technique used to combine information from multiple predictive models to generate a new model. This competition was the first time I used [stacking](http://blog.kaggle.com/2017/06/15/stacking-made-easy-an-introduction-to-stacknet-by-competitions-grandmaster-marios-michailidis-kazanova/) and it was the most valuable learning experience of the competition. 

Since the data size is relatively small, the competition is the perfect candidate for stacking. We split the dataset into 10 folds, trained L1 models and produced out-of-fold predictions to the train L2 model. We had about 35 L1 models from a wide range of performances. The 2 L2 model we experimented with were XGBoost and LASSO. Their performance were comparable at `0.9871`. A simple average of the two stacking model gave us our best and final model at `0.9872`.

### 2.5 Selected Model Performance

| Model            | Private Leaderboard | Public Leaderboard | Overfitting Delta | 
| --------------   | :-------------: | :-------------: | :-------------: |
| Extra Tree         | 0.9792 | 0.9805 | -0.0013 |
| NB-SVM             | 0.9821 | 0.9813 | **+0.0008** |
| Shallow & Wide CNN | 0.9835 | 0.9846 | -0.0011 |
| GRU + Attention    | 0.9856 | 0.9864 | -0.0008 |
| GRU + Capsule Net  | 0.9857 | 0.9863 | -0.0006 |
| Lasso Stacking     | 0.9870 | 0.9874 | -0.0004 |
| XGBoost Stacking   | 0.9870 | 0.9873 | -0.0003 |
| **Average of Lasso & XGBoost Stacking**   | **0.9872** | **0.9875** | **-0.0003** |

## 3. Reflections

### 3.1 What We did well
- **Stacking**: Biggest learning of this competition. It also contributed a lot to our rank since our single model performance is lower than other teams around us.
- **Model Diversity**: We experimented with a wide range of models and helped with stacking. It was also nice to revisit all the different models.

### 3.2 Areas of Improvements
- **Single model performance**: Our single model performance is lower than top teams around us since we missed a few key techniques that other top teams used.
- **Team collaboration**: We literally had teammates all over the world (Netherlands, New York, Hawaii and California) and it was difficult to collaborate around the time difference. We never had a team call or a discussion when everybody were awake.

### 3.3. Learning From Top Solutions

#### 3.3.1 1st Place Solution 

The 1st place team had a very systematic approach to the problem with an scientific rigor that I truly admire. They even did an ablation analysis! An elaborate solution overview can be found [here](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52557). Their approach can be summarized as the following:

- **Diverse Embedding**: as described in the above embedding section
- **Train/Test Time Augmentation (TTA)**: They did data augmentation at both training and test time leveraging [translation](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/48038). Each input sentence was translated to French, German and Spanish and then back to English. At training time, the translations of a sentence stay in the same fold to avoid leakage. At test time, the prediction is made by averaging various translations. TTA had a big impact on the performance of their models, with the following ablation analysis

| Model            | Leaderboard Score |
| --------------   | :-------------: |
| Baseline Bi-GRU  | 0.9862 |
| + train-time augmentation      | 0.9867      |
| + test-time augmentation       | 0.9865      |
| + train/test-time augmentation | 0.9874      |

- **Pseudo Labeling**: Pseudo Labeling is a semi-supervised technique introduced in ICML 2013 by [Lee](http://deeplearning.net/wp-content/uploads/2013/03/pseudo_label_final.pdf) that improves generalization performance (just what we need for this competition) using unlabeled data. The procedure is to train a model on the training set, predict the "pseudo-labels" on the test set and treat the pseudo-labeled test set as part of the new training set. The method is similar to [Entropy Regularization](https://pdfs.semanticscholar.org/1ee2/7c66fabde8ffe90bd2f4ccee5835f8dedbb9.pdf). By minimizing the cross entropy for unlabeled data, the overlap of class probability distribution can be reduced and the decision boundary of the model becomes more accurate. The team reported a increase in leaderboard score from `0.9880` to `0.9885` with the Pseudo Labeling technique.

#### 3.3.2 3rd Place Solution 

The 3rd place team has the [strongest single model](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52644) I have read, with a private leaderboard score of `0.9869`. With the [embed-encode-attend-predict](https://explosion.ai/blog/deep-learning-formula-nlp) framework, it can be break down as following:

- **embed**: concatenated [fastText](https://github.com/facebookresearch/fastText) and [GloVe](https://nlp.stanford.edu/projects/glove/) twitter embeddings. The redundancy definitely helped
- **encode**: 2-layer RNN setting, with Bi-LSTM following by Bi-GRU. I also experimented brief with deeper RNN models during the competition but it didn't go anywhere
- **attend**: A concatenation of the last states, global maximum pool, global average pool and two features: "Unique words rate" and "Rate of all-caps words". Again, the redundancy helped
- **predict**: dense layer

The other interesting thing is that Alex trained with a larger batch size (512) but for many epochs (15 epochs). I did the exact opposite and trained with small batch size (64 or 128) for a small number of epochs (usually 4 - 6 epochs). I am not sure if the training schedule had any impact.

### 4. Summary

Another great experience and my best finish on Kaggle so far! Teaming up gave a entirely different experience and I learned a lot working with my teammates. While it was a pity that we didn't get a gold metal (we were very close with a difference of `-0.0002`), we did miss a few key techniques used by the gold-winning teams so we didn't really deserve it. The push for a gold medal and the kaggle master tier continues.