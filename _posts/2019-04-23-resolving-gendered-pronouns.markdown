---
layout: article
title: "Kaggle: Learning from the Gendered Pronoun Resolution Challenge"
tags: kaggle nlp deep-learning
---

With the goal to learning PyTorch and getting more hands-on experience with transfer learning via [pre-trained language models](https://sijunhe.github.io/blog/2019/01/20/transfer-learning-in-nlp/), I took part in the [Gendered Pronoun Resolution Competition](https://www.kaggle.com/c/gendered-pronoun-resolution) on Kaggle. The learning alone was quite worth it. And **I placed 30th solo out of 800+ teams with limited time invested**.

![gpr](https://s3-us-west-1.amazonaws.com/sijunhe-blog/kaggle/pronoun_resolution.png)

<!--more-->

## Intro

I entered the [Gendered Pronoun Resolution Competition](https://www.kaggle.com/c/gendered-pronoun-resolution) on Kaggle with two goals in mind:

1. **Learn PyTorch**: PyTorch has been giving Tensorflow a run for its run since its 1.0 launch. With many popular packages like [Hugging Face](https://github.com/huggingface), [fast.ai](https://github.com/fastai/fastai), [allennlp]() built on top of it, PyTorch is becoming the standard in Machine Learning research.
2. **Get experience with Language Model Finetuning**: transfer learning with pre-trained langauge models like BERT has been very popular in the NLP community and I haven't had much experience using them in a project.

## Model

I used the edge probing model architecture from this [ICLR 2019 paper](https://openreview.net/forum?id=SJzSgnRcKX). I experimented with OpenAI GPT, ELMo, BERT base and large as the pre-trained encoders. 

![gpr_model](https://s3-us-west-1.amazonaws.com/sijunhe-blog/plots/post23/gpr_model.png)

- **Input Tokens**: the respective tokenizer of the pre-trained language model were used
- **Span Representation**: extracting span representation with[allennlp.modules.span_extractors](https://allenai.github.io/allennlp-docs/api/allennlp.modules.span_extractors.html). I experimented with both [EndpointSpanExtractor](https://allenai.github.io/allennlp-docs/api/allennlp.modules.span_extractors.html#allennlp.modules.span_extractors.endpoint_span_extractor.EndpointSpanExtractor) and [SelfAttentiveSpanExtractor](https://allenai.github.io/allennlp-docs/api/allennlp.modules.span_extractors.html#allennlp.modules.span_extractors.self_attentive_span_extractor.SelfAttentiveSpanExtractor) but there doesn't seem to be much of a difference.
- **MLP**: very simple 1-layer MLP from the size of the concatenated span representation to the number of classes(3). A dropout of 0.1 is applied, as suggest from the BERT paper.

## Results & Analysis

In order to properly benchmark the performance of the fine-tuned pre-trained language models, I built two baseline models without transfer learning.

- **Baseline with random initialization**: Replace the pre-trained encoder in the edge probing model with a word embedding and a Bidirectional LSTM. The word embedding is initialized randomly and trainable
- **Baseline with Glove**: Same model as the **Baseline with random initialization**, except using [Glove](https://nlp.stanford.edu/projects/glove/) to initialize the word embeddings

| pre-trained LM      | log-loss |
| --------------      |:--------:|
| Baseline-Random Init| 0.7647   |
| Baseline-Glove      | 0.7328   |
| ELMo                | 0.6612   |
| OpenAI GPT          | 0.6170   |
| Bert-base-cased     | 0.4310   |
| Bert-base-uncased   | 0.4221   |
| Bert-large-cased    | 0.3592   |
| Bert-large-uncased  | 0.3530   |

As expected, the performance improves as we add transfer learning or increase the size and complexity of the pre-trained encoder. I was very surprised by the strong performance of the BERT models. Even the BERT base models significantly outperform ELMo and GPT. 

My final submission was a ensemble of BERT large models fine-tuned on the GAP dataset. It scores around 0.33 on the public test set and 0.26 on the private test set.

## Learning From Top Solutions

Things that I should have considered:

- **Truncation**: most of the top solutions involved truncating the text to a certain length. I didn't do it since I was worried that a mention will get truncated. This normally wouldn't matter but it did for this competition due to the size of BERT large. Without truncation, I was only able to run a batch size of 2 when fine-tuning BERT large on my 1080-Ti, which was very unstable and 20% of the models failed to learn.
- **Augmentation**: one of the challenges of the competition was the small size of the training data. [7th place solution](https://www.kaggle.com/c/gendered-pronoun-resolution/discussion/90334) involved data augmentation by replacing the A and B names with a sets of placeholder names. This gave an improvement of 0.02.
- **Intermediate layers of BERT**: A few solutions mentioned mentioned using intermediate layers. The [3rd place solution](https://www.kaggle.com/c/gendered-pronoun-resolution/discussion/90424) used -5th and -6th layer of BERT and the [11th place solution](https://www.kaggle.com/c/gendered-pronoun-resolution/discussion/90483) concatenates last 8 layers of BERT. The reasoning behind this is that the last layer of BERT is too close to the masked LM objective and intermediate layers might offer better generalizations. I actually have a parameter in my model configuration that specified which layer to fine-tune with but I didn't find a big difference when I experimented with -1 and -2. 
- **BertAdam Optimizer**: I didn't experiment with the [BertAdam](https://github.com/huggingface/pytorch-pretrained-BERT/blob/694e2117f33d752ae89542e70b84533c52cb9142/README.md#bertadam) optimizer that comes with the pre-trained BERT package in PyTorch and used vanilla Adam instead. The biggest challenge I faced for this competition was the unstable training due to small batch size and some solution mentioned that the warm-up portion of the BertAdam helped stabilizing the training.
