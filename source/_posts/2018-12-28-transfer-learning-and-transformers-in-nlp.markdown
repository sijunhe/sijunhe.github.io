---
layout: post
title: "Transfer Learning in NLP"
date: 2018-12-28 22:13:08 -0800
comments: true
published: false
categories: nlp deep-learning reading-notes
---

> NLP's ImageNet moment has arrived   
> 
> &mdash; [Sebastian Ruder](http://ruder.io/nlp-imagenet/)

Reading notes of major advances in NLP Transfer learning published in 2018:

- **ELMo from Allen NLP**:  [Deep Contextualized Word Representations](https://allennlp.org/elmo)
- **ULMFiT from fast.ai**: [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146)
- **Transformer from OpenAI**: [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) 
- **BERT from Google**: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

<!--more-->
## Transformer


## ELMo: Embeddings from Language Models

- Popular pre-trained [word representations]([embeddings](https://sijunhe.github.io/blog/2018/09/12/word-embeddings/) captures syntactic and semantic information of words. However, these approaches only allow a single context independent representation for each word. Ideal word representations should also model how word meaning vary across lingustic contexts (polysemy). 
- The paper introduces a deep contexualized word representation (ElMo) that directly address both challenges and improves the state of the art across a range of language understanding problems.
- ELMo word representations are functions of the entire input sentence, computed on top of two-layer biLMs. 

#### Bidirectional Language Models(biLM) and ELMo

The state-of-the-art bidirectional neural language models compute a token representation $\textbf{x}\_{k}\^{LM}$ then pass it through $L$ layers of forward LSTMs. At each position $k$, the $j$-th LSTM layer outputs a context-dependent representations $\textbf{h}\_{k,j}\^{LM}$, where each representation is a concatenation of the forward and backward representation$\[ \overrightarrow{\textbf{h}}\_{k,j}\^{LM}, \overleftarrow{\textbf{h}}\_{k,j}\^{LM}\]$.

ELMo is a task specific combination of the intermediate layer representations in the biLM. For each token $t\_k$, a $L$-layer biLM computes a set of $2L + 1$, where $\textbf{h}\_{k,0}\^{LM}$ is the token layer representation $x\_k\^{LM}$. For inclusion in a downstream model, ELMo collapses all layers into a single vector with task specific weighting:

$$\textbf{ELMo}\_k\^{task} = \gamma\^{task} \sum\_{j=0}\^L s\^{task}\_j \textbf{h}\_{k,j}\^{LM}$$

#### Using ELMo for NLP tasks

Most supervised NLP models form a **context-independent** token representations $\textbf{x}\_k$ for each token position using pre-trained word embedding and optionally character-based representations. Then the model forms a context-sensitive representations $\textbf{h}\_{k}$ with RNNs, CNNs or feed forward networks. 

To add ELMo to the supervised model, the authors

- Freeze the weights of the biLM, run the biLM and record all of the layer presentations for each word
- Replace $\textbf{x}\_k$ with $\left[\textbf{x}\_k; \textbf{ELMo}\_k\^{task}\right]$ and pass them into the task RNN
- Let the end task model learn a linear combination of these representations

For some tasks, the authors observe further improvement by introducing another set of output specific linear weights and replacing $\textbf{h}\_k$ with $\left[\textbf{h}\_k; \textbf{ELMo}\_k\^{task}\right]$. The authors also found it beneficial to add a moderate amount of dropout to ELMo and in some cases to regularize the ELMo weights by adding L2 loss, which forces the ELMo weights to stay close to an average of all biLM layers.

#### Evaluation and Analysis

In all six benchmark NLP tasks, simply adding ELMo gives a new state-of-the-art results, with relative error reductions ranging form 6-20%. There are some very impressive insights:

- Using deep contextual representations in downstream tasks improves performance over previous work that uses just the top layer
- Including ELMo at the output of biRNN in task-specific architectures improves overall results for some tasks (SRL and SNLI, but not SQuAD)
- Different layers in the biLM represent different types of information; The second layer contributes more to disambiguating meaning of the word while the first layer contributes more to capturing basic syntax.
- Adding ELMo to a model increases the sample efficiency considerably. The model requires less number of parameter updates and less overall training data to reach state-of-the-art performance.

## ULMFiT

Howard and Ruder propose Universal Language Model Fine-tuning (ULMFiT) that enables robust inductive transfer learning for any NLP tasks. The same 3-layer LSTM Language model with the same hyperparameter 