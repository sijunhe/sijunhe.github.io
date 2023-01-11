---
layout: article
title: "Sentence Embeddings and Similarity"
tags: reading-notes deep-learning nlp
---

Reading a few recent papers on Sentence Embedding and Semantic Sentence Similarity: 

- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
- [SimBERT: Integrating Retrieval and Generation into BERT](https://github.com/ZhuiyiTechnology/simbert)
- [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2104.08821)

![simbert](https://sijunhe-blog.s3.us-west-1.amazonaws.com/plots/post29/simbert.png)

<!--more-->

### Sentence-BERT

![sbert](https://sijunhe-blog.s3.us-west-1.amazonaws.com/plots/post29/sbert.png)

- When it comes to sentence pair task such as STS or NLI, BERT expects both sentence concaternated by a [SEP] token and uses cross encoding. This scales poorly due to the length of the input and also made it unpractical for retrieval tasks.
- Both average BERT embeddings and the BERT [CLS] token embeddings underperform average GloVe embeddings
- SBERT adds a pooling layer over the output of BERT/RoBERTa. Empirically, Mean Pooling outperforms Max Pooling and the [CLS] token embedding
- At training time, the sentence embeddings $u$, $v$ and theelement-wise difference $|u−v|$ are concaternated and feed into a dense layer to project into $k$ the number of classes
- At inference time, the cosine similarity is computed between he sentence embeddings $u$ and $v$.


### SimBERT

![simbert](https://sijunhe-blog.s3.us-west-1.amazonaws.com/plots/post29/simbert.png)

- SimBERT uses a cool multi-task setup where the STS objective is mixed with the seq2seq text generation obective. It is trained in a supervised fasion on similar sentence pairs.

<p align="center">
  <img style="display:inline" src="https://sijunhe-blog.s3.us-west-1.amazonaws.com/plots/post29/seq2seq-lm-attention.png" width="200" />
</p>

- The text generation objective follows the seq2seq LM introduced by [UniLM](https://arxiv.org/abs/1905.03197), where tokens in the source sentence can attend to all tokens in the source sentence and the tokens in the target sentence can only attend to the tokens to its left. 
- The STS objective utilizes in-batch similar sentences (due to the seq2seq task, both a[SEP]b and b[SEP]a are in the same batch) as positives and the rest as negatives. It accomplishes this in an efficient manner by taking the [CLS] embedding of the whole batch $V \in \mathcal{R}\^{bxd}$, normalize it by L2 distance along $d$ axis, takes an outer product $VV\^T$, pass each row through a softmax layer and optimize the cross entropy loss. This can also be seen as contrastive learning, as the in-batch positives are pulled together and the in-batch negatives are pushed apart.

### SimCSE

![simcse](https://sijunhe-blog.s3.us-west-1.amazonaws.com/plots/post29/simcse.png)

- SimCSE has a unsupervised component and a supervised component
- The unsupervised SimCSE creates positive pairs by comparing the sentences against themselves. By passing two of the same sentences through the same encoder, it relies on the dropout layer as data augmentation/noise. This outperforms the NSP pre-training objective on STS tasks (no surprise here. This objective is much closer to STS than NSP). It also outperforms other data augmentation technique such as word deletion
- For the supervised component, the authors tried various popular academic sentence pairs such as Quora question pairs, NLI and MNLI and a few ways to construct positive labels, such as entailment pairs as positives and contradition pairs as hard negatives.
