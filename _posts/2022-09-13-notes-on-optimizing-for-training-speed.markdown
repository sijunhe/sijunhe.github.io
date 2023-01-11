---
layout: article
title: "How to Train Transformers - Reading Notes on 【100亿模型计划】"
tags: reading-notes deep-learning nlp
---

![gpu](https://sijunhe-blog.s3.us-west-1.amazonaws.com/plots/post31/gpu.png)

<!--more-->

# Computation Complexity of Transformers

## Complexity of Matrix Multiplication
- dot product between 2 vectors of length *n* has a complexity of $2n$ : *n* multiplications and *n-1* additions in theory, but this is usually implemented as a single [Multiply–accumulate operation](https://en.wikipedia.org/wiki/Multiply%E2%80%93accumulate_operation)
- Matrix multiplication between two *n x n* matrix has a complexity of $2n^3$ because consists of $n^2$ pairs of dot products
- Benchmarking TFLOPS of GPU with matrix multiplication is lower than the theoretical TFLOPS of the GPU spec because of memory bandwidth. As an example, benchmarking with simple elementwise operation usually hit the memory bandwidth ceiling with very small TFLOPS

## Complexity of Feed-Forward Network

- Input to the encoder is of shape $[B, S, H]$, where each timestep slice is of shape $[B, H]$
- For each timestep, the input gets projected into a size of $[B, 4H]$ and then back into $[B, H]$. Each projection has a complexity of $8BH^2$, which sums up to $16BH^2$
- We need to do the same projections for each timestep, making the total complexity $16BH^2S$, or $O(BH^2S)$

## Complexity of Multi-head Self-Attention

- Input to the encoder is of shape $[B, S, H]$, where each data row is of shape $[H, S]$. The layer has $a$ attention heads
- Each attention head projects the input into query, key and value space with size of $H/a$ with a total complexity of $6H^2 S/a$
- Query, key and value matrice are of size $[S, H/a]$. Each timestep in the query matrix attends to each timestep in the key matrix through a dot product, which $S^2$ pairs of dot product between vectors of size $H/a$. Similar matrix multiplication also happens between the normalized attention scores and the value matrix. This sums up to a complexity of $4S^2H/a$
- Since we need to do this for each attention heads, the above two operations sum up to $6H^2 S + 4S^2H$. Afterwards we concatenate the outputs of all heads into a matrix of size $[S, H]$ and projects into a new matrix of size $[S, H]$, which is another $2H^2S$ and makes the total $8H^2 S + 4S^2H$
- Since the above computation is for each data row in the batch, the total computation is $8BH^2S + 4BS^2H$, or $O(BH^2S + BS^2H)$

## Complexity of Transformer Encoder/Decoder Layer

- The complexity of the encoder layer is *FFN* + *SelfAttention*, the complexity of the decoder layer in encoder-decoder models is approximately *FFN* + *2 * SelfAttention* to account for the Encoder-Decoder cross attention. Overall, both the encoder and the decoder has a complexity of $O(BH^{2}S + BS^{2}H)$
- In practice, Decoder-only models (e.g. GPT2) requires more computation than Encoder-only models (e.g. BERT) of the same size. This could be attributed to additional element-wise operation such as causal masking. Decoders in Encoder-Decoder (e.g. T5) is even slower due to the required cross attention
- Technically, transformers do scale *quadratically* with input sequence length. However, it also scales quadratically with the hidden space size, which usually dominates with $H=768,1024$. The $BS^{2}H$ term starts to matter when the sequence lengths get close to the maximum input sequence length such as 512
- All of the above calculations are theoretically and only takes matrix multiplication into account. In practice, computation-efficient operations such as GeLU activation and LayerNorm takes a significant amount of time outside of matrix multiplication due to the above-mentioned memory bandwidth ceiling



# Speed-up Training on GPUs

## Single GPU

- Use the largest batch size that fits into the GPU
- To save time on gradient updates, we can use gradient accumulation, which means running forward + backward for multiple times before apply the gradients
- If the model is super large and doesn't fit into the GPU memory, consider gradient checkpointing which trades compute for memory (~30% compute overhead due to having to re-compute activations)
- Use fp16 or bf16  to save on compute and memory. bf16 is preferred if available since we don't need to scale the loss like fp16
- Use a more optimized training library, e.g. [Megatron LM](https://github.com/NVIDIA/Megatron-LM) is at least 2x the speed of [HuggingFace Transformers]() due to the optimized fused kernels written in CUDA

## Multi GPU

### Data Parallel

- Each GPU keeps a copy of the model. It consumes part of the input batch, runs the forward + backward and sync the gradients with the All Reduce operation before applying the gradients
- It's important to measure the percentage of communication cost under the multi-gpu setup. We can only achieve close-to-linear speedup if we keep the communication cost low. 
- We can reduce communication cost by using hardware such as [NVLink](https://www.nvidia.com/en-us/data-center/nvlink/) which increases the communication bandwidth or by using gradient accumulation, which lowers the communication frequency.

### Tensor Parallel

- We split each tensor across each GPU, performs the computation separately and gather the results after each step
- Gradient accumulation no longer speeds up the training because communication is now involved in every forward + backward
- We can train larger models with Tensor Parallel than we can with Data Parallel because each tensor is splitted between GPU. 

### ZeRO ([DeepSpeed](https://www.deepspeed.ai/))

- ZeRO reduces the memory consumption of each GPU by partitioning the various model training states (weights, gradients, and optimizer states) across the available devices. ZeRO 1 partitions the optimizer states (e.g. Adam) and ZeRO 2 & 3 partitions the gradients and the model weights
- HuggingFace Transformers has good Deepspeed integrations

### Rule of Thumb

- Use Data Parallel if the model fits on a single GPU because its communciation only happens at gradient updates, which we can control with gradient accumulation
- If the model is too larger to fit on a single GPU, try [DeepSpeed](https://www.deepspeed.ai/) first
- When Tensor Parallel is eventually needed, it may be worth it to invest in hardware such as NVLink and GPU with large memory bandwidth