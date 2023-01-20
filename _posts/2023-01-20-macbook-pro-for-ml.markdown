---
layout: article
title: "Macbook Pro GPU for ML?"
tags: deep-learning ml-systems
publish: false
---

Before I left the Bay Area at the end of 2021, I splurged on a Macbook Pro with the new M1 Pro chip (8-Core CPU, 14-Core GPU). I had no idea that Apple was working on [MPS](https://developer.apple.com/metal/pytorch/) but I had a hunch that those GPU cores might be usable for ML at some point. PyTorch and Apple released [Accelerated PyTorch Training on Mac](https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/) around June 2022 and things quickly fall in line. As of the time of writing of this blogpost, MPS acceleration is available out-of-the-box in PyTorch on MacOS 12.3+. In this short post, I benchmarked two most common models (BERT for transformers, Stable Diffusion for Diffusion Models) on MPS and see how much speedup it can achieve out-of-the-box.

*Cover picture generated with [Stable Diffusion](https://huggingface.co/spaces/stabilityai/stable-diffusion) with the prompt "frying an egg on the keyboard of a macbook pro laptop"*

<p align="center">
  <img src="https://sijunhe-blog.s3.us-west-1.amazonaws.com/images/egg_mac.jpeg" />
</p>

<!--more-->


# Performance on BERT

I benchmarked BERT with the following code:

```python
from transformers import BertModel, BertTokenizer
import time
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()
model_mps = BertModel.from_pretrained("bert-base-uncased")
model_mps.to("mps").eval()

num_runs = 50
for seq_len in [128, 512]:
    for batch_size in [1, 4]:
        tokens = tokenizer(["[UNK]" * seq_len] * batch_size, add_special_tokens=False,return_tensors="pt")
        tokens_mps = tokenizer(["[UNK]" * seq_len] * batch_size, add_special_tokens=False, return_tensors="pt")
        tokens_mps.to("mps")
        
        start_time = time.time()
        for i in range(num_runs):
            outputs = model(**tokens)
        cpu_time = (time.time() - start_time) / num_runs

        start_time = time.time()
        for i in range(num_runs):
            outputs = model_mps(**tokens_mps)
        mps_time = (time.time() - start_time) / num_runs
        speedup = (cpu_time - mps_time) / mps_time
        print(f"Batch: {batch_size} Seq: {seq_len} CPU: {cpu_time * 1000:.0f} ms/it MPS: {mps_time * 1000:.0f} ms/it speedup: {speedup:.1%}")
```

Other than my Macbook Pro, I also ran a similar snippet on Kaggle Notebooks to order to compare Macbook Pro performance against T4 and P100.

| Batch Size | Sequence Length | Mac CPU  (ms/it) | Mac GPU (ms/it) | Mac GPU Speedup | T4 (ms/it) | P100 (ms/it) |
|------------|-----------------|------------------|-----------------|-----------------|------------|--------------|
| 1          | 128             | 59               | 29              | **203%**        | 12         | 13           |
| 4          | 128             | 154              | 50              | **308%**        | 26         | 16           |
| 1          | 512             | 200              | 56              | **357%**        | 30         | 18           |
| 4          | 512             | 752              | 200             | **376%**        | 114        | 65           |


# Performance on Stable Diffusion

I benchmarked BERT with the following snippet from [diffusers](https://huggingface.co/docs/diffusers/optimization/mps):

```
from diffusers import StableDiffusionPipeline
import torch
prompt = "a photo of an astronaut riding a horse on mars"
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

# First-time "warmup" pass (see explanation  above)
_ = pipe(prompt, num_inference_steps=1)
image = pipe(prompt, num_inference_steps=50).images[0]
```

| Mac CPU  (s/it)  | Mac GPU (s/it)  | Mac GPU Speedup | T4 (s/it)  | P100 (s/it) |
|------------------|-----------------|-----------------|------------|--------------|
| 3.88             | 1.18            | **299.6%**      | 0.50       | 0.29         |

# Conclusions

The good news is that the GPU cores on Apple silicon do speed up ML models as much as 300-400% on Macbook Pro. This is significant because you can get a 50-step Stable Diffusion inference from over 3 minutes to around 1 minute, which is quite nice.
However, this is still less than 1/2 of the performance on T4 and 1/4 of the performance on P100, which are at least two generations old. 

**In conclusion, you can get some nice inference speedup from the GPU cores on your Macbook Pro. But it is not for serious ML workloads since it falls far behind with older NVIDIA GPUs.**

