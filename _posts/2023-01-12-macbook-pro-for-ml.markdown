---
layout: article
title: "Does Macbook Pro M1 GPU Matter for ML?"
tags: deep-learning ml-systems
publish: false
---

Before I left the Bay Area at the end of 2021, I splurged on a Macbook Pro with the new M1 Pro chip (8-Core CPU, 14-Core GPU). I had no idea that Apple was working on [MPS](https://developer.apple.com/metal/pytorch/) but I had 

Cover picture generated with [Stable Diffusion](https://huggingface.co/spaces/stabilityai/stable-diffusion) with the prompt "frying an egg on the keyboard of a macbook pro laptop"
![mac-egg-frying](https://sijunhe-blog.s3.us-west-1.amazonaws.com/images/egg_mac.jpeg)

<!--more-->


# Performance on BERT
```
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
        speedup = cpu_time / mps_time
        print(f"Batch: {batch_size} Seq: {seq_len} CPU: {cpu_time * 1000:.0f} ms/it MPS: {mps_time * 1000:.0f} ms/it speedup: {speedup:.1%}")
```

| Batch Size | Sequence Length | Mac CPU  (ms/it) | Mac GPU (ms/it) | Mac GPU Speedup | T4 (ms/it) | P100 (ms/it) |
|------------|-----------------|------------------|-----------------|-----------------|------------|--------------|
| 1          | 128             | 59               | 29              | **199.6%**      | 12         | 13           |
| 4          | 128             | 154              | 50              | **310.2%**      | 26         | 16           |
| 1          | 512             | 200              | 56              | **355.7%**      | 30         | 18           |
| 4          | 512             | 752              | 200             | **375.9%**      | 114        | 65           |


# Performance on Stable Diffusion

```
from diffusers import StableDiffusionPipeline
import torch
prompt = "a photo of an astronaut riding a horse on mars"
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

# First-time "warmup" pass (see explanation  above)
_ = pipe(prompt, num_inference_steps=1)
image = pipe(prompt, num_inference_steps=50).images[0]
100%|██████████| 50/50 [00:58<00:00,  1.18s/it]
```

| Mac CPU  (s/it)  | Mac GPU (s/it)  | Mac GPU Speedup | T4 (s/it)  | P100 (s/it) |
|------------------|-----------------|-----------------|------------|--------------|
| 59               | 1.18            | **199.6%**      | 12         | 13           |
