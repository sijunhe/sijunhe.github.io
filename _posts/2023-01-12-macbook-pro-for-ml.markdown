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

```
Batch: 1 Seq: 128 CPU: 59 ms/it MPS: 29 ms/it speedup: 199.6%
Batch: 4 Seq: 128 CPU: 154 ms/it MPS: 50 ms/it speedup: 310.2%
Batch: 1 Seq: 512 CPU: 200 ms/it MPS: 56 ms/it speedup: 355.7%
Batch: 4 Seq: 512 CPU: 752 ms/it MPS: 200 ms/it speedup: 375.9%
```

```
# V100
Batch: 1 Seq: 128 T4 GPU: 12 ms/it
Batch: 4 Seq: 128 T4 GPU: 26 ms/it
Batch: 1 Seq: 512 T4 GPU: 30 ms/it
Batch: 4 Seq: 512 T4 GPU: 114 ms/it    

# P100
Batch: 1 Seq: 128 P100 GPU: 13 ms/it
Batch: 4 Seq: 128 P100 GPU: 16 ms/it
Batch: 1 Seq: 512 P100 GPU: 18 ms/it
Batch: 4 Seq: 512 P100 GPU: 65 ms/it
```

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