---
layout: article
title: "Graph, Eager and JIT"
subtitle: "Thoughts on Static and Dynamic Graphs while learning PaddlePaddle"
tags: paddle deep-learning ml-systems
---

Some thoughts on Graph Execution (Static Graph), Eager Exeution (Dynamic Graph) and a benchmark between them on [PaddlePaddle](https://github.com/PaddlePaddle/Paddle).

Cover picture generated with [Stable Diffusion](https://huggingface.co/spaces/stabilityai/stable-diffusion) with the prompt "a static robot and a dynamic robot in steampunk environment"

![robot](https://sijunhe-blog.s3.us-west-1.amazonaws.com/plots/post32/robot.jpg)

<!--more-->

## The Good Ol' Days of the Static Graph and Graph Execution

It was the spring of 2016 and Deep Learning was the hottest buzzword at Stanford. Certainly, I was not immune to it and I enrolled in [CS224N](https://web.stanford.edu/class/cs224n/), which kicked off my Deep Learning journey. In addition to the theory work, the course had a significant portion of hands-on work, which was done in Tensorflow (version 0.6.0 IIRC). Back then, Tensorflow was a fledging framework which had been released only a few months before the course started. The API was quite barebone with only the necessary tensor math operations such as `tf.add` and `tf.matmul`. Even early layer API such as `tf.layers` came much later. For example, this code to define a Dense Layer would be something like this:

```python
in_dim = 200
out_dim = 100
x = tf.placeholder('int32', [None, in_dim], name='x')
with tf.variable_scope("dense_layer"):
    W = tf.get_variable("W", [in_dim, out_dim])
    b = tf.get_variable("b", [out_dim])
out = tf.add(tf.matmul(W, x), b)

with tf.session() as sess:
    print(sess.run([out]), feed_dict={ x: np.random.normal((2, in_dim)) })
```

The short snippet above also illustrated the core principle of Tensorflow's static computational graph: you first define a computational graph with operation nodes of different types and then you pump data through the graph with specified input and output nodes. On the other hand, the dynamic graph approach builds its computational graph at runtime and construct/deconstruct objects on the fly. The benefit of the static graph approach is obvious: by forcing the users to define the graph before running it, it allows for a compilation step that optimizes the graph. The downside is that this declarative, un-pythonic way of programming creates a really bad developer experience. It (along with the API that was too low-level) created such a steep learning curve for us that only a small portion of the students in CS224N was able to fully reproduce the [BiDAF model](https://arxiv.org/abs/1611.01603) (SOTA model then, still a classic now).

While it was difficult learning TF 0.x, it was an invaluable experience for me as I built my understanding of Tensorflow from the low level. This knowledge quickly started to pay dividends: the first model I worked on at Twitter was written in the same TF 0.x API so I was able to hit the ground running. And when people at Twitter were complaining about the TF Estimator APIs in TF 1.X, I was having a great time because it was a much better experience than the one I started with! 

## Revolution of the Dynamic Graph and Eager Execution

While I kept my head down and worked on improving the Twitter product with NLP and the static graph of TF 1.x, Facebook released PyTorch 1.0 in late 2018 and it quickly picked up steam. The research community had been complaining about bad developer experience in TF 1.x and PyTorch perfectly addressed that pain point with its dynamic graph and eager execution. By late 2019, it was clear that PyTorch and its dynamic graph won the race, with Tensorflow joining the other side by releasing its 2.0 with eager execution. After all, most people don't need to squeeze every last drop of performance to train Google-scale models. But everybody can feel the joy of the imperative, pythonic PyTorch API.

When Twitter finally got its head out of the sand in 2020, it decided to adopt the dynamic graph to increase engineering productivity. The logical choice was to upgrade from TF 1.14 to TF 2 due to the hundreds of models already running in TF. However, the actual migration was a painful uphill battle. It was already difficult to migrate the code of hundreds of engineers in a [monorepo](https://en.wikipedia.org/wiki/Monorepo) setup. On top of that, TF 2's Eager Execution had serious performance regression compared with Graph Execution, especially on sparse models (very important for the many recommendation models at Twitter, see [Twitter Engineering blog](https://blog.twitter.com/engineering/en_us/topics/insights/2020/distributed-training-of-sparse-machine-learning-models-1) if interested). It took so long to address that we ended up migrating to TF 2.2 instead of the planned TF 2.0. To look at the migration from another perspective, the performance gap between Static and Dynamic graphs can be significant.

## Static vs Dynamic Graph on PaddlePaddle

Now that I have talked about Static and Dynamic graphs so much, let's put them up for a race to see how much difference there is. Since I am learning [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) and [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP), I will benchmark `bert-base-uncased` in an inference setup in Paddle. The dynamic model is created from `BertModel.from_pretrained("bert-base-uncased")` and the static model is converted via Paddle's JIT trace capability `paddle.jit.to_static`. The full benchmark script can be found [here](https://gist.github.com/sijunhe/e78380bcf507e10f913d1e6e1b08fae7).

During the static model conversion, we see Paddle's JIT doing some graph optimization by pass through the graph and fusing some operator, which should give some performance gain.
```
--- Running IR pass [gpu_cpu_squeeze2_matmul_fuse_pass]
--- Running IR pass [gpu_cpu_reshape2_matmul_fuse_pass]
--- Running IR pass [gpu_cpu_flatten2_matmul_fuse_pass]
--- Running IR pass [gpu_cpu_map_matmul_v2_to_mul_pass]
I1026 08:41:21.743461    23 fuse_pass_base.cc:57] ---  detected 73 subgraphs
--- Running IR pass [gpu_cpu_map_matmul_v2_to_matmul_pass]
I1026 08:41:21.748471    23 fuse_pass_base.cc:57] ---  detected 24 subgraphs
--- Running IR pass [matmul_scale_fuse_pass]
--- Running IR pass [multihead_matmul_fuse_pass_v3]
--- Running IR pass [gpu_cpu_map_matmul_to_mul_pass]
--- Running IR pass [fc_fuse_pass]
--- Running IR pass [fc_elementwise_layernorm_fuse_pass]
```

### CPU - 1 Thread

```
Batch: 1 Seq: 32 Dynamic: 146.47 ms, Static with 1 Thread: 141.89 ms Speedup: 3.1%
Batch: 1 Seq: 128 Dynamic: 504.29 ms, Static with 1 Thread: 436.91 ms Speedup: 13.4%
Batch: 1 Seq: 512 Dynamic: 2384.24 ms, Static with 1 Thread: 1772.91 ms Speedup: 25.6%
Batch: 4 Seq: 32 Dynamic: 437.25 ms, Static with 1 Thread: 419.09 ms Speedup: 4.2%
Batch: 4 Seq: 128 Dynamic: 1724.65 ms, Static with 1 Thread: 1526.03 ms Speedup: 11.5%
Batch: 4 Seq: 512 Dynamic: 9090.28 ms, Static with 1 Thread: 6879.77 ms Speedup: 24.3%
```

We see a significant speedup of up to 25% with the static graph in the single-thread CPU scenario. The performance gap also increases as we pump more data through the model with a bigger batch size and sequence length. I hypothesize that since IO takes a large percentage of the runtime when the input data is small, the performance gain is minimal. And when we increase the input data, we can closer to the real performance gap.

### CPU - 2 Thread

```
Batch: 1 Seq: 32 Dynamic: 162.81 ms Static with 2 Thread: 109.33 ms Speedup: 32.9%
Batch: 1 Seq: 128 Dynamic: 556.30 ms Static with 2 Thread: 269.69 ms Speedup: 51.5%
Batch: 1 Seq: 512 Dynamic: 2390.78 ms Static with 2 Thread: 1258.76 ms Speedup: 47.3%
Batch: 4 Seq: 32 Dynamic: 436.09 ms Static with 2 Thread: 420.18 ms Speedup: 3.6%
Batch: 4 Seq: 128 Dynamic: 1734.58 ms Static with 2 Thread: 1095.62 ms Speedup: 36.8%
Batch: 4 Seq: 512 Dynamic: 9097.34 ms Static with 2 Thread: 4045.13 ms Speedup: 55.5%
```

The two-thread CPU scenario is an unfair comparison between the static and the dynamic graphs because the static model benefits from multithreading but the dynamic model in Paddle does not (nor does torch I think). Clearly, we see a much larger gap compared with the single-thread runs.

### GPU
```
Batch: 1 Seq: 32 Dynamic: 9.77 ms Static with 4 Thread: 4.55 ms Speedup: 53.4%
Batch: 1 Seq: 128 Dynamic: 7.66 ms Static with 4 Thread: 6.11 ms Speedup: 20.2%
Batch: 1 Seq: 512 Dynamic: 21.25 ms Static with 4 Thread: 19.72 ms Speedup: 7.2%
Batch: 4 Seq: 32 Dynamic: 7.04 ms Static with 4 Thread: 5.90 ms Speedup: 16.3%
Batch: 4 Seq: 128 Dynamic: 18.20 ms Static with 4 Thread: 17.30 ms Speedup: 4.9%
Batch: 4 Seq: 512 Dynamic: 71.72 ms Static with 4 Thread: 69.05 ms Speedup: 3.7%
```

Even though this run is for GPU inference, it should also reflect the performance of GPU training. As shown above, we still see a speedup with the static graph under the GPU inference scenario, but the performance gap narrows as we increase the input data size. It's a bit weird that this is the exact opposite compared with CPU and I think it might have something to do with the percentage of time spent between compute and IO and the fact that GPU is so much faster. Without a profile, I don't have a good hypothesis now.