---
layout: article
title: "ChatGPT Series: Learning from Human Preferences"
tags: deep-learning nlp reading-notes chatgpt
---


## Early RL work on Learning from Human Preferences

- [Deep reinforcement learning from human preferences](https://arxiv.org/abs/1909.08593) by from OpenAI and DeepMind researchers Christiano et al.,  published in 2017

![basic_rl](/assets/images/posts/rlhf/basic_rl.jpeg)

- **Basic RL setup**: Agent receives State from the Environment. Based on the State, the Agent takes an action, arrives at a new State in the Environment and receive an Reward. Reward is the central idea in RL because it’s the only feedback for the agent, whose goal is to maximize its comulative reward.
- A lot of the early RL work is done on games because they are ideal environments with well-specified reward function to maximize for. It's difficult to design reward for even simple real world tasks. For example, suppose that we train a robot to clean a table or scramble an egg, it’s not clear how to construct a suitable reward function. One way is to allow a human to provide feedback and use this feedback to define the task, but using human feedback directly as a reward function is prohibitively expensive. Christiano et al. proposed an approach to learn a reward function from human feedback and use it to train an agent that optimizes that reward function

![learning_from_human_preference](/assets/images/posts/rlhf/learning_from_human_preference.png)

- The method maintains a policy $\pi: O → A$ and a reward function estimate $\hat{r} : O × A → R$, each parametrized by deep neural networks.
These networks are updated by three processes:
1. The policy $\pi$ interacts with the environment to produce a set of trajectories $\[ \tau^{1}, ..., \tau^{i} \]$. The parameters of $\pi$ are updated by a traditional reinforcement learning algorithm, in order
to maximize the sum of the predicted rewards $r_t = \hat{r}(o_{t}, a_{t})$. 
2. We select pairs of segments $\sigma_{1}, \sigma_{2}$ from the trajectories $\[ \tau^{1}, ..., \tau^{i} \] in step 1, and send them to a human for comparison.
3. The parameters of the mapping rˆ are optimized via supervised learning to fit the comparisons collected from the supervised learning to fit the comparisons collected from the human so far

- Results: without the assess the true reward (i.e. the score), the agent can learn from human feedback to achieve strong and somtimes superhuman performance in many of the environments.
- Challenges: The algorithm’s performance is only as good as the human evaluator’s intuition about what behaviors look correct, so if the human doesn’t have a good grasp of the task they may not offer as much helpful feedback.

​## Apply the above RL technique on Language Models?

- [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593) by OpenAI researchers Ziegler et al., published in 2019
- The authors finetuned pretrained LM with RL using a reward
model trained from human preferences on text continuations. *The work is mostly in the domain of RL with NLP being a medium to make RL practical and safe for real-world tasks.*


![finetuning_lm_human_preference](/assets/images/posts/rlhf/finetuning_lm_human_preference.png)

- https://arxiv.org/abs/2009.01325
- https://arxiv.org/abs/2109.10862
- https://openai.com/blog/instruction-following/