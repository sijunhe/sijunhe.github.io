---
layout: article
title: "Instruction Tuning and RLHF"
tags: deep-learning nlp reading-notes
---

<!--more-->

## Instruction Tuning and FLAN

![instruction-tuning](/assets/images/posts/instruction-tuning/instruction_tuning.png)

- [Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652) was published at ICLR 2022 and introduced **Instruction Finetuning**
- Background: LMs have shown good performances as few-shot learning but not as much at zero-shot learning. Without few-shot examples, it's hard for the model to perform well on prompts that are not similar to the pretraining data
- In order to improve the zero-shot performance, Wei et al. turned different NLP tasks into natural language instructions, e.g. *Is the sentiment positive or negative*, *translate "how are you" into Chinese*. All tasks are formatted into the Causal LM task. 
- Running Instruction Tuning on a large LM with 60+ NLP datasets expressed via instructions resulted in **FLAN**, which substantially improves the zero-shot performance
    - Ablation observation 1: increasing the number of task types in instruction tuning improves performance on unseen tasks
    - Ablation observation 2: the benefits of instruction tuning emerge only with sufficient model scale, as instruction tuning hurts the performance for models that are 8B or smaller. One potential explanation is that for small-scale models, learning the ∼40 tasks used during instruction tuning fills the entire model capacity, causing these models to perform worse on new tasks
- Instruction Tuning vs. finetuning vs. prompt tuning
    - **Finetuning**: Model learns one specific task. Requires many task-specific examples. Runs on seen task and unseen data at inference time.
    - **Prompt Tuning**: Model learns one specific task. Prompting improves few-shot performance by reducing the number of required task-speicic examples. Runs on seen task and unseen data at inference time.
    - **Instruction Finetuning**: model learns to perform many tasks via natural language instructions. Runs on unseen task and unseen data at inference time

## Advancing Instruction-Finetuned Models

![flan-t5](/assets/images/posts/instruction-tuning/flan_t5.png)

- Google reseachers Chung et al. published [Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) in late 2022, which showed that instruction finetuning can improve performance across a range of model architectures, model sizes, prompting setups, and evaluation tasks
- Key contributions on top of FLAN:
    - Scaled number of tasks from 62 to 1836
    - Added T5, PaLM and U-PaLM architecture (FLAN was on GPT architecture)
    - Introduced chain-of-thought data, which improves multi-step reasoning ability





## Fine-Tuning Language Models from Human Preferences

https://arxiv.org/abs/1909.08593

