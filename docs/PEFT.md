
# Parameter-Efficient Fine-Tuning (PEFT)

## 1. Overview

**Parameter-Efficient Fine-Tuning (PEFT)**, a family of methods designed to adapt large pretrained models to downstream tasks while updating only a **small subset of parameters**.

Instead of fine-tuning all model weights, PEFT methods:

* Freeze the majority of pretrained parameters
* Introduce a small number of task-specific trainable parameters
* Achieve competitive performance with significantly reduced computational cost

PEFT has become a standard approach for adapting large language models under limited memory and compute budgets.

---

## 2. Problem Formulation

Let a pretrained model be parameterized by:

```math
\theta \in \mathbb{R}^{N}
```

Given a downstream task with dataset $( \mathcal{D} )$, full fine-tuning solves:

```math
\theta^{*} = \arg\min_{\theta} \; \mathcal{L}(\theta; \mathcal{D})
```

where all parameters $( \theta )$ are updated.

In contrast, PEFT decomposes the parameters as:

```math
\theta = \theta_{0} + \Delta \theta
```

where:

* $( \theta_{0} )$ denotes frozen pretrained parameters
* $( \Delta \theta )$ denotes a small set of trainable parameters

The optimization objective becomes:

```math
\Delta \theta^{*} = \arg\min_{\Delta \theta} \; \mathcal{L}(\theta_{0} + \Delta \theta; \mathcal{D})
```

with:

```math
\|\Delta \theta\| \ll \|\theta\|
```

---

## 3. Design Principles

PEFT methods are guided by the following principles:

### 3.1 Parameter Efficiency

Aim to minimize the number of trainable parameters while preserving task performance:

```math
\frac{|\Delta \theta|}{|\theta|} \ll 1
```

---

### 3.2 Knowledge Preservation

By freezing pretrained weights, PEFT preserves general-purpose knowledge learned during large-scale pretraining, reducing catastrophic forgetting.

---

### 3.3 Modular Adaptation

Task-specific parameters can be stored, swapped, or composed without modifying the base model.

---

## 4. Categories of PEFT Methods

PEFT methods can be broadly categorized as follows.

---

### 4.1 Adapter-Based Methods

Adapters insert small trainable modules between layers:

```math
h' = h + f_{\text{adapter}}(h)
```

Examples:

* Adapters
* Prefix-tuning
* Prompt-tuning

---

### 4.2 Low-Rank Adaptation

Low-rank methods approximate weight updates using low-rank matrices:

```math
W' = W + B A
```

Examples:

* LoRA
* QLoRA
* AdaLoRA
* DenseLoRA
* etc 
---

### 4.3 Prompt-Based Optimization

Prompt-based methods optimize virtual tokens prepended to the input:

```math
x' = [p_1, \dots, p_m, x]
```

Examples:

* Soft prompts
* Prefix tuning

---

## 5. Training Characteristics

* Pretrained parameters: frozen
* Trainable parameters: PEFT-specific modules
* Optimizer states: proportional to trainable parameters
* Compatible with quantization and distributed training

PEFT enables fine-tuning models with billions of parameters on limited hardware.

---

## 6. Comparison with Full Fine-Tuning

| Aspect               | Full Fine-Tuning | PEFT         |
| -------------------- | ---------------- | ------------ |
| Trainable parameters | All              | Small subset |
| Memory usage         | High             | Low          |
| Training cost        | High             | Low          |
| Multi-task support   | Limited          | Strong       |
| Deployment           | Heavy            | Modular      |

---

## 7. Practical Advantages

PEFT allows to:

* Fine-tune large models on a single GPU
* Maintain a shared frozen backbone across tasks
* Rapidly iterate and deploy task-specific adapters

---

## 8. Limitations

* Expressiveness may be limited by parameter budget
* Method selection is task-dependent
* Some PEFT methods introduce additional architectural complexity

Nevertheless, PEFT provides a strong trade-off between efficiency and performance.

---

## 9. Summary

**Parameter-Efficient Fine-Tuning as a principled approach to adapting large pretrained models by learning a compact set of task-specific parameters while preserving the knowledge encoded in the base model.**

