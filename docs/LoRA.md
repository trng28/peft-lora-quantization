
# LoRA (Low-Rank Adaptation)

## 1. What is LoRA?

**LoRA (Low-Rank Adaptation)** is a **parameter-efficient fine-tuning (PEFT)** technique designed for large neural networks such as **Transformers and Large Language Models (LLMs)**.

Instead of fine-tuning all model parameters, LoRA:

* **Freezes the original pretrained weights**
* **Injects small trainable low-rank matrices** to capture task-specific updates

This enables efficient training with **significantly fewer parameters, lower memory usage, and faster convergence**.

---

## 2. Core Idea

Given a pretrained weight matrix:


```math
W \in \mathbb{R}^{d \times k}
```


LoRA avoids updating ( W ) directly.
Instead, it learns a low-rank update:

```math
W' = W + \Delta W
```

where:

```math
\Delta W = B A
```

with:

* $(A \in \mathbb{R}^{r \times k})$
* $(B \in \mathbb{R}^{d \times r})$
* $(r \ll \min(d, k))$

> Only **A** and **B** are trainable; **W remains frozen**.

---

## 3. Forward Computation

For an input vector (x):

$
\begin{aligned}
y &= W' x \
&= W x + B A x
\end{aligned}
$

In practice, a scaling factor ( \alpha ) is applied:

$
y = W x + \frac{\alpha}{r} B A x
$

where:

* $( \alpha )$ controls the magnitude of the LoRA update
* Common choice: $( \alpha = r )$

---

## 4. Why Low-Rank Works

Large pretrained models are often **over-parameterized**.
Task-specific updates tend to lie in a **low-dimensional subspace**, making a low-rank approximation sufficient.

### Parameter Comparison

For a linear layer $(W \in \mathbb{R}^{4096 \times 4096})$:

* Full fine-tuning:

  $4096 \times 4096 \approx 16M \text{ parameters}$
 

* LoRA with (r=8):
  $4096 \times 8 + 8 \times 4096 \approx 65K \text{ parameters}$


> **~250× fewer trainable parameters**

---

## 5. Where LoRA is Applied

LoRA is typically inserted into **Linear layers**, especially:

* Self-attention:

  * Query (`q_proj`)
  * Key (`k_proj`)
  * Value (`v_proj`)
  * Output (`o_proj`)
* Feed-forward layers:

  * `fc_in`, `fc_out`

You can selectively apply LoRA to **only the most impactful layers**.

---

## 6. Key Hyperparameters

| Parameter        | Description                    |
| ---------------- | ------------------------------ |
| `r`              | Rank of low-rank matrices      |
| `alpha`          | Scaling factor                 |
| `dropout`        | Dropout applied to LoRA branch |
| `target_modules` | Layers where LoRA is injected  |

Typical values:

* `r ∈ {4, 8, 16}`
* `alpha = r` or `2r`

---

## 7. Training Behavior

* Original weights: **frozen**
* Trainable parameters: **LoRA adapters only**
* Optimizer states: drastically reduced
* Compatible with **quantization (e.g., 4-bit / 8-bit)**

---

## 8. LoRA vs Full Fine-Tuning

| Aspect              | Full Fine-Tuning | LoRA        |
| ------------------- | ---------------- | ----------- |
| Trainable params    | Very large       | Very small  |
| GPU memory          | High             | Low         |
| Training speed      | Slow             | Fast        |
| Multi-task learning | Hard             | Easy        |
| Model storage       | Large            | Lightweight |

---

## 9. Minimal PyTorch Example

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=8):
        super().__init__()
        self.weight = nn.Linear(in_features, out_features, bias=False)
        self.A = nn.Linear(in_features, r, bias=False)
        self.B = nn.Linear(r, out_features, bias=False)
        self.scaling = alpha / r

        self.weight.weight.requires_grad = False

    def forward(self, x):
        return self.weight(x) + self.scaling * self.B(self.A(x))
```

---

## 10. Limitations

* Limited expressiveness for very small ranks
* Not always sufficient for tasks far from the pretraining domain
* Requires careful selection of target layers

Despite this, LoRA is widely considered a **default fine-tuning strategy for LLMs**.

---

## 11. When Should You Use LoRA?

### Recommended when:

* Fine-tuning large Transformer models
* GPU memory is limited
* Multiple tasks or adapters are required
* Efficient deployment is important

### Less useful when:

* Models are small
* Full fine-tuning is affordable and necessary

---

## 12. Summary

> **LoRA enables efficient fine-tuning of large models by learning low-rank weight updates while keeping pretrained weights frozen.**

---
