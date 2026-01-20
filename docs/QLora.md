# QLoRA (Quantized Low-Rank Adaptation)

## 1. Overview

**QLoRA** is a parameter-efficient fine-tuning method that extends **LoRA** by enabling training on **quantized pretrained weights**.
It makes it possible to fine-tune large language models using **4-bit or 8-bit quantization**, while maintaining performance comparable to full fine-tuning.

The key principle is to:

* Quantize and freeze the pretrained model weights
* Train low-rank adaptation matrices in higher precision

---

## 2. Core Formulation

Let the pretrained weight matrix be:

```math
W \in \mathbb{R}^{d \times k}
```

QLoRA applies a quantization operator $( Q(\cdot) )$ to obtain:

```math
\hat{W} = Q(W)
```

The effective weight used during training is:

```math
W' = \hat{W} + \Delta W
```

where the LoRA update is defined as:

```math
\Delta W = B A
```

with:

* $( A \in \mathbb{R}^{r \times k} )$
* $( B \in \mathbb{R}^{d \times r} )$
* $( r \ll \min(d, k) )$

Only the matrices ( A ) and ( B ) are trainable.
The quantized weight ( \hat{W} ) remains frozen.

---

## 3. Forward Computation

Given an input vector ( x ):

```math
\begin{aligned}
y &= W' x \\
  &= \hat{W} x + B A x
\end{aligned}
```

A scaling factor is applied to stabilize training:

```math
y = \hat{W} x + \frac{\alpha}{r} B A x
```

where:

* $( \alpha )$ controls the magnitude of the LoRA update
* A common setting is $( \alpha = r )$

---

## 4. Quantization Strategy

QLoRA typically uses **4-bit NormalFloat (NF4)** quantization:

```math
\hat{W} = \text{NF4}(W)
```

NF4 is designed to better approximate normally distributed weights compared to uniform INT4 quantization.

To further reduce memory usage, **double quantization** may be applied, where quantization constants themselves are quantized.

---

## 5. Training Characteristics

* Pretrained weights: quantized and frozen
* Trainable parameters: LoRA adapters only
* Adapter precision: FP16 or BF16
* Optimizer: paged optimizers to control peak memory usage

This design enables stable training even with aggressive quantization.

---

## 6. Memory Efficiency

For large-scale models, QLoRA significantly reduces GPU memory requirements:

| Method           | Base Weight Precision | GPU Memory |
| ---------------- | --------------------- | ---------- |
| Full fine-tuning | FP16                  | Very high  |
| LoRA             | FP16                  | Medium     |
| QLoRA            | 4-bit / 8-bit         | Low        |

This makes single-GPU fine-tuning feasible for multi-billion-parameter models.

---

## 7. Comparison with Related Methods

| Aspect               | Full Fine-Tuning | LoRA          | QLoRA         |
| -------------------- | ---------------- | ------------- | ------------- |
| Base weights         | FP16             | FP16          | Quantized     |
| Trainable parameters | All              | Low-rank only | Low-rank only |
| Memory usage         | High             | Medium        | Low           |
| Hardware requirement | Multi-GPU        | Single GPU    | Single GPU    |

---

## 8. Practical Considerations

* QLoRA relies on specialized quantization kernels (e.g., bitsandbytes)
* Slight performance degradation may occur compared to FP16 LoRA
* Proper selection of target modules and rank remains critical

Despite these considerations, QLoRA provides a strong balance between efficiency and performance.

---

## 9. Summary

**QLoRA enables efficient fine-tuning of large models by combining low-rank adaptation with low-bit quantized pretrained weights, dramatically reducing memory requirements without sacrificing model quality.**

---
