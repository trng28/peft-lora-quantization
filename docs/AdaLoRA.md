
# AdaLoRA 

## 1. Overview

**AdaLoRA** is an extension of **LoRA** that introduces **adaptive rank allocation** during fine-tuning.
Instead of using a fixed low-rank dimension for all layers, AdaLoRA dynamically **allocates and prunes rank capacity** based on the importance of each layer.

This allows the model to:

* Use parameter budget more effectively
* Allocate higher rank to more important layers
* Achieve better performance under the same or lower parameter constraints

---

## 2. Core Formulation

Let the pretrained weight matrix be:

```math
W \in \mathbb{R}^{d \times k}
```

As in LoRA, AdaLoRA models the weight update as:

```math
W' = W + \Delta W
```

with the low-rank decomposition:

```math
\Delta W = B A
```

where:

* $( A \in \mathbb{R}^{r \times k} )$
* $( B \in \mathbb{R}^{d \times r} )$
* $( r )$ is the effective rank

The difference from LoRA is that **( r ) is no longer fixed** and can vary across layers and training steps.

---

## 3. Adaptive Rank Parameterization

AdaLoRA introduces a **rank importance vector** $( s \in \mathbb{R}^{r} ) $ and reparameterizes the update as:

```math
\Delta W = B \, \text{diag}(s) \, A
```

where:

* ( s_i ) represents the importance of the ( i )-th rank component
* Components with small ( s_i ) contribute less to the update

During training, rank components with low importance are gradually **pruned**.

---

## 4. Forward Computation

Given an input vector ( x ):

```math
\begin{aligned}
y &= W' x \\
  &= W x + B \, \text{diag}(s) \, A x
\end{aligned}
```

As in LoRA, a scaling factor is applied:

```math
y = W x + \frac{\alpha}{r} B \, \text{diag}(s) \, A x
```

where $( \alpha )$ controls the update magnitude.

---

## 5. Rank Allocation Strategy

AdaLoRA training proceeds in three phases:

1. **Warm-up**
   Train with a relatively large initial rank $( r_{\text{init}} )$

2. **Importance Estimation**
   Estimate rank importance using sensitivity-based metrics (e.g., gradient statistics)

3. **Rank Pruning**
   Gradually prune low-importance rank components to reach a target rank budget $( r_{\text{target}} )$

This process allows rank capacity to concentrate on the most influential layers.

---

## 6. Training Characteristics

* Pretrained weights: frozen
* Trainable parameters: adaptive low-rank matrices and importance scores
* Rank varies across layers
* Parameter budget is explicitly controlled

AdaLoRA achieves better trade-offs between **model capacity and efficiency** compared to fixed-rank LoRA.

---

## 7. Comparison with LoRA

| Aspect               | LoRA    | AdaLoRA          |
| -------------------- | ------- | ---------------- |
| Rank                 | Fixed   | Adaptive         |
| Parameter allocation | Uniform | Importance-based |
| Parameter efficiency | Good    | Better           |
| Training complexity  | Low     | Moderate         |

---

## 8. Practical Considerations

* Requires rank scheduling and pruning strategy
* Slightly higher training overhead than LoRA
* Particularly effective when the parameter budget is constrained

AdaLoRA is well-suited for fine-tuning large models where uniform rank allocation is suboptimal.

---

## 9. Summary

**AdaLoRA improves parameter-efficient fine-tuning by dynamically allocating low-rank capacity according to layer importance, achieving higher performance under the same parameter budget.**

