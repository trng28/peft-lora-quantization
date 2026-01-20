# Model Quantization

### 1. Definition & Objectives
Model Quantization reduces the numerical precision required to represent a model's parameters (weights and activations).
*   **Conversion:** It maps parameters from high-precision floating-point formats (FP32, FP16) to lower-precision representations like 8-bit integers (INT8), 4-bit integers (INT4), or even single bits.
*   **Goal:** This significantly reduces the model's memory footprint (by 4x to 32x), speeds up inference through efficient integer arithmetic, and lowers energy consumption, making LLMs viable for mobile and edge devices.

### 2. Primary Approaches
The paper categorizes quantization into two main strategies:

*   **Post-Training Quantization (PTQ):**
    *   Converts a pre-trained model to lower precision after training is complete.
    *   **Pros:** Fast and does not require retraining the model.
    *   **Cons:** Can lead to accuracy loss. Advanced PTQ methods use calibration datasets to determine optimal scaling factors to minimize this loss.

*   **Quantization-Aware Training (QAT):**
    *   Simulates quantization operations and errors during the model's training process (in the forward pass).
    *   **Pros:** Generally achieves much higher accuracy than PTQ, especially at very low bit-widths (like INT4), as the model learns to be robust to quantization noise.
    *   **Cons:** Computationally expensive, takes longer to train, and is more complex to implement.

### 3. Specialized Strategies
*   **Mixed Precision Quantization:**
    *   Assigns different bit-precisions to different parts of the model based on sensitivity. For example, sensitive layers (like attention outputs) might remain in FP16, while less sensitive layers are compressed to INT4. This offers a better balance between efficiency and accuracy.
*   **Binary & Ternary Quantization:**
    *   Extreme compression techniques using 1-bit (values +1, -1) or 2-bit (values +1, 0, -1) per parameter. This can reduce model size by 16x to 32x but requires specialized handling to maintain performance.
*   **Distillation and Quantization:**
    *   Combines Knowledge Distillation with quantization, where a full-precision teacher model helps train a low-precision student model to recover accuracy lost during compression.
