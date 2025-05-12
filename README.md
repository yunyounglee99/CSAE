# CSAE: Continual Sparse AutoEncoder

<img width="990" alt="스크린샷 2025-05-13 오전 1 58 25" src="https://github.com/user-attachments/assets/79b2c201-e8f5-440d-bd93-d65db54450a9" />


A novel continual learning architecture that addresses the scalability and forgetting issues in vision transformer-based models using sparse autoencoding and delta-interpolation

## 🔍 Overview

**CSAE** (Continual Sparse AutoEncoder) is a lightweight, memory-efficient continual learning method designed to mitigate catastrophic forgetting without growing the model size. It integrates a single sparse autoencoder (SAE) module in parallel with ViT layers and leverages top-k gating for selective parameter activation. Furthermore, CSAE adopts a delta interpolation strategy, which blends the previous task’s parameters with the current task’s updates using a learnable gating tensor $\alpha$, enabling smooth parameter transitions while preserving past knowledge.

## 🚨 Motivation

Traditional continual learning approaches like C-ADA suffer from:

- ⚠️ **Model size explosion**: Parameters grow linearly with each new task.
- ⚠️ **Over-constrained learning**: Gram-Schmidt orthogonalization can limit expressive learning by forcing unnecessary independence between learned weights.

To overcome these, CSAE introduces a **single shared SAE module** with **learnable residual blending** to reuse and adapt previous parameters efficiently.

## 🧠 Key Ideas

- **Pretrained ViT Backbone**: Used in a frozen state for feature extraction.
- **Parallel SAE Module**: Sparse AutoEncoder attached to the 0th–4th layers of the ViT in parallel.
- **Top-k Gating**: Selects only the top-k latent components, zeroing out the rest.
- **Residual Update**: Output of SAE is added back to the main network.
- **Delta interpolation Formula**:
  
  For task $\ t \$:

  $\ W_t = (1 - \alpha_t) W_{t-1} + \alpha_t W_t^{\text{new}} \$
  
  where $\alpha_t\$ is a learnable gate.

- **No parameter explosion**: One SAE is shared across all tasks, and only a small number of parameters are activated via top-k gating.

## 🧪 Datasets & Settings

CSAE is evaluated under:

- **Class-Incremental Learning**
  - CIFAR-100
  - CUB200 (future work)

## ⚙️ Implementation Details

- Optimizer: Adam (β₁=0.9, β₂=0.999)
- Batch Size: 128
- Learning Rate: 8e-5
- Total SAE middle dimension: 30

## 🏆 Results

![continual_learning_metrics](https://github.com/user-attachments/assets/87b8bd10-b114-4edc-99ed-62132c4993d4)


- Maintains competitive performance while **greatly reducing model size growth**.
- Outperforms traditional CAL-based methods in memory-constrained settings.
- Enables interpretable and selective feature reuse via **sparse latent gating**.

## 📌 Why CSAE?

| Feature | CSAE | C-ADA |
|--------|------|-------|
| Model growth | ❌ Fixed | ✅ Grows linearly |
| Interpretability | ✅ Sparse latent units | ❌ Opaque |
| Forgetting control | ✅ α-gated reuse | ⚠️ Orthogonal constraint |


## 📉 Limitation of CSAE

• While CSAE achieves strong average performance, the per-task accuracy trends (see “Task Accuracies”) exhibit noticeable fluctuations across tasks, suggesting sensitivity to task order and potential instability in latent representation learning.
• The “Average Forgetting” curve increases sharply after certain tasks, indicating that long-term knowledge retention is not always guaranteed, particularly under class-imbalanced or distribution-shifted settings.
• Future improvements may include dynamic task-aware gating, better regularization for stability, and adaptive α-tuning to balance between reuse and plasticity.
