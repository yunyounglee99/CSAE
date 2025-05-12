# CSAE: Continual Sparse AutoEncoder

<img width="1034" alt="스크린샷 2025-05-13 오전 1 58 25" src="https://github.com/user-attachments/assets/79b2c201-e8f5-440d-bd93-d65db54450a9" />


A novel continual learning architecture that addresses the scalability and forgetting issues in vision transformer-based models using sparse autoencoding and top-k gating.

## 🔍 Overview

**CSAE** (Continual Sparse AutoEncoder) is a lightweight, memory-efficient continual learning method designed to mitigate catastrophic forgetting without growing the model size. It integrates a single sparse autoencoder (SAE) module in parallel with ViT layers and leverages top-k gating for selective parameter activation.

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
- **Parameter Reuse Formula**:
  
  For task \( t \):
  \[
  W_t = (1 - \alpha_t) W_{t-1} + \alpha_t W_t^{new}
  \]
  where \( \alpha_t \) is a learnable gate.

- **No parameter explosion**: One SAE is shared across all tasks, and only a small number of parameters are activated via top-k gating.

## 🧪 Datasets & Settings

CSAE is evaluated under:

- **Class-Incremental Learning**
  - CIFAR-100
  - ImageNet-R
- **Domain-Incremental Learning**
  - CORe50
  - DomainNet

## ⚙️ Implementation Details

- Optimizer: Adam (β₁=0.9, β₂=0.999)
- Batch Size: 128
- Learning Rate: 5e-5
- Total SAE middle dimension: 60
- Softmax suppression for previous tasks during training

## 🏆 Results

- Maintains competitive performance while **greatly reducing model size growth**.
- Outperforms traditional CAL-based methods in memory-constrained settings.
- Enables interpretable and selective feature reuse via **sparse latent gating**.

## 📌 Why CSAE?

| Feature | CSAE | C-ADA |
|--------|------|-------|
| Model growth | ❌ Fixed | ✅ Grows linearly |
| Interpretability | ✅ Sparse latent units | ❌ Opaque |
| Forgetting control | ✅ α-gated reuse | ⚠️ Orthogonal constraint |
| Parallelization | ✅ ViT-parallel | ❌ Mostly sequential |

## 📁 Structure
