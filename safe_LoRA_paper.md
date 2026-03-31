# Safe LoRA Methodology Summary

## 1. Overview

Safe LoRA is a method designed to preserve the safety alignment of Large Language Models (LLMs) during fine-tuning, particularly when using parameter-efficient methods such as LoRA (Low-Rank Adaptation).

The key idea is to interpret alignment as a geometric property in weight space and constrain LoRA updates to remain within a safety-aligned subspace.

---

## 2. Problem Definition

Fine-tuning aligned LLMs often leads to **alignment degradation**, even when:

* The fine-tuning dataset is benign
* The number of harmful samples is small

This indicates that:

> Fine-tuning updates can move model weights away from the alignment-related directions.

---

## 3. Core Idea

Safe LoRA assumes that:

* Alignment information is encoded in a specific direction (or subspace) in weight space
* This direction can be extracted from the difference between aligned and unaligned models

Thus, by projecting fine-tuning updates onto this alignment subspace, safety can be preserved.

---

## 4. Alignment Matrix Construction

### 4.1 Definition

For each layer (i):

[
V_i = W^{aligned}_i - W^{unaligned}_i
]

* (W^{aligned}): aligned model (e.g., chat/instruct model)
* (W^{unaligned}): base model

---

### 4.2 Interpretation

The alignment matrix (V_i) represents:

* Instruction tuning effects
* Safety alignment (e.g., RLHF)
* Behavioral transformation from base model to aligned assistant

This vector captures the **direction of alignment** in weight space.

---

### 4.3 Projection Matrix

A projection operator is constructed as:

[
\hat{C}_i = V_i (V_i^T V_i)^{-1} V_i^T
]

This projects any update onto the subspace spanned by (V_i).

---

## 5. LoRA Update and Projection

### 5.1 LoRA Update

LoRA introduces a low-rank update:

[
\Delta W_i = A_i B_i^T
]

This update improves task performance but may harm alignment.

---

### 5.2 Similarity-Based Selection

Instead of projecting all updates, Safe LoRA applies projection selectively.

Define cosine similarity:

[
\text{sim} = \frac{\langle \Delta W_i, \hat{C}_i \Delta W_i \rangle_F}{|\Delta W_i|_F |\hat{C}_i \Delta W_i|_F}
]

---

### 5.3 Projection Rule

[
\Delta W_i = \hat{C}_i \Delta W_i \quad \text{if } \text{sim} < \tau
]

* If similarity is high → keep original update
* If similarity is low → project onto alignment subspace

---

### 5.4 Intuition

* High similarity → update is aligned with safety → no change
* Low similarity → update deviates from safety → correction needed

---

## 6. Theoretical Interpretation

### 6.1 Weight Space View

The weight space is treated as a vector space with Frobenius inner product.

---

### 6.2 Key Assumption

The alignment vector:

[
V = W_{aligned} - W_{unaligned}
]

encodes safety-related information.

---

### 6.3 Projection Meaning

The projection matrix defines a **safety subspace**.

Safe LoRA ensures that:

[
\text{Final Update} \in (\text{Low-Rank Space}) \cap (\text{Safety Subspace})
]

---

### 6.4 Interpretation

* LoRA searches for solutions in low-rank space
* Projection restricts solutions to remain aligned with safety

---

## 7. Efficient Approximation

### 7.1 Problem

Exact projection requires computing:

[
(V^T V)^{-1}
]

which is computationally expensive.

---

### 7.2 Approximation

A faster alternative is:

[
C_i = \frac{V_i V_i^T}{|V_i|_F}
]

---

### 7.3 Trade-off

* Exact projection: better safety-utility balance
* Approximation: significantly faster (practical use)

---

## 8. Extension to Full Fine-tuning

Safe LoRA can also be applied to full parameter updates:

[
W^{fine}_i = W^{pre}_i + C_i (W^{fine}_i - W^{pre}_i)
]

Instead of projecting full weights, the method projects the **residual update**.

---

## 9. Key Properties

| Property       | Description                    |
| -------------- | ------------------------------ |
| Data-free      | No additional dataset required |
| Training-free  | No retraining needed           |
| Model-agnostic | Applicable to various LLMs     |
| Lightweight    | Minimal computation overhead   |

---

## 10. Summary

Safe LoRA reformulates alignment preservation as a geometric constraint in weight space.

By projecting LoRA updates onto an alignment-derived subspace, the method:

* Prevents safety degradation
* Preserves downstream performance
* Requires no additional data or training

---

## One-line Summary

Safe LoRA preserves LLM safety by projecting fine-tuning updates onto an alignment subspace derived from the difference between aligned and unaligned model
