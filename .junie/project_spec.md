### Project: MulT-EHR (Multi-Task EHR)

This document provides a detailed specification of the MulT-EHR framework as described in the paper "Multi-task heterogeneous graph learning on electronic health records" published in *Neural Networks* (2024).

### 1. Goals of the Code
The primary goal of the MulT-EHR framework is to facilitate accurate medical diagnosis by learning representations from Electronic Health Records (EHRs). Specific objectives include:
- **Modeling Heterogeneity**: Effectively mining complex relations and semantic information in EHRs using a heterogeneous graph structure.
- **Denoising and Confounding Adjustment**: Mitigating heavy noise and severe confounding effects (e.g., diverse patient backgrounds, varying treatment plans) using a causal inference framework.
- **Multi-Task Learning**: Leveraging inter-task knowledge to regularize the training process and improve generalizability across different clinical tasks using a single shared-weight model.
- **Handling Sparsity and Complexity**: Overcoming the challenges of sparse and complex EHR data structures.

### 2. Principles
The framework is built on the following core principles:
- **Relational Feature Learning**: Prioritizing relational features (how entities interact) over purely temporal or spatial features.
- **Causal Disentanglement**: Separating causal features (invariant across distributions) from trivial features (shortcuts/noise) based on the causal inference framework.
- **Environment Invariance**: Minimizing task-level variance to control extrapolation risk and ensure that the learned representations are consistent across different clinical "environments" (tasks).
- **Self-Supervision**: Using graph contrastive learning/translational methods (TransE) for node embedding pre-training to encode relational features before supervised training.

### 3. Architecture
The MulT-EHR architecture follows a multi-stage pipeline:

#### A. Graph Construction
- **Node Types (6)**: Patients, Visits, Diagnoses, Prescriptions, Procedures, and Lab Events.
- **Edge Types (5)**: Patient—Visit, Visit—Diagnosis, Visit—Prescription, Visit—Procedure, Visit—Lab Events.
- **Data Structure**: A heterogeneous graph representing meta-relations between medical entities.

#### B. Pre-training Module
- **Method**: Self-supervised learning using **TransE**.
- **Objective**: Translate relational features into node embeddings using a contrastive learning-based score function.

#### C. Graph Learning Module
- **Base Model**: **Heterogeneous Graph Transformer (HGT)**.
- **Mechanism**: Node-level aggregation using multi-head attention to align embedding distributions from different node and edge types into a unified latent space.

#### D. Causal Denoising Module
- **Disentanglement**: Splits features into **Causal Representations** and **Trivial Representations**.
- **Uniform Loss ($L_{unif}$)**: Ensures trivial representations match a uniform distribution (randomness), thereby eliminating backdoor paths.
- **Jensen-Shannon Divergence**: Used to measure the difference between trivial representations and noise.

#### E. Multi-Task Aggregation Module
- **Strategy**: **Hard Parameter Sharing**.
- **Objective**: Minimizes both the mean and the **variance** of task-specific losses to encourage task-invariant representations.
- **Optimization**: Uses an environment-invariant objective to reduce extrapolation risk.

### 4. Performance Metrics
The model is evaluated across four major EHR analysis tasks using several metrics:

#### Tasks:
1. **In-hospital Mortality Prediction (MORT)**: Binary classification.
2. **Readmission Prediction (READM)**: Binary classification.
3. **Length of Stay Prediction (LoS)**: Multi-class classification (10 classes).
4. **Drug Recommendation (DR)**: Multi-label classification (351 labels for MIMIC-III, 501 for MIMIC-IV).

#### Evaluation Metrics:
- **AUROC**: Area Under the Receiver Operating Curve (Primary for MORT, READM, DR, LoS).
- **AUPR**: Area Under the Precision-Recall Curve.
- **Accuracy**: Fraction of correct predictions (LoS).
- **F1-Score**: Weighted F1 for multi-class/multi-label tasks.
- **Jaccard Index**: Similarity measure for drug recommendation.

### 5. Implementation & Hyperparameters (Key Details)
- **Frameworks**: PyTorch, PyTorch Geometric (migrated from DGL), PyHealth.
- **Optimizer**: Adam (Learning rate: $5 \times 10^{-5}$, Weight decay: $1 \times 10^{-5}$).
- **Dropout**: 0.2 (robust across 0.1–0.6).
- **Hidden Dimensions**: 128 (performance improves with width).
- **Layers**: 2 (to avoid over-smoothing).
- **Sampling**: Subgraph sampling with $n_{visit} = 2000$.
- **Training Techniques**: Temperature annealing for classification loss, downsampling for imbalanced mortality tasks.

### 6. Datasets
- **MIMIC-III**
- **MIMIC-IV** (Default for current project implementation)
