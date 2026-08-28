# AMICO: Comprehensive Project Knowledge & Scientific Summary
**Adaptive Multi-Instance & Chemical Cross-Attention for Protein-Cofactor Specificity Prediction**

---

## 1. Executive Summary & Core Scientific Findings

The AMICO project aims to solve a fundamental challenge in computational structural biology: **accurately predicting the specific cofactor bound by a protein** (among 5 major cofactors: `ATP`, `NAD`, `FAD`, `B12`, `acetyl-CoA`) **under strict cross-fold generalization (zero-shot structural fold transfer)**, while localizing the exact binding pocket and estimating prediction uncertainty.

### 🏆 Benchmark Ranking Across Cross-Fold Splits

| Model Architecture | Input Modality | Test Accuracy | Test Macro F1 | Min / Max F1 | Inference Speed |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **🥇 `ligand_cross_mil`** | **ESM Pockets + Full Protein + ECFP4 Ligands** | **81.56 %** | **79.73 %** | **76.81 % / 83.36 %** | **66.7 s (Ultra-fast)** |
| **🥈 `egnn_ligand_cross_mil`** | **3D EGNN Pockets + Protein + ECFP4 Ligands** | **81.55 %** | **79.63 %** | **75.72 % / 85.87 %** | 704.3 s |
| **🥉 `foldseek`** (Baseline) | Full 3D PDB Structure (1-NN Alignment) | 78.85 % | 77.44 % | 72.26 % / 81.92 % | 287.3 s |
| **`cross_attention_mil`** | ESM Pockets + Full Protein Query | 79.01 % | 76.87 % | 69.81 % / 82.15 % | 53.6 s |
| **`self_attention_mil`** | ESM Pockets + Protein CLS Token | 78.19 % | 75.64 % | 68.84 % / 84.97 % | 56.6 s |
| **`encoder_mil`** | 3D EGNN Pockets Only (Embedding-level) | 44.02 % | 40.89 % | 34.50 % / 51.36 % | 313.5 s |
| **`standard_mil`** | ESM Pockets Only (Attention MIL) | 42.44 % | 40.61 % | 32.94 % / 50.30 % | 24.6 s |
| **`egnn_mil`** | 3D EGNN Pockets Only (Score-level) | 37.05 % | 35.10 % | 31.86 % / 42.43 % | 194.6 s |

---

## 2. Biological Problem & The "Pocket-Only Collapse" Phenomenon

### A. The Chemical Overlap Problem
Three of the most abundant cofactors in biology—**ATP, NAD, and FAD**—share identical chemical sub-structures:
* **ADP moiety:** Adenine base + Ribose sugar + Pyrophosphate chain.
* In local binding pockets, residues interacting with the phosphate backbone (e.g., P-loop / Walker A motif, Rossmann fold GxGxxG) look nearly identical in terms of local geometry and electrostatic charge.

### B. Why Traditional MIL Models Collapsed (~40% F1)
* When deep learning models (`standard_mil`, `encoder_mil`, `egnn_mil`) are given **only isolated pocket embeddings**, they achieve high training accuracy but completely fail to generalize across structural clusters (dropping to ~35–41% Test Macro F1).
* **Cause:** Without the global enzyme architecture, the model cannot disambiguate whether a generic nucleotide-binding sub-pocket belongs to a kinase (ATP), a dehydrogenase (NAD), or an oxidoreductase (FAD).

### C. The Solution: Multi-Level Context & Chemical Queries (+40% Boost)
1. **Global Protein Context:** Providing the whole-protein sequence embedding (from ESM-2) as a context anchor on token index 0 lifts performance from 40% to ~77% (matching Foldseek).
2. **Chemical Ligand Cross-Attention:** Supplying 1024-bit Morgan ECFP4 fingerprints of the cofactors as **Queries** that attend over both the protein context and the candidate pockets allows the model to surpass Foldseek, reaching **79.73% average Macro F1 (peaking at 85.87%)**.

---

## 3. Structural Clustering & The Data Split Trap

### A. The Problem with Naive Representative Splitting
* Clustering protein structures via Foldseek/TM-score produces clusters of wildly varying sizes (e.g., large ATP kinase clusters containing hundreds of structures vs. small B12 clusters).
* Performing naive `train_test_split` on cluster representatives breaks the label distribution on the instance level, leading to severe test set class skew (e.g., test set having 70% ATP and almost 0% FAD/acetyl-CoA).

### B. The Solution: `StratifiedGroupKFold`
* Replaced naive splitting with `StratifiedGroupKFold(n_splits=10)` (80% Train, 10% Val, 10% Test).
* **Property:** Guarantees zero data leakage between train/val/test by keeping structural clusters intact (`groups`), while simultaneously optimizing the balance of all 5 cofactor classes across folds.

---

## 4. Architecture of the Winning Model: `LigandCrossAttentionMIL`

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       LigandCrossAttentionMIL                                │
│                                                                             │
│  [5 Cofactor ECFP4 Fingerprints] ──────────► [Linear + LN] ──► Queries (Q) │
│                                                                   │         │
│  [ESM-2 Full Protein Embedding]  ──────────► [Token 0 (Context)]  │         │
│                                                   │               ▼         │
│  [ESM-2 Pocket 1..N Embeddings]  ──────────► [Tokens 1..N] ──► Keys/Values │
│                                                                   │         │
│                                                                   ▼         │
│                                            ┌─────────────────────────────┐  │
│                                            │ Multi-Head Cross-Attention  │  │
│                                            │ (dim=256, heads=4, drop=0.2)│  │
│                                            └──────────────┬──────────────┘  │
│                                                           ▼                 │
│                                            ┌─────────────────────────────┐  │
│                                            │ Residual 1 + LayerNorm 1    │  │
│                                            │ FFN (256->512->256, GELU)   │  │
│                                            │ Residual 2 + LayerNorm 2    │  │
│                                            └──────────────┬──────────────┘  │
│                                                           ▼                 │
│                                            ┌─────────────────────────────┐  │
│                                            │ Linear Scorer -> [B, 5]     │  │
│                                            └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Mathematical & Architectural Components:
1. **Keys/Values Sequence:** $\mathbf{K}, \mathbf{V} = [\mathbf{z}_{\text{protein}}, \mathbf{z}_{\text{pocket}_1}, \dots, \mathbf{z}_{\text{pocket}_N}] \in \mathbb{R}^{B \times (N+1) \times d}$.
2. **Query Vectors:** $\mathbf{Q} = [\mathbf{q}_{\text{acetyl-CoA}}, \mathbf{q}_{\text{ATP}}, \mathbf{q}_{\text{B12}}, \mathbf{q}_{\text{FAD}}, \mathbf{q}_{\text{NAD}}] \in \mathbb{R}^{B \times 5 \times d}$.
3. **Cross-Attention Weights:** $\mathbf{A} = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d}}\right) \in \mathbb{R}^{B \times 5 \times (N+1)}$.
   * $\mathbf{A}[c, 0]$: Attention paid by cofactor $c$ to the global enzyme sequence context.
   * $\mathbf{A}[c, 1..N]$: Attention paid by cofactor $c$ to individual 3D candidate pockets (enables direct pocket localization).
4. **Canonical Transformer FFN Block:**
   $$\mathbf{x}_1 = \text{LayerNorm}(\mathbf{Q} + \text{CrossAttn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}))$$
   $$\mathbf{x}_2 = \text{LayerNorm}(\mathbf{x}_1 + \text{FFN}(\mathbf{x}_1))$$
   $$\text{Logits} = \mathbf{W}_s \mathbf{x}_2 \in \mathbb{R}^{B \times 5}$$

---

## 5. Hyperparameter Tuning Recipes (Optuna Best Practices)

Through automated Bayesian hyperparameter search (Optuna with SQLite persistence and MedianPruner), the optimal regime was established:

* **Optimizer:** AdamW with cosine / ReduceLROnPlateau decay.
* **Learning Rate:** $4.5 \cdot 10^{-5}$ to $5.0 \cdot 10^{-5}$ (preventing rapid overfitting on small bags).
* **Weight Decay:** $10^{-5}$ to $10^{-4}$.
* **Label Smoothing:** $0.15 - 0.20$ (essential for zero-shot fold generalization across overlapping nucleotide classes).
* **Dropout:** $0.15 - 0.30$ in projections and FFN layers.
* **Hidden Dimension:** $256$ (with 4 attention heads).
* **Batch Size:** $64$.
* **Early Stopping:** Patience = 12 epochs on validation loss.

---

## 6. Uncertainty Estimation & True Negative Detection (MC Dropout)

Standard Softmax forces probabilities to sum to 100%, causing dangerous overconfidence on non-binding or out-of-distribution proteins. 

### Monte Carlo Dropout Solution ($T = 30$ Stochastic Passes):
1. **Mean Prediction:** $\bar{p}_c = \frac{1}{T} \sum_{t=1}^T p_c^{(t)}$.
2. **Epistemic Uncertainty:** $\sigma_c = \sqrt{\frac{1}{T} \sum_{t=1}^T (p_c^{(t)} - \bar{p}_c)^2}$.
3. **Predictive Entropy:** $H(y \mid \mathbf{x}) = -\sum_{c=1}^5 \bar{p}_c \log \bar{p}_c$.
4. **Decision Boundary for Non-Binders (True Negatives):**
   $$\text{If } \max_c(\bar{p}_c) < 0.50 \quad \text{OR} \quad \sigma_{\text{top}} > 0.15 \implies \mathbf{\text{Classified as NON\_BINDER / UNKNOWN}}$$

---

## 7. Migration Checklist for `AMICOfin`

The clean production repository `AMICOfin` contains only the verified, production-grade components:

- [x] **`model_ligand_cross_attention_mil.py`**: Core winning model.
- [x] **`dataset_cross_mil.py`**: Clean dataset collator for pockets + protein embeddings.
- [x] **`train_ligand_cross_attention_mil.py`**: Production training pipeline with JSON config loader.
- [x] **`optuna_ligand_cross_attention.py`**: Automated hyperparameter tuning with SQLite persistence.
- [x] **`predict_ligand_cross_attention.py`**: Production inference with MC Dropout & pocket localization.
- [x] **`benchmark_all_models.py`**: Master multi-split benchmark runner.
- [x] **`model_sequence_mlp.py` & `train_sequence_mlp.py`**: Pure sequence baseline.
- [x] **`benchmarks/foldseek_benchmark.py`**: 1-NN 3D structural baseline.
- [x] **`data_prep/two_level_structure_clustering.py`**: StratifiedGroupKFold clustering pipeline.
