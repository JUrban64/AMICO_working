# AMICO Residue-Level Attention Package (`residue_attention`)

This package implements **Residue-Level Multi-Instance Learning (MIL)** with **3D Geometric Distance Attention Bias** for fine-grained protein-cofactor specificity prediction and **direct key residue identification**.

---

## 🔬 Core Concepts & Architecture

Unlike coarse pocket-level models that average all residues in a binding pocket into a single vector, this module preserves individual amino acid tokens:

$$\mathbf{X} = [\underbrace{\mathbf{z}_{\text{protein}}}_{\text{Token 0 (Context)}}, \underbrace{\mathbf{r}_1, \mathbf{r}_2, \dots, \mathbf{r}_M}_{\text{Tokens 1..M (Residues)}}] \in \mathbb{R}^{(M+1) \times 1280}$$

1. **Token 0 (Global Context):** Whole-protein ESM-2 sequence embedding (essential anchor preventing pocket-only collapse).
2. **Tokens 1..M (Binding Residues):** Per-residue ESM-2 embeddings extracted from P2Rank-predicted binding pockets ($M \approx 30–120$ residues).
3. **3D Geometric Attention Bias:** Spatial distance matrix $D_{ij} = \|\mathbf{x}_i - \mathbf{x}_j\|_2$ computed between $C_\alpha$ atoms modifies attention logits, enabling residues in physical 3D contact (e.g. catalytic dyads / triads, salt bridges) to exchange structural context.

---

## 🏛️ Implemented Models

### 1. `ResidueLigandCrossAttentionMIL` (Recommended 🏆)
* **Queries ($Q$):** 5 Cofactor ECFP4 fingerprints (2048-dim) projected into latent dimension.
* **Stage 1 (3D Spatial Encoder):** Pocket residues undergo 3D geometric self-attention with pairwise distance bias.
* **Stage 2 (Chemical Cross-Attention):** The 5 cofactors query the refined 3D residues.
* **Key Residue Attribution:** The cross-attention weights $\mathbf{A}[b, c, 1..M]$ directly quantify the attention paid by cofactor $c$ (e.g. ATP) to each individual residue $r_i$.

```
[5 Cofactor ECFP4 Fingerprints] ────────► Queries (Q) ─────────────────────┐
                                                                           ▼
[Protein Context (Token 0)]      ┐                                 ┌──────────────────────────────┐
[Residues 1..M (Tokens 1..M)]    ├─► [3D Geometric Self-Attn] ───► │ Chemical Cross-Attention    │
[Residue C-alpha 3D Coords]      ┘   (Pairwise RBF Distance Bias)  │ (Q=Cofactors, K/V=Residues)  │
                                                                   └──────────────┬───────────────┘
                                                                                  ▼
                                                                     [FFN + Residuals + Scorer]
                                                                                  ▼
                                                                           Logits [B, 5]
```

### 2. `ResidueSelfAttentionMIL`
* **Tokens:** $[\mathbf{z}_{\text{protein}}, \mathbf{r}_1, \dots, \mathbf{r}_M]$.
* **Self-Attention:** Bidirectional interaction between global context and pocket residues modulated by 3D geometric bias.
* **Classification:** Projects contextualized Token 0 into cofactor logits.
* **Key Residue Attribution:** Attention paid by Token 0 to residues $\mathbf{A}[b, :, 0, 1..M]$.

---

## 📐 3D Geometric Distance Bias (`Pairwise3DDistanceBias`)

Supports two continuous distance kernels:
* **`rbf_gaussian` (Default):** Multi-kernel Radial Basis Function (RBF) expansion over 16 distance bins (0 to 25 Å) projected into attention heads:
  $$\text{Bias}_{h, i, j} = \sum_{k=1}^{16} w_{h, k} \exp\left(-\frac{(D_{ij} - \mu_k)^2}{2\sigma^2}\right)$$
* **`linear_decay`:** Exponential distance penalty with learnable positive slope $\gamma_h > 0$:
  $$\text{Bias}_{h, i, j} = -\gamma_h \cdot D_{ij}$$

---

## 🔍 Key Residue Identification & Visualization

After inference, `interpretability.py` extracts the top residues and exports visualization files:

```python
from residue_attention.interpretability import (
    rank_key_residues,
    format_key_residues_summary,
    export_pymol_script,
    write_pdb_with_attention_bfactor
)

# 1. Rank top residues
result = rank_key_residues(
    model=model,
    residue_features=res_feats,
    padding_mask=pad_mask,
    full_protein_feature=full_prot,
    residue_metadata=metadata,
    residue_coords=coords,
    target_class=1,  # ATP
    top_k=7
)

# Print markdown summary
print(format_key_residues_summary(result))

# 2. Export PyMOL script with stick highlights and labels
export_pymol_script("protein.pdb", result, "highlight_key_residues.pml", top_k=5)

# 3. Export PDB with B-factor heat map (0 to 100)
write_pdb_with_attention_bfactor("protein.pdb", "protein_attention.pdb", result)
```

In PyMOL:
```bash
pymol highlight_key_residues.pml
```
Or for the B-factor heatmap:
```python
load protein_attention.pdb
spectrum b, blue_white_red, all
```

---

## 📁 Package Structure

| File | Description |
| :--- | :--- |
| `geometric_bias.py` | 3D C-alpha Euclidean distance attention bias (RBF / Linear decay) |
| `model_residue_self_attention.py` | `ResidueSelfAttentionMIL` with 3D spatial bias |
| `model_residue_ligand_cross_att.py` | `ResidueLigandCrossAttentionMIL` (3D Spatial Encoder + Ligand Cross-Attn) |
| `dataset_residues.py` | `ResidueDataset` and `residue_collate_fn` with dynamic padding & 3D coords |
| `feature_extractor_residues.py` | `ResidueFeatureExtractor` for extracting ESM-2 features + 3D coords from PDB + P2Rank |
| `interpretability.py` | Key residue ranking, markdown table formatter, PyMOL & B-factor PDB exporters |
| `train_residue_models.py` | Complete training script with early stopping & evaluation |
| `demo_and_test.py` | End-to-end component unit tests & real structure verification |

---

## 🚀 How to Run (`residue_attention`)

### 1. Verification & End-to-End Demo (Unit Tests + Real Structure)
Runs component checks, forward/backward passes with 3D distance bias, P2Rank extraction on `AF-Q9NRG4-F1-model_v6.pdb`, and generates PyMOL `.pml` and B-factor `.pdb`:
```bash
python residue_attention/demo_and_test.py
```

### 2. Generating the Residue Dataset (`build_residue_dataset.py`)
Processes structures through P2Rank and ESM-2, saving per-residue embeddings, 3D $C_\alpha$ coordinates, whole-protein context embeddings, and metadata into `data_prep/residues_dataset.pt`:

#### A. From a folder of PDB structures:
```bash
python residue_attention/build_residue_dataset.py \
    --pdb-dir structures/all_pdbs \
    --metadata structures/dataset_metadata.tsv \
    --prank-out-dir structures/p2rank_outputs \
    --output-path data_prep/residues_dataset.pt \
    --min-prob 0.10 \
    --max-pockets 5
```

#### B. Quick test on a single PDB (reusing existing P2Rank outputs):
```bash
python residue_attention/build_residue_dataset.py \
    --single-pdb AF-Q9NRG4-F1-model_v6.pdb \
    --skip-p2rank \
    --prank-out-dir temp_p2rank \
    --output-path data_prep/test_single_residue_dataset.pt
```

#### C. Fast Re-partitioning existing dataset (e.g. after extraction):
```bash
python residue_attention/build_residue_dataset.py \
    --resplit-only \
    --output-path data_prep/residues_dataset.pt \
    --split-suffix mil_0.5_nr0.9
```

### 3. Training Models (`train_residue_models.py`)

#### A. Full Dataset Training (with auto-partitioning via `--split-suffix`):
```bash
python residue_attention/train_residue_models.py \
    --model-type residue_ligand_cross \
    --dataset-path data_prep/residues_dataset.pt \
    --split-suffix mil_0.5_nr0.9 \
    --epochs 50 \
    --batch-size 32 \
    --lr 5e-5 \
    --bias-3d-mode rbf_gaussian \
    --save-path weights/residue_ligand_cross_best.pt
```

#### B. Smoke-test Training (Synthetic Data):
```bash
python residue_attention/train_residue_models.py --synthetic-demo --epochs 5
```

#### Available CLI Options:
* `--model-type`: `residue_ligand_cross` (default) or `residue_self_attention`
* `--dataset-path`: Path to `.pt` residue dataset file
* `--split-suffix`: Clustering split suffix (e.g. `mil_0.5_nr0.9` or `mil0.5_nr0.9`)
* `--synthetic-demo`: Runs quick smoke-test on synthetic data
* `--bias-3d-mode`: `rbf_gaussian` (default) or `linear_decay`
* `--no-3d-bias`: Disable 3D geometric attention bias
* `--epochs`, `--batch-size`, `--lr`, `--weight-decay`, `--dropout`, `--hidden-dim`, `--num-heads`, `--patience`, `--save-path`

### 4. PyMOL 3D Key Residue Visualization
Inspect the highlighted catalytic / binding residues:
```bash
pymol residue_attention/demo_visualization.pml
```
Or view the continuous attention B-factor heatmap:
```python
load residue_attention/demo_annotated.pdb
spectrum b, blue_white_red, all
```
