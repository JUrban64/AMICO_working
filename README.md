# AMICO: Cofactor Specificity Prediction via Attention MIL

AMICO predicts enzyme cofactor binding specificity (`ATP`, `NAD`, `FAD`, `B12`, `acetyl-CoA`) from protein 3D structures under strict cross-fold structural generalization.

---

## 🔄 End-to-End Pipeline

```text
[Raw PDB Structure] 
        │
        ├── 1. Pocket Detection (P2Rank) ──────────> 3D Binding Pockets (Residues + Coordinates)
        │                                                          │
        ├── 2. Representation Learning (ESM-2)                     │
        │      ├─ Whole-protein sequence ──────────> Global Context Vector [1280]
        │      └─ Pocket residue segments ─────────> Multi-Instance Pocket Bag [N, 1280]
        │                                                          │
        ├── 3. AMICO Model (LigandCrossAttentionMIL / SelfAttentionMIL)
        │      ├─ ECFP4 Chemical Fingerprints [5, 1024] cross-attend over [N+1, D]
        │      ├─ Multi-Head Attention identifies pocket relevance weights
        │      └─ Monte Carlo Dropout (T=30) -> Probabilities + Epistemic Uncertainty
        │                                                          │
        └── 4. Downstream Docking (AutoDock Vina, Optional)
               └─ Docks 3D cofactor conformer into top P2Rank pocket center
```

### Pipeline Stages:
1. **Data Curation & Diversity Sampling (`data_prep/alphafoldDB_APi.py`)**:
   Downloads AlphaFold structures across cofactors using UniProt cursor pagination with taxonomic (`taxonId`) and functional (`EC`) throttling to eliminate redundancy.
2. **Homology-Free Structural Splitting (`data_prep/structure_clustering.py`)**:
   Clusters structures with Foldseek to construct cross-fold train/validation/test splits, preventing data leakage across structural superfamilies.
3. **Feature Preprocessing (`data_prep/build_esm_dataset.py`)**:
   Executes P2Rank pocket discovery and ESM-2 (`esm2_t33_650M_UR50D`) extraction in batch to produce `esm_dataset.pt` (pocket bags) and `esm_full_proteins.pt` (global sequence context).
4. **Attention MIL Classification (`model_ligand_cross_att.py`, `model_self_attention.py`)**:
   - `LigandCrossAttentionMIL`: Morgan ECFP4 cofactor fingerprints act as queries cross-attending over structural pocket representations and global context.
   - `SelfAttentionMIL`: Self-attention pooling across pocket instances without chemical queries.
5. **Inference & AutoDock Vina Docking (`predict.py`, `docking_utils.py`)**:
   Accepts a raw PDB, runs end-to-end pocket detection + ESM-2 embedding + AMICO inference, quantifies uncertainty, and optionally docks the predicted ligand into the pocket center.

---

## 📁 Repository Structure

```text
AMICO/
├── README.md                            # Documentation
├── requirements.txt                     # Dependencies
│
├── model_ligand_cross_att.py            # LigandCrossAttentionMIL architecture
├── model_self_attention.py              # SelfAttentionMIL architecture
│
├── dataset.py                           # Dataset loader & collator for MIL bags
├── train_ligand_cross_att.py            # Trainer for LigandCrossAttentionMIL
├── train_self_attention.py              # Trainer for SelfAttentionMIL
│
├── predict.py                           # End-to-end inference CLI & API
├── p2rank_utils.py                      # P2Rank execution & output parsing
├── esm_extractor.py                     # ESM-2 feature extractor
├── docking_utils.py                     # AutoDock Vina preparation & docking
├── tune_optuna.py                       # Hyperparameter optimization (Optuna)
│
└── data_prep/
    ├── alphafoldDB_APi.py               # AlphaFold DB downloading & metadata curation
    ├── structure_clustering.py          # Foldseek clustering & cluster-split generation
    ├── build_esm_dataset.py             # Batch P2Rank + ESM-2 extraction pipeline
    ├── generate_pocket_embeddings.py    # Pocket-only embedding builder
    └── generate_full_protein_embeddings.py # Full-protein embedding builder
```

---

## ⚙️ Installation

```bash
git clone https://github.com/JUrban64/AMICO.git
cd AMICO
pip install -r requirements.txt
```

*Requirements:* Python ≥ 3.10, PyTorch ≥ 2.0, [P2Rank](https://github.com/rdkit/p2rank) (optional for inference from raw PDBs), and AutoDock Vina (optional for docking).

---

## 🚀 Usage

### 1. End-to-End Inference from a PDB Structure
```bash
# Predict cofactor specificity and estimate MC Dropout uncertainty:
python predict.py \
    --pdb /path/to/protein.pdb \
    --checkpoint ligand_cross_mil_best.pt \
    --mc-samples 30

# Predict and automatically dock the predicted cofactor into the top pocket:
python predict.py \
    --pdb /path/to/protein.pdb \
    --checkpoint ligand_cross_mil_best.pt \
    --dock \
    --dock-out ./docking_results
```

### 2. Model Training
```bash
# Train Ligand Cross-Attention MIL:
python train_ligand_cross_att.py --split-suffix mil_0.5 --epochs 50

# Train with Optuna-tuned hyperparameters:
python train_ligand_cross_att.py --config-json best_params_ligand_cross_mil_0.5.json --epochs 50

# Train Self-Attention MIL:
python train_self_attention.py --split-suffix mil_0.5 --epochs 50
```

### 3. Hyperparameter Tuning
```bash
python tune_optuna.py --model ligand_cross_mil --split-suffix mil_0.5 --n-trials 50
```

### 4. Data Preparation Pipeline

#### A. Download AlphaFold Structures (`alphafoldDB_APi.py`)
Downloads cofactor-binding structures from AlphaFold DB with taxonomic and functional diversity controls. Configured via parameters at the top of `data_prep/alphafoldDB_APi.py`:
- `TARGET_PER_CLASS`: Target quotas per cofactor (e.g. `{'ATP': 17000, 'NAD': 14000, 'FAD': 10000, 'acetyl-CoA': 6500, 'B12': 2500}`).
- `MAX_PER_EC`: Max enzymes per primary EC number (default: `40`).
- `MAX_PER_ORG_EC`: Max enzymes per organism-EC combination (default: `1`).
- `MIN_LENGTH` / `MAX_LENGTH`: Sequence length bounds (default: `60` – `1400` aa).
- `NUM_WORKERS`: Parallel download threads (default: `16`).

```bash
# Run structure download and metadata generation (resumes automatically):
python data_prep/alphafoldDB_APi.py
```

#### B. Structural Clustering & Zero-Leakage Splits (`structure_clustering.py`)
Clusters structures with **Foldseek** (`easy-cluster`) and performs stratified group splitting to guarantee that no structural folds or superfamilies leak across splits.

**Clustering Options:**
- `--tmscore-threshold`, `--tmscore` *(float, default: `0.5`)*: TM-score threshold for Foldseek easy-cluster (e.g. `0.5` enforces distinct structural fold separation).
- `--nr-threshold` *(float, optional)*: Non-redundant pre-filtering threshold (e.g. `--nr-threshold 0.9` discards structures with TM-score ≥ 0.9 prior to clustering).
- `--metadata` *(str, optional)*: Path to `dataset_metadata.tsv` or `master_dataset_cache.json` for class-stratified splitting.
- `--test`: Quick dry-run on 30 structures.

```bash
# Standard fold-level clustering (TM-score 0.5):
python data_prep/structure_clustering.py --tmscore-threshold 0.5

# Two-level clustering (90% non-redundancy pre-filter + TM-score 0.5 clustering):
python data_prep/structure_clustering.py --tmscore-threshold 0.5 --nr-threshold 0.9

# Outputs generated:
#   train_mil_0.5.txt, validation_mil_0.5.txt, test_mil_0.5.txt, clusters_mil_0.5.json
```

#### C. Pocket Detection & ESM-2 Extraction (`build_esm_dataset.py`)
Processes PDB structures through P2Rank and ESM-2 in batch to produce training datasets (`esm_dataset.pt` and `esm_full_proteins.pt`).

**Extraction Options:**
- `--pdb-dir` *(str)*: Directory containing `.pdb` files (default: `structures/all_pdbs`).
- `--metadata` *(str)*: Path to `dataset_metadata.tsv`.
- `--min-prob` *(float, default: `0.30`)*: Minimum P2Rank pocket probability threshold.
- `--threads` *(int, default: `8`)*: CPU threads for P2Rank execution.
- `--skip-p2rank`: Skip running P2Rank and reuse existing pocket CSV files.
- `--pockets-only` / `--full-only`: Generate only `esm_dataset.pt` or `esm_full_proteins.pt`.

```bash
# Full extraction pipeline (P2Rank + ESM-2):
python data_prep/build_esm_dataset.py --threads 8 --min-prob 0.30

# Reuse existing P2Rank pocket predictions:
python data_prep/build_esm_dataset.py --skip-p2rank
```

---

## 💻 Python API Example

```python
from predict import AMICOPredictor

predictor = AMICOPredictor(checkpoint_path="ligand_cross_mil_best.pt")

# End-to-end: PDB -> P2Rank -> ESM-2 -> AMICO -> Prediction
result = predictor.predict_from_pdb("protein.pdb", mc_samples=30)

print(f"Predicted Cofactor: {result['predicted_cofactor']}")
print(f"Confidence:         {result['confidence'] * 100:.1f} %")
print(f"Uncertainty (std):  ±{result['uncertainty_std'] * 100:.2f} %")
print(f"Pocket Center:      {result['best_pocket_center']}")
```
