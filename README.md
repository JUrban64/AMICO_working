# AMICO: Cofactor Specificity Prediction via Attention MIL

AMICO predicts enzyme cofactor binding specificity (`ATP`, `NAD`, `FAD`, `B12`, `acetyl-CoA`) from protein 3D structures under strict cross-fold structural generalization.

---

## ⚙️ Installation

### Option A: Via Conda / Mamba (Recommended, includes Foldseek)
```bash
git clone https://github.com/JUrban64/AMICO.git
cd AMICO
conda env create -f environment.yml
conda activate amico
```

### Option B: Via pip
```bash
git clone https://github.com/JUrban64/AMICO.git
cd AMICO
pip install -r requirements.txt
conda install -c conda-forge -c bioconda foldseek
```

*Requirements:* Python ≥ 3.10, PyTorch ≥ 2.0.
*External tools:*
- [Foldseek](https://github.com/steineggerlab/foldseek) (included in `environment.yml`; required for structural clustering and Foldseek benchmarks).
- [P2Rank](https://github.com/rdkit/p2rank) (bundled in `p2rank_2.5.1/`, requires Java JRE/JDK ≥ 11 for pocket prediction from raw PDBs).
- [AutoDock Vina](https://vina.scripps.edu/) (optional for downstream docking: `pip install vina` or system binary).

---

## 🚀 Usage & Workflow

The AMICO workflow runs end-to-end from raw data collection to cross-fold structural clustering, representation extraction, model training, and inference with molecular docking:

```text
1. Download Structures ──> 2. Foldseek Clustering ──> 3. P2Rank & ESM-2 ──> 4. Tuning & Training ──> 5. Inference & Docking
   (alphafoldDB_APi.py)      (structure_clustering)      (build_esm_dataset)    (train_*.py)           (predict.py)
```

---

### Step 1: Download AlphaFold Structures (`data_prep/alphafoldDB_APi.py`)

Downloads cofactor-binding structures from AlphaFold DB with taxonomic and functional diversity controls to prevent dataset bias. Configured via parameters at the top of `data_prep/alphafoldDB_APi.py`:

- `TARGET_PER_CLASS`: Target quotas per cofactor (e.g. `{'ATP': 17000, 'NAD': 14000, 'FAD': 10000, 'acetyl-CoA': 6500, 'B12': 2500}`).
- `MAX_PER_EC`: Maximum enzymes per primary EC number (default: `40`).
- `MAX_PER_ORG_EC`: Maximum enzymes per organism-EC combination (default: `1`).
- `MIN_LENGTH` / `MAX_LENGTH`: Sequence length bounds (default: `60` – `1400` aa).
- `NUM_WORKERS`: Parallel download threads (default: `16`).

```bash
# Run structure download and metadata generation (resumes automatically):
python data_prep/alphafoldDB_APi.py
```
*Outputs:* PDB files in `structures/all_pdbs/`, metadata in `structures/dataset_metadata.tsv`, and cache in `structures/master_dataset_cache.json`.

---

### Step 2: Structural Clustering & Zero-Leakage Splits (`data_prep/structure_clustering.py`)

Clusters structures using **Foldseek** (`easy-cluster`) and performs class-stratified group splitting to guarantee that no structural folds or superfamilies leak across train, validation, and test splits.

**Clustering Options:**
- `--tmscore-threshold`, `--tmscore` *(float, default: `0.5`)*: TM-score threshold for Foldseek clustering (`0.5` enforces distinct structural fold separation).
- `--nr-threshold` *(float, optional)*: Non-redundant pre-filtering threshold (e.g. `--nr-threshold 0.9` discards structures with TM-score ≥ 0.9 prior to clustering).
- `--metadata` *(str, optional)*: Path to `dataset_metadata.tsv` or `master_dataset_cache.json` for class-stratified splitting.
- `--test`: Quick dry-run on 30 structures.

```bash
# Standard fold-level clustering (TM-score 0.5):
python data_prep/structure_clustering.py --tmscore-threshold 0.5

# Two-level clustering (90% non-redundancy pre-filter + TM-score 0.5 clustering):
python data_prep/structure_clustering.py --tmscore-threshold 0.5 --nr-threshold 0.9
```
*Outputs generated:* `train_mil_0.5.txt`, `validation_mil_0.5.txt`, `test_mil_0.5.txt`, and `clusters_mil_0.5.json`.

---

### Step 3: Pocket Discovery & ESM-2 Feature Extraction (`data_prep/build_esm_dataset.py`)

Executes P2Rank pocket discovery and ESM-2 (`esm2_t33_650M_UR50D`) embedding extraction in batch to produce the multi-instance pocket bags and global protein representations.

**Extraction Options:**
- `--pdb-dir` *(str)*: Directory containing `.pdb` files (default: `structures/all_pdbs`).
- `--metadata` *(str)*: Path to `dataset_metadata.tsv`.
- `--min-prob` *(float, default: `0.30`)*: Minimum P2Rank pocket probability threshold.
- `--threads` *(int, default: `8`)*: CPU threads for P2Rank execution.
- `--skip-p2rank`: Skip running P2Rank and reuse existing pocket CSV predictions.
- `--pockets-only` / `--full-only`: Generate only `esm_dataset.pt` or `esm_full_proteins.pt`.

```bash
# Full extraction pipeline (P2Rank + ESM-2):
python data_prep/build_esm_dataset.py --threads 8 --min-prob 0.30

# Reuse existing P2Rank pocket predictions (extract ESM-2 only):
python data_prep/build_esm_dataset.py --skip-p2rank
```
*Outputs generated:* `data_prep/esm_dataset.pt` (pocket instances) and `data_prep/esm_full_proteins.pt` (global sequence embeddings).

---

### Step 4: Hyperparameter Optimization (`tune_optuna.py`)

Tunes architecture and training hyperparameters using Optuna with median pruning:

```bash
# Optimize LigandCrossAttentionMIL on the 0.5 TM-score split:
python tune_optuna.py --model ligand_cross_mil --split-suffix mil_0.5 --n-trials 50

# Optimize SelfAttentionMIL:
python tune_optuna.py --model self_attention_mil --split-suffix mil_0.5 --n-trials 50
```
*Outputs generated:* `best_params_<model>_<suffix>.json` and SQLite study database.

---

### Step 5: Model Training

Train the selected attention architecture using the cross-fold structural splits:

```bash
# 1. Train Ligand Cross-Attention MIL (chemical queries cross-attending over pockets):
python train_ligand_cross_att.py --split-suffix mil_0.5 --epochs 50 --save-model weights/ligand_cross_mil_best.pt

# 2. Train with Optuna-tuned hyperparameters:
python train_ligand_cross_att.py --config-json best_params_ligand_cross_mil_0.5.json --epochs 50 --save-model weights/ligand_cross_mil_best.pt

# 3. Train Self-Attention MIL (baseline without chemical queries):
python train_self_attention.py --split-suffix mil_0.5 --epochs 50 --save-model weights/self_attention_mil_best.pt
```
*Outputs generated:* Trained model checkpoints in `weights/` (e.g. `weights/ligand_cross_mil_best.pt`, `weights/self_attention_mil_best.pt`).

---

### Step 6: End-to-End Inference & Molecular Docking (`predict.py`)

Run cofactor specificity prediction on any raw PDB structure with Monte Carlo dropout epistemic uncertainty estimation and optional automated docking into the top predicted binding pocket:

```bash
# Predict cofactor specificity and estimate MC Dropout uncertainty:
python predict.py \
    --pdb /path/to/protein.pdb \
    --checkpoint weights/ligand_cross_mil_best.pt \
    --mc-samples 30

# Predict and automatically dock the predicted cofactor into the top P2Rank pocket:
python predict.py \
    --pdb /path/to/protein.pdb \
    --checkpoint weights/ligand_cross_mil_best.pt \
    --dock \
    --dock-out ./docking_results
```

---

### Step 7: Python API Example

You can integrate AMICO directly into your Python scripts or analysis pipelines:

```python
from predict import AMICOPredictor

predictor = AMICOPredictor(checkpoint_path="weights/ligand_cross_mil_best.pt")

# End-to-end: Raw PDB -> P2Rank -> ESM-2 -> AMICO -> Cofactor Specificity
result = predictor.predict_from_pdb("protein.pdb", mc_samples=30)

print(f"Predicted Cofactor: {result['predicted_cofactor']}")
print(f"Confidence:         {result['confidence'] * 100:.1f} %")
print(f"Uncertainty (std):  ±{result['uncertainty_std'] * 100:.2f} %")
print(f"Top Binding Pocket: Pocket #{result['best_binding_pocket']}")
if result.get('best_p2rank_pocket_center'):
    print(f"Pocket Center (Å):  {result['best_p2rank_pocket_center']}")
```

---

## 🔄 End-to-End Architecture

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
        │      ├─ ECFP4 Chemical Fingerprints [5, 2048] cross-attend over [N+1, D]
        │      ├─ Multi-Head Attention identifies pocket relevance weights
        │      └─ Monte Carlo Dropout (T=30) -> Probabilities + Epistemic Uncertainty
        │                                                          │
        └── 4. Downstream Docking (AutoDock Vina, Optional)
               └─ Docks 3D cofactor conformer into top P2Rank pocket center
```

### Architecture Details:
- **LigandCrossAttentionMIL (`model_ligand_cross_att.py`)**: Uses Morgan ECFP4 chemical fingerprints of target cofactors as queries that cross-attend over structural pocket representations and the global protein context vector.
- **SelfAttentionMIL (`model_self_attention.py`)**: Self-attention pooling mechanism across pocket instances and full-protein context without chemical queries.
- **Monte Carlo Dropout**: Epistemic uncertainty estimation and rejection thresholds can be further optimized and calibrated on validation data (e.g., via BALD mutual information or temperature scaling).
- **AutoDock Vina Docking (`utils/docking_utils.py`)**: Automatically computes 3D cofactor conformers via RDKit and docks into the center of mass of the top P2Rank pocket.

---

## 📁 Repository Structure

```text
AMICO/
├── README.md                            # Documentation
├── requirements.txt                     # Dependencies
│
├── weights/                             # Directory for trained PyTorch checkpoints (.pt)
│
├── model_ligand_cross_att.py            # LigandCrossAttentionMIL architecture
├── model_self_attention.py              # SelfAttentionMIL architecture
│
├── dataset.py                           # Dataset loader & collator for MIL bags
├── train_ligand_cross_att.py            # Trainer for LigandCrossAttentionMIL
├── train_self_attention.py              # Trainer for SelfAttentionMIL
├── tune_optuna.py                       # Hyperparameter optimization (Optuna)
│
├── predict.py                           # End-to-end inference CLI & API
│
├── utils/
│   ├── p2rank_utils.py                  # P2Rank execution & output parsing
│   ├── esm_extractor.py                 # ESM-2 feature extractor
│   └── docking_utils.py                 # AutoDock Vina preparation & docking
│
└── data_prep/
    ├── preprocessing.py                 # Feature extraction & preprocessing config
    ├── alphafoldDB_APi.py               # AlphaFold DB downloading & metadata curation
    ├── structure_clustering.py          # Foldseek clustering & cluster-split generation
    ├── build_esm_dataset.py             # Batch P2Rank + ESM-2 extraction pipeline
    ├── generate_pocket_embeddings.py    # Pocket-only embedding builder
    └── generate_full_protein_embeddings.py # Full-protein embedding builder
```
