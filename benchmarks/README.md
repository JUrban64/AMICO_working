# AMICO Benchmarks Suite

Tato složka obsahuje dvě hlavní baseline metody pro predikci kofaktorové specificity proteinů (`acetyl-CoA`, `ATP`, `B12`, `FAD`, `NAD`):

---

## 1. Foldseek 1-NN Strukturní Benchmark (`foldseek_benchmark.py`)

### Metodika
* **Princip:** Neparametrický 1-Nearest Neighbor (1-NN) baseline založený na strukturním zarovnání celých 3D struktur proteinů.
* **Protokol:**
  * **Query (dotazovaná množina):** Test split.
  * **Database (referenční databáze):** Train split.
* **Přiřazení predikce:**
  * Pro každý testovací protein vyhledá Foldseek (`easy-search`) nejpodobnější protein v trénovací sadě.
  * Výsledky jsou řazeny primárně podle nejvyššího **TM-score (`qtmscore`)** a sekundárně podle nejnižšího **E-value**.
  * Predikovaný kofaktor je kofaktor strukturně nejbližšího trénovacího proteinu (Top-1 hit).
  * Pokud Foldseek nenalezne žádný hit (ani s volným E-value prahem 10.0), použije se fallback na převažující třídu trénovací sady (*majority class*).
* **Férové srovnání:** Automaticky filtruje PDB soubory podle přítomnosti v `esm_dataset.pt` (lze vypnout přepínačem `--all-pdbs`).

### Spuštění z příkazové řádky
```bash
# Standardní split (např. mil_0.5)
python benchmarks/foldseek_benchmark.py --split-suffix mil_0.5

# Shlukování kapes a struktur (struct_pocket_0.5_0.5) s 8 vlákny
python benchmarks/foldseek_benchmark.py --split-suffix struct_pocket_0.5_0.5 --threads 8

# Použití non-redundantního (NR) splitu
python benchmarks/foldseek_benchmark.py --split-suffix mil_0.5 --use-nr

# Specifikace vlastní cesty k Foldseeku nebo PDB složce
python benchmarks/foldseek_benchmark.py --foldseek-bin /path/to/foldseek --pdb-dir /path/to/pdbs
```

---

## 2. Sequence ESM-2 MLP Benchmark (`sequence_mlp_benchmark.py`)

### Metodika
* **Princip:** Čistě sekvenční parametrický baseline model bez znalosti 3D kapes a bez chemických ligandových queries.
* **Architektura (`SequenceMLPClassifier`):**
  * **Vstup:** Globální sekvenční embedding celého proteinu z ESM-2 (1280-dimenzionální vektor).
  * **Vrstvy:**
    $$\text{Linear}(1280 \to 256) \to \text{LayerNorm} \to \text{GELU} \to \text{Dropout}(0.3)$$
    $$\text{Linear}(256 \to 256) \to \text{LayerNorm} \to \text{GELU} \to \text{Dropout}(0.3)$$
    $$\text{Linear}(256 \to 5)$$
  * **Výstup:** Klasifikační logity pro 5 kofaktorových tříd.
* **Tréninkový protokol:**
  * Trénuje na Train splitu s vyváženými vahami tříd a Label Smoothingem (0.1).
  * Validuje na Validation splitu pomocí `ReduceLROnPlateau` a `EarlyStopping` (patience = 12).
  * Nejlepší checkpoint podle validační ztráty je následně vyhodnocen na Test splitu.

### Spuštění z příkazové řádky
```bash
# Spuštění přímo přes dedikovaný benchmark skript
python benchmarks/sequence_mlp_benchmark.py --split-suffix mil_0.5

# Spuštění s upravenými hyperparametry
python benchmarks/sequence_mlp_benchmark.py --split-suffix mil_0.5 --lr 5e-5 --epochs 50 --batch-size 64

# Alternativní spuštění přes hlavní train.py
python train.py --model sequence_mlp --split-suffix mil_0.5
```

---

## 3. Použití v Python kódu

Oba benchmarky lze importovat a spouštět přímo z Pythonu (např. v master benchmarkovacích skriptech):

```python
from benchmarks import run_foldseek_benchmark, run_sequence_mlp, SequenceMLPClassifier

# 1. Běh Foldseek benchmarku
fs_results = run_foldseek_benchmark(split_suffix="mil_0.5", threads=8)
print("Foldseek Test Acc:", fs_results["test_acc"])
print("Foldseek Test Macro F1:", fs_results["test_macro_f1"])

# 2. Běh Sequence MLP benchmarku
mlp_results = run_sequence_mlp(split_suffix="mil_0.5", epochs=50, lr=5e-5)
print("Sequence MLP Test Acc:", mlp_results["test_acc"])
print("Sequence MLP Test Macro F1:", mlp_results["test_macro_f1"])
```
