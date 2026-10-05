"""
Shared feature-extraction settings for AMICO.

The same preprocessing must be applied when building the training dataset and
at inference time. These settings are stamped onto every pocket record by the
dataset builders, copied into training checkpoints, and read back by predict.py.
"""

DEFAULT_ESM_MODEL = "facebook/esm2_t33_650M_UR50D"

# Minimum P2Rank pocket probability kept as a MIL instance.
DEFAULT_MIN_PROB = 0.30

# How pocket embeddings are computed:
#   'slice'  - one ESM pass over the full protein; each pocket = mean of its
#              residues' per-residue embeddings (keeps sequence context).
#   'concat' - legacy: pocket residues joined N->C into a pseudo-peptide and
#              embedded on their own (no sequence context).
POCKET_EMBEDDING_MODES = ("slice", "concat")
DEFAULT_POCKET_EMBEDDING = "slice"

# How sequences longer than ESM-2's context (1022 residues) are handled:
#   'chunk'    - overlapping windows, averaged where they overlap (full length).
#   'truncate' - legacy: only the first 1022 residues are embedded.
LONG_SEQUENCE_MODES = ("chunk", "truncate")
DEFAULT_LONG_SEQUENCES = "chunk"

CONFIG_KEYS = ("min_prob", "pocket_embedding", "long_sequences", "esm_model")

# Settings used to build datasets/checkpoints before this config existed.
LEGACY_PREPROCESSING = {
    "min_prob": DEFAULT_MIN_PROB,
    "pocket_embedding": "concat",
    "long_sequences": "truncate",
    "esm_model": DEFAULT_ESM_MODEL,
}


def make_preprocessing_config(min_prob=DEFAULT_MIN_PROB,
                              pocket_embedding=DEFAULT_POCKET_EMBEDDING,
                              long_sequences=DEFAULT_LONG_SEQUENCES,
                              esm_model=DEFAULT_ESM_MODEL):
    if pocket_embedding not in POCKET_EMBEDDING_MODES:
        raise ValueError(f"pocket_embedding must be one of {POCKET_EMBEDDING_MODES}, got {pocket_embedding!r}")
    if long_sequences not in LONG_SEQUENCE_MODES:
        raise ValueError(f"long_sequences must be one of {LONG_SEQUENCE_MODES}, got {long_sequences!r}")
    return {
        "min_prob": float(min_prob),
        "pocket_embedding": pocket_embedding,
        "long_sequences": long_sequences,
        "esm_model": esm_model,
    }


def config_from_records(records):
    """
    Returns the preprocessing config shared by all pocket records.
    Records without a 'preprocessing' field are treated as LEGACY_PREPROCESSING.
    Raises ValueError if the records were built with different settings.
    """
    seen = {}
    for item in records:
        cfg = item.get("preprocessing") or LEGACY_PREPROCESSING
        key = tuple(cfg.get(k) for k in CONFIG_KEYS)
        seen.setdefault(key, dict(cfg))
    if not seen:
        return None
    if len(seen) > 1:
        raise ValueError(
            "Pocket dataset mixes records built with different preprocessing settings: "
            f"{list(seen.values())}. Rebuild the dataset with a single configuration."
        )
    return next(iter(seen.values()))


def describe(cfg):
    return (f"min_prob={cfg['min_prob']}, pocket_embedding={cfg['pocket_embedding']}, "
            f"long_sequences={cfg['long_sequences']}, esm_model={cfg['esm_model']}")
