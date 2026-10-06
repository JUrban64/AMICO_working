import torch
import numpy as np

from data_prep.preprocessing import (
    DEFAULT_ESM_MODEL,
    DEFAULT_LONG_SEQUENCES,
    LONG_SEQUENCE_MODES,
)

# ESM-2 has 1026 positions; <cls> and <eos> take two, leaving 1024. We keep the
# historical 1024-token limit (incl. special tokens) -> 1022 residues per pass.
ESM_MAX_RESIDUES = 1022
CHUNK_OVERLAP = 256


class ESMFeatureExtractor:
    """
    ESM-2 embedding extractor for whole-protein sequences and predicted binding pockets.
    Default model: facebook/esm2_t33_650M_UR50D (1280 dimensions).
    """
    def __init__(self, model_name=DEFAULT_ESM_MODEL, device=None,
                 long_sequences=DEFAULT_LONG_SEQUENCES, verbose=True):
        try:
            from transformers import AutoTokenizer, EsmModel
        except ImportError:
            raise ImportError(
                "The 'transformers' package is not installed. "
                "Please install it using: pip install transformers"
            )

        if long_sequences not in LONG_SEQUENCE_MODES:
            raise ValueError(f"long_sequences must be one of {LONG_SEQUENCE_MODES}")

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
        else:
            self.device = torch.device(device)

        self.long_sequences = long_sequences
        self.verbose = verbose
        self.num_long_sequences = 0  # how many sequences exceeded ESM_MAX_RESIDUES

        print(f"-> Loading ESM-2 model ({model_name}) on device {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.hidden_size = self.model.config.hidden_size

    def _forward(self, sequence):
        """Single ESM pass; caller guarantees len(sequence) <= ESM_MAX_RESIDUES."""
        inputs = self.tokenizer(sequence, return_tensors="pt", add_special_tokens=True, truncation=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[0, 1:-1, :]  # drop <cls>/<eos> -> [L, D]

    def extract_sequence_embeddings_raw(self, sequence, long_sequences=None):
        """
        Per-residue embeddings for an amino acid sequence.

        Sequences longer than ESM_MAX_RESIDUES are either embedded in overlapping
        windows (long_sequences='chunk', full length returned) or truncated
        (long_sequences='truncate', only the first ESM_MAX_RESIDUES returned).
        Both cases are counted in self.num_long_sequences and reported when verbose.

        Returns:
            torch.Tensor: [L, D] (L = len(sequence), or ESM_MAX_RESIDUES if truncated)
        """
        if not sequence:
            raise ValueError("Empty sequence provided to extract_sequence_embeddings_raw")

        mode = long_sequences or self.long_sequences
        L = len(sequence)
        if L <= ESM_MAX_RESIDUES:
            return self._forward(sequence)

        self.num_long_sequences += 1
        if mode == 'truncate':
            if self.verbose:
                print(f"⚠️  [ESM] Sequence of {L} aa truncated to the first {ESM_MAX_RESIDUES} residues "
                      f"({L - ESM_MAX_RESIDUES} residues not embedded).")
            return self._forward(sequence[:ESM_MAX_RESIDUES])

        window = ESM_MAX_RESIDUES
        step = window - CHUNK_OVERLAP
        starts = list(range(0, L - window, step)) + [L - window]
        if self.verbose:
            print(f"ℹ️  [ESM] Sequence of {L} aa exceeds {window} residues; "
                  f"embedding in {len(starts)} overlapping windows.")

        acc = torch.zeros(L, self.hidden_size, device=self.device)
        counts = torch.zeros(L, 1, device=self.device)
        for s in starts:
            acc[s:s + window] += self._forward(sequence[s:s + window])
            counts[s:s + window] += 1
        return acc / counts

    def extract_sequence_embedding(self, sequence, long_sequences=None):
        """
        Global protein embedding via mean pooling across residues.

        Returns:
            torch.Tensor: [D]
        """
        raw_emb = self.extract_sequence_embeddings_raw(sequence, long_sequences=long_sequences)
        return torch.mean(raw_emb, dim=0).cpu()

    @staticmethod
    def pocket_residue_indices(pocket):
        """0-based positions of the pocket's residues in the full parsed sequence."""
        if pocket.get('residue_indices'):
            return list(pocket['residue_indices'])
        return [r['seq_idx'] for r in pocket.get('residues', []) if 'seq_idx' in r]

    def extract_features(self, parsed_data, long_sequences=None):
        """
        Computes the global protein embedding and one contextual embedding per pocket.

        A single ESM pass is computed over the full protein sequence. Each pocket's
        embedding is the mean of its residues' per-residue embeddings ('slice' mode),
        preserving real 3D and sequence context.

        Returns:
            dict with
              'full_protein_feature': [D] tensor
              'pocket_features':      [N, D] tensor (row i <-> pockets[i])
              'pockets':              list of pocket dicts that received a feature
              'dropped_pocket_ids':   pocket ids that could not be embedded
        """
        full_seq = parsed_data['full_sequence']
        pockets = parsed_data.get('pockets', [])
        kept, feats, dropped = [], [], []

        per_res = self.extract_sequence_embeddings_raw(full_seq, long_sequences=long_sequences)  # [L, D]
        full_feature = per_res.mean(dim=0).cpu()
        L = per_res.size(0)
        for p in pockets:
            idx = sorted({i for i in self.pocket_residue_indices(p) if 0 <= i < L})
            if not idx:
                dropped.append(p.get('pocket_id'))
                continue
            idx_t = torch.tensor(idx, dtype=torch.long, device=per_res.device)
            feats.append(per_res.index_select(0, idx_t).mean(dim=0).cpu())
            kept.append(p)

        if dropped and self.verbose:
            print(f"⚠️  [ESM] {len(dropped)} pocket(s) had no residues mappable to the sequence and were skipped: {dropped}")

        pocket_features = torch.stack(feats, dim=0) if feats else torch.empty((0, full_feature.numel()), dtype=torch.float32)
        return {
            'full_protein_feature': full_feature,
            'pocket_features': pocket_features,
            'pockets': kept,
            'dropped_pocket_ids': dropped,
        }

    def extract_all_from_parsed(self, parsed_data, long_sequences=None):
        """
        Wrapper around extract_features().

        Returns:
            tuple: (pocket_features [N, D], full_protein_feature [D])
        """
        out = self.extract_features(parsed_data, long_sequences=long_sequences)
        return out['pocket_features'], out['full_protein_feature']
