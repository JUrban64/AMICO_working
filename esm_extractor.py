import torch
import numpy as np


class ESMFeatureExtractor:
    """
    ESM-2 embedding extractor for whole-protein sequences and predicted binding pockets.
    Default model: facebook/esm2_t33_650M_UR50D (1280 dimensions).
    """
    def __init__(self, model_name="facebook/esm2_t33_650M_UR50D", device=None):
        try:
            from transformers import AutoTokenizer, EsmModel
        except ImportError:
            raise ImportError(
                "The 'transformers' package is not installed. "
                "Please install it using: pip install transformers"
            )

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
        else:
            self.device = torch.device(device)

        print(f"-> Loading ESM-2 model ({model_name}) on device {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def extract_sequence_embeddings_raw(self, sequence, max_length=1024):
        """
        Extracts per-residue embeddings for a given amino acid sequence.
        
        Returns:
            torch.Tensor: [L, 1280]
        """
        if not sequence or len(sequence) == 0:
            raise ValueError("Empty sequence provided to extract_sequence_embeddings_raw")

        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Remove <cls> and <eos> tokens
        embeddings = outputs.last_hidden_state[0, 1:-1, :]  # [L, 1280]
        return embeddings

    def extract_sequence_embedding(self, sequence):
        """
        Extracts global sequence embedding for a protein via mean pooling across residues.
        
        Returns:
            torch.Tensor: [1280]
        """
        raw_emb = self.extract_sequence_embeddings_raw(sequence)
        mean_emb = torch.mean(raw_emb, dim=0)  # [1280]
        return mean_emb.cpu()

    def extract_pocket_embeddings(self, pocket_sequences):
        """
        Extracts embeddings for a list of binding pocket sequences.
        Performs mean pooling across residues for each individual pocket.
        
        Args:
            pocket_sequences: list of strings (amino acid sequences for each pocket)
            
        Returns:
            torch.Tensor: [N_pockets, 1280]
        """
        if not pocket_sequences:
            return torch.empty((0, 1280), dtype=torch.float32)

        pocket_embs = []
        for seq in pocket_sequences:
            if not seq or len(seq) == 0:
                continue
            raw_emb = self.extract_sequence_embeddings_raw(seq)
            mean_emb = torch.mean(raw_emb, dim=0)  # [1280]
            pocket_embs.append(mean_emb.cpu())

        if not pocket_embs:
            return torch.empty((0, 1280), dtype=torch.float32)

        return torch.stack(pocket_embs, dim=0)  # [N, 1280]

    def extract_all_from_parsed(self, parsed_data):
        """
        Processes parsed P2Rank output and returns paired tensors for AMICO models.
        
        Returns:
            tuple: (pocket_features [N, 1280], full_protein_feature [1280])
        """
        full_seq = parsed_data['full_sequence']
        full_protein_feature = self.extract_sequence_embedding(full_seq)

        pocket_seqs = [p['sequence'] for p in parsed_data['pockets']]
        pocket_features = self.extract_pocket_embeddings(pocket_seqs)

        return pocket_features, full_protein_feature
