import os
from pathlib import Path
import torch
import numpy as np

TARGET_NAMES = ['acetyl-CoA', 'ATP', 'B12', 'FAD', 'NAD']


def rank_key_residues(model, residue_features, padding_mask, full_protein_feature, 
                      residue_metadata, residue_coords=None, target_class=None, top_k=10):
    """
    Ranks residues by attention importance for a given or predicted cofactor.
    
    Args:
        model: ResidueSelfAttentionMIL or ResidueLigandCrossAttentionMIL
        residue_features: [1, M, 1280] or [M, 1280]
        padding_mask: [1, M] or [M]
        full_protein_feature: [1, 1280] or [1280]
        residue_metadata: list of dicts for the M residues
        residue_coords: [1, M, 3] or [M, 3] optional
        target_class: int (0..4) or None (uses top predicted class)
        top_k: int, how many top residues to return
        
    Returns:
        dict:
            'predicted_class_idx': int,
            'predicted_cofactor': str,
            'probabilities': dict of cofactor -> prob,
            'ranked_residues': list of dicts for top_k residues,
            'all_scores': np.ndarray [M]
    """
    model.eval()
    
    # Ensure batch dimension
    if residue_features.dim() == 2:
        residue_features = residue_features.unsqueeze(0)
    if padding_mask.dim() == 1:
        padding_mask = padding_mask.unsqueeze(0)
    if full_protein_feature.dim() == 1:
        full_protein_feature = full_protein_feature.unsqueeze(0)
    if residue_coords is not None and residue_coords.dim() == 2:
        residue_coords = residue_coords.unsqueeze(0)

    device = next(model.parameters()).device
    residue_features = residue_features.to(device)
    padding_mask = padding_mask.to(device)
    full_protein_feature = full_protein_feature.to(device)
    if residue_coords is not None:
        residue_coords = residue_coords.to(device)

    with torch.no_grad():
        if hasattr(model, 'cross_attn'):
            logits, cross_weights, _ = model(
                residue_features, padding_mask, full_protein_feature, residue_coords
            )
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            pred_class = int(np.argmax(probs)) if target_class is None else target_class
            importance = model.get_residue_importance(cross_weights, target_class=pred_class)[0].cpu().numpy()
        else:
            logits, self_weights = model(
                residue_features, padding_mask, full_protein_feature, residue_coords
            )
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            pred_class = int(np.argmax(probs)) if target_class is None else target_class
            importance = model.get_residue_importance(self_weights)[0].cpu().numpy()

    M = len(residue_metadata)
    scores = importance[:M]

    # Combine metadata with attention scores
    annotated = []
    for i, meta in enumerate(residue_metadata):
        score_val = float(scores[i])
        entry = dict(meta)
        entry['attention_score'] = score_val
        entry['attention_percent'] = score_val * 100.0
        annotated.append(entry)

    # Sort descending by attention score
    annotated.sort(key=lambda x: x['attention_score'], reverse=True)
    top_residues = annotated[:top_k]

    for rank_idx, r in enumerate(top_residues, 1):
        r['rank'] = rank_idx

    prob_dict = {TARGET_NAMES[i]: float(probs[i]) for i in range(len(TARGET_NAMES))}

    return {
        'predicted_class_idx': pred_class,
        'predicted_cofactor': TARGET_NAMES[pred_class],
        'probabilities': prob_dict,
        'confidence': float(probs[pred_class]),
        'ranked_residues': top_residues,
        'all_scores': scores
    }


def format_key_residues_summary(ranking_result):
    """Formats ranked key residues into a readable markdown table."""
    cofactor = ranking_result['predicted_cofactor']
    conf = ranking_result['confidence'] * 100.0
    lines = [
        f"### Predicted Cofactor: **{cofactor}** (Confidence: {conf:.1f}%)",
        "",
        "| Rank | Residue | Chain | PDB Pos | Sequence Idx | Attention Share | Pocket(s) | P2Rank Score |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for r in ranking_result['ranked_residues']:
        pockets_str = ",".join(str(p) for p in r.get('pockets', []))
        lines.append(
            f"| **#{r['rank']}** | **{r.get('resname', 'UNK')}** ({r.get('one_letter', '')}) | "
            f"{r.get('chain', 'A')} | {r.get('resseq', '')} | {r.get('seq_idx', '')} | "
            f"**{r['attention_percent']:.2f} %** | {pockets_str} | {r.get('p2rank_score', 0.0):.2f} |"
        )

    return "\n".join(lines)


def export_pymol_script(pdb_path, ranking_result, out_pml_path, top_k=5):
    """
    Generates an automated PyMOL visualization script (.pml).
    Highlights and sticks the top-k key residues with custom 3D labels.
    """
    pdb_path = Path(pdb_path).resolve()
    out_pml_path = Path(out_pml_path)
    cofactor = ranking_result['predicted_cofactor']
    top_res = ranking_result['ranked_residues'][:top_k]

    sel_strings = []
    label_commands = []
    
    for r in top_res:
        c = r.get('chain', 'A')
        num = r.get('resseq', '')
        sel_name = f"key_{r.get('resname', 'RES')}_{num}"
        sel_strings.append(sel_name)
        label_commands.append(
            f"select {sel_name}, chain {c} and resi {num}\n"
            f"show sticks, {sel_name}\n"
            f"color brightorange, {sel_name} and name C*\n"
            f'label {sel_name} and name CA, "{r.get("resname","")}{num} ({r["attention_percent"]:.1f}%)"\n'
        )

    combined_sel = " + ".join(sel_strings) if sel_strings else "none"

    pml_content = f"""# ==============================================================================
# AMICO Residue-Level Attention Visualization for {pdb_path.name}
# Predicted Cofactor: {cofactor}
# ==============================================================================

reinitialize
load {str(pdb_path)}, protein

# Base protein styling
hide everything, all
show cartoon, protein
color palecyan, protein
set cartoon_transparency, 0.25

# Styling labels
set label_size, 16
set label_color, white
set label_font_id, 7
set label_outline_color, black
set label_shadow_mode, 2

# Highlight Key Residues Identified by AMICO Attention
{''.join(label_commands)}

# Center on key binding site residues
select all_key_residues, {combined_sel}
zoom all_key_residues, 6.0
orient all_key_residues

# Background and lighting
bg_color black
set ray_shadows, 0
set antialias, 2
"""
    with open(out_pml_path, 'w', encoding='utf-8') as f:
        f.write(pml_content)

    return out_pml_path


def write_pdb_with_attention_bfactor(pdb_in_path, pdb_out_path, ranking_result):
    """
    Writes a new PDB file where the B-factor column of each residue is replaced
    by the normalized AMICO attention importance score (0.00 to 100.00).
    Enables instant heatmap visualization via 'spectrum b, blue_white_red'.
    """
    # Build mapping (chain, resseq) -> b_factor
    scores_map = {}
    all_scores = ranking_result.get('all_scores', [])
    max_score = float(np.max(all_scores)) if len(all_scores) > 0 and np.max(all_scores) > 0 else 1.0

    for r in ranking_result['ranked_residues']:
        c = r.get('chain', 'A').strip()
        num = str(r.get('resseq', '')).strip()
        # Scale score relative to maximum (0 to 100)
        norm_val = (r['attention_score'] / max_score) * 100.0
        scores_map[(c, num)] = norm_val
        scores_map[num] = norm_val

    with open(pdb_in_path, 'r', encoding='utf-8', errors='replace') as f_in, \
         open(pdb_out_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            if line.startswith(('ATOM  ', 'HETATM')):
                chain_id = line[21].strip()
                resseq = line[22:26].strip()
                b_val = scores_map.get((chain_id, resseq), scores_map.get(resseq, 0.00))
                # Format into cols 61-66 (6 characters, 2 decimals)
                b_str = f"{b_val:6.2f}"[:6]
                new_line = line[:60] + b_str + line[66:]
                f_out.write(new_line)
            else:
                f_out.write(line)

    return pdb_out_path
