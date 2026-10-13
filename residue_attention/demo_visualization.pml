# ==============================================================================
# AMICO Residue-Level Attention Visualization for AF-Q9NRG4-F1-model_v6.pdb
# Predicted Cofactor: ATP
# ==============================================================================

reinitialize
load /Users/jachymurban/Desktop/AMICO/AF-Q9NRG4-F1-model_v6.pdb, protein

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
select key_ASN_380, chain A and resi 380
show sticks, key_ASN_380
color brightorange, key_ASN_380 and name C*
label key_ASN_380 and name CA, "ASN380 (2.3%)"
select key_ALA_40, chain A and resi 40
show sticks, key_ALA_40
color brightorange, key_ALA_40 and name C*
label key_ALA_40 and name CA, "ALA40 (2.1%)"
select key_SER_239, chain A and resi 239
show sticks, key_SER_239
color brightorange, key_SER_239 and name C*
label key_SER_239 and name CA, "SER239 (2.1%)"
select key_PRO_124, chain A and resi 124
show sticks, key_PRO_124
color brightorange, key_PRO_124 and name C*
label key_PRO_124 and name CA, "PRO124 (2.0%)"
select key_ILE_118, chain A and resi 118
show sticks, key_ILE_118
color brightorange, key_ILE_118 and name C*
label key_ILE_118 and name CA, "ILE118 (2.0%)"


# Center on key binding site residues
select all_key_residues, key_ASN_380 + key_ALA_40 + key_SER_239 + key_PRO_124 + key_ILE_118
zoom all_key_residues, 6.0
orient all_key_residues

# Background and lighting
bg_color black
set ray_shadows, 0
set antialias, 2
