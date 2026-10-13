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
select key_TYR_217, chain A and resi 217
show sticks, key_TYR_217
color brightorange, key_TYR_217 and name C*
label key_TYR_217 and name CA, "TYR217 (2.3%)"
select key_ILE_241, chain A and resi 241
show sticks, key_ILE_241
color brightorange, key_ILE_241 and name C*
label key_ILE_241 and name CA, "ILE241 (2.1%)"
select key_MET_85, chain A and resi 85
show sticks, key_MET_85
color brightorange, key_MET_85 and name C*
label key_MET_85 and name CA, "MET85 (2.1%)"
select key_ALA_203, chain A and resi 203
show sticks, key_ALA_203
color brightorange, key_ALA_203 and name C*
label key_ALA_203 and name CA, "ALA203 (2.1%)"
select key_PRO_37, chain A and resi 37
show sticks, key_PRO_37
color brightorange, key_PRO_37 and name C*
label key_PRO_37 and name CA, "PRO37 (2.1%)"


# Center on key binding site residues
select all_key_residues, key_TYR_217 + key_ILE_241 + key_MET_85 + key_ALA_203 + key_PRO_37
zoom all_key_residues, 6.0
orient all_key_residues

# Background and lighting
bg_color black
set ray_shadows, 0
set antialias, 2
