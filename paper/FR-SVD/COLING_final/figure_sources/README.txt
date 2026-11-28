Framework artwork
-----------------
overview_original.pdf is an unchanged copy of the canonical 10.7 artwork.
FR-SVD_overview.drawio is its matching source page (page 7 of the original
FR-SVD.drawio), with the Stage-1 target corrected to G_l(beta_hat) and two
map-back labels. The production PDF preserves the original vector artwork;
local updates are applied with overview_overlay.tex and build_overview.py.
From this directory, run pdflatex overview_overlay.tex twice, then run
build_overview.py with a Python interpreter providing pypdf. Output is
../figs/Over_view_of_FR-SVD.pdf. Cropping removes only outer whitespace.

Efficiency plot
---------------
Run plot_frsvd_weight_speedup.py with matplotlib. The source data match the
corrected 10.7 manuscript. The PDF is sized at 3.35 by 2.40 inches, with 8.2–9 pt
labels for one ACL column. Output is ../figs/frsvd_weight_speedup.pdf.
