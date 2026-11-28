"""Restore the original vector overview with a local Stage-1 math correction.

Source artwork is preserved byte-for-byte as overview_original.pdf.
Run pdflatex overview_overlay.tex first; this script merges vector annotations
and removes only the large outer whitespace of the original A4 export.
"""
from pathlib import Path
from pypdf import PdfReader, PdfWriter
from pypdf.generic import RectangleObject

ROOT = Path(__file__).resolve().parent
page = PdfReader(ROOT / 'overview_original.pdf').pages[0]
page.merge_page(PdfReader(ROOT / 'overview_overlay.pdf').pages[0])
# Original content is x=22..816, y(top)=18..383; retain a visible safety margin.
page.cropbox = RectangleObject((18, 207.96, 820, 580.96))
writer = PdfWriter()
writer.add_page(page)
writer.add_metadata({'/Title': 'FR-SVD framework', '/Author': 'Anonymous'})
with (ROOT.parent / 'figs' / 'Over_view_of_FR-SVD.pdf').open('wb') as f:
    writer.write(f)
