#!/usr/bin/env python3
"""Render a PPTX and create five-slide contact sheets for visual QA."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import fitz
from PIL import Image, ImageDraw


def render(pptx: Path, out_dir: Path, only_last: int = 0):
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "libreoffice", "--headless", "--convert-to", "pdf", "--outdir",
        str(out_dir), str(pptx)
    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    pdf = out_dir / (pptx.stem + ".pdf")
    doc = fitz.open(pdf)
    first = max(0, len(doc)-only_last) if only_last else 0
    rendered = []
    for i in range(first, len(doc)):
        pix = doc[i].get_pixmap(matrix=fitz.Matrix(1.55, 1.55), alpha=False)
        path = out_dir / f"slide-{i+1:02d}.png"
        pix.save(path)
        rendered.append(path)
    doc.close()
    thumbs = []
    for p in rendered:
        im = Image.open(p).convert("RGB")
        im.thumbnail((640, 360))
        canvas = Image.new("RGB", (660, 398), "white")
        canvas.paste(im, ((660-im.width)//2, 18))
        ImageDraw.Draw(canvas).text((16, 375), p.stem, fill=(48, 58, 72))
        thumbs.append(canvas)
    cols = 2
    rows = (len(thumbs)+cols-1)//cols
    sheet = Image.new("RGB", (cols*660, rows*398), (235, 239, 244))
    for i, im in enumerate(thumbs):
        sheet.paste(im, ((i % cols)*660, (i // cols)*398))
    contact = out_dir / "contact.png"
    sheet.save(contact)
    return pdf, contact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pptx", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--only-last", type=int, default=0)
    args = ap.parse_args()
    pdf, contact = render(args.pptx, args.out_dir, args.only_last)
    print(pdf)
    print(contact)


if __name__ == "__main__":
    main()
