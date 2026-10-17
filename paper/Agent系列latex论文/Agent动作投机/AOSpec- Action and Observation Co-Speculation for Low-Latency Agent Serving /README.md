# Action–Observation Co-Speculation for Low-Latency Agentic Serving — AAAI-2027 submission

LaTeX source for the AAAI-2027 anonymous submission. This repository is the
**Overleaf-synced project**; it contains only what is needed to compile the paper.

> ⚠️ **Anonymous submission.** Keep this repository **private** until the paper is
> public. The commit history and account name are de-anonymizing.

## Layout

| Path | What it is |
|---|---|
| `main.tex` | Preamble, macros, title, abstract, and `\input{}` of each section. |
| `sections/*.tex` | One file per section: `introduction`, `related_work`, `motivation`, `methodology`, `experiments`, `conclusion`. **Edit these.** |
| `flatten.py` | Produces the single-file source AAAI requires (see below). |
| `references.bib` | Bibliography. Entries marked `VERIFY` / `PLACEHOLDER` still need filling. |
| `aaai2027.sty`, `aaai2027.bst` | From the official AAAI-2027 author kit — do not edit. |
| `ReproducibilityChecklist.tex` | Author-kit checklist; `\input` near the end of `main.tex` when the CFP requires it. |
| `figs/*.pdf` | Generated figures embedded by the sections. |

## ⚠️ Submitting: AAAI requires a SINGLE .tex file

The author kit is explicit:

> "Your LaTeX source file submitted as a **single** .tex file (do not use the `input`
> command to include sections of your paper — every section must be in the single source
> file). (The only allowable exception is .bib file.)" … "**Do not submit your source in
> multiple text files.**"

Non-compliant source is **returned to the authors to fix**. We nevertheless draft with
`sections/` because it is far easier to edit and review; before submitting, run:

```
python3 flatten.py          # main.tex + sections/*.tex -> main_submission.tex
python3 flatten.py --check  # verify it flattens cleanly, write nothing
```

Submit `main_submission.tex` (not `main.tex`) together with `references.bib`,
`aaai2027.sty`/`.bst`, the `figs/` actually used, and the compiled PDF.
`main_submission.tex` is git-ignored on purpose — regenerate it rather than keeping a
second copy of the paper that can drift out of sync.

## Building

Compiles with **pdfLaTeX** (the AAAI style requires it — not XeLaTeX/LuaLaTeX):

```
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

On Overleaf: set *Menu → Compiler* to **pdfLaTeX**. Do **not** set
`\bibliographystyle` — `aaai2027.sty` sets it to `aaai2027.bst` automatically.

## Figures

Figure PDFs in `figs/` are **generated**, not hand-edited. Their plotting code lives in
the main research repository under `paper/figures/` and in the local `plot_*.py`
scripts; frozen inputs live in `paper/figures/data/`. To change a figure, edit the
data/code and run `bash paper/figures/make_all.sh`.

`system_overview` does not exist yet — the system-overview diagram is currently an
`\fbox` placeholder in `main.tex` and still needs to be drawn.

## Status

**Full draft — all six sections written** (Introduction, Related Work, Motivation,
Methodology, Experiments, Conclusion + Limitations). Roughly 6 pages of the 7-page
budget, leaving room for the numbers to grow.

Still outstanding:
- `system_overview` — the system diagram is an `\fbox` placeholder and needs drawing.
- Baselines must be **re-scored under the task-disjoint holdout** before submission
  (marked `TODO(Q1)` in `main.tex`); the current baseline numbers come from an earlier
  split.
- PASTE is cited but not yet reimplemented as a baseline row (`TODO(Q3)`).
- Six-metric macro/median values in Table 2 are carried over from the earlier split
  (`TODO(Q1/Q5)`).
- Appendix (hyperparameters, per-cell latency provenance, full six-metric tables) not
  yet written, though the text refers to it.

**All numbers are old-data placeholders** pending the final experiment sweep. Every one is
wrapped in the `\old{}` macro, so `grep -n '\\old{' main.tex` lists them all; to highlight
them in a build, temporarily use:

```latex
\renewcommand{\old}[1]{\textbf{#1}}
```

Citations use placeholder keys (`TODOagents`, `TODOserving1`, `TODOserving2`) with stub
`.bib` entries so the draft compiles; replace them with the real works.
