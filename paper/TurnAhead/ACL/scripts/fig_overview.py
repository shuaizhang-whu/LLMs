#!/usr/bin/env python3
"""Draft of Figure 1 as a PNG; the final vector version is redrawn by hand."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "figures"
W, H = 10.0, 5.15                       # inches; axes units are inches
X0, UNIT = 1.72, 0.73                   # left edge of the timeline, inches per time unit
R, D, L, B = 6.0, 1.5, 1.0, 4.0         # illustrative durations
TAU = D + L

INK, MUTED, GUIDE = "#1F2A44", "#6B7280", "#B8BDC9"
TURN_F, TURN_E, CALL_F = "#DCE8F7", "#4A78B0", "#A9C6EA"
TOOL_F, TOOL_E = "#FCE0BC", "#C9772B"
NEXT_F, NEXT_E = "#D3ECD5", "#3F8F4C"
BOX_F, BOX_E, SAVE = "#F5F6F9", "#8A8FA0", "#1B7F5C"

plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 12,
    "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans",
    "mathtext.it": "Liberation Sans:italic", "mathtext.bf": "Liberation Sans:bold",
    "mathtext.fallback": "stixsans",
})


def tx(t):
    return X0 + t * UNIT


def bar(ax, t0, t1, y, h, face, edge, label="", dashed=False, size=12, weight="normal"):
    ax.add_patch(FancyBboxPatch(
        (tx(t0), y - h / 2), tx(t1) - tx(t0), h,
        boxstyle="round,pad=0,rounding_size=0.045", facecolor=face, edgecolor=edge,
        linewidth=1.3, linestyle=(0, (4, 2.2)) if dashed else "-", zorder=3))
    if label:
        ax.text((tx(t0) + tx(t1)) / 2, y, label, ha="center", va="center",
                fontsize=size, color=INK, fontweight=weight, zorder=4)


def turn(ax, y, with_call_label=True):
    """Parent turn: reasoning followed by the final tool call."""
    bar(ax, 0, R, y, 0.36, TURN_F, TURN_E)
    ax.add_patch(FancyBboxPatch(
        (tx(R - 0.62), y - 0.18), tx(R) - tx(R - 0.62), 0.36,
        boxstyle="round,pad=0,rounding_size=0.045", facecolor=CALL_F, edgecolor=TURN_E,
        linewidth=1.3, zorder=3))
    ax.text(tx((R - 0.62) / 2), y, r"turn $t$:  reasoning", ha="center", va="center",
            fontsize=12, color=INK, zorder=4)
    ax.text(tx(R - 0.31), y, "call", ha="center", va="center", fontsize=11, color=INK, zorder=4)


def arrow(ax, p0, p1, color=INK, style="-|>", lw=1.2, scale=9, z=5):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=scale,
                                 color=color, linewidth=lw, shrinkA=0, shrinkB=0, zorder=z))


def span(ax, t0, t1, y, text, color=MUTED, size=11, dy=0.0, weight="normal"):
    arrow(ax, (tx(t0), y), (tx(t1), y), color=color, style="<|-|>", lw=1.1, scale=8)
    ax.text((tx(t0) + tx(t1)) / 2, y + dy, text, ha="center",
            va="center" if dy == 0 else "bottom", fontsize=size, color=color, fontweight=weight,
            zorder=6, bbox=dict(facecolor="white", edgecolor="none", pad=1.6) if dy == 0 else None)


def badge(ax, x, y, n, r=0.105):
    ax.add_patch(Circle((x, y), r, facecolor=INK, edgecolor="white", linewidth=0.8, zorder=8))
    ax.text(x, y - 0.004, str(n), ha="center", va="center", fontsize=10.5, color="white",
            fontweight="bold", zorder=9)


def saved(ax, t0, t1, y, text, inside=True):
    """Time that the row no longer needs, relative to the row above."""
    ax.add_patch(FancyBboxPatch(
        (tx(t0) + 0.03, y - 0.18), tx(t1) - tx(t0) - 0.03, 0.36,
        boxstyle="round,pad=0,rounding_size=0.045", facecolor="#EEF7F0", edgecolor=SAVE,
        linewidth=0.0, hatch="////", alpha=0.55, zorder=2))
    if inside:
        ax.text((tx(t0) + tx(t1)) / 2, y, text, ha="center", va="center", fontsize=11.5,
                color=SAVE, fontweight="bold", zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", pad=2.0, alpha=0.9))
    else:
        ax.text(tx(t1), y - 0.24, text, ha="right", va="top", fontsize=11.5, color=SAVE,
                fontweight="bold", zorder=6)


def row_label(ax, y, tag, name):
    ax.text(0.12, y, tag, ha="left", va="center", fontsize=12, color=MUTED)
    ax.text(0.47, y, name, ha="left", va="center", fontsize=12, color=INK, fontweight="bold",
            linespacing=1.15)


def end_tick(ax, t, y):
    ax.plot([tx(t), tx(t)], [y - 0.25, y + 0.25], color=INK, linewidth=1.6, zorder=6,
            solid_capstyle="round")


def main():
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    ya, yb, yc = 4.80, 3.86, 2.52          # main lanes
    yb_tool = 3.42
    box_lo, box_hi = 1.22, 2.14
    yc_tool, yc_next = 1.92, 1.50

    for y in (4.20, 3.14):                 # row separators
        ax.plot([0.12, W - 0.12], [y, y], color="#E5E7EB", linewidth=0.9, zorder=1)

    # (a) Serial
    row_label(ax, ya, "(a)", "Serial")
    turn(ax, ya)
    bar(ax, R, R + L, ya, 0.36, TOOL_F, TOOL_E, "tool")
    bar(ax, R + L, R + L + B, ya, 0.36, NEXT_F, NEXT_E, r"turn $t{+}1$")
    end_tick(ax, R + L + B, ya)
    span(ax, 0, R, 4.43, r"$R_t$")
    span(ax, R, R + L, 4.43, r"$L_t$")
    span(ax, R + L, R + L + B, 4.43, r"$B_{t+1}$")

    # (b) Speculative tool dispatch
    row_label(ax, yb, "(b)", "Speculative\ntool dispatch")
    turn(ax, yb)
    bar(ax, D, TAU, yb_tool, 0.26, TOOL_F, TOOL_E, "tool", dashed=True, size=11)
    arrow(ax, (tx(D), yb - 0.18), (tx(D), yb_tool + 0.13), color=TOOL_E)
    ax.text(tx(D) - 0.07, yb_tool, r"predict $\hat{a}_t$", ha="right", va="center",
            fontsize=11, color=MUTED)
    bar(ax, R, R + B, yb, 0.36, NEXT_F, NEXT_E, r"turn $t{+}1$")
    end_tick(ax, R + B, yb)
    saved(ax, R + B, R + L + B, yb, r"saved: $L_t$", inside=False)

    # (c) TurnAhead
    row_label(ax, yc, "(c)", "TurnAhead\n(ours)")
    ax.add_patch(FancyBboxPatch(
        (tx(0.18), box_lo), tx(TAU + B + 0.22) - tx(0.18), box_hi - box_lo,
        boxstyle="round,pad=0,rounding_size=0.08", facecolor=BOX_F, edgecolor=BOX_E,
        linewidth=1.1, linestyle=(0, (5, 3)), zorder=2))
    turn(ax, yc)
    bar(ax, D, TAU, yc_tool, 0.26, TOOL_F, TOOL_E, "tool", dashed=True, size=11)
    arrow(ax, (tx(D), yc - 0.18), (tx(D), yc_tool + 0.13), color=TOOL_E)
    ax.text(tx(D) - 0.07, yc_tool, r"predict $\hat{a}_t$", ha="right", va="center",
            fontsize=11, color=MUTED)
    bar(ax, TAU, TAU + B, yc_next, 0.34, NEXT_F, NEXT_E, r"turn $t{+}1$  (successor)",
        dashed=True)
    # launch: observation ready -> successor starts
    arrow(ax, (tx(TAU), yc_tool - 0.13), (tx(TAU), yc_next + 0.17), color=NEXT_E)
    ax.text(tx(TAU) + 0.09, yc_tool, r"observation $\hat{o}_t$ ready", ha="left", va="center",
            fontsize=11, color=MUTED)
    badge(ax, tx(TAU) - 0.2, yc_next, 1)
    badge(ax, tx(0.18) + 0.2, box_lo + 0.2, 2)
    ax.text(tx(0.18) + 0.36, box_lo + 0.2, "sandbox", ha="left", va="center", fontsize=11,
            color=MUTED, style="italic")
    # commit check: the parent's real output against what the successor assumed
    xc = tx(R - 0.31)
    arrow(ax, (xc, yc - 0.18), (xc, yc_next + 0.17), color=INK, style="<|-|>", lw=1.3, scale=8)
    badge(ax, xc, (yc - 0.18 + yc_next + 0.17) / 2 - 0.02, 3)
    span(ax, TAU, R, 2.84, r"slack $\sigma_t$", color=MUTED)
    end_tick(ax, TAU + B, yc)
    ax.plot([tx(TAU + B), tx(TAU + B)], [yc_next + 0.17, yc - 0.25], color=GUIDE, linewidth=0.9,
            linestyle=(0, (2, 2)), zorder=4)
    saved(ax, TAU + B, R + B, yc, r"saved: $\min(B_{t+1},\,\sigma_t)$")

    # guides shared by the rows
    for t, lo, hi in ((R + L + B, yb, ya - 0.25), (R + B, yc, yb - 0.25)):
        ax.plot([tx(t), tx(t)], [lo, hi], color=GUIDE, linewidth=0.9, linestyle=(0, (2, 2)), zorder=1)

    # time axis
    arrow(ax, (tx(0), 0.98), (W - 0.2, 0.98), color=MUTED, lw=1.0, scale=10)
    ax.text(W - 0.22, 0.86, "time", ha="right", va="top", fontsize=11, color=MUTED)

    # legend for the three strategies
    ax.plot([0.12, W - 0.12], [0.74, 0.74], color="#E5E7EB", linewidth=0.9, zorder=1)
    items = (("Anchored Lookahead", r"start turn $t{+}1$ once the observation is ready"),
             ("Sandboxed Execution", "run speculative tools on environment copies"),
             ("Exact-Input Commit", r"keep turn $t{+}1$ only if its input is unchanged"))
    for i, (name, text) in enumerate(items):
        x = 0.22 + i * 3.27
        badge(ax, x + 0.1, 0.50, i + 1)
        ax.text(x + 0.29, 0.50, name, ha="left", va="center", fontsize=12, color=INK,
                fontweight="bold")
        ax.text(x + 0.29, 0.24, text, ha="left", va="center", fontsize=10.5, color=MUTED)

    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fig1_overview.png", dpi=300)
    fig.savefig("/tmp/fig1_overview_preview.png", dpi=110)


if __name__ == "__main__":
    main()
