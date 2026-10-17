"""Build explicitly labelled paper mockups from local logs and supplied tables.

No model is loaded and no experiment is run. Predictions are illustrative
scenarios, not estimates with validated statistical uncertainty.
"""
from pathlib import Path
import csv
import json
import math
import os
import re
import tempfile

OUT = Path(__file__).resolve().parent
FR = OUT.parent
SAES = FR.parent / "SAES-SVD"
PHASE = FR / "result/phase2"


def metric(path):
    text = path.read_bytes().decode("utf-8", errors="replace")
    hits = list(re.finditer(r"PPL after pruning:.*?'wikitext2':\s*(?:np\.float64\()?([0-9.eE+-]+)", text))
    if not hits:
        raise ValueError(f"No PPL in {path}")
    m = hits[-1]
    return float(m.group(1)), str(path), text[:m.start()].count("\n") + 1


def write_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


sources = {}
source_files = {
    "center": PHASE / "r1abl_rho035_a04_06_r1_0.92.log",
    "rho025": PHASE / "fisher_direct_rho0.25_a04_06.log",
    "rho035": PHASE / "fisher_direct_rho0.35_a04_06.log",
    "rho042": PHASE / "fisher_direct_rho0.42_a04_06.log",
    "rho048": PHASE / "fisher_direct_rho0.48_a04_06.log",
    "rho030_a03": PHASE / "fisher_direct_rho0.3_a03_07.log",
    "rho040_a03": PHASE / "fisher_direct_rho0.4_a03_07.log",
    "rho030_a04": PHASE / "fisher_direct_rho0.3_a04_06.log",
    "rho060_a03": PHASE / "fisher_direct_rho0.6_a03_07.log",
    "rho060_a04": PHASE / "fisher_direct_rho0.6_a04_06.log",
    "ratio04_full": PHASE / "ratio04_g1_rho042_a045055_l1e5_ns512_r1_0.95.log",
    "saes_a00": SAES / "result/alpha_ablation_gpu0/Exp-A1_rho0.8_eps1e-3_alpha0.0-1.0_20251223.log",
    "saes_a02": SAES / "result/alpha_ablation_gpu0/Exp-A3_rho0.8_eps1e-3_alpha0.2-0.8_20251223.log",
    "saes_a03": SAES / "result/alpha_ablation_gpu0/Exp-A4_rho0.8_eps1e-3_alpha0.3-0.7_20251223.log",
    "saes_a04": SAES / "result/alpha_ablation_gpu0/（8.06）Exp-A5_rho0.8_eps1e-3_alpha0.4-0.6_20251223.log",
}
for key, path in source_files.items():
    ppl, source, line = metric(path)
    sources[key] = {"id": key, "ppl": ppl, "status": "measured_in_log", "source": source, "line": line}
write_csv("measured_evidence.csv", list(sources.values()))


def value(key):
    return sources[key]["ppl"]


methods = ["SAES-SVD", "FR-SVD (F)", "FR-SVD (R)", "FR-SVD"]
provided02 = [7.17, 7.08, 7.12, 7.02]
provided06 = [22.01, 20.42, 21.13, 19.36]
predicted04 = [math.sqrt(a * b) for a, b in zip(provided02, provided06)]
rows = []
for ratio, values, status in [
    (0.2, provided02, "user_screenshot_unverified"),
    (0.4, predicted04, "predicted_log_interpolation_not_measured"),
    (0.6, provided06, "user_screenshot_unverified"),
]:
    for method, ppl in zip(methods, values):
        rows.append({"compression_ratio": ratio, "method": method, "ppl": ppl,
                     "ppl_display": f"{ppl:.2f}", "reduction_percent": 100 * (1 - ppl / values[0]),
                     "status": status,
                     "basis": "user_supplied_20pct_and_60pct_tables"})
write_csv("table3_placeholder.csv", rows)

# Interpolate FR's local alpha sensitivity at rho=.35 from rho=.3 and .4.
# Transfer that DELTA from r1=.90 to .92, assuming no interaction.
delta03 = (value("rho030_a03") + value("rho040_a03")) / 2 - value("rho035")
# For unobserved wide alpha ranges, borrow the SAES A-series shape and
# scale it to the small local FR delta. This extrapolation is deliberately
# documented; it is not a fitted/predictively validated model.
shape_scale = delta03 / (value("saes_a03") - value("saes_a04"))
delta00 = shape_scale * (value("saes_a00") - value("saes_a04"))
delta02 = shape_scale * (value("saes_a02") - value("saes_a04"))
delta025 = value("rho025") - value("rho035")
delta045 = (value("rho042") + value("rho048")) / 2 - value("rho035")
center = value("center")
paper_center = 7.02  # Additional scenario assumption, not verified configuration.
sensitivity = []
specs = [
    (0.35, 0.00, 1.00, delta00, "scaled_SAES_shape_extrapolation"),
    (0.35, 0.20, 0.80, delta02, "scaled_SAES_shape_extrapolation"),
    (0.35, 0.30, 0.70, delta03, "FR_rho_interpolation_and_r1_delta_transfer"),
    (0.35, 0.40, 0.60, 0, "measured_center"),
    (0.25, 0.40, 0.60, delta025, "FR_r1_delta_transfer"),
    (0.45, 0.40, 0.60, delta045, "FR_rho_interpolation_and_r1_delta_transfer"),
]
for rho, amin, amax, delta, basis in specs:
    sensitivity.append({"rho": rho, "alpha_min": amin, "alpha_max": amax,
                        "compression_ratio": 0.2, "r1_ratio": 0.92,
                        "local_log_scale_ppl": center + delta,
                        "local_status": "measured" if basis == "measured_center" else "predicted_not_measured",
                        "paper_scenario_ppl": paper_center + delta,
                        "paper_status": "hypothetical_reanchoring_not_measured",
                        "delta_from_center": delta, "avg_accuracy": "",
                        "accuracy_status": "unavailable_no_prediction_basis", "basis": basis})
write_csv("sensitivity_placeholder.csv", sensitivity)

assumptions = {
    "table3": "log PPL is linear between the user-provided ratios 0.2 and 0.6; endpoints unverified in current logs",
    "table3_formula": "ppl(0.4)=sqrt(ppl(0.2)*ppl(0.6))",
    "sensitivity_local_center": center,
    "sensitivity_paper_center_assumed": paper_center,
    "sensitivity_FR_delta_alpha03": delta03,
    "sensitivity_SAES_shape_scale": shape_scale,
    "sensitivity_transfer": "additive PPL differences; no rho-alpha-r1 interaction; unvalidated across versions",
    "accuracy": "no compatible accuracy measurements; left blank",
    "uncertainty": "not quantified; no synthetic SD, CI, significance or repeated runs",
}
(OUT / "assumptions.json").write_text(json.dumps(assumptions, indent=2, ensure_ascii=False) + "\n")

table_tex = [
    "% Requires: \\usepackage{booktabs,xcolor}",
    "% ALL ratio=0.4 entries are predictions, NOT experimental measurements.",
    r"\begin{table}[t]", r"\centering",
    r"\caption{Predicted placeholders for the 40\% compression ablation on WikiText-2. Values are interpolated from the supplied 20\% and 60\% tables and are not measured results.}",
    r"\label{tab:ablation-predicted-placeholder}",
    r"\begin{tabular}{cccc}", r"\toprule",
    r"SAES-SVD & FR-SVD (F) & FR-SVD (R) & FR-SVD \\", r"\midrule",
]
cells = [f"{predicted04[0]:.2f}" + r"$^{\dagger}$"]
for ppl in predicted04[1:]:
    improvement = 100 * (1 - ppl / predicted04[0])
    cells.append(f"{ppl:.2f}" + r"$^{\dagger}$ " + r"(\textcolor{green!45!black}{$\downarrow " + f"{improvement:.1f}" + r"\%$})")
table_tex += [" & ".join(cells) + r" \\", r"\bottomrule", r"\end{tabular}",
              r"\par\smallskip\footnotesize $\dagger$: Predicted placeholder; replace after evaluation.", r"\end{table}", ""]
(OUT / "table3_placeholder.tex").write_text("\n".join(table_tex))

sens_tex = [
    "% Requires: \\usepackage{booktabs}",
    r"\begin{table}[t]", r"\centering",
    r"\caption{Illustrative sensitivity placeholders at 20\% compression and $r_1/r=0.92$. All PPL entries are hypothetical values reanchored to 7.02; accuracy is pending.}",
    r"\label{tab:sensitivity-predicted-placeholder}",
    r"\begin{tabular}{ccccc}",
    r"\toprule", r"$\rho$ & $\alpha_{\min}$ & $\alpha_{\max}$ & PPL$^{\dagger}\downarrow$ & Avg. Acc. \\",
    r"\midrule",
]
for row in sensitivity:
    sens_tex.append(f"{row['rho']:.2f} & {row['alpha_min']:.2f} & {row['alpha_max']:.2f} & {row['paper_scenario_ppl']:.3f} & --" + r" \\")
sens_tex += [r"\bottomrule", r"\end{tabular}",
             r"\par\smallskip\footnotesize $\dagger$: Hypothetical placeholders, not measured results. Decimal precision is for layout only.",
             r"\end{table}", ""]
(OUT / "sensitivity_placeholder.tex").write_text("\n".join(sens_tex))

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="paper-placeholder-mpl-"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5), constrained_layout=True)
alpha_rows = sensitivity[:4]
rho_rows = sorted(sensitivity[3:], key=lambda r: r["rho"])
axes[0].plot(range(4), [r["paper_scenario_ppl"] for r in alpha_rows],
             "o--", color="#226699", markerfacecolor="white", linewidth=1.6)
axes[0].set_xticks(range(4))
axes[0].set_xticklabels(["[0.0, 1.0]", "[0.2, 0.8]", "[0.3, 0.7]", "[0.4, 0.6]"])
axes[0].set_xlabel(r"$[\alpha_{\min},\alpha_{\max}]$ at $\rho=0.35$")
axes[1].plot([r["rho"] for r in rho_rows], [r["paper_scenario_ppl"] for r in rho_rows],
             "s--", color="#996622", markerfacecolor="white", linewidth=1.6)
axes[1].set_xticks([0.25, 0.35, 0.45])
axes[1].set_xlabel(r"$\rho$ at $\alpha\in[0.4,0.6]$")
for ax in axes:
    ax.set_ylim(7.0, 7.08)
    ax.set_ylabel("WikiText-2 PPL (hypothetical)")
    ax.grid(axis="y", alpha=0.2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
fig.suptitle("PREDICTED PLACEHOLDERS - NOT MEASURED", fontsize=12, weight="bold")
fig.savefig(OUT / "sensitivity_placeholder.pdf", bbox_inches="tight")
fig.savefig(OUT / "sensitivity_placeholder.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Lightweight output validation, without running any ML experiment.
assert len(rows) == 12 and len(sensitivity) == 6
assert all(math.isfinite(r["ppl"]) for r in rows)
assert sum(r["local_status"] == "measured" for r in sensitivity) == 1
assert all(not r["avg_accuracy"] for r in sensitivity)
print(json.dumps({"output_dir": str(OUT), "table3_predicted_ppl": predicted04,
                  "sensitivity": sensitivity, "assumptions": assumptions}, indent=2))
