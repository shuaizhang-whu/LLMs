#!/usr/bin/env python3
"""Section 3 statistics from the logged runs (reads logs only, CPU)."""
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import runs
from runs import input_status

ROOT = Path(__file__).resolve().parent.parent
ARMS = ("spork", "raw")


def pct(x):
    return 100.0 * float(x)


def collect(grouped):
    """Per-turn rows of the dispatch-only runs and window counts of the method's runs."""
    valid = [(task, group) for task, items in grouped.items() for group in items]
    turns, tasks = [], []
    check = Counter()
    launch = Counter()
    for g, group in valid:
        base, ours = group["spork"], group["raw"]
        task = {"task": g, "split": base["split"], "base_s": base["wall_s"],
                "ours_s": ours["wall_s"], "ideal_s": 0.0}
        seq = base["turns"]
        for i, t in enumerate(seq):
            row = {"task": g, "turn": t["turn"], "gen_s": t["main_end_ms"] / 1e3,
                   "turn_s": t["turn_end_ms"] / 1e3, "tool": t["baseline_has_tool_call"],
                   "dispatched": t["spec_dispatched"], "anchor": bool(t["gate_match"]),
                   "next_s": seq[i + 1]["turn_end_ms"] / 1e3 if i + 1 < len(seq) else None}
            if t["baseline_has_tool_call"]:
                row["tool_s"] = (t["spec_wall_s"] if t["gate_match"] else
                                 (t["canonical_tool_end_ms"] - t["canonical_tool_dispatch_ms"]) / 1e3)
            if row["anchor"]:
                ready = t["spec_end_ms"] / 1e3
                call = {"name": t["committed_probe_name"], "arguments": t["committed_probe_args"]}
                row.update(dispatch_s=t["spec_dispatch_ms"] / 1e3, ready_s=ready,
                           slack_s=max(0.0, row["gen_s"] - ready),
                           copy_s=ready - t["spec_dispatch_ms"] / 1e3 - t["spec_wall_s"],
                           input=input_status(call, t["baseline_text"]))
                # Disjoint windows start at odd turns, as in the evaluated method.
                if t["turn"] % 2 == 1 and row["next_s"] is not None:
                    task["ideal_s"] += min(row["next_s"], row["slack_s"])
            turns.append(row)
        tasks.append(task)
        # Cross-check the text-level input test against the logged commit decisions.
        for pair in ours["prefetch"]["pairs"]:
            launch["windows"] += 1
            if not pair.get("child_started"):
                continue
            launch["launched"] += 1
            launch["adopted"] += bool(pair.get("adopted"))
            launch["reject:" + str(pair.get("rejection"))] += 1
            first = pair["first_turn"]
            if pair.get("adopted"):
                before, after = (ours["turns"][first - 1 + k]["environment_after"] for k in (0, 1))
                launch["state_changed_committed"] += before != after
            else:
                seen, start = pair.get("child_environment_observed"), pair.get("candidate_parent_state")
                launch["state_changed_discarded"] += seen is not None and seen != start
            if pair.get("rejection") in (None, "next_prompt_tokens"):
                parent = ours["turns"][first - 1]
                same = input_status(pair["candidate_call"], parent["baseline_text"]) == "identical"
                check["agree" if same == (pair["rejection"] is None) else "disagree"] += 1
    return valid, turns, tasks, check, launch


def summarize(valid, turns, tasks, check, launch):
    tool = [r for r in turns if r["tool"]]
    anchors = [r for r in turns if r["anchor"]]
    pairs = [r for r in anchors if r["next_s"] is not None]
    slack = np.array([r["slack_s"] for r in anchors])
    share = np.array([r["slack_s"] / r["gen_s"] for r in anchors])
    ratio = np.array([r["slack_s"] / r["next_s"] for r in pairs])
    wall = sum(t["base_s"] for t in tasks)
    status = Counter(r["input"] for r in anchors)
    ideal = np.array([t["ideal_s"] / t["base_s"] for t in tasks])
    real = np.array([(t["base_s"] - t["ours_s"]) / t["base_s"] for t in tasks])
    return {
        "tasks": len({g for g, _ in valid}), "runs": len(valid),
        "turns": len(turns), "tool_turns": len(tool),
        "dispatched_tool_turns": sum(r["dispatched"] for r in tool),
        "anchor_turns": len(anchors), "anchor_pct_of_tool_turns": pct(len(anchors) / len(tool)),
        "mean_task_s": wall / len(tasks),
        "tool_pct_of_generation": pct(sum(r["tool_s"] for r in tool) / sum(r["gen_s"] for r in turns)),
        "generation_pct_of_wall": pct(sum(r["gen_s"] for r in turns) / wall),
        "dispatch_pos_median_pct": pct(np.median([r["dispatch_s"] / r["gen_s"] for r in anchors])),
        "ready_pos_median_pct": pct(np.median([r["ready_s"] / r["gen_s"] for r in anchors])),
        "copy_median_s": float(np.median([r["copy_s"] for r in anchors])),
        "slack_median_s": float(np.median(slack)),
        "slack_q25_s": float(np.percentile(slack, 25)), "slack_q75_s": float(np.percentile(slack, 75)),
        "slack_positive_pct": pct(np.mean(slack > 0)),
        "slack_share_median_pct": pct(np.median(share)),
        "slack_pct_of_wall": pct(slack.sum() / wall),
        "pairs": len(pairs),
        "next_median_s": float(np.median([r["next_s"] for r in pairs])),
        "slack_over_next_median": float(np.median(ratio)),
        "next_fits_fully_pct": pct(np.mean(ratio >= 1)),
        "next_fits_half_pct": pct(np.mean(ratio >= 0.5)),
        "next_covered_mean_pct": pct(np.mean(np.minimum(ratio, 1))),
        "input_identical": status["identical"], "input_visible_text": status["visible_text"],
        "input_stray_tag": status["stray_tag"],
        "input_identical_pct": pct(status["identical"] / len(anchors)),
        "ideal_window_saving_pct": pct(sum(t["ideal_s"] for t in tasks) / wall),
        "realized_saving_pct": pct(1 - sum(t["ours_s"] for t in tasks) / wall),
        "ideal_vs_realized_pearson": float(np.corrcoef(ideal, real)[0, 1]),
        "input_test_crosscheck": dict(check), "logged_windows": dict(launch),
    }


def draft_stats(valid):
    """Upper bound on what reusing probe tokens as drafts can save."""
    base, ours = ([group[c] for _, group in valid] for c in ARMS)
    tokens = sum(runs.tokens(r) for r in base)
    s_per_token = sum(t["main_end_ms"] for r in base for t in r["turns"]) / 1e3 / tokens
    drafts = [sum(r["d3_counters"]["accepted_draft_tokens"] for r in arm) for arm in (base, ours)]
    wall = [sum(r["wall_s"] for r in arm) for arm in (base, ours)]
    return {
        "draft_token_pct": pct(drafts[0] / tokens),
        "draft_time_bound_pct": pct(drafts[0] * s_per_token / wall[0]),
        "draft_forgone_bound_pct": pct((drafts[0] - drafts[1]) * s_per_token / wall[1]),
    }


MACROS = (  # (macro, key, format)
    ("MotTasks", "tasks", "{:d}"), ("MotTurns", "turns", "{:d}"),
    ("MotToolTurns", "tool_turns", "{:d}"), ("MotAnchorTurns", "anchor_turns", "{:d}"),
    ("MotAnchorPct", "anchor_pct_of_tool_turns", "{:.1f}"),
    ("MotToolGenPct", "tool_pct_of_generation", "{:.1f}"),
    ("MotReadyPosPct", "ready_pos_median_pct", "{:.1f}"),
    ("MotSlackMedian", "slack_median_s", "{:.1f}"),
    ("MotSlackSharePct", "slack_share_median_pct", "{:.1f}"),
    ("MotSlackWallPct", "slack_pct_of_wall", "{:.1f}"),
    ("MotPairs", "pairs", "{:d}"), ("MotNextMedian", "next_median_s", "{:.1f}"),
    ("MotSlackOverNext", "slack_over_next_median", "{:.2f}"),
    ("MotNextFitsPct", "next_fits_fully_pct", "{:.1f}"),
    ("MotNextHalfPct", "next_fits_half_pct", "{:.1f}"),
    ("MotNextCoveredPct", "next_covered_mean_pct", "{:.1f}"),
    ("MotInputSame", "input_identical", "{:d}"),
    ("MotInputSamePct", "input_identical_pct", "{:.1f}"),
    ("MotInputText", "input_visible_text", "{:d}"), ("MotInputTag", "input_stray_tag", "{:d}"),
    ("MotInputDiff", "input_different", "{:d}"),
    ("MotStateCommitted", "state_changed_committed", "{:d}"),
    ("MotStateDiscarded", "state_changed_discarded", "{:d}"),
    ("MotStateTotal", "state_changed_total", "{:d}"),
    ("MotPlotOutside", "pairs_outside_plot", "{:d}"),
    ("MotCopyMedian", "copy_median_s", "{:.2f}"),
    ("MotDraftTokenPct", "draft_token_pct", "{:.1f}"),
    ("MotDraftTimePct", "draft_time_bound_pct", "{:.1f}"),
    ("MotDraftForgonePct", "draft_forgone_bound_pct", "{:.1f}"),
)


def macros(stats, prefix):
    """Macro names start with Mot for deterministic serving and MotStd for standard serving."""
    return {prefix + name[3:]: (stats[key], fmt) for name, key, fmt in MACROS}


def plot(turns, stats, path):
    pairs = [r for r in turns if r["anchor"] and r["next_s"] is not None]
    nxt = np.array([r["next_s"] for r in pairs])
    slack = np.array([r["slack_s"] for r in pairs])
    fits = slack >= nxt
    plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 8,
                         "axes.linewidth": 0.6, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.03, 2.55))
    top = 200
    ax.fill_between([0, top], [0, top], [top, top], color="#3F8F4C", alpha=0.07, linewidth=0)
    ax.plot([0, top], [0, top], color="#1F2A44", linewidth=0.8)
    ax.plot([0, top], [0, top / 2], color="#1F2A44", linewidth=0.7, linestyle=(0, (4, 3)))
    ax.scatter(nxt[~fits], slack[~fits], s=7, color="#4A78B0", alpha=0.55, linewidths=0)
    ax.scatter(nxt[fits], slack[fits], s=7, color="#3F8F4C", alpha=0.65, linewidths=0)
    ax.text(5, top - 5, "above the diagonal:\nwhole next turn fits (%.0f%%)" % stats["next_fits_fully_pct"],
            ha="left", va="top", color="#2F6F3A")
    ax.text(top - 3, 3, "above the dashed line:\nat least half fits (%.0f%%)" % stats["next_fits_half_pct"],
            ha="right", va="bottom", color="#1F2A44")
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_aspect("equal")
    ax.set_xticks(range(0, top + 1, 50))
    ax.set_yticks(range(0, top + 1, 50))
    ax.set_xlabel("duration of the next turn (s)")
    ax.set_ylabel("slack of the current turn (s)")
    ax.tick_params(width=0.6, length=2.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(pad=0.3)
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path)
    return int(np.sum((nxt > top) | (slack > top)))


def main():
    values, report = {}, {}
    for prefix, paths in (("Mot", runs.DETERMINISTIC), ("MotStd", runs.STANDARD)):
        data = collect(runs.groups(runs.load(paths), ARMS))
        stats = summarize(*data)
        stats.update(draft_stats(data[0]))
        stats.update({k: data[4][k] for k in ("state_changed_committed", "state_changed_discarded")})
        stats["state_changed_total"] = stats["state_changed_committed"] + stats["state_changed_discarded"]
        stats["input_different"] = stats["anchor_turns"] - stats["input_identical"]
        # The figure shows deterministic serving, the setting of the paired comparisons.
        stats["pairs_outside_plot"] = (plot(data[1], stats, ROOT / "figures" / "fig2_slack.pdf")
                                       if prefix == "Mot" else 0)
        values.update(macros(stats, prefix))
        report[prefix] = stats
    out = ROOT / "generated"
    out.mkdir(exist_ok=True)
    (out / "motivation_stats.json").write_text(json.dumps(report, indent=2) + "\n")
    runs.write_macros(values, out / "motivation_numbers.tex", "motivation_stats.py")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
