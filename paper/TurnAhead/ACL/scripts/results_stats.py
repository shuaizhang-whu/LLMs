#!/usr/bin/env python3
"""Section 5 numbers, the breakdown table and the result figure from the logged runs (reads logs only, CPU)."""
import difflib
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import runs

ROOT = Path(__file__).resolve().parent.parent
NAME = {"nospec": "None", "spork": "Disp", "raw": "Ours",
        "actor_only": "Defer", "raw_no_probe": "NoProbe", "serial": "Late"}
ABLATION_ARMS = ("nospec", "spork", "raw", "actor_only", "raw_no_probe", "serial")
PAIRS = (("spork", "raw"), ("nospec", "raw"), ("nospec", "spork"))
ONE, TWO, INT = "{:.1f}", "{:.2f}", "{:d}"
MAX_TURNS = 8


def put(out, name, value, fmt=ONE):
    out[name] = (int(value) if fmt == INT else float(value), fmt)


def put_interval(out, name, estimate, fmt=ONE):
    for suffix, value in zip(("", "Lo", "Hi"), estimate):
        put(out, name + suffix, value, fmt)


def items_of(grouped):
    return [g for v in grouped.values() for g in v]


def latency(grouped, arms, pairs, prefix, out):
    items, k = items_of(grouped), len(arms)
    rows = [[sum(g[a]["wall_s"] for g in v) for a in arms] + [sum(runs.tokens(g[a]) for g in v) for a in arms]
            for v in grouped.values()]
    put(out, prefix + "Tasks", len(grouped), INT)
    put(out, prefix + "Groups", len(items), INT)
    put(out, prefix + "Turns", np.mean([len(g[arms[0]]["turns"]) for g in items]))
    for a in arms:
        wall = np.array([g[a]["wall_s"] for g in items])
        put(out, prefix + "Mean" + NAME[a], wall.mean())
        put(out, prefix + "Med" + NAME[a], np.percentile(wall, 50))
        put(out, prefix + "Tail" + NAME[a], np.percentile(wall, 95))
    for base, ours in pairs:
        b, o, key = arms.index(base), arms.index(ours), NAME[ours] + NAME[base]
        estimate = runs.interval(rows, lambda s, b=b, o=o: 100 * (1 - s[..., o] / s[..., b]))
        put_interval(out, prefix + "Save" + key, estimate)
        put(out, prefix + "Speed" + key, 1 / (1 - estimate[0] / 100), TWO)
        # Time per committed token removes differences in trajectory length between configurations.
        put_interval(out, prefix + "TokSave" + key, runs.interval(
            rows, lambda s, b=b, o=o: 100 * (1 - (s[..., o] / s[..., k + o]) / (s[..., b] / s[..., k + b]))))
        put(out, prefix + "Faster" + key, sum(row[o] < row[b] for row in rows), INT)
        for q, tag in ((50, "Med"), (95, "Tail")):
            put(out, prefix + tag + "Save" + key, 100 * (1 - np.percentile([g[ours]["wall_s"] for g in items], q)
                                                          / np.percentile([g[base]["wall_s"] for g in items], q)))


def quality(grouped, arms, prefix, out):
    k = len(arms)
    rows = [[sum(g[a]["action_reward"]["n_matched"] for g in v) for a in arms]
            + [sum(g[a]["action_reward"]["n_golden"] for g in v) for a in arms] for v in grouped.values()]
    total = np.sum(rows, axis=0)
    for i, a in enumerate(arms):
        put(out, prefix + "Match" + NAME[a], 100 * total[i] / total[k + i])
        put(out, prefix + "Solved" + NAME[a], 100 * np.mean([g[a]["action_reward"]["reward"] for g in items_of(grouped)]))
    for base, ours in PAIRS[:2]:
        b, o = arms.index(base), arms.index(ours)
        put_interval(out, prefix + "MatchDiff" + NAME[ours] + NAME[base], runs.interval(
            rows, lambda s, b=b, o=o: 100 * (s[..., o] / s[..., k + o] - s[..., b] / s[..., k + b])))


def failures(records, arms, prefix, out):
    by_task = defaultdict(lambda: defaultdict(list))
    for (task, _, arm), rec in records.items():
        if arm in arms:
            by_task[task][arm].append(rec)
    k = len(arms)
    rows = [[sum(bool(r["errors"]) for r in v[a]) for a in arms] + [len(v[a]) for a in arms] for v in by_task.values()]
    total = np.sum(rows, axis=0)
    other = sum(1 for r in records.values() if r["config"] in arms and r["errors"]
                and not all("finish_reason='length'" in e for e in r["errors"]))
    put(out, prefix + "FailOther", other, INT)
    for i, a in enumerate(arms):
        put(out, prefix + "Runs" + NAME[a], total[k + i], INT)
        put(out, prefix + "Fail" + NAME[a], total[i], INT)
        put(out, prefix + "FailPct" + NAME[a], 100 * total[i] / total[k + i])
    for base, ours in PAIRS[:2]:
        b, o = arms.index(base), arms.index(ours)
        put_interval(out, prefix + "FailDiff" + NAME[ours] + NAME[base], runs.interval(
            rows, lambda s, b=b, o=o: 100 * (s[..., o] / s[..., k + o] - s[..., b] / s[..., k + b])))


def signature(rec):
    return [(t["baseline_has_tool_call"], t["baseline_tool_name"], json.dumps(t["baseline_tool_args"], sort_keys=True),
             t["tool_result_sha256"], json.dumps(t.get("environment_after"), sort_keys=True)) for t in rec["turns"]]


def identity(records, arms, prefix, out):
    """Token-level identity of all arms on the tasks of a single-repetition run."""
    n = Counter()
    for task in sorted({g for g, _, _ in records}):
        recs = [records.get((task, 0, a)) for a in arms]
        if any(r is None for r in recs):
            continue
        n["planned"] += 1
        same = (all([t["main_token_ids"] for t in r["turns"]] == [t["main_token_ids"] for t in recs[0]["turns"]]
                    for r in recs) and all(signature(r) == signature(recs[0]) for r in recs))
        errors = [bool(r["errors"]) for r in recs]
        if not any(errors):
            n["clean"] += 1
            n["same"] += same
            n["turns"] += len(recs[0]["turns"]) if same else 0
        elif all(errors):
            n["failed"] += 1
            n["failed_same"] += (same and recs[0]["failed_tokens"] is not None
                                 and all((r["failed_turn"], r["failed_tokens"]) ==
                                         (recs[0]["failed_turn"], recs[0]["failed_tokens"]) for r in recs))
        else:
            n["mixed"] += 1
    for key, name in (("planned", "Planned"), ("clean", "Clean"), ("same", "Same"), ("turns", "Turns"),
                      ("failed", "Failed"), ("failed_same", "FailedSame"), ("mixed", "Mixed")):
        put(out, prefix + "Ident" + name, n[key], INT)


def divergence(records, prefix, out):
    """How often two runs of a task follow the same trajectory under standard kernels."""
    tasks = sorted({g for g, _, _ in records})
    def same(a, b):
        return (a is not None and b is not None and not a["errors"] and not b["errors"]
                and [t["main_token_ids"] for t in a["turns"]] == [t["main_token_ids"] for t in b["turns"]])
    for base, ours in PAIRS:
        both = [(records.get((t, r, base)), records.get((t, r, ours))) for t in tasks for r in (0, 1)]
        both = [(a, b) for a, b in both if a is not None and b is not None and not a["errors"] and not b["errors"]]
        put(out, prefix + "SamePairs" + NAME[ours] + NAME[base], len(both), INT)
        put(out, prefix + "Same" + NAME[ours] + NAME[base], sum(same(a, b) for a, b in both), INT)
    for arm in runs.MAIN_ARMS:
        both = [(records.get((t, 0, arm)), records.get((t, 1, arm))) for t in tasks]
        both = [(a, b) for a, b in both if a is not None and b is not None and not a["errors"] and not b["errors"]]
        put(out, prefix + "RepeatPairs" + NAME[arm], len(both), INT)
        put(out, prefix + "RepeatSame" + NAME[arm], sum(same(a, b) for a, b in both), INT)


def successor_windows(rec):
    return [p for p in rec["prefetch"].get("pairs", []) if p.get("kind") == "successor_pair"]


def windows(grouped, arm, prefix, out):
    pairs = [p for g in items_of(grouped) for p in successor_windows(g[arm])]
    started = [p for p in pairs if p.get("child_started")]
    committed = [p for p in started if p.get("adopted")]
    reasons = Counter(p.get("rejection") for p in started if not p.get("adopted"))
    overlap = lambda p: p["parent_finished_s"] - p["child_start_s"]
    put(out, prefix + "Windows", len(pairs), INT)
    put(out, prefix + "Started", len(started), INT)
    put(out, prefix + "Committed", len(committed), INT)
    put(out, prefix + "Discarded", len(started) - len(committed), INT)
    put(out, prefix + "CommitPct", 100 * len(committed) / len(started))
    put(out, prefix + "RejCall", reasons.pop("parent_call", 0), INT)
    put(out, prefix + "RejReply", reasons.pop("parent_has_no_tool", 0), INT)
    put(out, prefix + "RejInput", reasons.pop("next_prompt_tokens", 0), INT)
    put(out, prefix + "RejOther", sum(reasons.values()), INT)
    put(out, prefix + "OverlapMed", np.median([overlap(p) for p in started]))
    put(out, prefix + "OverlapCommitPct", 100 * sum(overlap(p) for p in committed) / sum(overlap(p) for p in started))
    put(out, prefix + "CommittedPerTask", len(committed) / len(items_of(grouped)), TWO)


def overhead(grouped, arms, prefix, out):
    for a in arms:
        recs = [g[a] for g in items_of(grouped)]
        committed = sum(runs.tokens(r) for r in recs)
        probes = sum(p["tokens"] for r in recs for t in r["turns"] for p in t.get("probes") or [])
        lost = [p for r in recs for p in successor_windows(r) if p.get("child_started") and not p.get("adopted")]
        lost_probes = sum((q.get("usage") or {}).get("completion_tokens") or 0
                          for p in lost for q in p.get("child_prediction_requests") or [])
        lost_tokens = sum(p.get("observed_wasted_child_tokens") or 0 for p in lost)
        put(out, prefix + "Gen" + NAME[a], (committed + probes + lost_probes + lost_tokens) / committed, TWO)
        put(out, prefix + "ProbePct" + NAME[a], 100 * (probes + lost_probes) / committed)
        put(out, prefix + "LostPct" + NAME[a], 100 * lost_tokens / committed)


def speed(grouped, prefix, out):
    def rate(turns):
        return (sum(t["main_end_ms"] - (t["main_first_token_ms"] or 0.0) for t in turns)
                / sum(t["total_decode_tokens"] for t in turns))
    solo = [t for g in items_of(grouped) for t in g["nospec"]["turns"]]
    busy = [g["raw"]["turns"][p["first_turn"] - 1] for g in items_of(grouped) for p in successor_windows(g["raw"])
            if p.get("child_started") and p.get("child_launch_phase") == "before_parent_completion"]
    put(out, prefix + "MsSolo", rate(solo))
    put(out, prefix + "MsBusy", rate(busy))
    put(out, prefix + "BusyRatio", rate(busy) / rate(solo), TWO)
    put(out, prefix + "TurnTokens", np.mean([t["total_decode_tokens"] for t in solo]), "{:.0f}")
    put(out, prefix + "TurnSeconds", np.mean([t["main_end_ms"] / 1e3 for t in solo]))


def concurrency(out):
    slowdown = {}
    for prefix, path in runs.CONCURRENCY.items():
        summary = json.loads(path.read_text())["summary"]
        put(out, "Con" + prefix + "Ms", summary["1"]["ms_per_token"])
        put(out, "Con" + prefix + "ExtraMs", summary["2"]["ms_per_token"] - summary["1"]["ms_per_token"])
        for streams, name in (("2", "Two"), ("3", "Three"), ("4", "Four")):
            put(out, "Con" + prefix + name, summary[streams]["slowdown"], TWO)
        slowdown[prefix] = summary["2"]["slowdown"]
    return slowdown


def window_terms(rec, typical):
    """Windows of a dispatch-only run as the method would form them: (overlap in s, would commit)."""
    seq, terms = rec["turns"], []
    for i in range(0, min(len(seq), MAX_TURNS - 1), 2):
        t = seq[i]
        if not t.get("spec_dispatched") or t.get("spec_end_ms") is None:
            continue
        first = t.get("main_first_token_ms") or 0.0
        reasoning_end = first + (t["main_end_ms"] - first) * t["cot_tokens"] / max(1, t["total_decode_tokens"])
        if t["spec_end_ms"] >= reasoning_end:
            continue  # The observation arrives after the reasoning: no successor is started.
        commit = False
        if t["gate_match"] and i + 1 < len(seq):
            call = {"name": t["committed_probe_name"], "arguments": t["committed_probe_args"]}
            commit = runs.input_status(call, t["baseline_text"]) == "identical"
        slack = (t["main_end_ms"] - t["spec_end_ms"]) / 1e3
        terms.append((min(slack, seq[i + 1]["turn_end_ms"] / 1e3 if commit else typical), commit))
    return terms


def cost_model(grouped, slowdown, prefix, out):
    """Saving predicted from the dispatch-only runs and the measured two-stream slowdown."""
    typical = float(np.median([t["turn_end_ms"] / 1e3 for g in items_of(grouped) for t in g["spork"]["turns"]]))
    rows = []
    for v in grouped.values():
        terms = [term for g in v for term in window_terms(g["spork"], typical)]
        rows.append([sum(g["spork"]["wall_s"] for g in v), sum(g["raw"]["wall_s"] for g in v),
                     sum((2 - slowdown) * o if c else -(slowdown - 1) * o for o, c in terms),
                     sum(o for o, c in terms if c), len(v)])
    a = np.array(rows)
    predicted, ideal = 100 * a[:, 2].sum() / a[:, 0].sum(), 100 * a[:, 3].sum() / a[:, 0].sum()
    put(out, "Cost" + prefix + "Pred", predicted)
    put(out, "Cost" + prefix + "Ideal", ideal)
    # Difference of the two printed values, so that the text adds up.
    put(out, "Cost" + prefix + "Contention", round(ideal, 1) - round(predicted, 1))
    put(out, "Cost" + prefix + "Measured", 100 * (1 - a[:, 1].sum() / a[:, 0].sum()))
    put(out, "Cost" + prefix + "Corr", np.corrcoef(a[:, 2] / a[:, 0], 1 - a[:, 1] / a[:, 0])[0, 1], "{:.3f}")
    put(out, "Cost" + prefix + "ErrMed", np.median(np.abs(a[:, 2] - (a[:, 0] - a[:, 1])) / a[:, 4]))
    put(out, "Cost" + prefix + "Breakeven", 100 * (slowdown - 1))
    return a


def reuse(grouped, prefix, out):
    """Token overlap between a discarded successor and the turn that replaced it."""
    n = replaced = prefix_tokens = shared = 0
    committed = sum(runs.tokens(g["raw"]) for g in items_of(grouped))
    for g in items_of(grouped):
        rec = g["raw"]
        for p in successor_windows(rec):
            draft = p.get("discarded_child_token_ids")
            if not draft or p.get("adopted") or len(rec["turns"]) <= p["first_turn"]:
                continue
            real = rec["turns"][p["first_turn"]]["main_token_ids"]
            n += 1
            replaced += len(real)
            prefix_tokens += next((i for i, (x, y) in enumerate(zip(draft, real)) if x != y), min(len(draft), len(real)))
            blocks = difflib.SequenceMatcher(None, draft, real, autojunk=False).get_matching_blocks()
            shared += sum(b.size for b in blocks if b.size >= 8)
    put(out, "Reuse" + prefix + "Turns", n, INT)
    put(out, "Reuse" + prefix + "ReplacedPct", 100 * replaced / committed)
    put(out, "Reuse" + prefix + "PrefixPct", 100 * prefix_tokens / replaced)
    put(out, "Reuse" + prefix + "SharedPct", 100 * shared / replaced)
    put(out, "Reuse" + prefix + "SharedAllPct", 100 * shared / committed)


def ablation(grouped, out):
    arms = ABLATION_ARMS
    rows = [[v[0][a]["wall_s"] for a in arms] for v in grouped.values()]
    put(out, "AblTasks", len(grouped), INT)
    base, full = arms.index("spork"), arms.index("raw")
    for i, a in enumerate(arms):
        put(out, "AblMean" + NAME[a], np.mean([row[i] for row in rows]))
        if a not in ("spork",):
            put_interval(out, "AblSave" + NAME[a], runs.interval(
                rows, lambda s, i=i: 100 * (1 - s[..., i] / s[..., base])))
        if a in ("actor_only", "raw_no_probe", "serial"):
            put_interval(out, "AblSlower" + NAME[a], runs.interval(
                rows, lambda s, i=i: 100 * (s[..., i] / s[..., full] - 1)))


def tool_latency(abl, out):
    arms = ABLATION_ARMS[:-1]
    fig, ax = plt.subplots(figsize=(3.2, 2.25))
    floors = []
    series = defaultdict(list)
    for tag, path in (("Low", runs.DETERMINISTIC[0]),
                      ("Mid", runs.RESULTS / "e5_tool10_qwen32_invariant_v1"),
                      ("High", runs.RESULTS / "e5_tool20_qwen32_invariant_v1")):
        floor = json.loads((path / "manifest.json").read_text())["tool_floor_s"]
        groups = abl if tag == "Low" else runs.groups(runs.load((path,)), arms)
        if groups.keys() != abl.keys():
            raise ValueError(f"Different or incomplete tool-latency cohort: {path}")
        same = 0
        for task, entries in groups.items():
            for arm in arms:
                current, reference = entries[0][arm], abl[task][0][arm]
                same += (signature(current) == signature(reference)
                         and [t["main_token_ids"] for t in current["turns"]]
                         == [t["main_token_ids"] for t in reference["turns"]])
        prefix = "Tool" + tag
        put(out, prefix + "Floor", floor, INT)
        put(out, prefix + "Identical", same, INT)
        put(out, prefix + "Comparisons", len(groups) * len(arms), INT)
        rows = [[entries[0][a]["wall_s"] for a in arms] for entries in groups.values()]
        floors.append(floor)
        for base, method, name in (("nospec", "spork", "Dispatch"), ("spork", "raw", "Increment"),
                                   ("nospec", "raw", "Total")):
            i, j = arms.index(base), arms.index(method)
            est = runs.interval(rows, lambda s: 100 * (1 - s[..., j] / s[..., i]))
            put_interval(out, prefix + name, est)
            series[name].append(est)
        for i, name in ((arms.index("actor_only"), "Defer"), (arms.index("raw_no_probe"), "NoProbe")):
            j = arms.index("raw")
            put_interval(out, prefix + name, runs.interval(rows, lambda s: 100 * (s[..., i] / s[..., j] - 1)))
    for name, label, marker in (("Total", "TurnAhead vs. serial", "o"),
                                ("Increment", "TurnAhead vs. dispatch", "s"),
                                ("Dispatch", "Dispatch vs. serial", "^")):
        est, lo, hi = np.array(series[name]).T
        ax.plot(floors, est, marker=marker, label=label, markersize=4)
        ax.fill_between(floors, lo, hi, alpha=0.12)
    ax.set(xticks=floors, xlabel="Tool latency floor (s)", ylabel="Task time saved (%)", ylim=(0, None))
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(pad=0.45)
    fig.savefig(ROOT / "figures" / "fig4_tool_latency.pdf")
    plt.close(fig)


def breakdown_table(det, std, path):
    """Appendix table: savings of the method over speculative tool dispatch by domain."""
    def num(x):
        return ("$-$%.1f" % -x) if round(x, 1) < 0 else "%.1f" % abs(x)

    def cell(grouped, keep):
        sub = {t: v for t, v in grouped.items() if keep(v[0]["spork"])}
        rows = [[sum(g[a]["wall_s"] for g in v) for a in ("spork", "raw")] for v in sub.values()]
        n = sum(len(v) for v in sub.values())
        if n < 5:
            return "-- & -- & -- & --"
        est = runs.interval(rows, lambda s: 100 * (1 - s[..., 1] / s[..., 0]))
        return "%d & %.1f & %.1f & %s\\,[%s,\\,%s]" % (
            n, sum(r[0] for r in rows) / n, sum(r[1] for r in rows) / n, *map(num, est))
    subsets = [("All tasks", lambda r: True), ("Airline", lambda r: r["domain"] == "airline"),
               ("Retail", lambda r: r["domain"] == "retail")]
    lines = ["% Generated by scripts/results_stats.py. Do not edit by hand.",
             "\\begin{tabular}{lrrrlrrrl}", "\\toprule",
             " & \\multicolumn{4}{c}{Deterministic serving} & \\multicolumn{4}{c}{Standard serving} \\\\",
             "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
             "Tasks & Runs & Dispatch (s) & \\method (s) & Saving (\\%) "
             "& Runs & Dispatch (s) & \\method (s) & Saving (\\%) \\\\", "\\midrule"]
    lines += ["%s & %s & %s \\\\" % (label, cell(det, keep), cell(std, keep)) for label, keep in subsets]
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def plot(det, model, path):
    rows = [(v[0]["spork"]["wall_s"], v[0]["raw"]["wall_s"]) for v in det.values()]
    base, ours = np.array(rows).T
    plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5))
    ax = axes[0]
    top = float(np.ceil(1.03 * max(base.max(), ours.max()) / 100) * 100)
    ax.plot([0, top], [0, top], color="#1F2A44", linewidth=0.8, label="equal time")
    ax.plot([0, top], [0, 0.8 * top], color="#1F2A44", linewidth=0.7, linestyle=(0, (4, 3)), label="20% less time")
    ax.scatter(base, ours, s=9, color="#3F8F4C", alpha=0.7, linewidths=0)
    ax.legend(loc="upper left", frameon=False, handlelength=2.2, borderaxespad=0.2)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_aspect("equal")
    ax.set_xlabel("task time with speculative tool dispatch (s)")
    ax.set_ylabel("task time with TurnAhead (s)")
    ax.set_title("(a) per-task time, deterministic serving", fontsize=8)
    ax = axes[1]
    predicted, measured = model[:, 2], model[:, 0] - model[:, 1]
    lo, hi = min(predicted.min(), measured.min(), 0.0), max(predicted.max(), measured.max())
    hi = float(np.ceil(hi / 50) * 50)
    lo = float(np.floor(lo / 50) * 50)
    ax.plot([lo, hi], [lo, hi], color="#1F2A44", linewidth=0.8)
    ax.scatter(predicted, measured, s=9, color="#4A78B0", alpha=0.7, linewidths=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("time saved per task, predicted (s)")
    ax.set_ylabel("time saved per task, measured (s)")
    ax.set_title("(b) predicted and measured saving", fontsize=8)
    for ax in axes:
        ax.tick_params(width=0.6, length=2.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout(pad=0.4, w_pad=1.5)
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path)


def main():
    out = {}
    std_records = runs.load(runs.STANDARD)
    det_records = runs.load(runs.DETERMINISTIC)
    abl_records = {k: r for k, r in det_records.items() if (k[0], 0, "serial") in det_records}
    std = runs.groups(std_records, runs.MAIN_ARMS)
    det = runs.groups(det_records, runs.MAIN_ARMS)
    abl = runs.groups(abl_records, ABLATION_ARMS)
    slowdown = concurrency(out)
    for prefix, records, grouped in (("Std", std_records, std), ("Det", det_records, det)):
        put(out, prefix + "Planned", len({g for g, _, _ in records}), INT)
        latency(grouped, runs.MAIN_ARMS, PAIRS, prefix, out)
        quality(grouped, runs.MAIN_ARMS, prefix, out)
        failures(records, runs.MAIN_ARMS, prefix, out)
        windows(grouped, "raw", "Win" + prefix, out)
        overhead(grouped, runs.MAIN_ARMS, "Ovh" + prefix, out)
        speed(grouped, "Spd" + prefix, out)
        reuse(grouped, prefix, out)
    identity(det_records, runs.MAIN_ARMS, "Det", out)
    identity(abl_records, ABLATION_ARMS, "Abl", out)
    divergence(std_records, "Std", out)
    ablation(abl, out)
    model = cost_model(det, slowdown["Det"], "Det", out)
    cost_model(std, slowdown["Std"], "Std", out)
    generated = ROOT / "generated"
    generated.mkdir(exist_ok=True)
    breakdown_table(det, std, generated / "tab_breakdown.tex")
    plot(det, model, ROOT / "figures" / "fig3_results.pdf")
    tool_latency(abl, out)
    runs.write_macros(out, generated / "results_numbers.tex", "results_stats.py")
    (generated / "results_stats.json").write_text(json.dumps({k: v[0] for k, v in out.items()}, indent=2) + "\n")
    print(json.dumps({k: v[0] for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
