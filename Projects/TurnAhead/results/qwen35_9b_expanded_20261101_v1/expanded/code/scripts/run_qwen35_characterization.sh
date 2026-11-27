#!/usr/bin/env bash
# Diagnostic ablation only: retains trajectory drift and failures as outcomes.
# The strict preflight/ablation entry point and its acceptance gate stay intact.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/qwen35_env.sh"
cd "$SPORK_CODEX_ROOT"
export CUDA_VISIBLE_DEVICES=''
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --run ) ]]; then
    printf 'Usage: bash scripts/run_qwen35_characterization.sh [--run]\n' >&2
    exit 2
fi
SPORK_Q35_CHARACTERIZATION="$SPORK_CODEX_ROOT/results/${SPORK_Q35_RUN_TAG}_characterization"
SPORK_Q35_ARGS=(
    --model "$SPORK_CODEX_MODEL" --model-family qwen35
    --base-url "http://127.0.0.1:$SPORK_CHECK_PORT/v1"
    --task-split "$SPORK_CODEX_ROOT/prefix_data/datasets/controller_stage3_20261008_213013_wi87e7du/task_split.json"
    --scope train --configs spork,serial,actor_only,raw
    --task-ids airline:2,airline:3,airline:4,airline:6,retail:0,retail:1,retail:3,retail:4
    --max-turns 2 --max-tokens 8192 --tool-floor 2 --seed 42 --repetitions 2
    --presence-penalty "$SPORK_Q35_PRESENCE_PENALTY"
    --output "$SPORK_Q35_CHARACTERIZATION"
)
"$SPORK_CODEX_ENV/bin/python" scripts/run_successor_full_eval.py "${SPORK_Q35_ARGS[@]}"
"$SPORK_CODEX_ENV/bin/python" - "$SPORK_Q35_CHARACTERIZATION" <<'PY'
import hashlib, json, pathlib, sys
output = pathlib.Path(sys.argv[1])
sources = [pathlib.Path('scripts/run_qwen35_characterization.sh'),
           pathlib.Path('scripts/analyze_qwen35_characterization.py')]
protocol = {
    'purpose': 'Diagnostic four-arm ablation after batch-sensitive GDN preflight',
    'formal_gate_overridden': False,
    'formal_quality_or_speedup_claim': False,
    'reason_for_continuing': 'Quantify failures, same-seed repeatability, trajectory changes and useful child work across eight training tasks; do not select only matching runs.',
    'planned_runs': 64,
    'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
}
path = output / 'diagnostic_protocol.json'
if path.exists() and json.loads(path.read_text()) != protocol:
    raise SystemExit('Diagnostic protocol changed; use a new run tag')
path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n')
for source in sources:
    target = output / 'diagnostic_code' / source
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != source.read_bytes():
        raise SystemExit('Diagnostic code changed; use a new run tag')
    target.write_bytes(source.read_bytes())
PY
if [[ $# -eq 0 ]]; then
    exit 0
fi
"$SPORK_CODEX_ENV/bin/python" scripts/check_qwen35_experiment.py --server
"$SPORK_CODEX_ENV/bin/python" -u scripts/run_successor_full_eval.py \
    "${SPORK_Q35_ARGS[@]}" --run 2>&1 | tee -a "${SPORK_Q35_CHARACTERIZATION}.launch.log"
"$SPORK_CODEX_ENV/bin/python" scripts/analyze_qwen35_characterization.py "$SPORK_Q35_CHARACTERIZATION"
