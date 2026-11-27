#!/usr/bin/env bash
# Optional diagnostic arm; does not bypass the original formal ablation gate.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/qwen35_env.sh"
cd "$SPORK_CODEX_ROOT"
export CUDA_VISIBLE_DEVICES=''
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --run ) ]]; then
    printf 'Usage: bash scripts/run_qwen35_recovery_check.sh [--run]\n' >&2
    exit 2
fi
SPORK_Q35_RECOVERY_OUTPUT="$SPORK_CODEX_ROOT/results/${SPORK_Q35_RECOVERY_TAG:-qwen35_9b_20261101_recovery_check}"
SPORK_Q35_RECOVERY_ARGS=(
    --model "$SPORK_CODEX_MODEL" --model-family qwen35
    --base-url "http://127.0.0.1:$SPORK_CHECK_PORT/v1"
    --task-split "$SPORK_CODEX_ROOT/prefix_data/datasets/controller_stage3_20261008_213013_wi87e7du/task_split.json"
    --scope train --task-ids airline:2,airline:4 --configs spork,raw,raw_fallback
    --repetitions 2 --max-turns 2 --max-tokens 8192 --tool-floor 2 --seed 42
    --presence-penalty "$SPORK_Q35_PRESENCE_PENALTY" --output "$SPORK_Q35_RECOVERY_OUTPUT"
)
"$SPORK_CODEX_ENV/bin/python" scripts/run_successor_full_eval.py "${SPORK_Q35_RECOVERY_ARGS[@]}"
"$SPORK_CODEX_ENV/bin/python" - "$SPORK_Q35_RECOVERY_OUTPUT" <<'PY'
import hashlib, json, pathlib, sys
output = pathlib.Path(sys.argv[1])
sources = [pathlib.Path('scripts/run_qwen35_recovery_check.sh'),
    pathlib.Path('scripts/analyze_qwen35_recovery.py'),
    pathlib.Path('scripts/analyze_qwen35_characterization.py')]
protocol = {'purpose': 'Bounded opt-in child-failure recovery stress check',
    'planned_runs': 12, 'train_tasks_selected_from_observed_failures': True,
    'formal_gate_overridden': False, 'formal_quality_or_speedup_claim': False,
    'extra_target_attempts_counted': True,
    'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
path = output / 'diagnostic_protocol.json'
if path.exists() and json.loads(path.read_text()) != protocol:
    raise SystemExit('Recovery protocol changed; use a new SPORK_Q35_RECOVERY_TAG')
path.write_text(json.dumps(protocol, indent=2) + '\n')
for source in sources:
    target = output / 'diagnostic_code' / source
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != source.read_bytes():
        raise SystemExit('Recovery source snapshot changed')
    target.write_bytes(source.read_bytes())
PY
if [[ $# -eq 0 ]]; then
    exit 0
fi
"$SPORK_CODEX_ENV/bin/python" scripts/check_qwen35_experiment.py --server
"$SPORK_CODEX_ENV/bin/python" -u scripts/run_successor_full_eval.py \
    "${SPORK_Q35_RECOVERY_ARGS[@]}" --run 2>&1 | tee -a "${SPORK_Q35_RECOVERY_OUTPUT}.launch.log"
"$SPORK_CODEX_ENV/bin/python" scripts/analyze_qwen35_recovery.py "$SPORK_Q35_RECOVERY_OUTPUT"
