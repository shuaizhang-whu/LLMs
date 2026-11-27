#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/codex_env.sh"
cd "$SPORK_CODEX_ROOT"
export CUDA_VISIBLE_DEVICES=''
stage=${1:-all}
mode=${2:---prepare}
case "$mode" in --run|--prepare) ;; *) exit 2 ;; esac
case "$stage" in
    all)
        bash "$SPORK_CODEX_ROOT/scripts/run_e6_telecom.sh" smoke "$mode"
        exec bash "$SPORK_CODEX_ROOT/scripts/run_e6_telecom.sh" main "$mode" ;;
    smoke) args=(--scope train --task-ids telecom:278 --require-identical-trajectories) ;;
    main) args=(--scope test) ;;
    *) exit 2 ;;
esac
output="$SPORK_CODEX_ROOT/results/e6_telecom_solo_qwen32_${stage}_invariant_v1"
if [[ "$mode" == --run ]]; then
    "$SPORK_CODEX_ENV/bin/python" scripts/check_codex_runtime.py --server --serving-mode invariant
    args+=(--run)
fi
"$SPORK_CODEX_ENV/bin/python" -u scripts/run_successor_full_eval.py --workload telecom-solo \
    --output "$output" --configs nospec,spork,raw --model "$SPORK_CODEX_MODEL" \
    --base-url "http://127.0.0.1:$SPORK_CHECK_PORT/v1" --serving-mode invariant \
    --max-turns 32 --tool-floor 2 --repetitions 1 "${args[@]}" 2>&1 | tee -a "$output.launch.log"
if [[ "$stage" == smoke && "$mode" == --run ]]; then
    "$SPORK_CODEX_ENV/bin/python" - "$output/summary.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
if not report["complete"] or report["failures"]:
    sys.exit("E6 smoke failed; see " + sys.argv[1])
PY
fi
