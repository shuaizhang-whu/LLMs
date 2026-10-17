#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/result/phase2"
LOG_FILE="${LOG_DIR}/phase2_gpu67_round.log"

mkdir -p "${LOG_DIR}"

cd "${SCRIPT_DIR}"
"/data/zhangs/anaconda3/envs/compress/bin/python" "${PROJECT_ROOT}/phase2_manager.py" --run-round 2>&1 | tee -a "${LOG_FILE}"
