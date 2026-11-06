#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="${WORKSPACE_ROOT}/policies/realtime-vla-flash"
PYTHON_BIN="${PI0_ABLATION_ORCHESTRATOR_PYTHON:-${REPO_ROOT}/.venv/bin/python}"

exec "${PYTHON_BIN}" "${REPO_ROOT}/scripts/experiment2/run_layer_denoise_ablation.py" "$@"
