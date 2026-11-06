#!/usr/bin/env bash
# Source this file from any directory to activate OxyGen and reuse local assets.
_oxygen_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_oxygen_repo}/.venv/bin/activate"
export OPENPI_DATA_HOME="${_oxygen_repo}/../../shared/checkpoints/openpi"
export UV_LINK_MODE=copy
unset _oxygen_repo
