#!/usr/bin/env bash
set -euo pipefail
ENV_DIR="$(cd "$(dirname "$0")" && pwd)"

EXE="${CONDA_EXE:-$(command -v mamba || command -v conda)}"
if [ -z "${EXE:-}" ]; then
  echo "ERROR: conda/mamba not found. Install Miniconda (or Miniforge/Mambaforge) first."
  exit 1
fi

for f in "$ENV_DIR"/*.yaml; do
  echo "==> $f"
  "$EXE" env create -f "$f" || "$EXE" env update -f "$f" --prune
done
