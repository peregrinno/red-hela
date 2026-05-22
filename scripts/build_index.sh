#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f resources/references.json.gz ]]; then
  echo "resources/references.json.gz ausente (copie do repo da rinha)" >&2
  exit 1
fi

export RED_HELA_ROOT="$PWD"
echo "Gerando data/red_hela.idx..."
uv run python -m red_hela.infrastructure.pack_index
ls -lh data/red_hela.idx
