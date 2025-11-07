#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT_DIR/datasets/steel/likely"
OUTPUT_DIR="$ROOT_DIR/results"
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

declare -a PORTFOLIOS=(
  "finished_steel_portfolio.yml finished_portfolio_bf.json"
  "finished_steel_portfolio_eaf.yml finished_portfolio_eaf.json"
  "finished_steel_portfolio_bf_charcoal.yml finished_portfolio_bf_charcoal.json"
)

for entry in "${PORTFOLIOS[@]}"; do
  read -r spec output <<<"$entry"
  python3 -m forge.cli.steel_batch_cli run \
    --spec "$ROOT_DIR/configs/$spec" \
    --data-dir "$DATA_DIR" \
    --output "$OUTPUT_DIR/$output"
done

python3 "$ROOT_DIR/scripts/summarize_portfolios.py" \
  "$OUTPUT_DIR/finished_portfolio_bf.json" \
  "$OUTPUT_DIR/finished_portfolio_eaf.json" \
  "$OUTPUT_DIR/finished_portfolio_bf_charcoal.json" \
  -o "$OUTPUT_DIR/summary.json"
