#!/usr/bin/env bash
# Ejecuta varias carreras en Webots sin interfaz (rotando la parrilla de salida)
# Uso: scripts/webots_batch.sh carrera_S 4 [max_time]
set -e
WORLD=${1:-carrera_S}; N=${2:-4}; MAXT=${3:-900}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/results/webots"; mkdir -p "$OUT"
for g in $(seq 0 $((N-1))); do
  echo "== carrera $g en $WORLD =="
  RACE_GRID=$g RACE_MAXTIME=$MAXT RACE_QUIT=1 RACE_OUT="$OUT" \
    ${XVFB:-xvfb-run -a} webots --batch --mode=fast --no-rendering --stdout --stderr --minimize \
    "$ROOT/webots/worlds/$WORLD.wbt" 2>&1 | grep "race_brain\]" || true
done
