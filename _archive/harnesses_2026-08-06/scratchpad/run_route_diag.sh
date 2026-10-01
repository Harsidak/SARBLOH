#!/bin/sh
# Per-branch action accounting across the DEV split only (the 8 held-out games
# are deliberately absent -- no routing change may be derived from them).
# Single arm: this is a measurement, not a comparison, so there is nothing to
# pair against. Same 5-way shard layout and timeout as run_pursuit_diag.sh so
# the wall-clock per shard stays comparable with earlier runs.
export PYTHONIOENCODING=utf-8
export PYTHONHASHSEED=0
PY=/c/Users/Banwa/anaconda3/python.exe
D=scratchpad/diag
mkdir -p $D
"$PY" -u scratchpad/route_diag.py bp35 cn04 dc22 ka59 --seed 0 --timeout 390 --out $D/route_0.json > $D/route_0.log 2>&1 &
"$PY" -u scratchpad/route_diag.py lf52 lp85 ls20 r11l --seed 0 --timeout 390 --out $D/route_1.json > $D/route_1.log 2>&1 &
"$PY" -u scratchpad/route_diag.py re86 sb26 sc25 sp80 --seed 0 --timeout 390 --out $D/route_2.json > $D/route_2.log 2>&1 &
"$PY" -u scratchpad/route_diag.py su15 tn36 tr87 tu93 wa30 --seed 0 --timeout 390 --out $D/route_3.json > $D/route_3.log 2>&1 &
wait
echo "ROUTE DIAG DONE"
