#!/bin/sh
# Paired A/B for the phase-3 ladder change: _explore_action's fallback picks the
# LEAST-SPENT (state, action) pair instead of a uniform-random one.
# `lon` = least-spent (new behaviour), `loff` = ARC_NO_LEASTSPENT=1 (old uniform).
# Only the 7 dev games where that branch fires; the other 10 cannot differ.
export PYTHONIOENCODING=utf-8
export PYTHONHASHSEED=0
PY=/c/Users/Banwa/anaconda3/python.exe
D=scratchpad/diag
mkdir -p $D
run () {
  "$PY" -u scratchpad/route_diag.py su15 cn04 --seed 0 --timeout 390 --out $D/$1_0.json > $D/$1_0.log 2>&1 &
  "$PY" -u scratchpad/route_diag.py lf52 dc22 --seed 0 --timeout 390 --out $D/$1_1.json > $D/$1_1.log 2>&1 &
  "$PY" -u scratchpad/route_diag.py bp35 sc25 --seed 0 --timeout 390 --out $D/$1_2.json > $D/$1_2.log 2>&1 &
  "$PY" -u scratchpad/route_diag.py sb26 --seed 0 --timeout 390 --out $D/$1_3.json > $D/$1_3.log 2>&1 &
  wait
}
export ARC_NO_LEASTSPENT=1
run loff
echo "LOFF DONE"
unset ARC_NO_LEASTSPENT
run lon
echo "LON DONE"
