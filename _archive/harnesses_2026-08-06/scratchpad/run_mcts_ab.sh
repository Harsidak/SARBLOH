#!/bin/sh
# Paired A/B for COMPONENT 4 (MCTS): identical code, ARC_NO_MCTS off vs on.
# Only the 8 dev games where the branch actually fires -- the other 9 cannot
# differ, so running them would spend 20 minutes proving nothing.
# `mon` = MCTS on (the control, = current behaviour), `moff` = MCTS disabled.
export PYTHONIOENCODING=utf-8
export PYTHONHASHSEED=0
PY=/c/Users/Banwa/anaconda3/python.exe
D=scratchpad/diag
mkdir -p $D
run () {   # $1 = tag
  "$PY" -u scratchpad/route_diag.py bp35 dc22 --seed 0 --timeout 390 --out $D/$1_0.json > $D/$1_0.log 2>&1 &
  "$PY" -u scratchpad/route_diag.py sb26 sc25 --seed 0 --timeout 390 --out $D/$1_1.json > $D/$1_1.log 2>&1 &
  "$PY" -u scratchpad/route_diag.py sp80 tr87 --seed 0 --timeout 390 --out $D/$1_2.json > $D/$1_2.log 2>&1 &
  "$PY" -u scratchpad/route_diag.py wa30 su15 --seed 0 --timeout 390 --out $D/$1_3.json > $D/$1_3.log 2>&1 &
  wait
}
unset ARC_NO_MCTS
run mon
echo "MON DONE"
export ARC_NO_MCTS=1
run moff
echo "MOFF DONE"
