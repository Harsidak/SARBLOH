#!/bin/sh
# Controlled pair: IDENTICAL code, goals off vs goals on. The `sac` baseline is a
# different build, so a delta against it also contains the refactor; this pair
# isolates the goal model itself.
export PYTHONIOENCODING=utf-8
export PYTHONHASHSEED=0
PY=/c/Users/Banwa/anaconda3/python.exe
D=scratchpad/diag
run () {   # $1 = tag
  "$PY" -u scratchpad/diag_eval.py ar25 bp35 cd82 cn04 dc22 --seed 0 --timeout 390 --out $D/$1_0.json > $D/$1_0.log 2>&1 &
  "$PY" -u scratchpad/diag_eval.py ft09 g50t ka59 lf52 lp85 --seed 0 --timeout 390 --out $D/$1_1.json > $D/$1_1.log 2>&1 &
  "$PY" -u scratchpad/diag_eval.py ls20 m0r0 r11l re86 s5i5 --seed 0 --timeout 390 --out $D/$1_2.json > $D/$1_2.log 2>&1 &
  "$PY" -u scratchpad/diag_eval.py sb26 sc25 sk48 sp80 su15 --seed 0 --timeout 390 --out $D/$1_3.json > $D/$1_3.log 2>&1 &
  "$PY" -u scratchpad/diag_eval.py tn36 tr87 tu93 vc33 wa30 --seed 0 --timeout 390 --out $D/$1_4.json > $D/$1_4.log 2>&1 &
  wait
}
export ARC_NO_GOALS=1
run goff
echo "GOFF DONE"
unset ARC_NO_GOALS
run gon
echo "GON DONE"
