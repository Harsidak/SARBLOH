#!/bin/sh
# Controlled pair for Section 9 step 6 (PURSUIT wiring): identical code, goals off
# vs goals on. Same shard layout as run_goal_diag.sh so the arms stay comparable
# with the pre-pursuit `goff`/`gon` pair; new tags so that pair is not overwritten
# -- goff should come back byte-identical, which is itself a check that
# ARC_NO_GOALS still covers every new call site.
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
run poff
echo "POFF DONE"
unset ARC_NO_GOALS
run pon
echo "PON DONE"
