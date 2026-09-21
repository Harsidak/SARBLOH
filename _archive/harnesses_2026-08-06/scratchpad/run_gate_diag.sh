#!/bin/sh
# Verification of the steers() gate: with Channel B held out of the potential and
# A=0 everywhere, _active() must be False in every game -> frontier_ranker() is
# None -> rank_frontier is identity -> the run must be BYTE-IDENTICAL to goff.
# That is the falsifiable claim; this run is what decides it.
export PYTHONIOENCODING=utf-8
export PYTHONHASHSEED=0
PY=/c/Users/Banwa/anaconda3/python.exe
D=scratchpad/diag
"$PY" -u scratchpad/diag_eval.py ar25 bp35 cd82 cn04 dc22 --seed 0 --timeout 390 --out $D/ggate_0.json > $D/ggate_0.log 2>&1 &
"$PY" -u scratchpad/diag_eval.py ft09 g50t ka59 lf52 lp85 --seed 0 --timeout 390 --out $D/ggate_1.json > $D/ggate_1.log 2>&1 &
"$PY" -u scratchpad/diag_eval.py ls20 m0r0 r11l re86 s5i5 --seed 0 --timeout 390 --out $D/ggate_2.json > $D/ggate_2.log 2>&1 &
"$PY" -u scratchpad/diag_eval.py sb26 sc25 sk48 sp80 su15 --seed 0 --timeout 390 --out $D/ggate_3.json > $D/ggate_3.log 2>&1 &
"$PY" -u scratchpad/diag_eval.py tn36 tr87 tu93 vc33 wa30 --seed 0 --timeout 390 --out $D/ggate_4.json > $D/ggate_4.log 2>&1 &
wait
echo "GGATE DONE"
