#!/bin/sh
# Paired A/B for the CERTIFY green-gate (COMPONENT 5.7 / the executable world model).
#
# Measured 2026-08-01 across the 17 dev games: ExecPlanner.act took the GREEN
# exits 119 times and the RED exits 1658 -- the theory is certified only ~7% of
# the time, so `_commit` almost never runs and `react.plan` is 45 of 18,504
# actions. `certify()` only restores green on a throttled full replay triggered
# by a CHANGE to the theory key, so a converged-but-once-falsified theory stays
# red for the rest of the level.
#
# `con` = certify on (control, current behaviour); `coff` = ARC_NO_CERT=1, always
# green, so the planner commits multi-step plans through an uncertified model.
# This is deliberately the UNSAFE direction: committing on a falsified model is
# the hazard the gate exists for. The question the run answers is whether the
# red verdict is EARNED (coff loses levels) or STALE (coff is free or better).
#
# Only the games where ExecPlanner actually drives -- click-only games route to
# the ClickPlanner and cannot differ.
export PYTHONIOENCODING=utf-8
export PYTHONHASHSEED=0
PY=/c/Users/Banwa/anaconda3/python.exe
D=scratchpad/diag
mkdir -p $D
run () {   # $1 = tag
  "$PY" -u scratchpad/route_diag.py cn04 dc22 --seed 0 --timeout 390 --out $D/$1_0.json > $D/$1_0.log 2>&1 &
  "$PY" -u scratchpad/route_diag.py ls20 re86 --seed 0 --timeout 390 --out $D/$1_1.json > $D/$1_1.log 2>&1 &
  "$PY" -u scratchpad/route_diag.py sp80 tr87 --seed 0 --timeout 390 --out $D/$1_2.json > $D/$1_2.log 2>&1 &
  "$PY" -u scratchpad/route_diag.py tu93 wa30 --seed 0 --timeout 390 --out $D/$1_3.json > $D/$1_3.log 2>&1 &
  wait
}
unset ARC_NO_CERT
run con
echo "CON DONE"
export ARC_NO_CERT=1
run coff
echo "COFF DONE"
