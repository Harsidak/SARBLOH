# E03 — Segmentation + ASCII grid alongside the image
- **Axis varied:** perception

> Logged retrospectively on 2026-10-01 from the owner's report. Qualitative result; no held-out RHAE yet (Rule 2 open).

## Mechanism
Objects segmented and the grid serialised as ASCII text (implemented in `intuition.py`), next to the E02 PNG. Inspired by Tufa Labs' harness, which does the equivalent as segmentation.

## Why it should work
The image gives the global picture; text gives exact positions and counts the vision path gets wrong.

## Prediction
More precise understanding of object positions and sizes than image alone.
