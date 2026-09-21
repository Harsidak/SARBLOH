# sarbloh.surat — Perception and retrieval

Owns everything between the raw environment frame and the rest of the system.

Responsibilities:
- Store every frame losslessly outside the context window, keyed by (turn, frame index).
- Serve retrieval at frame, region and pixel granularity, on request, as an explicit operation.
- Encode observations for the model: grid serialisation, frame deltas, optional render.
- Object segmentation and identity tracking across frames.

Forbidden: using the context window as the primary frame store. The ARC-AGI-3 technical report names naive
rolling-window context management as a leading cause of agent failure; retrieval is a choice the agent makes,
not a side effect of message history.

Open question to settle by experiment: whether hand-built segmentation helps. Tufa Labs reported that
hand-crafted tools *hindered* their agent and that improvisation did better. Do not assume.

Planned units: `frame_store.py`, `retrieval.py`, `encoding.py`, `segmentation.py`
