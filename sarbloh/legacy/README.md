# sarbloh/legacy — frozen

Pre-reorganisation monolithic agents, preserved verbatim:

- `my_agent.py` (465 KB)
- `my_agent_llm.py` (188 KB)
- `my_agent_original.py` (181 KB)
- `check_statencoder.py`

These are here to be *read and harvested*, not run and not imported. Working behaviour is extracted into the
proper modules one component at a time, each extraction accompanied by a test in `tests/component/` that pins
the behaviour before the move.

Do not refactor these files in place. Do not import from this package. When a component has been extracted and
pinned by a test, note it below.

## Extraction log

| Component | From | To | Test | Date |
| --- | --- | --- | --- | --- |
| _(none yet)_ | | | | |
