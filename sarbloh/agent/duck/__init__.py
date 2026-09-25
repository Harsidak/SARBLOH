"""The Duck: Tufa Labs' Milestone #1 agent core, ported verbatim as SARBLOH's baseline (B000).

Provenance: Kaggle dataset jeroencottaar/taaf-kaggle-source-share (bundle 2026-06-12, ARC3-Inference aa69123),
snapshot in resources/repos/duck-kaggle-bundle/. Files copied from ``inference/agent/*`` and
``inference/utils/{grid_utils,openai_compat,segmentation}.py``. The only edits are:
- imports rewritten from ``inference.*`` to ``sarbloh.agent.duck.*``;
- ``python_tool_sandbox._kill_process_group`` also catches AttributeError (no ``os.killpg`` on Windows).

Licence: UNCONFIRMED (MIT classifier in pyproject, no LICENSE file). Must be settled before a prize submission.

Configuration is read from environment variables AT IMPORT TIME (``LOCAL_ANALYZER_*``). Call
``sarbloh.config.apply_llm_env(config)`` before importing ``tool_agent``. Do not change behaviour here: this
package is the control arm. New mechanisms go in the sarbloh modules and are switched in from CONFIG.
"""
