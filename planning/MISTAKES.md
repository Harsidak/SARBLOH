# Mistakes log

Append-only. Anything that cost more than an hour. Newest first.

The entry that matters is the **cause**, not the symptom. "Run crashed" is not an entry. "Reward function failed
silently because the Kaggle environment swallows exceptions in the scoring callback" is an entry.

---

## 2026-09-25 — MTP speculative decoding on a checkpoint that has no MTP weights

**Cost:** one 52-min RTX Pro 6000 quick-check run (kernel `banwait13/sarbloh-main` v1) served with a useless draft
head; single-stream decode 27.3 tok/s, and ~40k tokens per game in 40 min.

**What happened:** the `fast` profile (`--speculative-config {"method":"mtp","num_speculative_tokens":3}`) started,
passed the tool-call smoke test and was accepted. The vLLM log then showed `Mean acceptance length: 1.00` and ~0%
draft acceptance for the whole run: every step paid for 3 draft tokens and kept none. The same profile also turned
prefix caching off, so each agent turn re-prefilled its whole context (prefix cache hit rate 0.0% throughout).

**Cause:** the attached weights are `vrfai/Qwen3.6-27B-FP8`, an llm-compressor requant saved as a single
`model.safetensors`. The official `Qwen/Qwen3.6-27B-FP8` ships a separate `mtp.safetensors` (477 MB); the vrfai
export has none (checked on HF 2026-09-25), so vLLM ran an MTP layer with no trained weights. The `fast` flags were
copied from DuckQwen's setup, which used a different checkpoint. The smoke test checks correctness, not speed, so it
could not catch this.

**Rule adopted:** before trusting a speculative-decoding profile, read the vLLM `SpecDecoding metrics` line
(acceptance must be well above 0), and check the checkpoint's file list for draft weights before enabling MTP.
`fast` is removed from the chain until it runs on a checkpoint that has MTP weights.

---

## 2026-09-21 — Reorg moved files without running the tests

**Cost:** the test suite was silently un-runnable for three days. Found 2026-09-24 when setting up the uv env.

**What happened:** the SARBLOH reorg moved `my_agent.py` to `sarbloh/legacy/`, `Kaggle_test.py` to `sarbloh/harness/`
and the suites to `tests/component/`. Every suite's path shim, the runner's glob and `eval/components.py` still
pointed at the old locations: `run_all.py` reported "no suites selected", and each suite failed on import.
The documented command (`pytest tests/`) cannot run these suites at all, because they are scripts, not pytest tests.

**Cause:** the reorg was checked by eye, not by running `tests/run_all.py` afterwards, and the documented test
command had never been executed.

**Rule adopted:** any commit that moves files runs `uv run python tests/run_all.py` and must stay 16/16 green. Every
command documented in `CLAUDE.md` §6 has been executed at least once (verified 2026-09-24).

---

## Prior Kaggle competition — RL forced where hardware made it suboptimal

**Cost:** the competition result; LoRA SFT, not RL, produced the best score.

**What happened:** the owner, whose research background is RL, applied RL to a Kaggle problem because the papers
argue RL is what drives advanced agentic systems. Under Kaggle's hardware limits RL underperformed, and LoRA SFT
won.

**Cause:** the method was chosen from the literature, which assumes cluster-scale compute, rather than from the
competition's actual GPU and wall-clock budget, and it was not tested early against a simpler baseline.

**Rule adopted:** method follows hardware. RL has to beat an SFT or no-training baseline inside the Kaggle budget,
in a measured run, before it is adopted. The default training route is LoRA SFT on verified trajectories.
(`CLAUDE.md` §0)

---

## 2026-08 — Tests moved to the archive

**Cost:** repository verifiability, for roughly six weeks.

**What happened:** fifteen test files were moved to `_archive/tests_2026-08-10/` rather than fixed. Four tests
remained live against an 830 KB agent codebase.

**Cause:** tests were treated as blocking work rather than as the thing that makes work cumulative. Under
schedule pressure the cheapest-looking move was to remove the blocker.

**Rule adopted:** a broken test is fixed or deleted with a recorded reason. Never archived. (`CLAUDE.md` §4.6)

---

## 2026-08 — Results recorded without attribution

**Cost:** three months of experimental results are not reconstructable.

**What happened:** fourteen result files accumulated in `scores/` — `battle2`, `champion`, `memo`,
`llmagent_route` — with no mapping from any of them to the configuration or hypothesis that produced it.

**Cause:** logging was treated as a write-up step to be done later rather than as part of the run.

**Rule adopted:** the ledger row is written by the run. A run that is not logged did not happen. (`CLAUDE.md` §4.1)

---

## 2026-09-21 — Wall-clock ceiling assumed, not verified

**Cost:** low, caught early.

**What happened:** planning proceeded on a 9-hour Kaggle ceiling. The ARC Prize testing policy states 12 hours.

**Cause:** a working figure carried forward across sessions without being checked against the source.

**Rule adopted:** competition constants live in `CLAUDE.md` §2 with a verification date, and anything unverified
is marked UNCONFIRMED rather than used silently.

## 2026-09-28 — E003: vLLM "Engine core initialization failed" hid a linker error (v1 lost ~1 h)
- Symptom: every vLLM profile failed; the notebook showed only the APIServer "see root cause above" traceback.
- Cause: FlashInfer JIT-links its sm120 NVFP4 GEMM with `-lcuda`; the Kaggle image has only `libcuda.so.1`. Our
  failure print showed the log tail, not the engine's error lines.
- Fix: `prime/vllm.py` symlinks `libcuda.so` + sets LIBRARY_PATH (confirmed in v2); failure print greps engine errors.

## 2026-09-28 — E003: shipped a harness whose distinguishing features were opt-in
- Symptom: v2 ran cleanly but 0 compactions, 0 spawns, 0 Continual Harness writes: we tested bare Gemma + REPL.
- Cause: the port copied the paper's tools and prompt but not a guarantee they get used; with Gemma-31B nothing
  forced memory writes, and the 300-action cap ended games before compaction could trigger. We did not check
  locally that a model actually calls `rlm.harness` before spending an RTX Pro hour.

## 2026-09-28 — E003/E004: the "port" was not checked line by line against upstream (two runs misread)
- Symptom: E004 showed a reset after every compaction, memories of false facts, and the cap message filed as a game
  rule. We blamed the 4B model and the prompt, and E003's 0 harness writes on Gemma opting out.
- Cause: `prime/` was written as a reduced reimplementation of ~151k lines of TypeScript and labelled a "port" without
  a side-by-side audit. Compaction kept only the task + a summary (upstream keeps the newest 20k tokens verbatim, with
  a separate structured summarizer); auto-refine was off by default (upstream: on); the digest went into the system
  prompt instead of a message; refine prompts were labelled "verbatim" but had five lines missing. Our own ARC text
  used "cell" for two things and described `transitions()` as a tuple.
- Fix: E005 audit (`experiments/E005_prime_fidelity/audit.md`) and port. Rule: a module that claims to port an
  upstream file cites it and is diffed against it before any run is interpreted; "verbatim" is checked by a script.

## 2026-09-28 — `arc.step(0)` was a silent RESET (E004 and the E005 v1 smoke misread)
- Symptom: the E005 v1 local smoke spent 100 of 150 actions on RESET; E004's "reset after every compaction" was
  read as a compaction symptom.
- Cause: our ARC interface exposed the SDK's raw ids. RESET is id 0 and was listed in `obs.available_actions`, so
  the natural loop `for a in range(5): step(a)` reset the level on every pass. The model also reset before any
  compaction, so the post-compaction reset count never isolated compaction.
- Fix: RESET only through `arc.reset()` (own host request); `arc.step(0)` raises with a pointer to `arc.reset()`;
  0 is hidden from `available_actions`. Rule: an action interface never lets a destructive action hide behind a
  default index; count a behaviour's baseline rate before attributing it to a mechanism.

## 2026-09-30 — E007 bench: two wasted Kaggle runs (JSON in Python, then the wrong GPU)
- Symptom: v1 died after 8 s with `NameError: name 'true' is not defined`. v2 ran 11 min and every one of the 6
  profiles OOM'd while loading weights ("GPU 0 has a total capacity of 14.56 GiB"): it had 2x T4, not the RTX Pro 6000.
- Cause: v1 put `json.dumps(meta)` into a Python cell, and the pre-push check only exec'd the `SOURCES` line. v2's
  kernel metadata had `machine_shape: NvidiaRtxPro6000` but `competition_sources: []`; every run that got the card
  (sarbloh-main, sarbloh-prime) had the ARC-AGI-3 competition attached. That this is the cause is UNCONFIRMED.
- Fix: `repr()` for embedded values, and the check compiles every cell. The bench kernel attaches the competition and
  its environment cell asserts >90 GB VRAM before any model loads, so a wrong machine fails in seconds. Rule: a GPU-shaped run asserts its
  GPU before doing anything.

## 2026-10-02 — E112 v13: the SGLang wheelhouse could not install (Kaggle strips `+` from file names)
- Symptom: every NVFP4 SGLang profile failed in 0-4 s: `flashinfer-jit-cache was not found ... you require
  flashinfer-jit-cache==0.6.17+cu130`. The file was in the dataset.
- Cause: the Kaggle dataset upload renamed every local-version wheel, dropping the `+`
  (`torch-2.13.0cu130-...whl`, `sglang-0.5.19gd00d88efc8d6-...whl`). That is no longer a valid wheel version, so uv
  ignored those files and the lock's `==X+local` pins could not resolve. The unit test fixture used the pre-upload
  name with `+`, so it could not catch this.
- Fix: `sglang.relink()` builds a find-links dir of symlinks with each lock pin's `+local` restored, and uv
  installs from that. Rule: a wheelhouse test uses the file names *as listed on Kaggle* (`kaggle datasets files`),
  not the names the build produced.

## 2026-10-02 — E109 ls20: the watchdog killed a healthy SGLang server twice ("frozen")
- Symptom: the ls20 run on SGLang FP8 restarted the server at 21:25 and fell back to vLLM at 21:30. In 16 min the
  agent took 1 action. The freeze dumps showed "Fatal Python error: Aborted" in torch inductor threads.
- Cause: the watchdog measures progress as `prompt_tokens_total + generation_tokens_total`. SGLang increments both
  only when a request finishes, so one thinking turn longer than `freeze_after_s` (120 s) looked frozen while the
  log showed steady decoding at 43 tok/s. The "Aborted" stacks were our own kill signal. The E109 tests used counter
  values from finished requests only.
- Fix: also count `sglang:realtime_tokens_total`, which moves every decode step. Rule: a liveness signal is
  validated against a long in-flight request on each backend, not only against counters after requests finish.

## 2026-10-20 — Flash-Next submission run: the server took 3 h 15 min to start (8 min in earlier runs)
- Symptom: run 20261020_130421 was ready after 11715 s. `uv pip install -r requirements.lock` ran 18:34 → 21:27
  (10450 s; 52-83 s in every earlier run), then the weight load took 1042 s (210-242 s in the 10-17 and 10-19 morning
  runs). The 10-19 evening run had a normal install but a 1199 s weight load (ready at 1504 s, just under the 1500 s
  startup timeout).
- Cause (partly UNCONFIRMED): `qwen_flash_next.flash_sglang_runtime` started a 181 GB precache of the target and
  drafter *before* the install, so it read the same /kaggle/input mount as uv and later as SGLang's loader. 181 GB
  does not fit in 176.9 GB of RAM beside the server's 105 GB, so it could not stay cached anyway. Why uv kept going
  slowly for 2.8 h after the precache finished at 18:40 is not explained by the logs. The earlier fix that moved the
  precache after the install was made in `sglang.runtime`, which the Flash-Next path does not use.
- Fix: the Flash-Next install runs first and alone, and only the 4 GB drafter is precached after it. Every install
  step is killed after 300 s and retried from a clean venv; the third try has no limit, so a slow mount costs time
  but never fails the run. Rule: a startup change is checked on the code path of the profile that actually runs.

## 2026-10-20 (later): the precache was not why uv hung
- **Symptom:** after the weight precache was moved behind the install, the next run's `uv pip install` still hung: try 1 and try 2 were each killed at 300 s, and try 3 ran with no limit.
- **Cause (corrected):** the earlier entry blamed the 181 GB precache fighting for /kaggle/input. That was wrong. On 10-19 the install finished in 53 s with the same precache running beside it, and on 10-20 uv kept hanging for 2.8 h after the precache had ended. The hang is in uv reading wheels on the /kaggle/input mount, while plain sequential reads of the same mount ran at about 300 MB/s. Exactly why uv hangs is UNCONFIRMED.
- **Fix:** copy the wheels to /tmp in big sequential reads first, so that uv only ever reads local disk. Run uv with `-v` and print the tail of its log when a try times out.

## 2026-10-20 (evening): removing the target precache made the weight load cold
- **Symptom:** the install finished in under 160 s, then the log went quiet for many minutes while the weights loaded.
- **Cause:** db04b1d stopped precaching the 181 GB target, based on the wrong idea that the precache made uv hang. The
  fast starts (10-17, 10-19 morning: weights in about 4 min) all ran that precache beside the install. Without it the
  server reads the weights cold from /kaggle/input (17-21 min in the slow runs).
- **Fix:** the target and drafter precache starts first again, as in Sarbloh/; the local wheel copy stays, so uv no
  longer reads the mount. UNCONFIRMED until the next Kaggle run.
- **Lesson:** diff against the code that was fast (Sarbloh/) before changing what ran beside the slow step.

## 2026-10-20 (night): a fixed 25 min start timeout killed loads that were still going
- **Symptom:** games started 2 h 21 min into the run, on the 27B fallback. Every Flash-Next profile was killed at
  exactly 1500 s while still loading (frspec at 37/38 shards, then 15, 34 and 14 of 38), then the four 27B NVFP4
  profiles failed at once (banwait13/qwencoder is not attached), then the 27B FP8 on vLLM came up on its second try.
- **Cause:** `wait_ready` used a fixed deadline, `startup_timeout_s` 1500. The 38-shard load takes 3-4 min on a fast
  mount day (10-17, 10-19 morning) and 17-20 min on a slow one (10-19 evening, the 125-level run), so the slow-day
  starts only just fit under 25 min. Today the mount was slower still. Each retry loaded the same 181 GB from shard 1
  again, colder and slower. The previous entry is wrong on one point: the slow 10-19 evening runs had the same target
  precache as the fast ones (no serving change between fc6f84b and af6b06f), so the precache does not explain
  the fast/slow split; the mount speed does (UNCONFIRMED why it varies).
- **Fix:** a start fails only when the server log stops growing for `startup_stall_s` (600 s; the slowest shard
  seen took 110 s), or at `startup_timeout_s` (now 3600 s). The notebook prints the server's newest log line every
  minute. The unattached NVFP4 profiles are out of the Flash-Next chain.
- **Lesson:** a start timeout must follow progress, not a clock: the milestone-2 reference never kills a loading
  server at its deadline.

## 2026-10-23: two Phase B "system error" failures of v52 (cause UNCONFIRMED)
- **Symptom:** version 52 of `sarbloh-experimentation` (scriptVersionId 361194559) passed Save & Run in 8.95 h, then
  failed Phase B twice (10-23 03:40 and 08:34 UTC) with "A system error. Please try resubmitting". Phase B has no log.
- **Most likely cause: Kaggle's side.** Kaggle staff wrote on the forum (topic 751756, 10-23 23:54 UTC) that failed
  ARC-AGI-3 reruns in those days came from "dataset/model cache instability", escalated, with reruns of affected
  submissions under evaluation. Other teams got the same error on unchanged notebooks from 10-22 on (751631, 752253).
- **Our own risks, found while checking:**
  1. Run-time. The competition's Code Requirements page says "GPU Notebook <= 9 hours run-time"; commit ab61612 gave
     Submit 10 h, and v52 always plays to the cap (games stall near the 2000-action cap). Against it: versions with
     11.5 h (10-16) and 10 h (10-22, LoRA) completed, but whether those reruns actually ran past 9 h is unknown.
  2. Server start. `startup_timeout_s` is 3600 s; v52 was ready 2405 s after launch, the 10-22 1-hour run after
     3158 s, and teams report > 1 h model loads in those days. No fallback model, so a timeout fails the rerun.
  3. Gateway inactivity. Forum reports say the gateway closes a run after about 15 min without actions (UNCONFIRMED).
     In v52's Save & Run the last two games stalled without acting for the final 17 min.
- **Fix so far:** Submit budget 8.75 h (Save & Run stays 9 h). CLAUDE.md §2 and `resources/official/COMPETITION-RULES.md`
  record the 9-hour Kaggle rule; the 2026-09-21 entry took 12 h from the ARC Prize policy, not the Kaggle page.
- **Lesson:** before calling a Phase B error our bug, read the competition forum (`competition_list_topics`) for the
  same days; read limits from the Kaggle pages (`competition_list_pages`), not the organiser's general policy.

---

## 2026-11-04 — Six test suites left pinned to retired code

**Cost:** 8 of 40 suites red at the final cleanup; about an hour to sort real breaks from stale tests.

**What happened:** `tests/run_all.py` showed 32/40 green. Six suites failed on import or setup because the code
they test is gone from both `Sarbloh/` and `Sarbloh-Experimentation/`: `act()` and the `recall` tool
(`test_prime_tools`), `parse_hypothesis` (`test_prime_memory`), `GameMemory.write_act` (`test_prime_curator`),
`nemotron.NANO_OMNI` (`test_model_fallback`), the `lessons` stat (`test_prime_context`) and the
`keep_recent_tokens` config key (`test_prime_fidelity`). Two more failed on wording drift only: the stored picture
label became `[picture of current state(board)` (`test_append_only_context`), and `obs.history` is newest first
by design (`test_e116_perception`).

**Cause:** the harness was rewritten quickly between 10-16 and 10-28 (one-tool play loop, kernel-variable memory,
code graph), and the suites for the old design were not run before each commit.

**Resolution:** the two drift suites were fixed. The six suites for removed code were deleted (tests are never
archived; they stay in git history before this commit). 34/34 green after.

**Lesson:** run `tests/run_all.py --quick` before each commit that removes a tool or a config key.
