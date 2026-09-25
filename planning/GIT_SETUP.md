# Git and GitHub setup

Run in this order. Step 0 is not optional — 787 files are about to move and there is currently no version
control on any of them.

---

## 0. Backup before anything

In File Explorer: right-click the `SARBLOH` folder, **Send to > Compressed (zipped) folder**. Move the zip
**outside** `D:\FORGE\Research\` — put it in `D:\FORGE\_backups\`.

This is the real recovery point. Git cannot be it, because `.gitignore` deliberately excludes ~20 MB of archived
traces and binaries that are not worth carrying in history but are worth not losing.

---

## 1. Move `avo` out of the root first

```
D:\FORGE\Research\SARBLOH\avo\   ->   D:\FORGE\Research\SARBLOH\resources\repos\avo\
```

Do this **before** `git init`. `avo/` contains its own `.git`, and initialising around it makes git record a
broken submodule reference. Under `resources/repos/` it is gitignored and the problem disappears.

---

## 2. Initialise

```bash
cd /d D:\FORGE\Research\SARBLOH
git init -b main
git add .
git commit -m "Initial commit: SARBLOH skeleton over existing ARC-AGI-3 work"
```

Check what git decided to track before you commit:

```bash
git status --short
git count-objects -vH        # keep the repo well under 100 MB
```

If anything large slipped through, add it to `.gitignore` and `git rm --cached <path>` before committing.

---

## 3. Identity

```bash
git config user.name  "Harsidak Singh Banwait"
git config user.email "harsidak.singh.banwait13@gmail.com"
```

Use the same email on GitHub, or commits will not attribute to your account — which matters, because this
repository is going to be the artefact someone reads.

---

## 4. Move, then commit in groups

Execute `MIGRATION.md` in order. Commit after each group rather than in one lump:

```bash
git add -A && git commit -m "reorg: freeze monolithic agents into sarbloh/legacy"
git add -A && git commit -m "reorg: reclaim archived tests into tests/component"
git add -A && git commit -m "reorg: papers and architecture notes into resources/ and planning/"
```

Git detects renames automatically when content is unchanged, so history survives the moves.

---

## 5. GitHub

```bash
gh repo create sarbloh --private --source=. --remote=origin --push
```

Or, without the CLI: create an empty repo named `sarbloh` on github.com, then

```bash
git remote add origin https://github.com/<user>/sarbloh.git
git push -u origin main
```

**Private now, public before 30 Sep.** Milestone #2 requires an open-source submission. Going public is a
one-click change; going public with an unreviewed history is not reversible, so do the licence audit first.

---

## 6. Before you make it public

- [ ] No API keys, tokens or `.env` files in history — `git log -p | findstr /i "sk- api_key token"`
- [ ] `LICENSE` present at root (MIT). **Verify MIT satisfies the competition's "permissive public-domain
      licence" wording** — if ARC Prize requires CC0 or Unlicense specifically, swap it now, not in November.
- [ ] Every third-party component under `resources/repos/` has its licence recorded in
      `resources/INDEX.md`
- [ ] No environment source from `eval/real_games/` redistributed in violation of ARC Prize terms — check before
      pushing, these were pulled from the toolkit
- [ ] `CLAUDE.md` UNCONFIRMED items resolved or still clearly marked

---

## 7. Daily rhythm

```bash
git add -A && git commit -m "E00X: <hypothesis in five words>"
```

Commit message carries the experiment id. The ledger carries the result. Together they make
`history/ledger.jsonl` reconstructable from git alone, which is what turns six weeks of work into a paper
instead of a folder.
