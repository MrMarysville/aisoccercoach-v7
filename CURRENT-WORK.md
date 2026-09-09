# Current work

Updated 2026-09-09. This is the canonical repository for future work.

The owner wants full-game calibration: a source-bound pixel-to-field mapping
maintained through both halves, with simple visual corrections, independently
checked accuracy and honest coverage. The Roseville 120 × 70 yd template stays fixed.

The editor currently opens two reviewed development clips: Granite first half
(351 sampled frames, 35.1 seconds) and Butte second half (300 frames, 30 seconds).
It supports drag adjustments, Undo/Redo, original comparison, seeking, saving
and lookup from the saved map. The owner's saved revisions were copied unchanged.
Automatic initialization and resumable full-game processing from the app are unbuilt.

Newer development work was recovered from a saved checkpoint during migration:
Granite H2-slot1 (2300–2330s) and Butte H1-slot2 (980–1010s), each 300 frames at
10 fps. The selected shared bounded fits had no geometry/motion/source-fit
warnings on those frames. Recorded independent check medians/p95 were
0.443/1.565 px for Granite and 0.482/1.795 px for Butte. Granite passed 39/39
marking/frame groups after a documented source-label correction; its original
checks passed 37/38. Butte's unchanged checks passed 41/41. These are prior
measured development results preserved in `data/development/propagation-attempt007/`,
not new migration measurements or untouched holdout results. They have not
replaced the owner's editor baselines.

The separate 60-second forward/backward propagation experiment uses a paint
lock. Its sampler gauge is fit evidence, not an independent accuracy certificate;
the Butte forward extension contains substantial drift. Full-game accuracy,
physical metric accuracy and processing time remain unverified. All four reserved
evaluation windows remain closed; their exact plan is `data/development-plan.json`.

Next: assess the latest opposite-half fits, then extend source-bound mapping
through longer pans, zooms and reference changes with independent paint checks,
bounded corrections and resumable processing. Keep geometry/support warnings and
coverage visible. Tracking, identity and analytics remain outside current scope.

## Repository migration

The active editor and required calibration tools/tests were selected into fresh
Git history. No old docs, branches, model weights or footage were imported into
Git. Runtime data is ignored. Source provenance is in `migration.json`; private
data provenance is in `data/migration-receipt.json` and `data/artifacts.json`.
The old repositories and worktrees remain untouched for recovery.

Migration validation: 184 Python tests and nine frontend tests passed, along
with TypeScript, lint, the production build, Python syntax and whitespace checks.
All 14 command-line entry points passed their import/help smoke check. Both
migrated saved states equal their originals, all 14 private review files passed
SHA-256 comparison, and the recovered mapping evaluator is byte-identical to its
checkpoint. Desktop Chromium checks at 1440 × 1000 exercised both clips' playback,
seeking, drag, Undo/Redo, original comparison, save/reload, video byte ranges,
stale writes (409) and cross-site writes (403), using isolated copies. A missing
browser icon found during QA was fixed. Final production verification on port 3100 passed with no browser errors
and the original saved revisions intact. QA receipts and screenshots are outside Git at
`/tmp/aisoccercoach-v7-migration-qa/`. This is migration/UI evidence, not a new
calibration accuracy result; Microsoft Edge itself was not tested in this run.
