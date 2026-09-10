# Current work

Updated 2026-09-10. This is the canonical repository for future work.

The owner wants full-game calibration: a source-bound pixel-to-field mapping
maintained through both halves, with simple visual corrections, independently
checked accuracy and honest coverage. The Roseville 120 × 70 yd template stays fixed.
The intended finished flow starts with a video upload, invokes Codex to select,
label and review source field markings, then runs fitting, propagation and
independent checks and presents uncertain sections for review. The owner wants
Codex Astra for labeling; record the actual runtime model when that integration
runs. Agent-performed development markup is not upload-triggered automation.

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

An earlier 60-second propagation experiment used a paint lock; its sampler gauge
was fitting evidence. The current experiment uses raw adjacent propagation plus
separate, bounded reference correction and frozen independent paint checks.
All four minute-long development sequences have been processed, but none has
qualified. Full-game accuracy, physical metric accuracy and production runtime
remain unverified. All four reserved evaluation windows remain closed; their
exact plan is `data/development-plan.json`.

Next: finish the fresh successful-v7-method development baseline;
the newer single-reference extension experiment is a different model, not a
qualification of the original three-chart v7 setup. See the setup audit below
before continuing the saved re-anchoring experiment.
App orchestration and automatic initialization follow calibration qualification.
Tracking, identity and analytics remain outside current scope.

Handoff order: read `AGENTS.md`, this status, `README.md`, then
`calibration/skills/assisted-field-recovery/SKILL.md`. The repository skill carries
the complete frame-markup/v7 recipe without requiring the workstation's personal
skill registrations. The extension runner now accepts three-chart maps, but this
has not qualified the four minute-long sequences. Older results below retain their
dated implementation and model scope; they are not the latest v7 acceptance state.

## Calibration code verification — 2026-09-10

Latest follow-up: a package preflight now reads installed distribution names and
versions from `requirements.txt` and rejects changed pins or conflicting OpenCV
wheels before experiment output creation. The regression check passes. Both
private setup scripts now use it; their prior versions and the interrupted setup
remain preserved. The earlier error was querying `opencv-python` metadata when
the installed wheel was `opencv-python-headless`, not a failed OpenCV import.
OpenCV 5's published features were reviewed; no upgrade was made and no numerical
improvement is claimed. See the dependency decision in `README.md`.

Since the code milestone below, Granite's right-end and six temporal far-boundary
views have completed actual browser-pointer markup and native-paint semantic
review. A fresh three-chart fit processed 1,611 frames in **488.221 seconds**, with
265 fitting frames and 1,346 check frames retained by exact source identity.
The initial map has warnings on all frames, including a midfield training-feature
p95 of **7.837 px**, and only **16/1,611** sampled geometry checks pass. Its temporal
far-boundary fit is absent. The prescribed boundary refinement is running before
independent scoring; these training diagnostics are not a qualification result.
Inputs, complete helper freeze, measurements and outputs are private under
`out/v7-control-20260910/granite-control/`. Butte's missing right view was found at
1470 seconds in a guarded fitting-only acquisition; source markup/review is underway.
All four reserved windows remain closed. The owner also requested a fresh
independent audit of implementation, evidence and possible better approaches.

Finished the code-review/test milestone before resuming footage experiments.
Two owner-requested parallel agents reviewed isolated worktrees; the primary
agent reviewed and integrated their fixes and ran all checks in the active checkout.
Both fresh fitters now restrict image anchors to declared fitting frames and rank
reference distance by rational source time. Boundary paint is validated against
the fit split before fitting. Far/near refinements and the shared-reference fit
preserve frozen parent roles; far refinement also rejects reindexed manifests.
Exact fitting evidence survives revisions, and frozen-plan scoring checks nested
parent provenance so inherited fitting frames cannot become independent checks.

All **200 Python tests passed in 124.53 seconds**, including forward/backward
resume equivalence, non-identity v7 gauge composition at 1e-12 tolerance,
mixed-cadence reference selection, fitting-evidence inheritance and rejected role
changes. All **9 web tests**, typecheck, lint, production build, Python compilation,
CLI help and whitespace checks passed. The maintained skill validator passed
using the already-installed system PyYAML with the project interpreter; no runtime
dependency was added. Logs: `out/code-verification-20260910/`.
The 18 frozen plan/catalog, parent atlas/manifest and editor artifacts in the
preservation check still match their recorded hashes. No footage was decoded or
new source markings approved during this code-verification milestone.

These checks establish code behavior, not a successful fresh footage fit.
The Granite three-chart control has prepared source-bound inputs and draft
right-end markup; that draft still needs paint-semantic corrections/review and
fresh boundary evidence before fitting. Butte's missing right-end reference is
still unresolved. The four development extensions remain unqualified and all
four reserved evaluation windows stay closed. Finish the fresh v7 control and
the unchanged-window qualification, then wire the upload-triggered Codex flow.
Automatic labeling, automatic initialization and full-game app orchestration
remain unbuilt; metric certification and full-game acceptance remain false.

## Three-chart extension compatibility — 2026-09-10

The re-anchoring runner now accepts the fixed v7 multi-chart atlas. It uses the
measured midfield view when available, preserving its possibly non-identity
reference-to-native transform and every chart, blend, support and boundary
parameter. The existing 0.75-second smoother and 12-native-pixel correction cap
remain in effect; parent joins and artifacts stay immutable.

Support now explicitly fails for missing v7 charts/boundary fits or frames outside
the far/near temporal model's fitted time domain. Carrying a boundary endpoint
offset into an extension is still diagnostic only. This change does **not** fit
new extension boundary paint or establish a fresh v7 accuracy result.

All 11 focused re-anchoring checks passed in 45.55 seconds, including forward and
backward three-chart resume equivalence, a non-identity gauge, unchanged fixed
models/joins and rejection outside boundary time support. All 198 Python regression
tests passed in 109.08 seconds; CLI help, compilation, skill validation and
whitespace checks also passed. Logs: `out/v7-control-20260910/regression-three-chart.log`.

Source-only acquisition is underway because the existing opposite-half windows
mostly show the left end and midfield. Additional fitting-only views were decoded
in guarded intervals, without replacing any qualification window: Granite
2390.004–2570.004 s and Butte 1070–1400 s, at one-second fitting cadence. Their
source files were rehashed, PTS/time bases retained, and all four reserved windows
remain closed. Granite's new views show a usable right end around 2500 s;
Butte's right-end acquisition is not yet complete. No acquired view is independent
check evidence. Private declarations, exact frames and source surveys are under
`out/v7-control-20260910/`.

The saved single-reference polynomial comparison is also being rerun with the
source-time-aware runner, separately from v7. These jobs have no acceptance claim;
the old scores below are retained until the complete updated comparison and
receipts exist. No owner revision or editor baseline has changed.

Remaining: complete the fresh source-reviewed v7 chart/boundary fit; evaluate the
same four windows; finish transition-neighbor evidence, full-sequence comparison
review and hash-bound qualification receipts. No window is qualified yet.

The owner now permits a better alternative when supported by comparative
evidence. Keep v7 as the benchmark; a method change must improve the same frozen
accuracy/coverage checks, with explicit model scope and unchanged thresholds.
The owner also authorized ongoing GitHub milestone updates for an external audit.

## Original v7 setup comparison — 2026-09-10

The frame-markup skill has also been recovered from the historical propagation
checkout and updated at `calibration/skills/assisted-field-recovery/SKILL.md`.
It now uses the active checkout/interpreter, actual browser pointer annotations,
native-paint refinement and semantic review, then the original v7 local-chart,
slow-drift and spatial/temporal boundary workflow. It explicitly distinguishes
the single-reference polynomial experiment and preserves the frozen evidence
requirements. On this workstation, the installed `assisted-field-recovery` entry links to this
maintained project copy; `soccer-field-alignment` routes to it, and the generic
OpenCV skill no longer overrides an owner-selected nonlinear model with a
mandatory single homography. Historical skill files remain unchanged.

Skill validation and all nine documented CLI help checks passed. The 39 focused
annotation, paint/turf, assembly and v7 projection checks passed in 8.75 seconds.
A desktop Chromium smoke check at 1440 × 1000 exercised actual pointer markup,
Undo, two successive saves and reopening on a copied eligible Granite fit frame
(source PTS 207000360 at 1/90000, 2300.004 s). Three coarse native points produced
102/102 sampler-selected near-touchline points; source and measured imagery were
reviewed, and assembly retained the reviewed feature with exact source binding.
The browser's measured-overlay view also loaded with its image hash and saved
revision preserved; both browser passes had no console/page errors.
This is a markup/measurement integration check on one marking, not a new complete
v7 fit or an independent accuracy result. Private browser scripts, screenshots,
annotations, measurements and receipts are under
`/tmp/v7-frame-markup-20260910-h6o5zrag/`. Browser plugin was unavailable; the
existing local Playwright/Chromium installation was used. No dependencies,
calibration algorithms, owner maps or development/check assignments were changed.
The temporary annotation/review server was stopped after verification.

The owner reaffirmed that development should use the successful v7 setup from
`aisoccercoach-repo/v2`. Read-only comparison found an experiment-path mismatch,
not a corrupted migration. The original `rough-motion-v7` combines separately
fitted left/midfield/right charts, measured image motion with 0.75-second drift
smoothing, a spatial far-boundary correction and a fitted temporal far-boundary
correction. Those temporal parameters belong only to its 620.033333–655.033333 s
Granite development episode; do not extrapolate them onto other footage.

Using the current converter/evaluator on the frozen original artifacts reproduced
**351,000 projections (351 frames × 1,000 field points) with exactly 0.0 native px
difference** from the original v7 evaluator. All 14 migrated private review files
also remain byte-identical to their original files and migration hashes. The
Granite editor's embedded parent retains `v7_fixed_blend_v1`, three charts, and
its later reviewed boundary refinements. The editor baseline is preserved.

The minute-extension parents instead use `polynomial_reference_residual_v1`:
one chart with a shared degree-3 polynomial for Granite and degree-2 for Butte,
with no separate spatial or temporal far-boundary model. They come from later
opposite-half development work, not the original preview. In addition,
the audited `reanchor_field_recovery.run` required exactly one chart, so that
version could not extend the original three-chart v7 setup. Its 12 px bounded correction
experiment must not be described as the same configuration or as proof that the
original v7 method fails. The generic v7 fitter and boundary-refinement helpers
are already present; no replacement solver is needed merely to recover them.

Next establish a control using the v7 fitting workflow on the unchanged allowed
development footage, with fresh source-reviewed references and boundary evidence
where visible. Missing required views remain unsupported; never manufacture a
third chart or copy the old episode's fitted time spline. Review the extension
runner's single-chart/gauge assumptions before adapting that path. Retain the
single-reference candidates and their failures as a separate comparison, with
the frozen fit/check roles, missing evidence and reserved-window restrictions.

This audit changes no mapping or owner revision and makes no new paint-accuracy
claim. The original preview remains a 35.1-second development result; it does not
establish accuracy on the four minute extensions. Reproducible local check:
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B out/v7-setup-audit-20260910/check_v7_parity.py`.
Measured receipt: `out/v7-setup-audit-20260910/comparison-526jgnz7/report.json`.
All audit inputs/outputs remain private and ignored; historical files were read
without executing their fitting/rendering scripts or modifying their artifacts.

## Independently checked re-anchoring — 2026-09-10 progress checkpoint

Implemented `reanchor_field_recovery` around the existing registration and
0.75-second drift helpers. It accepts fitting references but no independent check
paint. Parent polynomial geometry, observed support and boundary mapping remain
fixed; 12-native-pixel correction limits, temporal-support gaps and per-frame
geometry failures remain explicit. Inputs, source identities, fitting roles and
implementation snapshots are frozen in private artifacts. The scorer's optional
frozen check plan keeps missing/ambiguous groups in the denominator. A qualification
receipt helper requires paint, geometry, transition neighbors, visual review and
resume equivalence; physical certification and full-game acceptance stay false.

All four initial 10-second extensions were independently scored before correction.
These results use their frozen initial evidence and the saved initial runner:

| Sequence | Raw passing groups | Corrected passing groups | Supported frames |
| --- | ---: | ---: | ---: |
| Granite forward | 35/110 | 66/110 | 100/100 |
| Granite backward | 33/99 | 42/99 | 66/100 |
| Butte forward | 28/110 | 29/110 | 100/100 |
| Butte backward | 16/76 | 16/76 | 36/100 |

Expanded the same exact-source windows to 60 seconds in both directions for both
clips: 2,400 sampled frames at 10 fps. Existing fit/check roles and initial check
measurements were carried forward by exact PTS. Source-only paint proposals cover
five-second checks plus drift/reference candidates; rejected samples, uncertain
labels and unmeasured distant boundaries remain recorded. No reserved footage
was accessed. Pausing after seven frames and resuming produced exactly equal raw
maps and diagnostics to uninterrupted processing in all four sequences. Paused
checkpoint bytes/mtimes and parent atlas/manifest hashes remained unchanged.

The saved `reanchor60-*-v2` development candidates produced these results:

| Sequence | Raw passing groups | Corrected passing groups | Supported frames | Incomplete evidence groups |
| --- | ---: | ---: | ---: | ---: |
| Granite forward | 76/294 | 112/294 | 157/600 | 154 |
| Granite backward | 40/270 | 53/270 | 66/600 | 137 |
| Butte forward | 47/291 | 66/291 | 574/600 | 176 |
| Butte backward | 16/242 | 16/242 | 36/600 | 166 |

**These are failed exploratory candidates, not results for the latest runner.**
Three fresh frame-300 references passed their required paint checks. The Butte
backward halfway stroke needed a source-only coordinate correction; its revised
measurements are saved but have not been used in a new run. An earlier narrow
paint-search attempt was retained; its expensive preview rendering was stopped.
The revised search uses the existing measurement CLI widths and records source
occlusion rejections. Independent check measurements were not used to fit maps.

Review also found that dense connection IDs ranked backward references by list
position instead of elapsed source time. The current runner now uses source-time
offsets for that ranking. Its focused synthetic checks pass, but the four real
sequences still need rerunning with this fix. Actual transition-neighbor checks,
complete comparison review, corrected-map resume equivalence and final combined
qualification receipts are outstanding. No candidate has been accepted.

Measured v2 correction runtime was 87.6–97.5 seconds per 600-frame sequence,
excluding preview rendering. Raw pause+resume took 14.6–15.2 minutes per clip
(both directions, including decoding); uninterrupted repeats took 29.3–32.2
minutes while competing with other local work. These are measured experiment
times, not a production throughput benchmark. Evidence, source reviews, hashes
and runnable experiment scripts are ignored under `out/reanchor-20260910/`;
`protocol.json` binds windows and preserved editor artifacts, and the two
`*-raw60-equivalence.json` files record the completed raw comparisons.

Future work, in order:

1. Establish the original v7 method as the explicit control, following the setup
   audit above. Resolve the extension runner's one-chart assumption before
   treating an extension experiment as a test of that method.
2. For the retained single-reference comparison, run the current source-time-aware
   correction on all four unchanged sequences using the reviewed references,
   and compare resumed/uninterrupted corrected maps.
3. Freeze and measure the actual transition neighbors, resolving source-only paint
   coverage where possible without removing failed groups or relabeling checks.
   Render and review complete comparisons; issue hash-bound qualification receipts.
4. If a fixed parent model still fails, document the residual pattern and next
   discriminating experiment. Do not automatically add a distortion model or open
   reserved windows. Keep 120 × 70 yd dimensions and their provenance fixed.
5. After development qualification, integrate assisted reference review and
   resumable processing into the existing app. Automatic initialization remains
   a later calibration task; broader product features stay deferred.

Validation at this checkpoint: all 195 Python regression tests pass on the current
source-time-aware implementation, as do all eight focused re-anchor tests,
three calibration CLI help checks, Python compilation and whitespace checks.
The frozen plan/catalog, both parent atlas/manifest pairs and all 12 editor
artifacts in this experiment's preservation manifest still match their hashes.
No dependencies, licenses, owner revisions or app behavior were changed by
re-anchoring. Pre-push check logs remain in `out/reanchor-20260910/`.

Handoff inputs (private paths below are relative to `out/reanchor-20260910/`):

- Clip IDs are `13232938-h2-slot1` (Granite, `green_v1`) and
  `13217031-h1-slot2` (Butte, `warm_green_v1`). `protocol.json` contains the exact
  parent atlas/manifest paths and guarded extension windows.
- Both `<clip>-raw60-resumed/` and `<clip>-raw60-uninterrupted/` are complete.
  Reuse `<clip>-raw60-resumed/frames-<direction>/frames.json` for either run.
  The expensive raw equivalence experiment does not need repeating.
- `<clip>-evidence60-<direction>/` holds `split.json`, `frozen-check-plan.json`,
  `measurements-reviewed.json` and `score-raw.json`. Preserve all inherited groups
  when adding checks around the newly observed reference transitions.
- Use `<clip>-references60-<direction>-v2.json`, except Butte backward, which uses
  `13217031-h1-slot2-references60-backward-v3.json` (reviewed, not yet run).
  `<clip>-reanchor60-<direction>-v2/` contains the superseded candidate results
  tabulated above; its embedded implementation snapshot predates the time fix.
- Start the current CLI with fresh output directories, such as
  `<clip>-reanchor60-<direction>-source-time-v3/`, and set
  `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`. Keep mapping/scoring runs separate
  from the slow optional `--render` pass. Existing experiment scripts document
  prior invocations; do not overwrite their outputs or treat them as the latest run.
- No experiment jobs remain running at this handoff. App integration and final
  four-sequence qualification are unfinished; the next agent should first read
  the original-v7 setup audit and start at future-work item 1 above. All footage
  and experiment artifacts remain local.

## Resumable propagation — 2026-09-10

Re-scored the latest opposite-half candidates with the current evaluator after
verifying atlas, frame-manifest and check-file hashes against the saved scores.
Results reproduce exactly: Granite's original checks pass 37/38 marking/frame
groups, its documented label correction passes 39/39 (median/p95 0.443/1.565 px),
and Butte passes 41/41 (0.482/1.795 px). Each check set covers six frames. No
points, labels, thresholds or fit/check roles were changed in this run. Fresh
score artifacts are under `out/calibration-20260910/assessment/`.

The propagation CLI now saves immutable, hash-linked per-frame maps and
diagnostics, supports `--max-frames` and `--resume`, and exports a source-bound
extension atlas when processing completes. Resume checks input/configuration,
implementation, environment and completed-frame hashes; an output lock prevents
concurrent writers. Frozen plan/catalog and reserved-window checks apply to
reused frames too. Parent fit/check provenance stays explicitly attached to the
parent manifest. Paused or missing diagnostic samples cannot claim whole-window
survival. Existing warnings and observed-support gates remain active; exports
are approximate diagnostics and do not enable public lookup or certify accuracy.

Two new 10-second forward extensions were decoded locally after the opposite-half
clips' last sampled frames. Each contains 100 frames at 10 fps. Pausing after seven frames
and resuming produced exactly the same frame maps and diagnostics as an
uninterrupted run. The first checkpoint stayed byte-identical with an unchanged
modification time. Both extensions had zero failed adjacent-motion steps, but
direct-to-anchor registration disagreement exceeded the existing 6 px p95 target:
first at +4.967 s for Granite and +6.967 s for Butte (peaks 6.994 and 6.486 px).
This diagnoses accumulation drift; it is not an independent paint-accuracy
measurement. Sparse overlays were inspected. No paint lock or new paint fitting
was used for these extensions; the four reserved windows and owner editor
baselines remain unchanged.

Validation: 187 Python tests and nine frontend tests passed, including synthetic
pause/crash/resume checks in both directions, paint-lock state, failed motion,
missing diagnostics, changed evidence/checkpoints, reserved-window rejection and
concurrent-writer rejection. Python syntax, CLI help and whitespace checks passed.
Final real-frame verification is recorded in
`out/calibration-20260910/final/resume-verification.json`.
The exact tested runner is retained beside that receipt; the subsequent
paint-lock warning-inheritance fix passed the focused synthetic checks.
Full-game runtime/accuracy remain unverified. An interrupted decode still needs
a fresh output directory; resumable processing begins with a complete frame
manifest. No dependency or license changes were needed.

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
