# Current work

Updated 2026-09-10. This is the canonical repository and current handoff.

The owner wants full-game calibration: source-bound pixel-to-field mappings
through both halves, simple visual corrections, independently checked accuracy
and honest coverage. The outer field stays at **120 × 70 yd**
(109.728 × 64.008 m); standard interior markings remain assumptions.

The current phase checks four unchanged 60-second development extensions at
10 fps. **None is qualified.** The polynomial experiment now has all four failed
qualification receipts and complete comparison videos. Fresh three-chart work
has identified different geometry and source-evidence problems; its results must
not be described as the same model as the polynomial experiment.
Physical metric certification and full-game acceptance remain false.
All four reserved evaluation windows in `data/development-plan.json` stay closed.

Read `AGENTS.md`, this file, `README.md`, then
`calibration/skills/assisted-field-recovery/SKILL.md` before continuing.
The owner permits a better alternative when supported by comparative evidence
and has authorized ongoing GitHub milestone updates for external audit.

## Verified code and preserved state

The latest Python verification is **207 tests passed in 103.40 seconds**.
The 63 focused assembly/annotation/preparation/boundary checks passed in
4.83 seconds, with CLI help, Python compilation and whitespace checks also passing.
Logs and real assembly checks: `out/assembly-fix-validation-20260910/`.
The unchanged app previously passed all 9 web tests, typecheck, lint and production
build; those logs remain in `out/code-verification-20260910/`.

The fresh independent audit found a reproducible assembly provenance bug. It is
now repaired: measured references retain exact PTS/time base/image hash/native
size; assembly validates the full split-manifest hash and original paint identity;
legacy subset measurements require their hash-matched annotation manifest.
Verified source-identity remaps are recorded and cannot target check frames.
Preparation preserves its predecode role declaration before binding the completed
manifest, without changing the roles. Real Granite and Butte assemblies pass,
including Butte's 70→1900 remap. The audit independently checked the existing
fresh inputs and found no instance of the bug in those actual references.

The diagnostic tool now preserves empty required groups and explicitly unavailable
maps. Source/hash/count checks and its 1e-7 px agreement gate remain intact.
On the real Granite scores it retains all 294/270 groups, including 56/54 empty
groups, with **exactly 0.0 px** unsigned-distance disagreement across
15,607/14,501 samples. Original failed logs and helper versions are preserved.

Both fitters restrict image anchors to declared fitting frames and rank reference
distance by rational source time. Boundary/refinement paths preserve parent roles;
exact fitting evidence survives parent revisions and cannot become independent
check evidence. Re-anchoring accepts three-chart maps, composes a non-identity
midfield gauge correctly, and retains the 0.75-second smoother, 12-native-pixel
correction cap, immutable joins, boundary constraints and geometry warnings.

Run `.venv/bin/python -m calibration.tools.check_environment` before creating
experiment outputs. It reads pinned distribution names from `requirements.txt`
and rejects changed versions or conflicting OpenCV wheels. OpenCV's distribution
is `opencv-python-headless==4.10.0.84`; its import is `cv2`. OpenCV 5's published
features were reviewed, but no measured benefit was established here and no
upgrade was made. See the dependency decision in `README.md`.

The editor still opens the owner's Granite first-half 351-frame/35.1-second and
Butte second-half 300-frame/30-second baselines, with drag adjustments, Undo/Redo,
original comparison, seeking, saving and saved-map lookup. Owner revisions and
frozen experiment parents remain unchanged. The 18-artifact preservation set
includes plan/catalog, both propagation parents/manifests and 12 editor artifacts.

## Completed polynomial comparison

These are current source-time-aware reruns on the same four windows and frozen
check evidence. They use one shared polynomial reference, not three-chart v7.

| Sequence | Raw passing groups | Corrected passing groups | Supported frames | Incomplete groups |
| --- | ---: | ---: | ---: | ---: |
| Granite forward | 76/294 | 112/294 | 157/600 | 154 |
| Granite backward | 40/270 | 53/270 | 66/600 | 137 |
| Butte forward | 47/291 | 66/291 | 574/600 | 176 |
| Butte backward | 16/242 | 16/242 | 36/600 | 166 |

All four resumed/uninterrupted raw and corrected frame arrays and diagnostics
match exactly; parent artifacts and joins remain unchanged. Every final receipt
fails required paint, full sampled support, transition paint and full-sequence
visual review. Supported geometry and resume/parent-preservation gates pass.
Missing observations and transition neighbors remain in the evidence.

Each source/raw/corrected video contains all 600 samples: 10 fps, 60.000 seconds,
2880×720. Receipts bind source hashes, exact PTS/time bases, atlas/manifest/video
hashes and frozen renderer versions. Generation took 330.007 seconds. The primary
agent inspected the four middle-frame layouts; **full visual review is pending**.

Private deliverables:

- `out/v7-control-20260910/polynomial-deliverables/delivery-summary.json`
- Four `qualification-failed.json`, `source-raw-corrected.mp4` and
  `video-receipt.json` files beneath that directory.
- Exact equivalence checks: `out/v7-control-20260910/polynomial-audit/`.
- Original frozen protocol, inputs, source-only reviews and independent check
  plans: `out/reanchor-20260910/`.

## Fresh three-chart development work

The original saved v7 evaluator was reproduced on **351,000 projections with
exactly 0.0 native-pixel difference**; all 14 migrated private review files also
match their original bytes. The historical editor baseline remains preserved.
The fresh control restores its three-chart, fixed blend, slow image-drift and
spatial/temporal boundary architecture. Its generic fitting objective is **not an
exact replay of the historical fitting recipe**: finite extents, normalization,
fragment weighting and selected near-side evidence differ. Historical visual
success did not certify every marking at the newer median≤3/p95≤6 px targets.

### Granite

The source-bound control has 1,611 frames: 265 fitting and 1,346 check frames,
including additional guarded fitting-only views needed for the right reference.
Actual browser-pointer markup and native-paint semantic review are complete for
the right view and six additional far-boundary views. All three charts were fitted
fresh from source paint; no old H, polynomial or temporal spline was imported.

Base fitting took 488.221 seconds. Boundary refinement took 136.204 seconds and
fits eight temporal anchors over 2240.004–2389.870667 s. It preserves chart and
image-motion values. All 1,611 frames retain the midfield gauge-paint warning;
1,595 fail geometry. The 111 extra acquisition frames lie outside boundary time
support. Both fixed minute windows are inside that time domain.

| Sequence | Fresh three-chart passing groups | Polynomial passing groups | Fresh valid geometry frames |
| --- | ---: | ---: | ---: |
| Granite forward | 119/294 | 112/294 | 0/600 |
| Granite backward | 81/270 | 53/270 | 0/600 |

Scoring uses unchanged source measurements, manifests, plans and `central_v1`.
The same 154/137 incomplete groups remain. Box and halfway checks improve, but
the near touchline passes **0/28 forward and 0/27 backward** groups. Neither
minute qualifies. Detailed evidence:
`out/v7-control-20260910/granite-control/independent-score-v1/summary-v2.json`.

Two complete source/polynomial/fresh-three-chart videos, each retaining all 600
samples and warnings, are under `granite-control/comparison-previews-v1/`.
Generation took 177.046 seconds. Full visual review remains pending.

### Butte

The combined control has 1,901 frames: 555 fitting and 1,346 check frames.
The additional right view at 1470 seconds has 9 source-reviewed features and
146 native paint samples; its observed upper-right support is limited and it has
no near-touchline evidence. Actual pointer/measurement/browser receipts are under
`out/v7-control-20260910/butte-right-labels/`.

The first fresh fit correctly stopped before motion: its midfield circle and
halfway line did not identify eight homography parameters. The failed attempt
and receipt are retained. Eight additional source-reviewed far-line points on
the same exact midfield frame resolve that missing constraint. A short, newly
frozen preflight then fitted all three references in **15.944 seconds**; every
training feature passes median≤3/p95≤6 px, with maximum p95 **2.584 px**.
This is training evidence, not a propagated accuracy result.

Boundary review accepted 44 points on 7 portions. Counts on the six planned fit
frames are **11, 9, 8, 0, 6, 10**. Only two frames meet the fitter's ten-sample
temporal-anchor requirement; at least five anchors are required. A complete
temporal boundary model remains unsupported, so another long run has not started.
No threshold, turf assumption or source measurement was changed to force a pass.
The eight midfield points are counted once when assembling boundary evidence.

Private handoff: `out/v7-control-20260910/butte-control/` contains
`fresh-measurements-v2.json`, `reference-preflight-v2/report.json`, the failed first
attempt, and `boundary-labels/` with source reviews and frozen helper snapshots.

## Audit conclusions and next work

The [fresh independent audit](calibration/AUDIT-2026-09-10.md) has six ranked
findings and three discriminating next steps. Its complete evidence is private
under `out/fresh-calibration-audit-20260910/`.

The full-grid audit reproduced all **301,990** Granite folded samples: **301,947**
involve an active chart's opposite projective branch. **43 positive-branch folds
across 19 frames must still fail**. Of all folds, 191,724 lie inside declared
support hulls. Removing boundary corrections did not fix the sampled folds.
No branch mask, hull enlargement, weighting change or weakened geometry gate has
been applied. Fragment grouping is another measured objective difference: a
bounded training-only refit improved the long-halfway p95 to 6.458 px but still
failed. It is not an accepted candidate.

Next, complete source-only reference/coverage checks before another long fit.
Granite needs a better complementary midfield view and a separately declared,
source-established chart-branch experiment; retain the 43 remaining folds and
every lost region. Butte still needs sufficient observed temporal boundary
anchors. Keep the four windows, thresholds, exact source roles and missing-group
denominators fixed. Do not spend another full boundary run on a known-invalid
reference or conceal missing coverage with extrapolation.

Actual transition-neighbor paint, missing source evidence and full-sequence visual
review remain necessary for any future passing qualification. The failed receipts
do not imply those tasks were completed.

The intended product flow is upload → actual Codex invocation → source selection,
pointer labeling and semantic review → preflight → fitting/propagation/checks →
qualified regions or explicit uncertain sections in the existing editor.
The owner requested Codex Astra for labeling; current internal labeling receipts
distinguish the requested model from unavailable actual-runtime metadata.
Upload-triggered labeling, automatic initialization and resumable full-game app
orchestration remain unbuilt and follow calibration qualification. Browser
development markup is not an automated app invocation. Tracking, identity,
ball/events, analytics and broader product infrastructure remain deferred.
