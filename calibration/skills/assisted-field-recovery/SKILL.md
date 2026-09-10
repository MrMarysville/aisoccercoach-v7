---
name: assisted-field-recovery
description: Mark up soccer source frames through the browser, refine and review native paint, and fit the original v7 local-chart and boundary-correction workflow in AISoccerCoach. Use for assisted field alignment and recovery; excludes tracking, analytics and automatic app invocation.
---

# Assisted field recovery

Use `/root/aisoccercoach-v7`, its `.venv/bin/python`, and the existing local
helpers below. Read `AGENTS.md`, `CURRENT-WORK.md` and `README.md` first. This
recipe carries forward the browser-markup workflow from the historical
`aisoccercoach-propagation/v2/skills/assisted-field-recovery` skill. Old checkouts
and artifacts remain historical inputs; do not run their fixed-output fitting
scripts or ask the owner to do the agent's markup work.

The owner-selected baseline is **rough-motion-v7**: independently fitted
left/midfield/right charts with fixed field-coordinate blending, measured
adjacent motion with 0.75-second slow-drift correction, spatial far-boundary
fitting, and a support/MAD-weighted temporal far-boundary spline (lambda 0.5).
Use `v7_fixed_blend_v1` and `field_recovery.BLEND`; preserve the original smooth
far-edge taper and protected interior for the original-v7 control. The agent
selects and marks actual source paint, local code refines it, and the agent
reviews the measurements before fitting. Keep independent check paint separate.

Inspect each candidate's actual mode, charts, boundary models and time support.
`fit_field_propagation` produces the different `polynomial_reference_residual_v1`
model. The current `reanchor_field_recovery` runner accepts only one chart, and
the raw propagator carries an endpoint boundary offset rather than fitting v7's
temporal boundary. These are separate experiments, not the default v7 path.
Do not silently switch to them or remove a one-chart guard without fixing its
gauge and transform assumptions. Preserve explicit alternative candidates for
comparison; their failures do not establish that original v7 fails.

## Inputs and source selection

Use the source identity, fixed owner dimensions and reserved split from the
current task. The present recovery holds the outer field at 109.728×64.008m;
standard-yard interior dimensions remain assumptions. Do not open reserved
footage, use old fitted coordinates to initialize a fresh fit, or replace a
missing reference with a copy of another chart.

Run from the active checkout root with its `.venv/bin/python`. Existing
`calibration/alignment_trial.py` supplies source hashing, exact-PTS decoding and closed
window guards. `calibration/tools/prepare_field_recovery.py` validates the source catalog and frozen plan,
then decodes an exact declared development clip ID (or a historical control
alias) and freezes fit/check indices before annotation:

```bash
.venv/bin/python -m calibration.tools.prepare_field_recovery --plan CLIPS_JSON --episode 13232938-h2-slot1 --out NEW_EPISODE_DIR
```

Reserved IDs, changed source/catalog bindings and overlapping padded decode
intervals are rejected. Selected frames retain their measured source phase; do
not round a nonzero source phase to a nominal sample-grid timestamp.

The resulting `source/frames.json` binds each native PNG to source SHA, integer
PTS, rational time base, dimensions and image hash. For annotation, create a
subset manifest beside it containing only the designated fit rows and the
parent manifest SHA. Keep source paths relative to that directory. Give the
independent reviewer separate predeclared check frames. Source-only view
selection is allowed; fitted projections must not select check evidence.

## Browser proposal and paint review

The active agent performs source-only marking through the installed browser
capability, or local Playwright when that capability is unavailable. A separate
model invocation is not needed for routine markup. If the owner explicitly
requests a worker/model, follow current delegation rules and record only runtime
model metadata actually exposed. Do not claim a selected model ran without that
evidence. A screenshot or injected JSON is not an actual pointer annotation.

```bash
.venv/bin/python calibration/tools/field_annotation_server.py --frames FIT_FRAMES_JSON --out ANNOTATIONS_DIR --port 8767
```

Use the installed browser capability to choose complementary end/midfield
views, zoom, scroll and click visible named marking portions. Capture multiple
points along curved lines; split portions around players. After selecting or
scrolling to a feature, verify the active feature and frame before clicking.
Save revisions through
the UI. The server retains immutable pointer/action records, exact frame binding,
zoom and pixel-centre conversion. Source nanosecond modification times cross
both state and save-response browser boundaries as decimal strings; saved revisions retain the exact
manifest integer. Source/frame hashes and PTS remain exact. A missing or ambiguous marking stays explicit.

```bash
.venv/bin/python -m calibration.tools.measure_field_annotations --frames FIT_FRAMES_JSON --annotations SAVED_REVISION_JSON --out NEW_PAINT_DIR --chart INDEX:right --chart INDEX:mid --chart INDEX:left
```

The sampler reads only native images and coarse annotation strips. It saves
accepted/rejected cross sections, paint widths and a `review.json` allowlist.
Thin paint and thick nearby paint use the same half-height band-centre
convention. Named near touchlines permit one grass side because the exterior
apron can be brown; other markings require grass on both sides. These are
measurement assumptions, not acceptance evidence. Thin markings use a 6px search
radius to keep nearby box edges and penalty arcs separate; foreground halfway
uses 20px and the thick near touchline 36px. Inspect crossing/adjacent features
explicitly: a broader search can measure a different, brighter marking.

The default `green_v1` turf profile retains OpenCV HSV H>24 and H<100, S>22,
V>35. Visibly brown grass can fail this color assumption despite clear paint.
After source-only color and semantic review, `--turf-profile warm_green_v1`
uses H<100 or H>170 with the same S/V limits. All other ridge, width, crossing,
viewport and one-/two-side conditions stay unchanged. Record the profile and
source evidence explicitly; preserve old measurements. Warm hues can include
skin, so this option does not replace occlusion review or admit people as turf.
Freeze any revised check measurements before fitting; changing measurements
after viewing model residuals is development tuning, not independent evidence.

Restart the same server with `--review-manifest NEW_PAINT_DIR/review.json`.
Inspect source and measured overlays in the browser. A precise detection of the
wrong sports line is a failure. Correct proposals with actual pointer actions,
save a new revision, rerun measurement into a new directory and review again.
Save the reviewed marking IDs and any source-backed occlusion/wrong-line
exclusions. Do not remove a sample because a fitted model dislikes it.

`SOURCE_REVIEW_JSON` is the agent's semantic-review record, separate from the
sampler's image-path `review.json`. It must contain `source_sha256`,
`annotation_sha256`, `measurement_sha256` (hash of `measurements.json`) and
`accepted_feature_ids` using IDs actually present in that measurement file.
Retain review notes and rejected/ambiguous features with their source-based
reasons. Never mark every sampler result reviewed without inspecting it.

Assemble those accepted IDs with the frozen full-frame split; semantic approval
does not promote rejected cross sections or create a fit from a sparse portion.

```bash
.venv/bin/python -m calibration.tools.assemble_field_recovery --frames ALL_FRAMES_JSON --paint PAINT_MEASUREMENTS_JSON --review SOURCE_REVIEW_JSON --split SPLIT_JSON --out REVIEWED_MEASUREMENTS_JSON
```

When multiple reviewed frames show the same end, select the primary source view
explicitly with `--reference-frame INDEX`. Other accepted views remain in
`additional_reviewed_reference_observations`; they are not silently fitted or
renamed as missing spatial charts. One freshly fitted chart uses the explicit
`single_reference_homography_v1` diagnostic mode. It retains the actual chart
count and its measured support hull, and checks the full visible geometry.
This is a partial diagnostic for a clip showing one end, not the full v7 setup.
Reacquire missing views within permitted development footage. Never duplicate a
chart or silently replace the requested baseline to fill a gap.

For a boundary-only review, add `--parent-measurements PARENT_INPUT_JSON`; this
preserves the original chart reference values and adds separate boundary rows.

## Fit the declared method

`calibration/field_recovery.py` and `calibration/tools/fit_field_recovery.py` implement fresh local
reference fitting, v3 spatial chart blending, adjacent motion with disjoint-cell
checks, v5 slow reference-drift correction, and v6/v7 protected far-boundary
corrections. They do not import the old experiment's coordinates or coefficients.
The old `calibration/field_atlas.py` evaluates saved maps and
`calibration/tools/package_field_atlas.py` converts old maps; neither fits new footage.

Measurements use schema `field-recovery-measurements-v1`, source SHA,
`references[{frame_index,chart,features[{label,points_native}]}]`, and disjoint
`fit_frame_indices` / `check_frame_indices`. Add exact frame bindings to each
reference. Keep browser/paint review provenance and measured support polygons.
Optional `boundary_observations` contain only declared fit-frame paint. The
temporal correction needs at least five independently measured fit-frame anchors
spanning the interval; three local charts alone do not establish that it ran.
Preserve every-tenth-frame plus last-frame fitting eligibility and inherited
exact-PTS roles. Refit boundary coefficients and time splines from current footage;
never copy a previous clip's fitted spline or extrapolate it into an extension.

Freeze recipe/helper/input hashes and copies of the exact helper files in the
new attempt directory before fitting. Preserve the old files before changing
code for a subsequent attempt.

```bash
.venv/bin/python -m calibration.tools.fit_field_recovery --frames ALL_FRAMES_JSON --measurements REVIEWED_MEASUREMENTS_JSON --out NEW_CANDIDATE_DIR --skip-render
```

Reference fitting optimizes normalized image-to-world parameters and enforces
finite line extents after a supporting-line warm start. Infinite-line fit errors
can hide points outside the painted segment. Fragmented observations of one
circle share a fresh ellipse seed.

One independently fitted reference per spatial chart is used in each candidate.
A later reference requires new source paint and a new immutable fit. Image-only
registration cannot clear a failed motion chain. Preserve failures and warnings
until supported reacquisition; do not reuse an old H and call it refitting.

Image-anchor connections first try direct gauge registrations, then a single
independently measured bridge for disconnected paint references. Rank passing
routes by the weaker edge's observed ground coverage, consistency and residual;
never by fitted paint checks. Preserve failed direct and bridge attempts. An
adjacent-only fallback may transport a diagnostic chart but must not enter the
independent drift observations or become a trusted paint connection. Save the
actual route for every anchor. For drift observations, prefer the nearest independently connected paint view;
retain direct gauge registrations for bridge evidence and prevent routes that
loop through their target. Mere availability of a gauge match must not override
a nearer view with broader observed overlap. Rank reference distance by exact
source time, including backward runs, rather than arbitrary list IDs. Prefer a
base chart/motion run with `--skip-render`,
followed by the boundary-refinement command below for the final overlays; that keeps
the same reviewed boundary fitter in the workflow.

The saved `field-recovery-atlas-v1` map is the rendering contract. Multiple-chart maps
declare `projection_mode: v7_fixed_blend_v1`: diagnostic overlays use the actual
v7 fixed weights. Never multiply those weights by observed hull masks or
renormalize them; that introduced clipping and folds in candidate002. Public
coordinates separately require measured support for every contributing chart,
native visibility and an unflagged frame. Missing charts remain unavailable.
Older maps without the mode keep their original evaluation/render semantics.
Numerical orientation and inverse checks are not physical accuracy certificates.

When only boundary corrections or diagnostic blend semantics change, preserve
the measured image motion and fresh chart fits in a parent candidate. Add newly
reviewed `boundary_observations`, exact frame bindings, and the parent's input
path/hash to a new measurement revision. Keep `references` byte-equivalent as
JSON values; the refinement helper rejects silent changes to them.

```bash
.venv/bin/python -m calibration.tools.refine_field_recovery --parent PARENT_ATLAS_JSON --frames ALL_FRAMES_JSON --measurements NEW_BOUNDARY_REVISION_JSON --out NEW_CANDIDATE_DIR
```

This revision is not a new image-registration or chart-fit result. The helper
combines source-reviewed far-line fragments per fit frame, fits only declared
paint, preserves all image-motion warnings, and recomputes changed-map geometry
and boundary warnings. Temporal anchors must span the desired interval;
outside-anchor frames stay flagged. Freeze this helper with the other files.

The original smoothstep spatial taper can fold a compressed distant strip even
when both endpoint curves are reasonable. After an explicit geometry diagnostic,
`--spatial-profile linear_field_y_v1` selects a versioned spatial-only linear
taper. It preserves the far boundary and protected interior exactly; temporal
tapering remains unchanged. It has derivative kinks at strip endpoints, so retain
that limitation and rerun full visible geometry. A profile switch cannot clear
an image-motion failure or substitute for better source evidence.

## Optional near-boundary correction

Use this only for a source-observed residual pattern, retaining the parent fit.
Acquire and review near-line paint on declared fit frames, including actual
observations spanning the required interval. Never invent an invisible boundary
anchor. A direct vertical image warp can move an already-correct halfway line
off its paint. The tested near correction instead remaps field y before
projection, preserving x and protecting all y≤20.1168m exactly. Halfway and end
lines retain their original local homography line sets. Outer field dimensions
are unchanged; `near_boundary_delta_field_y_m` is a mapping parameter, not a
measured physical displacement.

```bash
.venv/bin/python -m calibration.tools.assemble_field_recovery --frames ALL_FRAMES_JSON --paint PAINT_MEASUREMENTS_JSON --review NEAR_SOURCE_REVIEW_JSON --split SPLIT_JSON --parent-measurements PARENT_INPUT_JSON --boundary-label touch_near --parent-atlas PARENT_ATLAS_JSON --out NEAR_INPUT_JSON
.venv/bin/python -m calibration.tools.refine_near_field_recovery --parent PARENT_ATLAS_JSON --frames ALL_FRAMES_JSON --measurements NEAR_INPUT_JSON --out NEW_CANDIDATE_DIR
```

The scalar fit uses native-y residuals to finite, increasing, positive-orientation
near curves, including correctly oriented predictions just outside image height.
Clipping those predictions to the viewport would discard the very boundary being
recovered. Raw residuals, offscreen predictions, rejected observations, parameter
bounds and remaining errors are saved. A temporal gap is interpolated evidence,
not observed coverage; no temporal extrapolation is allowed. All-frame geometry
is rechecked, and parent warnings, charts, motion and far corrections persist.
The optional remap is a development adjustment and does not establish metric
accuracy. Freeze this helper with the other files before running it.

## Check and iterate

Freeze spatially broad source-only check paint before candidate scoring. Include
the entire visible extent of long lines, particularly lower near touchlines,
and separate interior markings. Compare old v7 and the fresh Granite candidate
on identical evidence. Report native Euclidean marking distance, sample counts,
extent, missing support and every motion warning. Also inspect the signed residual
along the entire visible marking to expose systematic bowing or edge offsets.
Name the coordinate used (for example native-y or local normal); do not equate
a fitting residual with the Euclidean check metric. Review all rendered frames and
native-resolution failure strips, including pan transitions. For extensions,
freeze five-second checks, drift-onset checks and actual reference-change
neighbors before scoring. Use identical evidence for baseline and candidate;
rejected, missing and ambiguous groups remain in the denominator.

```bash
.venv/bin/python -m calibration.tools.score_field_recovery --atlas ATLAS_JSON --frames ALL_FRAMES_JSON --checks FROZEN_CHECK_MEASUREMENTS_JSON --check-plan FROZEN_CHECK_PLAN_JSON --orientation-policy domain_boundary_v2 --out NEW_SCORE_JSON
.venv/bin/python -m calibration.tools.diagnose_field_recovery --atlas ATLAS_JSON --frames ALL_FRAMES_JSON --score NEW_SCORE_JSON --out NEW_SIGNED_DIAGNOSTIC_JSON
```

The diagnostic records nearest model-to-paint native x/y, normal and tangential
residuals, including clamped endpoints and unknown directions. Its unsigned
distances must reproduce the frozen score within 1e-7px; it cannot reclassify
acceptance. Sign follows the named curve's sample direction.

`calibration/field_marking_check.py` keeps `central_v1` as the CLI/API default for old
score reproduction. Explicit `domain_boundary_v2` uses a second-order inward
derivative only when a central probe crosses a known fixed-chart projection
domain edge. It requires two finite in-domain inward probes; existing finite
central determinants, including folds, are unchanged. Actual poles, unknown
domains and insufficient stencils remain unavailable. Scores retain the policy,
helper hash and per-marking stencil counts. The signed diagnostic follows that
frozen policy. Preserve the old score and declare a new check-only revision;
this fixes a checker defect and does not improve or refit the saved map.
For reproducing an existing run, use its frozen orientation policy; the explicit
policy in the command example is not permission to change an old comparison.

Keep the owner's useful visual-quality judgment separate from the newer
per-marking median≤3/p95≤6px experiment target. Old v7's far-edge score is not an
all-marking certificate. A check-informed adjustment is development tuning:
retain the frozen candidate and checks, state the failing region and change one
discriminating hypothesis. Save a new recipe/helper version when code changes.
Do not weaken gates or silently omit failed intervals to obtain a pass.

For exact reproduction or migration, compare the frozen original-v7 mapping
against the current evaluator on identical source frames and field points before
refitting; unchanged parameters should give identical projections. The current
audit and its runnable check are recorded in `CURRENT-WORK.md`. This verifies
implementation compatibility, not new paint accuracy. Inspect the mapping schema
before choosing a loader: original atlas, recovery atlas and immutable adjustment
wrapper are distinct formats. The existing app supports nudges and saved-map
lookup; updating this skill does not implement automatic app-driven invocation.

Apply this workflow within the current authorized development scope. Keep all
four reserved windows closed until the qualification and freeze plan permits
evaluation. Preserve source hashes, exact time bases, observed support, geometry
and motion warnings, immutable maps, owner revisions and stale-write checks.
For extensions, verify unchanged joins and resumed/uninterrupted map equivalence;
review the complete sampled sequence and report unsupported intervals. Keep
`metric_certified` and full-game acceptance false; a useful visual fit is not an
all-marking certificate. Record measured counts, runtime, failures, limitations
and next work in `CURRENT-WORK.md`. Images and artifacts stay in ignored `data/`
or `out/`. This recipe grants no remote-compute, deployment, tracking or analytics
authorization.
