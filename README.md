# AI Soccer Coach — field alignment

This is the active repository for the v7-based assisted field-alignment app.
Open this folder for new work. Read [CURRENT-WORK.md](CURRENT-WORK.md) first.

The app lets you review the field overlay, drag a visible marking, Undo/Redo,
compare the original, seek the clip and save an adjustment. The saved mapping
also supplies pixel-to-field lookup. It runs locally at
**http://127.0.0.1:3100/alignment**.

Calibration development now includes resumable propagation and a bounded
re-anchoring CLI. Four 60-second extensions have been processed, with raw resume
equivalence verified; none is independently qualified yet. These extensions use
a later single-reference polynomial model, while the original v7 setup uses
three local charts and boundary corrections. The original v7 evaluator reproduces
exactly after migration; the extension runner now accepts its three-chart maps
and rejects missing charts or boundary fits and boundary time extrapolation.
The fresh v7 control and four-window qualification are still in progress.
See [the setup comparison and next work](CURRENT-WORK.md) before continuing an
experiment or planning app integration.

## Run

Requirements: Node.js 22+, pnpm 10.28.2, Python 3.11 or 3.12, and local
`ffmpeg`/`ffprobe`. Python uses OpenCV, NumPy and SciPy; no GPU is required.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
pnpm install --frozen-lockfile
pnpm dev
```

Dependencies must be installed once. The running app uses local files, including
fonts bundled by Next.js at build time; it has no cloud API or account requirement.
A fresh build fetches the selected Google Fonts when they are not cached.

## What lives here

- `web/`: the one Next.js alignment app and its local API.
- `calibration/`: source preparation, browser annotation, paint measurement,
  fitting, refinement, propagation, independent checks and immutable adjustments.
- `data/`: ignored private recordings, review state and experiment evidence.
- `out/`: ignored new experiment outputs.
- `migration.json`: source provenance for the selected imported code.

The migrated workstation has two prepared review clips under `data/review/`:
`granite-control` and `butte-h2-slot2`. Each retains its source video, packet,
manifest, base mapping, active revision and saved revision history. A fresh Git
clone intentionally has no footage: copy those private directories from the
owner's workstation before reviewing. Without them the app shows unprepared clips.
`ALIGNMENT_DATA_ROOT` can select a different absolute review directory when
starting the app, including an isolated copy for write tests.

The verified source catalog and frozen development plan are under `data/`.
Their hashes and original source locations must be preserved. The editor is
independent of the old repository. Rerunning previous calibration experiments
still needs their original private frames and input artifacts; their locations
are recorded in `data/artifacts.json` and the saved declarations.

## Calibration contract

The owner-confirmed Roseville template is fixed at **120 × 70 yd**
(**109.728 × 64.008 m**). That size is an owner instruction, not a new survey;
standard-yard interior markings remain assumptions. Internal field coordinates
use metres with the origin at centre, x along length and y along width, as
encoded by the named markings. Image coordinates use native pixel centres;
frame identity uses the exact source PTS, time base, image hash and source hash.
Never infer a game's identity from its filename or alias.

Saved maps remain approximate and `metric_certified: false`. A visual
confirmation cannot certify geometry. Observed-support, warnings, geometry,
source binding and stale-revision checks remain enforced. Previous paint scores
apply to their parent map; an owner's adjustment requires new independent checks.
Existing artifact schema/version names are intentionally retained.

For agent-led source-frame markup, use the maintained
[assisted-field-recovery skill](calibration/skills/assisted-field-recovery/SKILL.md).
It covers browser pointer annotations, native-paint measurement, semantic review,
the original v7 chart/boundary fitting recipe and independent checks. This
repository copy is authoritative and can be read directly by the next agent;
the current workstation's installed `soccer-field-alignment` skill routes to it.
Personal skill registrations are local configuration, not part of a fresh clone.
A separate model invocation is not required for the active agent to mark frames;
automatic app invocation remains unbuilt.

Run CLI tools from this directory, for example:

```bash
.venv/bin/python -m calibration.tools.prepare_field_recovery --help
.venv/bin/python -m calibration.tools.field_annotation_server --help
.venv/bin/python -m calibration.tools.fit_field_recovery --help
.venv/bin/python -m calibration.tools.fit_field_propagation --help
.venv/bin/python -m calibration.tools.propagate_field_recovery --help
.venv/bin/python -m calibration.tools.reanchor_field_recovery --help
.venv/bin/python -m calibration.tools.score_field_recovery --help
```

`fit_field_propagation` joins observed end/midfield references using independent
image connections and a shared bounded residual. `propagate_field_recovery`
extends an existing fit through time using adjacent motion and an optional paint lock;
its paint gauge participates in fitting and is not independent accuracy evidence.
Both are development tools; automatic full-game execution from the app is unbuilt.

Propagation saves each completed frame's mapping and diagnostics atomically.
Use `--max-frames N` to pause after N additional frames per direction, then repeat
the same command with `--resume` to continue (omit `--max-frames` to finish).
Resume requires the same atlas, source manifests, frozen plan/catalog, settings
and implementation. Completed checkpoints are immutable, changed inputs are
rejected, and a local file lock prevents concurrent writers to the same output.
The default catalog is `data/game-sources.json`; select another with `--catalog`.
Reserved-window guards also apply when using `--reuse-frames`.

`report.json` distinguishes paused from completed processing. On completion,
`atlas-forward.json` and/or `atlas-backward.json` contain the extension maps with
exact source PTS and frame hashes, the parent's geometry/support and all inherited
and propagation warnings. These are diagnostic maps: warnings still block public
lookup, `metric_certified` stays false, and independent paint checks remain
necessary. Missing diagnostic samples cannot count as whole-window survival.
Interrupted frame decoding cannot resume; keep the partial output and start a
fresh output directory. Processing resumes once a complete frame manifest exists.

`reanchor_field_recovery` corrects slow image-registration drift around a fixed
parent map, including the original three-chart v7 model. It selects the measured
midfield reference when available and composes its saved reference transform;
that transform need not be identity. All chart/blend/boundary parameters persist.
For v7, missing charts or boundary fits prevent supported status. Existing far
and near temporal models cannot support frames outside their fitted time domains;
carrying an endpoint offset does not establish new boundary evidence. This CLI
does not refit a temporal boundary for an extension.
It requires `--atlas`, `--frames` (parent manifest), `--propagation`
(completed raw run without paint lock), `--extension-frames`, `--split`, `--plan`,
`--direction` and a fresh `--out` directory. Use `--turf-profile green_v1` for
Granite or `warm_green_v1` for Butte, and `--render` for every-frame raw/corrected
comparison images. Optional `--references` supplies reviewed fitting paint;
this CLI accepts no independent check-paint input.

Registration uses eligible fitting views at one-second intervals, 1920-pixel
width and the existing 0.75-second drift smoother. Nearest-reference selection
uses elapsed source time in both directions. A new reference must connect
directly or through one checked image bridge and pass its required named-paint
checks (median ≤3 px, p95 ≤6 px). Corrections exceeding 12 native pixels over the
visible observed-support grid, missing temporal brackets longer than 1.1 seconds,
or invalid geometry leave the raw map in place and mark the frame unsupported.
The original boundary map, polynomial model, dimensions, support and warnings
remain attached. Output includes `atlas.json`, `reference-connections.json`,
per-frame bounds/geometry/support in `report.json`, and frozen input and
implementation hashes with a source snapshot in `declaration.json`.

The split and optional references use checksum envelopes with `schema`, `payload`
and `payload_sha256` (the existing `field_recovery.digest`):

- `field-recovery-reanchor-split-v1`: source/manifest hashes and every exact frame
  identity with `role: fit|check`. Every tenth frame and the last are fit-eligible;
  `inherited_splits` entries (`path`, `sha256`) preserve earlier roles by exact PTS.
- `field-recovery-reanchor-references-v1`: source/manifest/split hashes and
  `references`, each with the exact fitting frame identity,
  `review_status: source_reviewed`, `required_labels`, and named `features`
  (`feature_id`, `label`, `points_native`). Missing or failed paint rejects an anchor.
- `field-recovery-check-plan-v1`: source/manifest hashes, disjoint complete
  `fit_frame_indices`/`check_frame_indices`, and required `groups`, each with
  `frame_index`, exact frame identity, `label`, `feature_ids` and
  `observation_status`. Exact identity includes source PTS/time base, image hash
  and native size. Missing/ambiguous portions remain in the required groups.

Pass this last artifact to `score_field_recovery --check-plan`. Required missing
groups, unreviewed measurements and absent maps fail; registration fitting views
cannot become independent checks. Without that option, legacy scoring is retained.
The scorer's `qualification_receipt(...)` helper combines the score and re-anchor
report with optional baseline, visual-review and resume-equivalence files. It
requires complete paint evidence, supported geometry, transition-neighbor checks,
full-sequence review, equivalent maps and unchanged parents before setting
`pixel_qualified`. Missing evidence keeps that flag false. Physical metric
certification and full-game acceptance always remain false.

## Verify

```bash
pnpm test
pnpm typecheck
pnpm lint
pnpm build
pnpm test:python
.venv/bin/python -m compileall -q calibration
git diff --check
```

Tests establish implementation behavior, not full-match calibration accuracy.
Keep browser screenshots and real-footage outputs out of Git.

## Direct dependencies

No model weights, tracking stack or hosted services were migrated. Existing
versions were retained for the numerical stack and Next.js/React. The JavaScript
lockfile fixes transitive resolutions for this new repository.

| Component | License |
| --- | --- |
| Next.js, React, React DOM, ESLint, Vitest, pytest | MIT |
| TypeScript | Apache-2.0 |
| NumPy, SciPy | BSD-3-Clause |
| OpenCV 4.10 | Apache-2.0 (wheel packaging and bundled notices also apply) |
| Barlow, Barlow Condensed, IBM Plex Mono fonts | SIL Open Font License 1.1 |

FFmpeg/ffprobe are local executables. Their distribution license depends on build
options; this repository does not bundle them. Only permissive components may
ship in the paid service. No PyAV, AGPL components or unlicensed weights are included.
