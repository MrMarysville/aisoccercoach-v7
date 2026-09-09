# AI Soccer Coach — field alignment

This is the active repository for the v7-based assisted field-alignment app.
Open this folder for new work. Read [CURRENT-WORK.md](CURRENT-WORK.md) first.

The app lets you review the field overlay, drag a visible marking, Undo/Redo,
compare the original, seek the clip and save an adjustment. The saved mapping
also supplies pixel-to-field lookup. It runs locally at
**http://127.0.0.1:3100/alignment**.

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

Run CLI tools from this directory, for example:

```bash
.venv/bin/python -m calibration.tools.prepare_field_recovery --help
.venv/bin/python -m calibration.tools.field_annotation_server --help
.venv/bin/python -m calibration.tools.fit_field_recovery --help
.venv/bin/python -m calibration.tools.fit_field_propagation --help
.venv/bin/python -m calibration.tools.propagate_field_recovery --help
.venv/bin/python -m calibration.tools.score_field_recovery --help
```

`fit_field_propagation` joins observed end/midfield references using independent
image connections and a shared bounded residual. `propagate_field_recovery`
extends an existing fit through time using adjacent motion and a paint lock;
its paint gauge participates in fitting and is not independent accuracy evidence.
Both are development tools; automatic full-game execution from the app is unbuilt.

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
