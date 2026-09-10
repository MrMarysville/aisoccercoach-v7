# Working in this repository

Read `CURRENT-WORK.md` and `README.md` before choosing work. This repository is
only the v7-based assisted field-alignment workflow. The old repositories and
worktrees are preserved history, not additional active apps.

- For frame markup, fitting or recovery, follow
  [assisted-field-recovery](calibration/skills/assisted-field-recovery/SKILL.md).
  Preserve the original v7 local-chart and boundary-correction workflow; a
  single-reference polynomial candidate is a separate experiment. The repository
  skill is authoritative even when an agent has older installed instructions.
- Keep the owner-confirmed field fixed at 120 × 70 yd (109.728 × 64.008 m).
  Preserve its provenance and distinguish assumed markings from measurements.
- No real footage, frames, screenshots or identifiable children in Git. Use
  ignored `data/`, `out/` or an external scratch directory.
- No billable compute, deployment or remote mutation without explicit owner
  authorization in the current request. Local CPU/GPU work is allowed.
- Only permissive components may ship; no AGPL, noncommercial or unlicensed
  weights/datasets. Record dependency decisions and licenses in `README.md`.
- Preserve source hashes, exact PTS/time bases, frozen fit/check roles, immutable
  maps, observed-support gates, geometry warnings and stale-write protection.
  Visual confirmation never sets metric certification or full-game acceptance.
- Keep four reserved evaluation windows closed until development qualification
  is satisfied. Do not silently relabel check data or tune on reserved footage.
- Update `CURRENT-WORK.md` with dated measured results, limitations and next work
  at each milestone. Do not copy historical status documents into this project.
- Use pnpm for JavaScript and `.venv/bin/python` for Python. Run relevant checks
  from `README.md`; inspect failures instead of weakening tests or gates.
- For Next.js route, rendering or config changes, read the relevant guide under
  `web/node_modules/next/dist/docs/` before editing. Its conventions may differ
  from earlier releases.
- Inspect Git status first and preserve other people's work. Use one checkout
  and main branch for routine work. If workers are explicitly requested, give
  each an isolated worktree, exact paths/interpreter, numerical acceptance tests,
  runtime limits, and instructions not to commit. Review and rerun their checks.
- Never reset/delete old repositories, worktrees or private artifacts as cleanup
  without authorization for the exact targets. Never expose secrets.

The current task is calibration. Tracking, identity, ball/event analysis, reports,
billing and broader product infrastructure are deferred.
