"""Propagation test: carry one reviewed v7 fit beyond its clip by adjacent motion only.

Development diagnostic, not acceptance. It answers one question: starting from
the first or last frame of a reviewed recovery clip, how many seconds of
adjacent KLT motion can carry the saved mapping before the drawn markings
leave the paint. No new paint reference, boundary fit or chart is added.

Gauge: on every scored frame the projected regulation markings are searched
for a white ridge within `--gauge-radius` native pixels using the recipe's
own paint sampler. The residual is the accepted samples' normal offset. This
is a fit-quality gauge on source images, not an independent frozen check and
not a metric certificate; competing paint can pull the ridge picker.

Independent drift: every `--independent-stride` frames the extension frame is
registered directly to the anchor frame with SIFT and compared with the chained
motion on a native grid. Disagreement is reported, never applied.
"""
import argparse
from copy import deepcopy
import fcntl
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import cv2
import numpy as np
import scipy

from calibration import field_recovery as recovery
from calibration.alignment_trial import (decode_window, guard_interval, load_plan, sha256_file,
                                        source_time, validate_frame, verify_source)
from calibration.field_adjustment import atomic_json
from calibration.field_marking_check import checked_marking_curve
from calibration.tools.measure_field_annotations import paint_layers, refine_polyline

GAUGE_TARGET_MEDIAN = 3.
GAUGE_TARGET_P95 = 6.
MIN_GAUGE_SAMPLES = 40


def validate_manifest(manifest, game, interval, fps):
    """Check reused frames as strictly as newly decoded frames, before image access."""
    guard_interval(game, *interval)
    if (manifest.get("schema") != "alignment-trial-frames-v1"
            or manifest["source"]["sha256"] != game["source_sha256"]
            or manifest["source"]["game_id"] != game["game_id"]
            or manifest["sampling_fps"] != fps
            or manifest["requested_interval_s"] != list(interval)
            or not manifest["frames"]):
        raise ValueError("Frames do not match this source, cadence or window")
    previous, indices = None, set()
    size = manifest["frames"][0]["native_size"]
    for row in manifest["frames"]:
        stamp = validate_frame(row, game, *interval)
        if (type(row["index"]) is not int or row["index"] in indices
                or (previous is not None and stamp <= previous)
                or row["native_size"] != size):
            raise ValueError("Invalid frame order, index or dimensions")
        indices.add(row["index"])
        previous = stamp
    return manifest["frames"]


def read_image(directory, row):
    path = (Path(directory) / row["file"]).resolve()
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != row["image_sha256"]:
        raise ValueError("Decoded still hash mismatch")
    im = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if im is None or [im.shape[1], im.shape[0]] != row["native_size"]:
        raise ValueError("Native still dimensions mismatch")
    return im


def marking_runs(curve, native_size, margin=8., max_step=100.):
    """Split a projected marking into connected in-image polylines."""
    finite = np.isfinite(curve).all(axis=1)
    inside = finite.copy()
    inside[finite] &= ((curve[finite] >= margin) & (curve[finite] < np.asarray(native_size) - margin)).all(axis=1)
    with np.errstate(invalid="ignore", over="ignore"):
        connected = inside[:-1] & inside[1:] & (np.linalg.norm(np.diff(curve, axis=0), axis=1) <= max_step)
    edges = np.diff(np.r_[False, connected, False].astype(np.int8))
    return [curve[start:stop + 1] for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def gauge_frame(image, mapping, radius, policy="central_v1", layers=None):
    """Paint residual of every visible projected marking on one frame."""
    per_marking = {}
    offsets = []
    layers = layers or paint_layers(image)
    for name in recovery.world_markings():
        world = recovery.feature_curve(dict(label=name, points_native=[[0, 0]]), samples=2001)
        curve, _ = checked_marking_curve(mapping, world, policy)
        accepted = []
        for run in marking_runs(curve, mapping.record["native_size"]):
            if len(run) < 2:
                continue
            try:
                picked, samples = refine_polyline(image, run, radius=radius, spacing=8., layers=layers)
            except ValueError:
                continue
            accepted.extend(abs(s["normal_offset_px"]) for s in samples if s["accepted"])
        if accepted:
            per_marking[name] = dict(samples=len(accepted), median_px=float(np.median(accepted)),
                                     p95_px=float(np.percentile(accepted, 95)))
            offsets.extend(accepted)
    offsets = np.asarray(offsets)
    summary = dict(samples=int(len(offsets)),
                   median_px=float(np.median(offsets)) if len(offsets) else None,
                   p95_px=float(np.percentile(offsets, 95)) if len(offsets) else None,
                   measured=bool(len(offsets) >= MIN_GAUGE_SAMPLES))
    return summary, per_marking


def paint_observations(image, mapping, radius, layers, policy="central_v1"):
    """Accepted paint centres next to each drawn marking: (drawn point, paint point, unit normal, marking)."""
    rows = []
    for name in recovery.world_markings():
        world = recovery.feature_curve(dict(label=name, points_native=[[0, 0]]), samples=2001)
        curve, _ = checked_marking_curve(mapping, world, policy)
        for run in marking_runs(curve, mapping.record["native_size"]):
            if len(run) < 2:
                continue
            try:
                _, samples = refine_polyline(image, run, radius=radius, spacing=8., layers=layers)
            except ValueError:
                continue
            for sm in samples:
                if sm["accepted"]:
                    a, q = np.asarray(sm["coarse_xy"]), np.asarray(sm["native_xy"])
                    d = q - a
                    n = d / np.linalg.norm(d) if np.linalg.norm(d) > 1e-9 else None
                    rows.append((a, q, n, sm["normal_offset_px"], name))
    return rows


def solve_correction(observations, native_size, *, damping=.02, huber_px=2., iterations=4):
    """Damped, robust homography update C (native->native) from normal-offset paint residuals.

    Linearised at identity in centred, scaled coordinates. Directions that no
    visible marking constrains stay at the chained motion because of damping.
    """
    w, h = native_size
    scale = max(w, h) / 2.
    centre = np.array([w, h]) / 2.
    a = np.array([o[0] for o in observations])
    q = np.array([o[1] for o in observations])
    labels = [o[4] for o in observations]
    if len(a) < 8:
        return None, dict(reason="too_few_paint_observations", observations=len(a))
    an = (a - centre) / scale
    qn = (q - centre) / scale
    d = qn - an
    norm = np.linalg.norm(d, axis=1)
    keep = norm > 1e-12
    # Offsets of exactly zero carry no direction; treat them as zero residual along the local normal.
    # Use the polyline normal from the sampler instead: d/|d| where available, else skip.
    an, qn, d, norm = an[keep], qn[keep], d[keep], norm[keep]
    labels = [l for l, k in zip(labels, keep) if k]
    n = d / norm[:, None]
    off = norm  # normal offset magnitude in scaled units (signed along n)
    x, y = an[:, 0], an[:, 1]
    ju = np.c_[x, y, np.ones_like(x), np.zeros_like(x), np.zeros_like(x), np.zeros_like(x), -x * x, -x * y]
    jv = np.c_[np.zeros_like(x), np.zeros_like(x), np.zeros_like(x), x, y, np.ones_like(x), -x * y, -y * y]
    rows = n[:, 0:1] * ju + n[:, 1:2] * jv
    counts = {l: labels.count(l) for l in set(labels)}
    base_w = np.array([1. / np.sqrt(counts[l]) for l in labels])
    delta = np.zeros(8)
    huber = huber_px / scale
    for _ in range(iterations):
        r = rows @ delta - off
        wr = base_w * np.where(abs(r) <= huber, 1., huber / np.maximum(abs(r), 1e-12))
        lam = damping * wr.sum()
        A = rows * wr[:, None]
        delta = np.linalg.solve(rows.T @ A + lam * np.eye(8), A.T @ off)
    C = np.eye(3)
    C[0, :] += delta[:3]
    C[1, :] += delta[3:6]
    C[2, :2] += delta[6:8]
    N = np.array([[1 / scale, 0, -centre[0] / scale], [0, 1 / scale, -centre[1] / scale], [0, 0, 1]])
    C_native = recovery.normalize_h(np.linalg.inv(N) @ C @ N)
    r = rows @ delta - off
    return C_native, dict(observations=int(len(off)), markings=sorted(counts),
                          residual_median_px=float(np.median(abs(r)) * scale),
                          residual_p95_px=float(np.percentile(abs(r), 95) * scale))


def paint_lock(image, atlas, record, layers, *, radius, passes=2, max_move_px=12., **solve_kw):
    """Refine one frame's reference->native transform on the paint next to its drawn markings."""
    t = recovery.matrix(record["reference_to_native"])
    history = []
    for _ in range(passes):
        mapping = recovery.RecoveryFrame(atlas, {**record, "reference_to_native": t.tolist()})
        obs = paint_observations(image, mapping, radius, layers)
        C, stats = solve_correction(obs, record["native_size"], **solve_kw)
        history.append(stats)
        if C is None:
            break
        w, h = record["native_size"]
        gx, gy = np.meshgrid(np.linspace(.1, .9, 5) * w, np.linspace(.2, .95, 5) * h)
        grid = np.c_[gx.ravel(), gy.ravel()]
        move = np.linalg.norm(recovery.project(C, grid) - grid, axis=1)
        stats["grid_move_median_px"] = float(np.median(move))
        stats["grid_move_max_px"] = float(np.max(move))
        if not np.isfinite(move).all() or np.max(move) > max_move_px:
            stats["rejected"] = "correction_exceeds_max_move"
            break
        t = recovery.normalize_h(C @ t)
    return t, history


def grid_disagreement(t_chain, t_independent, t_anchor, native_size):
    """Native-pixel disagreement between chained and directly registered motion."""
    w, h = native_size
    gx, gy = np.meshgrid(np.linspace(.1, .9, 7) * w, np.linspace(.15, .95, 7) * h)
    native = np.c_[gx.ravel(), gy.ravel()]
    reference = recovery.project(np.linalg.inv(t_anchor), native)
    a = recovery.project(t_chain, reference)
    b = recovery.project(t_independent, reference)
    d = np.linalg.norm(a - b, axis=1)
    finite = np.isfinite(d)
    return dict(points=int(finite.sum()), median_px=float(np.median(d[finite])) if finite.any() else None,
                p95_px=float(np.percentile(d[finite], 95)) if finite.any() else None)


def propagate(direction, atlas, clip_dir, clip_rows, ext_dir, ext_rows, *, gauge_radius, score_every,
              independent_stride, render_every, out, binding, max_frames=None,
              lock=False, lock_radius=12., lock_first_radius=20.):
    """Chain adjacent motion from the clip's boundary frame through the extension frames."""
    forward = direction == "forward"
    anchor_row = clip_rows[-1] if forward else clip_rows[0]
    anchor_record = atlas.frames[anchor_row["source_pts"] * Fraction(anchor_row["source_time_base"])]
    t_anchor = recovery.matrix(anchor_record["reference_to_native"])
    anchor_image = read_image(clip_dir, anchor_row)
    cv2.setRNGSeed(0)
    anchor_features = recovery.reference_features(anchor_image)
    order = ext_rows if forward else list(reversed(ext_rows))
    anchor_t = float(anchor_row["source_seconds"])
    previous = anchor_image
    t_chain = t_anchor.copy()
    failed_steps = []
    rows_out = []
    records = []
    renders = []
    started = time.monotonic()
    progress = out / f"progress-{direction}"
    progress.mkdir(exist_ok=True)
    previous_hash = recovery.digest(dict(binding=binding, direction=direction))
    # One immutable file per completed frame avoids rewriting a growing full-game checkpoint.
    for k, path in enumerate(sorted(progress.glob("*.json"))):
        checkpoint = json.loads(path.read_text())
        saved = checkpoint["payload"]
        if (k >= len(order) or path.name != f"{k + 1:06d}.json"
                or checkpoint["payload_sha256"] != recovery.digest(saved)
                or saved["previous_sha256"] != previous_hash
                or saved["entry"]["step"] != k + 1
                or any(saved["record"][key] != order[k][key] for key in
                       ("index", "source_pts", "source_time_base", "native_size", "image_sha256"))):
            raise ValueError("Changed, missing or incompatible propagation checkpoint")
        if sha256_file(Path(ext_dir) / order[k]["file"]) != order[k]["image_sha256"]:
            raise ValueError("Completed frame hash mismatch")
        rows_out.append(saved["entry"])
        records.append(saved["record"])
        if saved["failed_step"] is not None:
            failed_steps.append(saved["failed_step"])
        if saved["render"] is not None:
            renders.append(saved["render"])
        previous_hash = checkpoint["payload_sha256"]
    resumed_frames = len(records)
    if records:
        t_chain = recovery.matrix(records[-1]["reference_to_native"])
        previous = read_image(ext_dir, order[resumed_frames - 1])
    stop = len(order) if max_frames is None else min(len(order), resumed_frames + max_frames)
    for k in range(resumed_frames, stop):
        row = order[k]
        cv2.setRNGSeed(0)
        current = read_image(ext_dir, row)
        layers = paint_layers(current)
        h, stats = recovery.adjacent_motion(previous, current)
        failed_step = None
        if h is None:
            failed_step = dict(step=k, source_seconds=row["source_seconds"], **stats)
            failed_steps.append(failed_step)
        else:
            t_chain = recovery.normalize_h(h @ t_chain)
        record = dict(index=row["index"], source_pts=row["source_pts"], source_time_base=row["source_time_base"],
                      native_size=row["native_size"], image_sha256=row["image_sha256"],
                      reference_to_native=t_chain.tolist(),
                      boundary_offset_native_y_px=anchor_record["boundary_offset_native_y_px"],
                      near_boundary_delta_field_y_m=0.,
                      warnings=list(anchor_record.get("warnings", []))
                      + (["chained_through_failed_adjacent_motion_step"] if failed_steps else [])
                      + ["paint_locked_per_frame_unreviewed" if lock else "propagated_without_paint_reference",
                         "boundary_offset_carried_from_anchor"])
        lock_history = None
        if lock:
            t_locked, lock_history = paint_lock(current, atlas, record, layers,
                                                radius=lock_first_radius if k == 0 else lock_radius)
            t_chain = t_locked  # the locked transform is the next frame's chain start
            record["reference_to_native"] = t_chain.tolist()
        mapping = recovery.RecoveryFrame(atlas, record)
        entry = dict(step=k + 1, source_seconds=row["source_seconds"],
                     seconds_from_anchor=abs(row["source_seconds"] - anchor_t),
                     adjacent=dict(image_registration_pass=stats.get("image_registration_pass"),
                                   check_median_px=stats.get("check_median_px"),
                                   fit_inliers=stats.get("fit_inliers")),
                     failed_steps_so_far=len(failed_steps))
        if lock_history is not None:
            entry["lock"] = lock_history
        if (k + 1) % score_every == 0 or k == len(order) - 1:
            summary, per_marking = gauge_frame(current, mapping, gauge_radius, layers=layers)
            entry.update(gauge=summary, gauge_markings=per_marking)
        if (k + 1) % independent_stride == 0 or k == len(order) - 1:
            h_ind, ind_stats = recovery.register_reference(anchor_features, recovery.reference_features(current),
                                                           row["native_size"])
            if h_ind is not None:
                entry["independent"] = dict(**grid_disagreement(t_chain, recovery.normalize_h(h_ind @ t_anchor),
                                                                t_anchor, row["native_size"]),
                                            registration_pass=True, fit_inliers=ind_stats.get("fit_inliers"))
            else:
                entry["independent"] = dict(registration_pass=False, reason=ind_stats.get("reason"))
        name = None
        if (k + 1) % render_every == 0 or k == len(order) - 1 or k == 0:
            overlay = recovery.render_frame(current, mapping)
            label = f"{direction} +{entry['seconds_from_anchor']:.1f}s"
            if "gauge" in entry and entry["gauge"]["p95_px"] is not None:
                label += f" gauge med/p95 {entry['gauge']['median_px']:.1f}/{entry['gauge']['p95_px']:.1f}px n={entry['gauge']['samples']}"
            cv2.putText(overlay, label, (18, 62), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2, cv2.LINE_AA)
            small = cv2.resize(overlay, (960, 540), interpolation=cv2.INTER_AREA)
            name = f"overlay-{direction}-{entry['step']:04d}.jpg"
            if not cv2.imwrite(str(out / name), small, [cv2.IMWRITE_JPEG_QUALITY, 88]):
                raise OSError("Could not save propagation overlay")
            renders.append(name)
        saved = dict(previous_sha256=previous_hash, record=record, entry=entry,
                     failed_step=failed_step, render=name)
        previous_hash = recovery.digest(saved)
        atomic_json(progress / f"{k + 1:06d}.json",
                    dict(payload=saved, payload_sha256=previous_hash), exclusive=True)
        rows_out.append(entry)
        records.append(record)
        previous = current
        if (k + 1) % 50 == 0:
            print(f"{direction} {k + 1}/{len(order)} +{entry['seconds_from_anchor']:.1f}s "
                  f"failed_steps={len(failed_steps)} gauge={entry.get('gauge')}", flush=True)
    scored = [r for r in rows_out if r.get("gauge", {}).get("measured")]
    independent = [r for r in rows_out if "independent" in r]
    complete = len(records) == len(order)

    def survival(rows, key, limit):
        measured = [r for r in rows if key(r) is not None and math.isfinite(key(r))]
        exceed = next((r["seconds_from_anchor"] for r in measured if key(r) > limit), None)
        missing = len(rows) - len(measured)
        return dict(limit_px=limit, seconds_until_first_exceed=exceed,
                    survived_whole_window=complete and exceed is None and bool(measured) and not missing,
                    measured_frames=len(measured), unmeasured_frames=missing,
                    last_measured_seconds=measured[-1]["seconds_from_anchor"] if measured else None,
                    fraction_under_limit=float(np.mean([key(r) <= limit for r in measured])) if measured else None)

    gauges = [r for r in rows_out if "gauge" in r]
    return dict(direction=direction, anchor=dict(index=anchor_row["index"], source_seconds=anchor_t,
                                                  source_pts=anchor_row["source_pts"],
                                                  source_time_base=anchor_row["source_time_base"]),
                frames=len(records), requested_frames=len(order), complete=complete,
                resumed_frames=resumed_frames, scored_frames=len(scored),
                unmeasured_scored_frames=sum(1 for r in rows_out if "gauge" in r and not r["gauge"]["measured"]),
                failed_adjacent_steps=failed_steps,
                survival_gauge_median=survival(gauges, lambda r: r["gauge"]["median_px"] if r["gauge"]["measured"] else None, GAUGE_TARGET_MEDIAN),
                survival_gauge_p95=survival(gauges, lambda r: r["gauge"]["p95_px"] if r["gauge"]["measured"] else None, GAUGE_TARGET_P95),
                survival_independent_p95=survival(independent, lambda r: r["independent"].get("p95_px") if r["independent"]["registration_pass"] else None, GAUGE_TARGET_P95),
                gauge_baseline_note="same gauge on the reviewed clip's own saved frames reads median 0.7-1.5 px and p95 9-17 px (sampler tail on far touchline and box edges); compare medians and independent drift, not p95 alone",
                independent=[dict(seconds_from_anchor=r["seconds_from_anchor"], **r["independent"])
                             for r in rows_out if "independent" in r],
                renders=renders, elapsed_seconds=time.monotonic() - started, rows=rows_out, records=records)


def contact_sheet(out, names, path, columns=4, width=480):
    tiles = [cv2.resize(cv2.imread(str(out / n)), (width, width * 9 // 16)) for n in names]
    if not tiles:
        return
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)]
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--atlas", type=Path, required=True, help="Reviewed field-recovery-atlas-v1")
    p.add_argument("--frames", type=Path, required=True, help="The clip's source frames.json")
    p.add_argument("--plan", type=Path, required=True, help="Frozen clip plan; reserved windows stay closed")
    p.add_argument("--catalog", type=Path, default=Path("data/game-sources.json"))
    p.add_argument("--seconds", type=float, default=60.)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--directions", nargs="+", choices=("forward", "backward"), default=("forward", "backward"))
    p.add_argument("--gauge-radius", type=float, default=20.)
    p.add_argument("--score-every", type=int, default=5)
    p.add_argument("--independent-stride", type=int, default=10)
    p.add_argument("--render-every", type=int, default=50)
    p.add_argument("--paint-lock", action="store_true", help="Refine each frame on nearby paint after chaining")
    p.add_argument("--lock-radius", type=float, default=12.)
    p.add_argument("--reuse-frames", type=Path, default=None,
                   help="Earlier run directory whose frames-<direction>/ decodes are reused unchanged")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--resume", action="store_true", help="Continue this output with identical inputs and settings")
    p.add_argument("--max-frames", type=int, help="Pause after this many additional frames per direction")
    a = p.parse_args()
    for name in ("seconds", "fps", "gauge_radius", "score_every", "independent_stride", "render_every", "lock_radius"):
        value = getattr(a, name)
        if not math.isfinite(value) or value <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive and finite")
    if a.max_frames is not None and a.max_frames <= 0:
        p.error("--max-frames must be positive")
    if len(set(a.directions)) != len(a.directions):
        p.error("Duplicate direction")
    if a.resume and not a.out.is_dir():
        p.error("--resume requires an existing output directory")
    a.out.mkdir(parents=True, exist_ok=a.resume)
    with (a.out / ".run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Propagation output is already in use") from exc
        run(a)


def run(a):
    started = time.monotonic()
    cv2.setNumThreads(4)
    atlas = recovery.RecoveryAtlas.load(a.atlas)
    manifest = json.loads(a.frames.read_text())
    if any(manifest["source"][key] != atlas.payload["source"][key] for key in ("sha256", "game_id")):
        raise ValueError("Atlas/frames source mismatch")
    plan = load_plan(a.plan, a.catalog)
    game = next(g for g in plan["payload"]["games"] if g["game_id"] == manifest["source"]["game_id"])
    clip_rows = validate_manifest(manifest, game, manifest["requested_interval_s"], manifest["sampling_fps"])
    for row in clip_rows:
        record = atlas.frame(row["source_pts"], row["source_time_base"],
                             source_sha256=game["source_sha256"], native_size=row["native_size"]).record
        if any(record.get(key) != row[key] for key in ("source_pts", "source_time_base", "image_sha256")):
            raise ValueError("Atlas frame does not match the exact source still")
    clip_dir = a.frames.parent
    clip_start, clip_end = clip_rows[0]["source_seconds"], clip_rows[-1]["source_seconds"]
    step = 1. / a.fps
    windows = dict(forward=(round(clip_end + step / 2, 6), round(clip_end + step / 2 + a.seconds, 6)),
                   backward=(round(clip_start - a.seconds, 6), round(clip_start, 6)))
    for direction in a.directions:
        guard_interval(game, *windows[direction])
    declaration = dict(schema="field-recovery-propagation-test-v1", accepted=False, status="development diagnostic",
                       atlas_sha256=hashlib.sha256(a.atlas.read_bytes()).hexdigest(),
                       frames_sha256=hashlib.sha256(a.frames.read_bytes()).hexdigest(),
                       plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),
                       catalog_sha256=sha256_file(a.catalog),
                       clip_interval_s=[clip_start, clip_end], windows={d: windows[d] for d in a.directions},
                       fps=a.fps, gauge_radius_px=a.gauge_radius, score_every=a.score_every,
                       independent_stride=a.independent_stride, render_every=a.render_every,
                       environment=dict(python=platform.python_version(), opencv=cv2.__version__,
                                        numpy=np.__version__, scipy=scipy.__version__, opencv_threads=4, frame_rng_seed=0),
                       implementation_sha256={str(path): sha256_file(path) for path in
                           sorted(Path(recovery.__file__).parent.glob("*.py"))},
                       paint_sampler_sha256=sha256_file(Path(__file__).with_name("measure_field_annotations.py")),
                       targets=dict(median_px=GAUGE_TARGET_MEDIAN, p95_px=GAUGE_TARGET_P95,
                                                                            min_gauge_samples=MIN_GAUGE_SAMPLES),
                       paint_lock=bool(a.paint_lock), lock_radius_px=a.lock_radius, reused_frames=str(a.reuse_frames) if a.reuse_frames else None,
                       method=("adjacent KLT motion chained from the clip boundary frame, then a damped robust per-frame homography refinement on paint next to the drawn markings; no new chart, drift smoothing or boundary fit"
                               if a.paint_lock else
                               "adjacent KLT motion chained from the clip boundary frame; no new paint reference, chart, drift smoothing or boundary fit"),
                       helper_sha256=hashlib.sha256(Path(recovery.__file__).read_bytes()).hexdigest(),
                       runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       limits=["Paint gauge searches near the projected marking, so it can lock onto competing paint or miss when error exceeds the radius; frames with fewer than 40 accepted samples are unmeasured.",
                               "Boundary offset carried from the anchor frame; near-boundary temporal correction absent outside its support.",
                               "Not an independent frozen check and not a metric certificate.",
                               "Survival describes scheduled diagnostic samples only, not continuously checked coverage.",
                               "Completed frame maps are approximate diagnostics; inherited and propagation warnings still block public lookup."])
    declaration_path = a.out / "declaration.json"
    if a.resume:
        if recovery.digest(json.loads(declaration_path.read_text())) != recovery.digest(declaration):
            raise ValueError("Resume inputs, implementation or settings differ from the frozen declaration")
    else:
        atomic_json(declaration_path, declaration, exclusive=True)
    results = {}
    receipt = None
    for direction in a.directions:
        start_s, end_s = windows[direction]
        ext_dir = a.out / f"frames-{direction}"
        if a.reuse_frames is not None:
            ext_dir = a.reuse_frames / f"frames-{direction}"
        elif not (a.resume and (ext_dir / "frames.json").exists()):
            if receipt is None:
                receipt = verify_source(game, manifest["source"]["path"])
            decode_window(game, receipt, start_s, end_s, a.fps, ext_dir)
        ext_manifest = ext_dir / "frames.json"
        ext_rows = validate_manifest(json.loads(ext_manifest.read_text()), game, (start_s, end_s), a.fps)
        if ext_rows[0]["native_size"] != clip_rows[0]["native_size"]:
            raise ValueError("Extension native dimensions differ from anchor")
        binding = dict(declaration_sha256=recovery.digest(declaration), frames_sha256=sha256_file(ext_manifest))
        binding_path = a.out / f"inputs-{direction}.json"
        if binding_path.exists():
            if json.loads(binding_path.read_text()) != binding:
                raise ValueError("Extension frames differ from frozen propagation inputs")
        else:
            atomic_json(binding_path, binding, exclusive=True)
        print(f"{direction}: decoded {len(ext_rows)} frames {start_s}-{end_s}s", flush=True)
        results[direction] = propagate(direction, atlas, clip_dir, clip_rows, ext_dir, ext_rows,
                                       gauge_radius=a.gauge_radius, score_every=a.score_every,
                                       independent_stride=a.independent_stride, render_every=a.render_every, out=a.out,
                                       binding=binding, max_frames=a.max_frames, lock=a.paint_lock, lock_radius=a.lock_radius)
        result = results[direction]
        if result["complete"]:
            payload = deepcopy(atlas.payload)
            payload["frames"] = sorted(result["records"], key=lambda r: source_time(r["source_pts"], r["source_time_base"]))
            payload["provenance"] = dict(parent_atlas_sha256=declaration["atlas_sha256"],
                parent_frames_sha256=declaration["frames_sha256"],
                parent_provenance=payload.get("provenance", {}), propagation=binding,
                note="Inherited chart reference indices and fit/check roles refer to the parent manifest. "
                     "Extension frames are propagation diagnostics, with no new independent paint checks.")
            document = recovery.RecoveryAtlas.from_payload(payload).document
            atlas_path = a.out / f"atlas-{direction}.json"
            if atlas_path.exists():
                if json.loads(atlas_path.read_text()) != document:
                    raise ValueError("Completed propagation atlas differs from saved progress")
            else:
                atomic_json(atlas_path, document, exclusive=True)
        contact_sheet(a.out, result["renders"], a.out / f"contact-{direction}.jpg")
        atomic_json(a.out / f"result-{direction}.json", result)
    summary = {d: {k: r[k] for k in ("frames", "requested_frames", "complete", "resumed_frames", "scored_frames", "unmeasured_scored_frames", "survival_gauge_median",
                                     "survival_gauge_p95", "survival_independent_p95")}
               | dict(failed_adjacent_steps=len(r["failed_adjacent_steps"]),
                      independent_p95_px=[i.get("p95_px") for i in r["independent"]])
               for d, r in results.items()}
    complete = all(r["complete"] for r in results.values())
    report = dict(declaration=declaration, complete=complete, summary=summary, elapsed_seconds=time.monotonic() - started)
    atomic_json(a.out / "report.json", report)
    print(("PROPAGATION_DONE " if complete else "PROPAGATION_PAUSED ") + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
