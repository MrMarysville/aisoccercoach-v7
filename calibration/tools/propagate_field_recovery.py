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
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from calibration import field_recovery as recovery
from calibration.alignment_trial import decode_window
from calibration.field_marking_check import checked_marking_curve
from calibration.tools.measure_field_annotations import paint_layers, refine_polyline

GAUGE_TARGET_MEDIAN = 3.
GAUGE_TARGET_P95 = 6.
MIN_GAUGE_SAMPLES = 40


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
              independent_stride, render_every, out, lock=False, lock_radius=12., lock_first_radius=20.):
    """Chain adjacent motion from the clip's boundary frame through the extension frames."""
    forward = direction == "forward"
    anchor_row = clip_rows[-1] if forward else clip_rows[0]
    anchor_record = atlas.frames[anchor_row["source_pts"] * Fraction(anchor_row["source_time_base"])]
    t_anchor = recovery.matrix(anchor_record["reference_to_native"])
    anchor_image = read_image(clip_dir, anchor_row)
    anchor_features = recovery.reference_features(anchor_image)
    order = ext_rows if forward else list(reversed(ext_rows))
    anchor_t = float(anchor_row["source_seconds"])
    previous = anchor_image
    t_chain = t_anchor.copy()
    failed_steps = []
    rows_out = []
    renders = []
    started = time.monotonic()
    for k, row in enumerate(order):
        current = read_image(ext_dir, row)
        layers = paint_layers(current)
        h, stats = recovery.adjacent_motion(previous, current)
        if h is None:
            failed_steps.append(dict(step=k, source_seconds=row["source_seconds"], **stats))
        else:
            t_chain = recovery.normalize_h(h @ t_chain)
        record = dict(index=row["index"], source_pts=row["source_pts"], source_time_base=row["source_time_base"],
                      native_size=row["native_size"], image_sha256=row["image_sha256"],
                      reference_to_native=t_chain.tolist(),
                      boundary_offset_native_y_px=anchor_record["boundary_offset_native_y_px"],
                      near_boundary_delta_field_y_m=0.,
                      warnings=(["chained_through_failed_adjacent_motion_step"] if failed_steps else [])
                      + ["propagated_without_paint_reference", "boundary_offset_carried_from_anchor"])
        lock_history = None
        if lock:
            t_locked, lock_history = paint_lock(current, atlas, record, layers,
                                                radius=lock_first_radius if k == 0 else lock_radius)
            t_chain = t_locked  # the locked transform is the next frame's chain start
            record["reference_to_native"] = t_chain.tolist()
            record["warnings"] = [w for w in record["warnings"] if w != "propagated_without_paint_reference"] + ["paint_locked_per_frame_unreviewed"]
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
        if (k + 1) % independent_stride == 0:
            h_ind, ind_stats = recovery.register_reference(anchor_features, recovery.reference_features(current),
                                                           row["native_size"])
            if h_ind is not None:
                entry["independent"] = dict(**grid_disagreement(t_chain, recovery.normalize_h(h_ind @ t_anchor),
                                                                t_anchor, row["native_size"]),
                                            registration_pass=True, fit_inliers=ind_stats.get("fit_inliers"))
            else:
                entry["independent"] = dict(registration_pass=False, reason=ind_stats.get("reason"))
        if (k + 1) % render_every == 0 or k == len(order) - 1 or k == 0:
            overlay = recovery.render_frame(current, mapping)
            label = f"{direction} +{entry['seconds_from_anchor']:.1f}s"
            if "gauge" in entry and entry["gauge"]["p95_px"] is not None:
                label += f" gauge med/p95 {entry['gauge']['median_px']:.1f}/{entry['gauge']['p95_px']:.1f}px n={entry['gauge']['samples']}"
            cv2.putText(overlay, label, (18, 62), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2, cv2.LINE_AA)
            small = cv2.resize(overlay, (960, 540), interpolation=cv2.INTER_AREA)
            name = f"overlay-{direction}-{entry['step']:04d}.jpg"
            cv2.imwrite(str(out / name), small, [cv2.IMWRITE_JPEG_QUALITY, 88])
            renders.append(name)
        rows_out.append(entry)
        previous = current
        if (k + 1) % 50 == 0:
            print(f"{direction} {k + 1}/{len(order)} +{entry['seconds_from_anchor']:.1f}s "
                  f"failed_steps={len(failed_steps)} gauge={entry.get('gauge')}", flush=True)
    scored = [r for r in rows_out if r.get("gauge", {}).get("measured")]
    independent = [r for r in rows_out if r.get("independent", {}).get("registration_pass")]

    def survival(rows, key, limit):
        exceed = next((r["seconds_from_anchor"] for r in rows if key(r) > limit), None)
        return dict(limit_px=limit, seconds_until_first_exceed=exceed,
                    survived_whole_window=exceed is None and bool(rows),
                    last_measured_seconds=rows[-1]["seconds_from_anchor"] if rows else None,
                    fraction_under_limit=float(np.mean([key(r) <= limit for r in rows])) if rows else None)

    return dict(direction=direction, anchor=dict(index=anchor_row["index"], source_seconds=anchor_t,
                                                  source_pts=anchor_row["source_pts"]),
                frames=len(order), scored_frames=len(scored),
                unmeasured_scored_frames=sum(1 for r in rows_out if "gauge" in r and not r["gauge"]["measured"]),
                failed_adjacent_steps=failed_steps,
                survival_gauge_median=survival(scored, lambda r: r["gauge"]["median_px"], GAUGE_TARGET_MEDIAN),
                survival_gauge_p95=survival(scored, lambda r: r["gauge"]["p95_px"], GAUGE_TARGET_P95),
                survival_independent_p95=survival(independent, lambda r: r["independent"]["p95_px"], GAUGE_TARGET_P95),
                gauge_baseline_note="same gauge on the reviewed clip's own saved frames reads median 0.7-1.5 px and p95 9-17 px (sampler tail on far touchline and box edges); compare medians and independent drift, not p95 alone",
                independent=[dict(seconds_from_anchor=r["seconds_from_anchor"], **r["independent"])
                             for r in rows_out if "independent" in r],
                renders=renders, elapsed_seconds=time.monotonic() - started, rows=rows_out)


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
    a = p.parse_args()
    started = time.monotonic()
    a.out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(4)
    atlas = recovery.RecoveryAtlas.load(a.atlas)
    manifest = json.loads(a.frames.read_text())
    if manifest["source"]["sha256"] != atlas.payload["source"]["sha256"]:
        raise ValueError("Atlas/frames source mismatch")
    plan = json.loads(a.plan.read_text())
    game = next(g for g in plan["payload"]["games"] if g["game_id"] == manifest["source"]["game_id"])
    if game["source_sha256"] != manifest["source"]["sha256"]:
        raise ValueError("Plan/source mismatch")
    receipt = dict(manifest["source"])
    clip_rows = manifest["frames"]
    clip_dir = a.frames.parent
    clip_start, clip_end = clip_rows[0]["source_seconds"], clip_rows[-1]["source_seconds"]
    step = 1. / a.fps
    windows = dict(forward=(round(clip_end + step / 2, 6), round(clip_end + step / 2 + a.seconds, 6)),
                   backward=(round(clip_start - a.seconds, 6), round(clip_start, 6)))
    declaration = dict(schema="field-recovery-propagation-test-v1", accepted=False, status="development diagnostic",
                       atlas_sha256=hashlib.sha256(a.atlas.read_bytes()).hexdigest(),
                       frames_sha256=hashlib.sha256(a.frames.read_bytes()).hexdigest(),
                       plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),
                       clip_interval_s=[clip_start, clip_end], windows={d: windows[d] for d in a.directions},
                       fps=a.fps, gauge_radius_px=a.gauge_radius, score_every=a.score_every,
                       independent_stride=a.independent_stride, targets=dict(median_px=GAUGE_TARGET_MEDIAN, p95_px=GAUGE_TARGET_P95,
                                                                            min_gauge_samples=MIN_GAUGE_SAMPLES),
                       paint_lock=bool(a.paint_lock), lock_radius_px=a.lock_radius, reused_frames=str(a.reuse_frames) if a.reuse_frames else None,
                       method=("adjacent KLT motion chained from the clip boundary frame, then a damped robust per-frame homography refinement on paint next to the drawn markings; no new chart, drift smoothing or boundary fit"
                               if a.paint_lock else
                               "adjacent KLT motion chained from the clip boundary frame; no new paint reference, chart, drift smoothing or boundary fit"),
                       helper_sha256=hashlib.sha256(Path(recovery.__file__).read_bytes()).hexdigest(),
                       runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       limits=["Paint gauge searches near the projected marking, so it can lock onto competing paint or miss when error exceeds the radius; frames with fewer than 40 accepted samples are unmeasured.",
                               "Boundary offset carried from the anchor frame; near-boundary temporal correction absent outside its support.",
                               "Not an independent frozen check and not a metric certificate."])
    (a.out / "declaration.json").write_text(json.dumps(declaration, indent=2) + "\n")
    results = {}
    for direction in a.directions:
        start_s, end_s = windows[direction]
        ext_dir = a.out / f"frames-{direction}"
        if a.reuse_frames is not None:
            ext_dir = a.reuse_frames / f"frames-{direction}"
            reused = json.loads((ext_dir / "frames.json").read_text())
            if (reused["source"]["sha256"] != receipt["sha256"] or reused["sampling_fps"] != a.fps
                    or [round(v, 6) for v in reused["requested_interval_s"]] != [round(start_s, 6), round(end_s, 6)]):
                raise ValueError("Reused frames do not match this source, cadence or window")
        else:
            decode_window(game, receipt, start_s, end_s, a.fps, ext_dir)
        ext_rows = json.loads((ext_dir / "frames.json").read_text())["frames"]
        print(f"{direction}: decoded {len(ext_rows)} frames {start_s}-{end_s}s", flush=True)
        results[direction] = propagate(direction, atlas, clip_dir, clip_rows, ext_dir, ext_rows,
                                       gauge_radius=a.gauge_radius, score_every=a.score_every,
                                       independent_stride=a.independent_stride, render_every=a.render_every, out=a.out,
                                       lock=a.paint_lock, lock_radius=a.lock_radius)
        contact_sheet(a.out, results[direction]["renders"], a.out / f"contact-{direction}.jpg")
        (a.out / f"result-{direction}.json").write_text(json.dumps(results[direction], indent=2, allow_nan=False) + "\n")
    summary = {d: {k: r[k] for k in ("frames", "scored_frames", "unmeasured_scored_frames", "survival_gauge_median",
                                     "survival_gauge_p95", "survival_independent_p95")}
               | dict(failed_adjacent_steps=len(r["failed_adjacent_steps"]),
                      independent_p95_px=[i.get("p95_px") for i in r["independent"]])
               for d, r in results.items()}
    report = dict(declaration=declaration, summary=summary, elapsed_seconds=time.monotonic() - started)
    (a.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("PROPAGATION_DONE " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
