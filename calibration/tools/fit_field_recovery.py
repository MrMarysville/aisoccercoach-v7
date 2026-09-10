#!/usr/bin/env python3
"""Fit a source-bound v7-method candidate from fresh named paint measurements.

Usage: python calibration/tools/fit_field_recovery.py --frames frames.json
       --measurements fresh.json --out /absolute/new/output
No source decoding, old artifact loading, cloud calls, or check-label fitting.
"""
import argparse
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import field_recovery as recovery


def load_inputs(frames_path, measurements_path):
    frames_path, measurements_path = Path(frames_path), Path(measurements_path)
    manifest = json.loads(frames_path.read_text())
    measurements = json.loads(measurements_path.read_text())
    if manifest.get("schema") != "alignment-trial-frames-v1":
        raise ValueError("Expected alignment-trial-frames-v1 manifest")
    if measurements.get("schema") != "field-recovery-measurements-v1":
        raise ValueError("Expected field-recovery-measurements-v1")
    source_hash = manifest["source"]["sha256"]
    if measurements["source_sha256"] != source_hash:
        raise ValueError("Measurement/source hash mismatch")
    by_index = {}
    previous = None
    for row in manifest["frames"]:
        i = row["index"]
        if type(i) is not int or i in by_index:
            raise ValueError("Invalid or duplicate frame index")
        if type(row["source_pts"]) is not int or not isinstance(row["source_time_base"], str):
            raise ValueError("Exact integer PTS and rational time base required")
        tb = Fraction(row["source_time_base"])
        t = row["source_pts"]*tb
        if tb <= 0 or (previous is not None and t <= previous):
            raise ValueError("Source PTS must be strictly ordered")
        if row["source_sha256"] != source_hash:
            raise ValueError("Frame/source hash mismatch")
        previous = t
        by_index[i] = row
    fit = set(measurements["fit_frame_indices"])
    check = set(measurements["check_frame_indices"])
    if not fit or fit & check or not (fit | check) <= set(by_index):
        raise ValueError("Declare nonempty fit frames and disjoint existing check frames")
    for ref in measurements["references"] + measurements.get("boundary_observations", []):
        if ref["frame_index"] not in fit:
            raise ValueError("Reference fit attempted outside declared fit-frame split")
        row = by_index[ref["frame_index"]]
        for feature in ref["features"]:
            p = recovery.points(feature["points_native"])
            if np.any(p < 0) or np.any(p >= row["native_size"]):
                raise ValueError("Reference annotation contains offscreen or preview coordinates")
        for key in ("source_pts", "source_time_base", "image_sha256"):
            if key in ref and ref[key] != row[key]:
                raise ValueError(f"Reference {key} does not match the exact still")
    # Check features are never returned as fit references; freeze their content in provenance.
    return manifest, measurements, by_index


def image_anchor_positions(rows, measurements, reference_stride):
    positions = {row["index"]: i for i, row in enumerate(rows)}
    fit = {positions[i] for i in measurements["fit_frame_indices"]}
    return sorted((set(range(0, len(rows), max(reference_stride, 1))) & fit) |
                  {positions[r["frame_index"]] for r in measurements["references"]} | {max(fit)})


def fitting_frame_bindings(manifest, indices):
    indices = set(indices)
    return [dict(source_sha256=manifest["source"]["sha256"],
                 **{k: row[k] for k in ("source_pts", "source_time_base", "image_sha256", "native_size")})
            for row in manifest["frames"] if row["index"] in indices]


def independent_reference_connections(gauge_index, paint_indices, image_anchor_indices, register, *, times=None):
    """Connect image anchors directly, then paint references through one measured hop.

    Adjacent accumulation is deliberately absent from this function. Every edge
    must pass the original registration checks, including the bridge's first leg.
    Among passing bridges, maximize the weaker edge's observed ground coverage;
    consistency and summed check medians break ties. No paint errors rank paths.
    """
    transforms = {gauge_index: np.eye(3)}
    distance = lambda a, b: abs(a-b) if times is None else abs(times[a]-times[b])
    paths = {gauge_index: dict(path=[gauge_index], kind="gauge", cost=None)}
    attempts, direct_stats = [], {}
    for i in sorted(set(paint_indices) | set(image_anchor_indices)):
        if i == gauge_index:
            continue
        h, stats = register(gauge_index, i)
        attempts.append(dict(from_index=gauge_index, to_index=i, role="gauge_direct", **stats))
        if h is not None:
            transforms[i] = h
            direct_stats[i] = stats
            paths[i] = dict(path=[gauge_index, i], kind="gauge_direct", cost=None)
    # Freeze first legs before adding bridges: no hidden multi-hop chain is allowed.
    direct = dict(transforms)
    for i in paint_indices:
        if i in direct:
            continue
        candidates = []
        for j in sorted(set(direct)-{gauge_index}, key=lambda j: (distance(j, i), j)):
            h, stats = register(j, i)
            attempts.append(dict(from_index=j, to_index=i, role="paint_reference_bridge", **stats))
            if h is None:
                continue
            first = direct_stats[j]
            cost = dict(bottleneck_inlier_hull_fraction=min(first["inlier_hull_fraction"], stats["inlier_hull_fraction"]),
                        bottleneck_check_consistent_fraction=min(first["check_consistent_fraction"], stats["check_consistent_fraction"]),
                        total_check_median_px=first["check_median_px"]+stats["check_median_px"])
            rank = (-cost["bottleneck_inlier_hull_fraction"], -cost["bottleneck_check_consistent_fraction"],
                    cost["total_check_median_px"], distance(j, i), j)
            candidates.append((rank, recovery.normalize_h(h @ direct[j]), j, cost))
        if candidates:
            _, h, j, cost = min(candidates, key=lambda x: x[0])
            transforms[i] = h
            paths[i] = dict(path=[gauge_index, j, i], kind="gauge_single_bridge", cost=cost)
    return transforms, paths, attempts


def independent_drift_observations(gauge_index, paint_indices, image_anchor_indices,
                                   registered, connection_paths, register, *, times=None):
    """Use the closest connected paint view, while keeping gauge bridge evidence.

    Direct gauge matches establish independent connections; their availability
    must not override a nearer paint view's overlap. A target on the reference's
    connection path cannot register back through that reference and form a loop.
    """
    paint = sorted(set(paint_indices) & set(registered))
    distance = lambda a, b: abs(a-b) if times is None else abs(times[a]-times[b])
    observations, paths, attempts = {}, {}, []
    for i in image_anchor_indices:
        if i in paint:
            observations[i] = registered[i]
            paths[i] = deepcopy(connection_paths[i])
            continue
        for j in sorted(paint, key=lambda j: (distance(j, i), j)):
            if i in connection_paths[j]["path"]:
                attempts.append(dict(from_index=j, to_index=i, role="drift_route_skipped",
                                     image_registration_pass=False, reason="target_already_on_reference_connection_path"))
                continue
            if j == gauge_index:
                # This pair was already tested during bridge construction.
                if i in registered:
                    observations[i] = registered[i]
                    paths[i] = deepcopy(connection_paths[i])
                    break
                continue
            h, stats = register(j, i)
            attempts.append(dict(from_index=j, to_index=i, role="independent_paint_to_image_anchor", **stats))
            if h is not None:
                observations[i] = recovery.normalize_h(h @ registered[j])
                paths[i] = dict(path=connection_paths[j]["path"]+[i],
                               kind="independent_paint_to_image_anchor", cost=None)
                break
    return observations, paths, attempts


def run(frames_path, measurements_path, out, *, render=True, reference_stride=10):
    started = time.monotonic()
    frames_path, measurements_path, out = Path(frames_path), Path(measurements_path), Path(out)
    manifest, measurements, by_index = load_inputs(frames_path, measurements_path)
    if out.exists():
        raise ValueError("Output directory must not exist; preserve earlier revisions")
    out.mkdir(parents=True)
    cv2.setNumThreads(4)
    rows = manifest["frames"]
    positions = {r["index"]: i for i, r in enumerate(rows)}
    source_times = {i: r["source_pts"]*Fraction(r["source_time_base"]) for i, r in enumerate(rows)}
    targets = image_anchor_positions(rows, measurements, reference_stride)
    def write(name, value):
        (out/name).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    def read(row):
        path = (frames_path.parent/row["file"]).resolve()
        if not path.is_relative_to(frames_path.parent.resolve()):
            raise ValueError("Frame path escapes source manifest directory")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["image_sha256"]:
            raise ValueError("Decoded still hash mismatch")
        im = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if im is None or [im.shape[1], im.shape[0]] != row["native_size"]:
            raise ValueError("Native still dimensions mismatch")
        return im
    write("fit-declaration.json", dict(source_sha256=manifest["source"]["sha256"],
          measurements_sha256=hashlib.sha256(measurements_path.read_bytes()).hexdigest(),
          frames_sha256=hashlib.sha256(frames_path.read_bytes()).hexdigest(),
          fit_frame_indices=measurements["fit_frame_indices"], check_frame_indices=measurements["check_frame_indices"],
          image_anchor_frame_indices=[rows[i]["index"] for i in targets],
          reference_distance="elapsed source time; image anchors restricted to declared fit frames",
          projection_mode=(recovery.SINGLE_REFERENCE_PROJECTION_MODE if len(measurements["references"]) == 1
                           else recovery.V7_PROJECTION_MODE),
          motion_reference_method="direct gauge registrations, one measured bridge per disconnected paint reference, then nearest connected paint reference for drift; adjacent-only chart fallback is not independent evidence",
          method="fresh local charts + v5 accumulated motion/slow drift + v6 protected spatial boundary + v7 native-y temporal correction",
          field=recovery.FIELD, reference_stride=reference_stride,
          helper_sha256=hashlib.sha256(Path(recovery.__file__).read_bytes()).hexdigest(),
          runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    fitted = []
    for ref in measurements["references"]:
        frame = by_index[ref["frame_index"]]
        read(frame)  # Verify the still before trusting source-bound samples.
        fit = recovery.fit_reference(ref["features"], frame["native_size"], ref.get("support_world_polygon"))
        fit.update(frame_index=ref["frame_index"], chart=ref["chart"])
        fitted.append(fit)
        write("reference-fits.json", fitted)
        print(f"paint fit {ref['frame_index']} {ref['chart']}: {fit['features']}", flush=True)
    if not fitted:
        raise ValueError("At least one new paint reference is required")
    # Each later reference is independently refitted. The fixed v7 atlas takes one
    # designated reference per chart; additional references are separate recoveries.
    names = [f["chart"] for f in fitted]
    if len(set(names)) != len(names):
        raise ValueError("One reference per spatial chart per candidate; save a later refit as a new candidate revision")
    gauge_fit = next((f for f in fitted if f["chart"] == "mid"), fitted[len(fitted)//2])
    gauge_index = positions[gauge_fit["frame_index"]]
    actual_refs = [positions[f["frame_index"]] for f in fitted]
    # A fit must meet observed marking error bounds before it can clear a chain warning.
    trusted_refs = [positions[f["frame_index"]] for f in fitted if all(
        x["finite_samples"] == x["samples"] and x["p95_px"] is not None and x["p95_px"] <= 6 for x in f["features"])]
    steps, motion_stats = [None], []
    previous = read(rows[0])
    cache = {}
    for i in actual_refs:
        cache[i] = recovery.reference_features(read(rows[i]))
    for i in range(1, len(rows)):
        current = read(rows[i])
        h, stats = recovery.adjacent_motion(previous, current)
        steps.append(h)
        motion_stats.append(dict(frame_index=rows[i]["index"], **stats))
        previous = current
        if i % 25 == 0:
            print(f"adjacent motion {i}/{len(rows)-1}", flush=True)
    # Chain flags are deliberately not cleared by image-only direct matches.
    accumulated, warnings = recovery.accumulate_motion(steps, gauge_index)
    for i in targets:
        if i not in cache:
            cache[i] = recovery.reference_features(read(rows[i]))
    def register(a, b):
        return recovery.register_reference(cache[a], cache[b], rows[b]["native_size"])
    independently_registered, reference_paths, attempts = independent_reference_connections(
        gauge_index, actual_refs, targets, register, times=source_times)
    direct_stats = [dict(from_frame_index=rows[s["from_index"]]["index"], frame_index=rows[s["to_index"]]["index"],
                         **{k:v for k,v in s.items() if k not in ("from_index", "to_index")}) for s in attempts]
    # A usable adjacent route can transport a chart for diagnostics, but cannot
    # become independent drift evidence or clear a failed chain's warning.
    fallback_chart_transport = {i: accumulated[i] for i in actual_refs
                                if i not in independently_registered and not warnings[i]}
    connected_paint_refs = sorted(set(trusted_refs) & set(independently_registered))
    accumulated, warnings = recovery.accumulate_motion(steps, gauge_index, connected_paint_refs, independently_registered)
    for i in actual_refs:
        if i not in independently_registered:
            warnings[i].append("paint_reference_not_independently_connected_to_gauge")
    if gauge_index not in trusted_refs:
        for warning in warnings:
            warning.append("gauge_paint_fit_exceeds_6px_p95")
    observations, drift_paths, drift_attempts = independent_drift_observations(
        gauge_index, actual_refs, targets, independently_registered, reference_paths, register, times=source_times)
    independent = [observations.get(i) for i in range(len(rows))]
    direct_stats.extend(dict(from_frame_index=rows[s["from_index"]]["index"], frame_index=rows[s["to_index"]]["index"],
                             **{k:v for k,v in s.items() if k not in ("from_index", "to_index")}) for s in drift_attempts)
    for i in targets:
        print(f"independent image reference {i}: {'available' if independent[i] is not None else 'unavailable'}", flush=True)
    times = [float(t) for t in source_times.values()]
    transforms, drift = recovery.smooth_reference_drift(accumulated, independent, times, gauge_index)
    for i in drift["unsupported_indices"]:
        warnings[i].append("no_independent_slow_drift_support")
    charts = []
    for f in fitted:
        i = positions[f["frame_index"]]
        if i not in independently_registered and i not in fallback_chart_transport:
            continue
        charts.append(dict(chart=f["chart"], reference_frame_index=f["frame_index"],
                           field_to_reference=recovery.normalize_h(np.linalg.inv(transforms[i]) @ np.asarray(f["field_to_native"])).tolist(),
                           support_world_polygon=f["support_world_polygon"],
                           transport_status="independent_image_reference" if i in independently_registered else "adjacent_only_diagnostic"))
    frame_records = [dict(index=r["index"], source_pts=r["source_pts"], source_time_base=r["source_time_base"],
                          native_size=r["native_size"], image_sha256=r["image_sha256"],
                          reference_to_native=h.tolist(), boundary_offset_native_y_px=0., warnings=w)
                     for r, h, w in zip(rows, transforms, warnings)]
    projection_mode = recovery.SINGLE_REFERENCE_PROJECTION_MODE if len(fitted) == 1 else recovery.V7_PROJECTION_MODE
    payload = dict(status="approximate", metric_certified=False, projection_mode=projection_mode,
                   source={k:manifest["source"][k] for k in ("game_id", "sha256")},
                   field=recovery.FIELD, blend=measurements.get("blend", recovery.BLEND), charts=charts,
                   protected_y_m=-recovery.BOX18_HALF_W, spatial_boundary=None, temporal_boundary=None,
                   frames=frame_records, provenance=dict(measurements_sha256=hashlib.sha256(measurements_path.read_bytes()).hexdigest(),
                                                       fit_frame_indices=measurements["fit_frame_indices"],
                                                       fitting_frames=fitting_frame_bindings(manifest,
                                                           {rows[i]["index"] for i in targets} | {r["frame_index"]
                                                               for r in measurements.get("boundary_observations", [])})))
    boundary = []
    # Extra boundary observations are accepted only on designated fit frames.
    for ref in measurements["references"] + measurements.get("boundary_observations", []):
        i = positions[ref["frame_index"]]
        if ref["frame_index"] not in measurements["fit_frame_indices"]:
            continue
        for f in ref.get("features", []):
            if f["label"] == "touch_far":
                boundary.append(dict(frame_index=ref["frame_index"], points_native=f["points_native"],
                                     reference_to_native=transforms[i].tolist()))
    if boundary:
        payload["spatial_boundary"] = recovery.fit_spatial_boundary(boundary, measurements["fit_frame_indices"])
    atlas = recovery.RecoveryAtlas.from_payload(payload)
    temporal_observations = []
    for row in boundary:
        i = positions[row["frame_index"]]
        mapping = atlas.frame(rows[i]["source_pts"], rows[i]["source_time_base"],
                              source_sha256=manifest["source"]["sha256"], native_size=rows[i]["native_size"])
        native = recovery.points(row["points_native"])
        # Native y correction follows v7. Interpolation is allowed only along a
        # monotone, visible projected far boundary; no nearest-fit paint search.
        field = np.c_[np.linspace(-recovery.HALF_L, recovery.HALF_L, 8001), np.full(8001, -recovery.HALF_W)]
        curve = mapping.project(field)
        keep = np.isfinite(curve).all(axis=1)
        curve = curve[keep]
        if len(curve) < 10:
            continue
        order = np.argsort(curve[:, 0])
        curve = curve[order]
        if np.max(np.diff(curve[:, 0])) > 100:
            continue
        selected = (native[:, 0] >= curve[0, 0]) & (native[:, 0] <= curve[-1, 0])
        if selected.sum() < 10:
            continue
        errors = native[selected, 1]-np.interp(native[selected, 0], curve[:, 0], curve[:, 1])
        med = np.median(errors)
        temporal_observations.append(dict(frame_index=row["frame_index"], time_s=times[i], samples=len(errors),
                                          median_offset_px=float(med), mad_px=float(np.median(abs(errors-med)))))
    # Merge duplicate fit-frame observations before spline fitting.
    temporal_observations = list({r["frame_index"]:r for r in temporal_observations}.values())
    temporal = recovery.fit_temporal_boundary(temporal_observations, measurements["fit_frame_indices"])
    payload["temporal_boundary"] = temporal
    if temporal:
        for f, t in zip(payload["frames"], times):
            try:
                f["boundary_offset_native_y_px"] = recovery.temporal_offset(temporal, t)
            except ValueError:
                f["warnings"].append("outside_temporal_boundary_fit_support")
    atlas = recovery.RecoveryAtlas.from_payload(payload)
    # Validate the saved nonlinear map across its visible diagnostic domain.
    grid_x, grid_y = np.meshgrid(np.linspace(-recovery.HALF_L, recovery.HALF_L, 181),
                                 np.linspace(-recovery.HALF_W, recovery.HALF_W, 107))
    grid = np.c_[grid_x.ravel(), grid_y.ravel()]
    geometry = []
    for row in rows:
        m = atlas.frame(row["source_pts"], row["source_time_base"], source_sha256=manifest["source"]["sha256"], native_size=row["native_size"])
        check = recovery.visible_geometry_check(m, grid)
        geometry.append(dict(frame_index=row["index"], **check))
        if not check["valid"]:
            payload["frames"][positions[row["index"]]]["warnings"].append("invalid_or_absent_visible_supported_geometry")
    atlas = recovery.RecoveryAtlas.from_payload(payload)
    atlas.save(out/"atlas.json")
    # Reload before every downstream report/render path: saved map is the contract.
    atlas = recovery.RecoveryAtlas.load(out/"atlas.json")
    if render:
        (out/"overlay").mkdir()
        for i, row in enumerate(rows):
            mapping = atlas.frame(row["source_pts"], row["source_time_base"], source_sha256=manifest["source"]["sha256"], native_size=row["native_size"])
            cv2.imwrite(str(out/"overlay"/f"{i+1:05d}.jpg"), recovery.render_frame(read(row), mapping), [cv2.IMWRITE_JPEG_QUALITY, 92])
            if i % 50 == 0:
                print(f"render saved mapping {i}/{len(rows)}", flush=True)
    report = dict(accepted=False, status="approximate candidate; independent paint checks pending", total_frames=len(rows),
                  frames_with_warnings=sum(bool(f["warnings"]) for f in payload["frames"]),
                  references=fitted, adjacent_motion=motion_stats, independent_registration=direct_stats,
                  reference_connections=[dict(frame_index=rows[i]["index"],
                        path=[rows[j]["index"] for j in record["path"]], kind=record["kind"], cost=record["cost"])
                        for i, record in sorted(reference_paths.items())],
                  independent_drift_routes=[dict(frame_index=rows[i]["index"],
                        path=[rows[j]["index"] for j in record["path"]], kind=record["kind"])
                        for i, record in sorted(drift_paths.items())],
                  adjacent_only_chart_frame_indices=[rows[i]["index"] for i in sorted(fallback_chart_transport)],
                  drift=drift, geometry=geometry, temporal_fit_observations=temporal_observations,
                  atlas_sha256=hashlib.sha256((out/"atlas.json").read_bytes()).hexdigest(),
                  elapsed_seconds=time.monotonic()-started,
                  limitations=["Reference fitting scores are training evidence, not held-out validation.",
                               "Ground image matches are checks of image motion, not metric certification.",
                               "Diagnostic rendering evaluates the declared projection mode; public coordinates remain restricted to measured support.",
                               "One freshly fitted reference per chart per candidate; later adjustments require a new immutable fit revision."])
    write("report.json", report)
    print(json.dumps({k:report[k] for k in ("accepted", "status", "total_frames", "frames_with_warnings", "elapsed_seconds")}), flush=True)
    print("FIELD_RECOVERY_COMPLETE", flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames", required=True)
    p.add_argument("--measurements", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--skip-render", action="store_true")
    p.add_argument("--reference-stride", type=int, default=10,
                   help="Independent image reference sampling interval, default 10 source samples")
    args = p.parse_args()
    run(args.frames, args.measurements, args.out, render=not args.skip_render,
        reference_stride=args.reference_stride)


if __name__ == "__main__":
    main()
