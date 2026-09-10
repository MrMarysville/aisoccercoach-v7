"""Bound image-registration drift around an immutable parent field model.

Only eligible fitting images and optional source-reviewed reference paint enter
this CLI. Independent check paint is deliberately a separate scoring input.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
import time

import cv2
import numpy as np
import scipy

from calibration import field_propagation as propagation
from calibration import field_recovery as recovery
from calibration.alignment_trial import guard_interval, load_plan, sha256_file, source_time
from calibration.field_adjustment import atomic_json
from calibration.tools.fit_field_recovery import independent_reference_connections, independent_drift_observations
from calibration.tools.propagate_field_recovery import read_image, validate_manifest
from calibration.tools.score_field_recovery import marking_errors


IDENTITY_KEYS = ("source_pts", "source_time_base", "image_sha256", "native_size")


def identity(row):
    return source_time(row["source_pts"], row["source_time_base"])


def frozen_payload(path, schema):
    document = json.loads(Path(path).read_text())
    if document.get("schema") != schema or document.get("payload_sha256") != recovery.digest(document["payload"]):
        raise ValueError("Invalid frozen evidence checksum or schema")
    return document["payload"]


def load_split(path, manifest, manifest_path):
    split = frozen_payload(path, "field-recovery-reanchor-split-v1")
    if (split["source_sha256"] != manifest["source"]["sha256"]
            or split["frames_manifest_sha256"] != sha256_file(manifest_path)):
        raise ValueError("Split/source/manifest mismatch")
    inherited = {}
    for entry in split.get("inherited_splits", []):
        if sha256_file(entry["path"]) != entry["sha256"]:
            raise ValueError("Changed inherited frame split")
        old = frozen_payload(entry["path"], "field-recovery-reanchor-split-v1")
        if old["source_sha256"] != split["source_sha256"]:
            raise ValueError("Inherited split belongs to another source")
        for row in old["frames"]:
            stamp = identity(row)
            if stamp in inherited and inherited[stamp] != row:
                raise ValueError("Conflicting inherited source-frame roles")
            inherited[stamp] = row
    if len(split["frames"]) != len(manifest["frames"]):
        raise ValueError("Split must cover every source frame")
    for i, (role, frame) in enumerate(zip(split["frames"], manifest["frames"])):
        if any(role[k] != frame[k] for k in (*IDENTITY_KEYS, "index")):
            raise ValueError("Split frame identity mismatch")
        expected = "fit" if i % 10 == 0 or i == len(split["frames"]) - 1 else "check"
        if identity(frame) in inherited:
            old = inherited[identity(frame)]
            if any(old[k] != frame[k] for k in IDENTITY_KEYS):
                raise ValueError("Inherited source still changed")
            expected = old["role"]
        if role["role"] != expected:
            raise ValueError("Frozen fit/check role changed")
    return split


def validate_completed_run(directory, direction, parent, frames_path, extension_path, rows, plan_path, catalog_path):
    """Verify immutable checkpoint links without changing or resuming the raw run."""
    directory = Path(directory)
    declaration_path = directory / "declaration.json"
    declaration = json.loads(declaration_path.read_text())
    if (declaration["atlas_sha256"] != sha256_file(parent)
            or declaration["frames_sha256"] != sha256_file(frames_path)
            or declaration["plan_sha256"] != sha256_file(plan_path)
            or declaration["catalog_sha256"] != sha256_file(catalog_path)
            or declaration["paint_lock"]):
        raise ValueError("Propagation must use these frozen parents and adjacent motion without paint lock")
    result = json.loads((directory / f"result-{direction}.json").read_text())
    if not result["complete"] or result["frames"] != len(rows) or result["requested_frames"] != len(rows):
        raise ValueError("Re-anchoring requires a completed propagation direction")
    binding = dict(declaration_sha256=recovery.digest(declaration), frames_sha256=sha256_file(extension_path))
    if json.loads((directory / f"inputs-{direction}.json").read_text()) != binding:
        raise ValueError("Propagation/extension input binding mismatch")
    ordered = rows if direction == "forward" else list(reversed(rows))
    checkpoints = sorted((directory / f"progress-{direction}").glob("*.json"))
    if len(checkpoints) != len(rows) or len(result["records"]) != len(rows) or len(result["rows"]) != len(rows):
        raise ValueError("Missing propagation checkpoints or diagnostics")
    previous = recovery.digest(dict(binding=binding, direction=direction))
    for i, (path, row, record, entry) in enumerate(zip(checkpoints, ordered, result["records"], result["rows"]), 1):
        checkpoint = json.loads(path.read_text())
        saved = checkpoint["payload"]
        if (path.name != f"{i:06d}.json" or saved["previous_sha256"] != previous
                or checkpoint["payload_sha256"] != recovery.digest(saved)
                or saved["record"] != record or saved["entry"] != entry
                or entry["step"] != i or any(record[k] != row[k] for k in (*IDENTITY_KEYS, "index"))):
            raise ValueError("Changed propagation checkpoint, map or diagnostic")
        previous = checkpoint["payload_sha256"]
    raw = recovery.RecoveryAtlas.load(directory / f"atlas-{direction}.json")
    if raw.payload["frames"] != sorted(result["records"], key=identity):
        raise ValueError("Propagation atlas differs from immutable frame maps")
    original = recovery.RecoveryAtlas.load(parent)
    if {k: v for k, v in raw.payload.items() if k not in ("frames", "provenance")} != {
            k: v for k, v in original.payload.items() if k not in ("frames", "provenance")}:
        raise ValueError("Propagation changed the fixed parent field model")
    return raw, declaration


def correction_bounds(atlas, record, transform, grid, max_px=12.):
    before = recovery.RecoveryFrame(atlas, record).project(grid)
    candidate = recovery.RecoveryFrame(atlas, dict(record, reference_to_native=transform.tolist()))
    after = candidate.project(grid)
    support = atlas.support_mask(grid)
    visible = lambda p: np.isfinite(p).all(axis=1) & (p >= 0).all(axis=1) & (p < record["native_size"]).all(axis=1)
    chosen = support & (visible(before) | visible(after))
    distance = np.linalg.norm(after[chosen] - before[chosen], axis=1)
    summary = recovery.error_summary(distance)
    valid = bool(len(distance) and np.isfinite(distance).all() and distance.max() <= max_px)
    return dict(**summary, limit_px=max_px, valid=valid,
                domain="union of raw/candidate visible observed-support world-grid points")


def bounded_correction(atlas, records, observations, times, boundary_index, grid):
    """Retain the boundary gauge and fast motion; reject gaps and excessive moves."""
    accumulated = [recovery.matrix(r["reference_to_native"]) for r in records]
    try:
        transforms, drift = recovery.smooth_reference_drift(accumulated, observations, times, boundary_index, sigma_s=.75)
    except (ValueError, np.linalg.LinAlgError) as exc:
        transforms = accumulated
        drift = dict(observations=sum(h is not None for h in observations),
                     unsupported_indices=list(range(len(records))), failure=str(exc))
    valid = [i for i, h in enumerate(observations) if h is not None]
    corrected, statuses = [], []
    for i, (record, transform) in enumerate(zip(records, transforms)):
        before = [j for j in valid if times[j] <= times[i]]
        after = [j for j in valid if times[j] >= times[i]]
        bracket = bool(before and after and times[after[0]] - times[before[-1]] <= 1.1)
        bounds = correction_bounds(atlas, record, transform, grid)
        reasons = []
        if atlas.projection_mode == recovery.V7_PROJECTION_MODE:
            if {c["chart"] for c in atlas.charts} != {"left", "mid", "right"}:
                reasons.append("incomplete_v7_spatial_charts")
            if atlas.payload.get("spatial_boundary") is None or atlas.payload.get("temporal_boundary") is None:
                reasons.append("missing_v7_boundary_fit")
        for name in ("temporal_boundary", "near_temporal_boundary"):
            model = atlas.payload.get(name)
            if model is not None and not model["domain_s"][0] <= times[i] <= model["domain_s"][1]:
                reasons.append(f"outside_{name}_fit_support")
        if i in drift["unsupported_indices"] or not bracket:
            reasons.append("missing_temporal_registration_support")
        if not bounds["valid"]:
            reasons.append("correction_exceeds_12px_or_has_no_finite_visible_support")
        candidate = deepcopy(record)
        if not reasons and i != boundary_index:
            candidate["reference_to_native"] = transform.tolist()
        geometry = recovery.visible_geometry_check(recovery.RecoveryFrame(atlas, candidate), grid)
        if not geometry["valid"] or geometry["nonfinite_supported_projections"]:
            reasons.append("invalid_or_absent_visible_supported_geometry")
        inherited_failures = [w for w in record.get("warnings", []) if w not in (
            "propagated_without_paint_reference", "boundary_offset_carried_from_anchor")]
        if inherited_failures:
            reasons.append("inherited_mapping_warning")
        candidate_geometry = geometry
        if reasons:
            candidate["reference_to_native"] = deepcopy(record["reference_to_native"])
            geometry = recovery.visible_geometry_check(recovery.RecoveryFrame(atlas, candidate), grid)
        candidate["warnings"] = list(dict.fromkeys(record.get("warnings", []) + reasons))
        corrected.append(candidate)
        statuses.append(dict(frame_index=record["index"], **{k: record[k] for k in IDENTITY_KEYS},
            source_seconds=times[i], supported=not reasons, reasons=reasons,
            correction_applied=not reasons and i != boundary_index, bounds=bounds, geometry=geometry,
            candidate_geometry=candidate_geometry,
            inherited_warnings=record.get("warnings", [])))
    # Exact equality matters here: averaging matrices must never nudge the join.
    corrected[boundary_index] = deepcopy(records[boundary_index])
    return corrected, statuses, drift


def unsupported_intervals(rows):
    intervals = []
    for i, row in enumerate(rows):
        if row["supported"]:
            continue
        if i == 0 or rows[i - 1]["supported"]:
            intervals.append(dict(first_frame_index=row["frame_index"], first_source_pts=row["source_pts"],
                                  source_time_base=row["source_time_base"], reasons=[]))
        intervals[-1].update(last_frame_index=row["frame_index"], last_source_pts=row["source_pts"],
                             reasons=sorted(set(intervals[-1]["reasons"]) | set(row["reasons"])))
    return intervals


def run(atlas_path, frames_path, propagation_path, extension_path, split_path, plan_path, catalog_path,
        direction, output, *, references_path=None, turf_profile="green_v1", render=False):
    started = time.monotonic()
    if direction not in ("forward", "backward") or turf_profile not in ("green_v1", "warm_green_v1"):
        raise ValueError("Invalid direction or turf profile")
    atlas_path, frames_path, propagation_path, extension_path, split_path, plan_path, catalog_path, output = map(
        Path, (atlas_path, frames_path, propagation_path, extension_path, split_path, plan_path, catalog_path, output))
    inputs = [atlas_path, frames_path, extension_path, split_path, plan_path, catalog_path,
              *sorted(propagation_path.glob("*.json")), *sorted((propagation_path / f"progress-{direction}").glob("*.json"))]
    if references_path:
        inputs.append(Path(references_path))
    hashes = {str(p.resolve()): sha256_file(p) for p in inputs}
    parent = recovery.RecoveryAtlas.load(atlas_path)
    manifest, extension = [json.loads(p.read_text()) for p in (frames_path, extension_path)]
    plan = load_plan(plan_path, catalog_path)
    game = next(g for g in plan["payload"]["games"] if g["game_id"] == parent.payload["source"]["game_id"])
    if parent.payload["source"]["sha256"] != game["source_sha256"]:
        raise ValueError("Parent source differs from the frozen plan")
    parent_rows = validate_manifest(manifest, game, manifest["requested_interval_s"], 10)
    rows = validate_manifest(extension, game, extension["requested_interval_s"], 10)
    for row in parent_rows:
        record = parent.frames[identity(row)]
        if any(record[k] != row[k] for k in IDENTITY_KEYS):
            raise ValueError("Parent atlas/source still mismatch")
    if (direction == "forward" and identity(rows[0]) <= identity(parent_rows[-1])) or (
            direction == "backward" and identity(rows[-1]) >= identity(parent_rows[0])):
        raise ValueError("Extension lies on the wrong side of its parent")
    catalog = next(g for g in json.loads(catalog_path.read_text())["games"] if g["game_id"] == game["game_id"])
    for window in (manifest["requested_interval_s"], extension["requested_interval_s"]):
        guard_interval(game, window[0] - .1, window[1] + .1)
    if not any(h["start_pts_s"] <= min(manifest["requested_interval_s"][0], extension["requested_interval_s"][0]) - .1
               and max(manifest["requested_interval_s"][1], extension["requested_interval_s"][1]) + .1 <= h["end_pts_s"]
               for h in catalog["halves"]):
        raise ValueError("Parent/extension decoder padding crosses a recorded half boundary")
    split = load_split(split_path, extension, extension_path)
    raw, declaration = validate_completed_run(propagation_path, direction, atlas_path, frames_path,
                                               extension_path, rows, plan_path, catalog_path)
    if extension["requested_interval_s"] != declaration["windows"][direction]:
        raise ValueError("Extension differs from the frozen propagation window")
    refs = []
    if references_path:
        reviewed = frozen_payload(references_path, "field-recovery-reanchor-references-v1")
        if (reviewed["source_sha256"] != game["source_sha256"]
                or reviewed["frames_manifest_sha256"] != sha256_file(extension_path)
                or reviewed["split_sha256"] != sha256_file(split_path)):
            raise ValueError("Reviewed reference/source/split mismatch")
        refs = reviewed["references"]
    row_by_index = {r["index"]: r for r in rows}
    fit_indices = {r["index"] for r in split["frames"] if r["role"] == "fit"}
    if len({r["frame_index"] for r in refs}) != len(refs):
        raise ValueError("Duplicate reviewed reference")
    for ref in refs:
        if ref["frame_index"] not in fit_indices or ref["review_status"] != "source_reviewed":
            raise ValueError("Reference must be source-reviewed fitting evidence")
        row = row_by_index[ref["frame_index"]]
        if any(ref[k] != row[k] for k in IDENTITY_KEYS):
            raise ValueError("Reviewed reference exact still mismatch")
        if not ref["required_labels"] or len({f["feature_id"] for f in ref["features"]}) != len(ref["features"]):
            raise ValueError("Reference needs required markings and unique paint portions")
        for feature in ref["features"]:
            if not feature["points_native"]:
                continue  # Missing paint remains a failed reference check below.
            pixels = recovery.points(feature["points_native"])
            if np.any(pixels < 0) or np.any(pixels >= row["native_size"]):
                raise ValueError("Reference paint must use native in-image pixels")
    hashes.update({str(Path(e["path"]).resolve()): e["sha256"] for e in split.get("inherited_splits", [])})
    output.mkdir(parents=True, exist_ok=False)
    helpers = sorted(Path(recovery.__file__).parent.rglob("*.py"))
    freeze = output / "implementation"
    freeze.mkdir()
    for path in helpers:
        relative = path.relative_to(Path(recovery.__file__).parent)
        target = freeze / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    atomic_json(output / "declaration.json", dict(schema="field-recovery-reanchor-run-v1", inputs=hashes,
        implementation_sha256={str(p): sha256_file(p) for p in helpers},
        environment=dict(python=platform.python_version(), opencv=cv2.__version__, numpy=np.__version__, scipy=scipy.__version__,
                         blas_threads={k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}),
        direction=direction, turf_profile=turf_profile, registration_width=1920, sigma_s=.75,
        max_displacement_native_px=12., maximum_connection_bridges=1, maximum_observation_gap_s=1.1,
        transform_direction="reference pixels to current native pixels", metric_certified=False), exclusive=True)
    cv2.setNumThreads(4)
    # Every chart is already expressed in the atlas's shared reference plane.
    # Use the measured midfield view when available; its saved transform need
    # not be identity, so compose it after each gauge-to-image registration.
    gauge_chart = next((c for c in parent.charts if c["chart"] == "mid"), parent.charts[0])
    gauge_row = next(r for r in parent_rows if r["index"] == gauge_chart["reference_frame_index"])
    gauge_transform = recovery.matrix(parent.frames[identity(gauge_row)]["reference_to_native"])
    # The reused helpers rank nearest anchors by ID distance. Use source-time
    # offsets so backward order and the gap to the parent gauge stay meaningful.
    views = {round((identity(r) - identity(gauge_row)) * 1_000_000_000): r
             for r in [gauge_row] + [r for r in rows if r["index"] in fit_indices]}
    targets = sorted(views)
    positions = {r["index"]: i for i, r in views.items() if i != 0}
    paint = [positions[r["frame_index"]] for r in refs]
    features = {}
    for i, row in views.items():
        cv2.setRNGSeed(0)
        features[i] = propagation.reference_features(read_image(frames_path.parent if i == 0 else extension_path.parent, row),
                                                       turf_profile=turf_profile, max_width=1920)

    def register(a, b):
        cv2.setRNGSeed(0)
        return recovery.register_reference(features[a], features[b], views[b]["native_size"])

    connected, paths, attempts = independent_reference_connections(0, paint, targets, register)
    xx, yy = np.meshgrid(np.linspace(-recovery.HALF_L, recovery.HALF_L, 181),
                          np.linspace(-recovery.HALF_W, recovery.HALF_W, 107))
    grid = np.c_[xx.ravel(), yy.ravel()]
    checked_refs, approved = [], [0]
    for ref in refs:
        i = positions[ref["frame_index"]]
        scores = []
        geometry = None
        record = deepcopy(raw.frames[identity(views[i])])
        if i in connected:
            record["reference_to_native"] = recovery.normalize_h(connected[i] @ gauge_transform).tolist()
            mapping = recovery.RecoveryFrame(parent, record)
            geometry = recovery.visible_geometry_check(mapping, grid)
            for feature in ref["features"]:
                pixels = recovery.points(feature["points_native"]) if feature["points_native"] else np.empty((0, 2))
                errors, _ = marking_errors(mapping, feature["label"], pixels)
                summary = recovery.error_summary(errors)
                scores.append(dict(feature_id=feature["feature_id"], label=feature["label"], **summary,
                    passed=summary["samples"] >= 3 and summary["finite_samples"] == summary["samples"]
                    and summary["median_px"] <= 3 and summary["p95_px"] <= 6))
        labels = {s["label"] for s in scores if s["passed"]}
        ok = (bool(scores) and all(s["passed"] for s in scores) and set(ref["required_labels"]) <= labels
              and geometry["valid"] and not geometry["nonfinite_supported_projections"])
        checked_refs.append(dict(frame_index=ref["frame_index"], connection_id=i, connected=i in connected,
                                 approved=ok, paint=scores, geometry=geometry, required_labels=ref["required_labels"]))
        if ok:
            approved.append(i)
    observations, drift_paths, drift_attempts = independent_drift_observations(0, approved, targets, connected, paths, register)
    # Failed reviewed anchors cannot quietly count as direct correction observations.
    for ref in checked_refs:
        if not ref["approved"]:
            observations.pop(ref["connection_id"], None)
    connections = dict(views=[dict(connection_id=i, manifest="parent" if i == 0 else "extension", **row)
                              for i, row in views.items()], paths=paths, attempts=attempts,
        reference_paint_checks=checked_refs, correction_paths=drift_paths, correction_attempts=drift_attempts,
        transforms={i: h.tolist() for i, h in connected.items()},
        observation_role="fitting evidence, never independent accuracy checks")
    connections["gauge"] = dict(chart=gauge_chart["chart"], frame_index=gauge_row["index"],
                               reference_to_native=gauge_transform.tolist())
    atomic_json(output / "reference-connections.json", connections, exclusive=True)
    boundary_row = parent_rows[-1] if direction == "forward" else parent_rows[0]
    boundary_record = parent.frames[identity(boundary_row)]
    records = sorted(raw.payload["frames"] + [boundary_record], key=identity)
    boundary_index = next(i for i, r in enumerate(records) if identity(r) == identity(boundary_row))
    by_time = {identity(views[i]): recovery.normalize_h(h @ gauge_transform) for i, h in observations.items()}
    by_time[identity(boundary_row)] = recovery.matrix(boundary_record["reference_to_native"])
    corrected, statuses, drift = bounded_correction(parent, records, [by_time.get(identity(r)) for r in records],
                                                    [float(identity(r)) for r in records], boundary_index, grid)
    payload = deepcopy(parent.payload)
    payload["frames"] = [r for i, r in enumerate(corrected) if i != boundary_index]
    payload["provenance"] = dict(parent_atlas_sha256=sha256_file(atlas_path), parent_frames_sha256=sha256_file(frames_path),
        parent_provenance=deepcopy(parent.payload.get("provenance", {})), raw_propagation_atlas_sha256=sha256_file(propagation_path / f"atlas-{direction}.json"),
        split_sha256=sha256_file(split_path), fitting_frames=[dict(source_sha256=game["source_sha256"], **{k: r[k] for k in IDENTITY_KEYS}) for r in views.values()],
        connections_sha256=sha256_file(output / "reference-connections.json"),
        note="Fixed parent model/support; bounded image correction only. Inherited indices refer to the parent manifest.")
    atlas = recovery.RecoveryAtlas.from_payload(payload)
    atlas.save(output / "atlas.json")
    ext_status = [r for i, r in enumerate(statuses) if i != boundary_index]
    transitions = []
    previous_reference = 0
    for i in sorted(observations, key=lambda i: identity(views[i]), reverse=direction == "backward"):
        if i == 0:
            continue
        path = drift_paths[i]["path"]
        anchor = i if i in approved else (path[-2] if len(path) > 1 else 0)
        if anchor != previous_reference:
            frame = views[i]
            check_rows = [r for r in split["frames"] if r["role"] == "check"]
            before = [r for r in check_rows if identity(r) < identity(frame)]
            after = [r for r in check_rows if identity(r) > identity(frame)]
            transitions.append(dict(source_pts=frame["source_pts"], source_time_base=frame["source_time_base"],
                frame_index=frame["index"], from_connection_id=previous_reference, to_connection_id=anchor,
                required_neighbor_check_indices=[before[-1]["index"] if before else None, after[0]["index"] if after else None],
                independent_neighbor_checks_pass=False))
            previous_reference = anchor
    if render:
        previews = output / "comparisons"
        previews.mkdir()
        for i, row in enumerate(rows):
            image = read_image(extension_path.parent, row)
            pair = [recovery.render_frame(image, recovery.RecoveryFrame(a, a.frames[identity(row)])) for a in (raw, atlas)]
            for label, im in zip(("raw propagation", "bounded re-anchor"), pair):
                cv2.putText(im, f"{label}  PTS={row['source_pts']}  {row['source_time_base']}", (15, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
            if not cv2.imwrite(str(previews / f"{i:05d}.jpg"), np.hstack([cv2.resize(im, (960, 540)) for im in pair])):
                raise OSError("Could not save comparison preview")
    for path, expected in hashes.items():
        if sha256_file(path) != expected:
            raise ValueError("Frozen input changed during re-anchoring")
    report = dict(schema="field-recovery-reanchor-report-v1", atlas_sha256=sha256_file(output / "atlas.json"),
        frames_sha256=sha256_file(extension_path), source_sha256=game["source_sha256"], direction=direction,
        frame_count=len(rows), supported_frames=sum(r["supported"] for r in ext_status),
        unsupported_intervals=unsupported_intervals(ext_status), reference_changes=transitions, frames=ext_status,
        drift=drift, boundary_record=boundary_record, boundary_mapping_unchanged=corrected[boundary_index] == boundary_record,
        grid=dict(length_samples=181, width_samples=107, includes_field_boundaries=True),
        independent_paint_qualification="pending separate frozen-plan scoring", metric_certified=False, full_game_accepted=False,
        elapsed_seconds=time.monotonic() - started, parent_inputs_unchanged=True)
    atomic_json(output / "report.json", report, exclusive=True)
    print(json.dumps({k: report[k] for k in ("frame_count", "supported_frames", "elapsed_seconds")}))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("atlas", "frames", "propagation", "extension-frames", "split", "plan", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--catalog", type=Path, default=Path("data/game-sources.json"))
    parser.add_argument("--direction", required=True, choices=("forward", "backward"))
    parser.add_argument("--references", type=Path, help="Frozen source-reviewed fit references; no check paint")
    parser.add_argument("--turf-profile", choices=("green_v1", "warm_green_v1"), default="green_v1")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    run(args.atlas, args.frames, args.propagation, args.extension_frames, args.split, args.plan, args.catalog,
        args.direction, args.out, references_path=args.references, turf_profile=args.turf_profile, render=args.render)


if __name__ == "__main__":
    main()
