"""Score a frozen atlas on separately selected source-only native paint evidence."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from calibration.alignment_trial import sha256_file, source_time
from calibration.field_atlas import FieldAtlas
from calibration.field_recovery import RecoveryAtlas, digest, error_summary, feature_curve, curve_distance, points
from calibration.field_marking_check import (ORIENTATION_POLICIES, checked_marking_curve,
                                    helper_sha256, validate_orientation_policy)


def marking_errors(mapping, label, pixels, orientation_policy="central_v1"):
    world = feature_curve(dict(label=label, points_native=[[0, 0]]), samples=8001)
    curve, orientation = checked_marking_curve(mapping, world, orientation_policy)
    return curve_distance(pixels, curve), orientation


def load_check_plan(path, manifest, frames_path, atlas):
    """A frozen denominator is bound to exact source frames, never just indices."""
    document = json.loads(Path(path).read_text())
    plan = document["payload"]
    if (document.get("schema") != "field-recovery-check-plan-v1"
            or document.get("payload_sha256") != digest(plan)
            or plan["source_sha256"] != manifest["source"]["sha256"]
            or plan["frames_manifest_sha256"] != sha256_file(frames_path)):
        raise ValueError("Frozen check plan/source/manifest mismatch")
    frames = {r["index"]: r for r in manifest["frames"]}
    fit, check = set(plan["fit_frame_indices"]), set(plan["check_frame_indices"])
    if fit & check or fit | check != set(frames):
        raise ValueError("Check plan must preserve a complete disjoint frame split")
    # The atlas stores fitting images by source identity. Image-registration
    # residuals and paint from those images cannot reappear as independent checks.
    evidence = atlas.payload.get("provenance", {}).get("fitting_frames", [])
    fitted = {source_time(r["source_pts"], r["source_time_base"]) for r in evidence}
    groups = {}
    for group in plan["groups"]:
        index, label = group["frame_index"], group["label"]
        key = index, label
        if key in groups or index not in check:
            raise ValueError("Duplicate check group or fitting frame used as a check")
        row = frames[index]
        if any(group[k] != row[k] for k in ("source_pts", "source_time_base", "image_sha256", "native_size")):
            raise ValueError("Check group does not match its exact source still")
        if source_time(row["source_pts"], row["source_time_base"]) in fitted:
            raise ValueError("Registration fitting evidence cannot be an independent paint check")
        feature_curve(dict(label=label, points_native=[[0, 0]]), samples=2)
        if len(set(group["feature_ids"])) != len(group["feature_ids"]):
            raise ValueError("Duplicate required paint portion")
        groups[key] = group
    if not groups:
        raise ValueError("Frozen check plan needs required marking/frame groups")
    return groups


def score(atlas_path, frames_path, checks_path, output, orientation_policy="central_v1", *, check_plan=None):
    validate_orientation_policy(orientation_policy)
    started = time.monotonic()
    document = json.loads(Path(atlas_path).read_text())
    atlas = (RecoveryAtlas(document) if document["schema"] == "field-recovery-atlas-v1"
             else FieldAtlas(document))
    manifest = json.loads(Path(frames_path).read_text())
    checks = json.loads(Path(checks_path).read_text())
    if checks["source_sha256"] != manifest["source"]["sha256"]:
        raise ValueError("Independent checks/source mismatch")
    frames = {r["index"]: r for r in manifest["frames"]}
    required = load_check_plan(check_plan, manifest, frames_path, atlas) if check_plan else {}
    if check_plan and checks.get("frames_manifest_sha256") != sha256_file(frames_path):
        raise ValueError("Independent checks/frame-manifest mismatch")
    grouped = {}
    for feature in checks["measurements"]:
        if check_plan or feature.get("selected_count", 0):
            grouped.setdefault((feature["frame_index"], feature["label"]), []).append(feature)
    if check_plan:
        extra = set(grouped) - set(required)
        if extra:
            raise ValueError("Paint groups outside the frozen check plan")
        for key in required:
            grouped.setdefault(key, [])
    rows = []
    all_errors = []
    for (index, label), features in sorted(grouped.items()):
        frame = frames[index]
        missing = []
        if check_plan:
            group = required[index, label]
            ids = [r["feature_id"] for r in features]
            if len(set(ids)) != len(ids) or set(ids) - set(group["feature_ids"]):
                raise ValueError("Unexpected or duplicate measured paint portion")
            missing = sorted(set(group["feature_ids"]) - set(ids))
            for feature in features:
                count = feature.get("selected_count", 0)
                if count != len(feature.get("points_native", [])):
                    raise ValueError("Paint selected count differs from saved points")
                if count:
                    pixels = points(feature["points_native"])
                    if np.any(pixels < 0) or np.any(pixels >= frame["native_size"]):
                        raise ValueError("Independent paint must use native in-image pixels")
                if count < 3 or feature.get("review_status") != "source_reviewed":
                    missing.append(feature["feature_id"])
        selected = [r for r in features if r.get("selected_count", 0)]
        pixels = np.concatenate([r["points_native"] for r in selected]) if selected else np.empty((0, 2))
        try:
            mapping = atlas.frame(frame["source_pts"], frame["source_time_base"],
                                  source_sha256=manifest["source"]["sha256"], native_size=frame["native_size"])
        except ValueError:
            if not check_plan:
                raise
            mapping = None
        if check_plan and mapping is not None and mapping.record.get("image_sha256") != frame["image_sha256"]:
            raise ValueError("Scored mapping belongs to a different exact source still")
        if mapping is None or not len(pixels):
            errors, orientation = np.full(len(pixels), np.inf), None
        else:
            errors, orientation = marking_errors(mapping, label, pixels, orientation_policy)
        summary = error_summary(errors)
        qualifies = (bool(summary["samples"]) and summary["finite_samples"] == summary["samples"]
                     and summary["median_px"] <= 3 and summary["p95_px"] <= 6)
        if check_plan:
            qualifies = qualifies and not missing and bool(group["feature_ids"]) and group["observation_status"] == "visible"
        rows.append(dict(frame_index=index, source_pts=frame["source_pts"],
                         source_time_base=frame["source_time_base"], label=label,
                         source_seconds=frame["source_seconds"], **summary,
                         orientation=orientation,
                         experiment_pixel_target_pass=bool(qualifies),
                         measured_native_bounds=[pixels.min(axis=0).tolist(), pixels.max(axis=0).tolist()] if len(pixels) else None,
                         searched_samples=sum(r.get("searched_count", 0) for r in features),
                         feature_ids=[r["feature_id"] for r in features],
                         raw_samples=[dict(native_xy=p.tolist(), error_px=float(e) if np.isfinite(e) else None,
                                       projection_available=bool(np.isfinite(e))) for p, e in zip(pixels, errors)]))
        if check_plan:
            rows[-1].update(required=True, observation_status=group["observation_status"],
                            missing_or_unmeasured_feature_ids=missing, mapping_available=mapping is not None,
                            evidence_complete=not missing and bool(group["feature_ids"])
                            and group["observation_status"] == "visible")
        all_errors.extend(errors)
    result = dict(schema="field-recovery-independent-score-v1", accepted=False,
                  source_sha256=manifest["source"]["sha256"],
                  atlas_sha256=sha256_file(atlas_path), checks_sha256=sha256_file(checks_path),
                  frames_sha256=sha256_file(frames_path),
                  orientation_policy=orientation_policy, orientation_helper_sha256=helper_sha256(),
                  check_frame_count=len({r["frame_index"] for r in rows}),
                  marking_frame_count=len(rows),
                  passing_marking_frames=sum(r["experiment_pixel_target_pass"] for r in rows),
                  all_marking_frames_pass=bool(rows) and all(r["experiment_pixel_target_pass"] for r in rows),
                  pooled_diagnostic=error_summary(all_errors), rows=rows,
                  elapsed_seconds=time.monotonic() - started,
                  limits=["Source-only image evidence on development check frames, not a physical metric certificate.",
                          "Raw points and missing projections retained; pooled error does not replace per-marking checks.",
                          "The current candidate retains motion/support warnings independently of paint scores."])
    if check_plan:
        result.update(check_plan_sha256=sha256_file(check_plan), required_marking_frames=len(required),
                      missing_or_unmeasured_marking_frames=sum(not r["evidence_complete"] for r in rows),
                      metric_certified=False, full_game_accepted=False)
    with Path(output).open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return {k: result[k] for k in ("check_frame_count", "marking_frame_count", "passing_marking_frames",
                                  "all_marking_frames_pass", "pooled_diagnostic", "elapsed_seconds")}


def qualification_receipt(score_path, report_path, output, *, baseline_path=None,
                          visual_review_path=None, equivalence_path=None):
    """Combine separately frozen evidence; missing review/equivalence never passes.

    This is a development pixel qualification only. The individual input files
    remain inspectable and hash-bound; no pooled score replaces a required group.
    """
    scored, report = [json.loads(Path(p).read_text()) for p in (score_path, report_path)]
    if (not scored.get("check_plan_sha256") or scored.get("required_marking_frames") != len(scored["rows"])
            or any(scored[k] != report[k] for k in ("atlas_sha256", "frames_sha256", "source_sha256"))):
        raise ValueError("Qualification requires matching frozen-plan scores and re-anchor evidence")
    frames = report["frames"]
    if len(frames) != report["frame_count"] or len({(r["source_pts"], r["source_time_base"]) for r in frames}) != len(frames):
        raise ValueError("Qualification report omits or duplicates sampled frames")
    paint_pass = bool(scored["rows"]) and all(row["experiment_pixel_target_pass"] and row["evidence_complete"]
        and row["samples"] > 0 and row["finite_samples"] == row["samples"]
        and row["median_px"] <= 3 and row["p95_px"] <= 6 for row in scored["rows"])
    groups = {}
    for row in scored["rows"]:
        groups.setdefault(row["frame_index"], []).append(row)
    transitions = []
    for change in report["reference_changes"]:
        neighbors = change["required_neighbor_check_indices"]
        passed = len(neighbors) == 2 and all(i is not None and i in groups
            and all(r["experiment_pixel_target_pass"] and r["evidence_complete"] for r in groups[i]) for i in neighbors)
        transitions.append(dict(change, independent_neighbor_checks_pass=passed))
    paths = [Path(score_path), Path(report_path)]
    optional = {}
    for name, path in (("visual_review", visual_review_path), ("equivalence", equivalence_path)):
        optional[name] = json.loads(Path(path).read_text()) if path else None
        if path:
            paths.append(Path(path))
            if any(optional[name].get(k) != report[k] for k in ("atlas_sha256", "frames_sha256", "source_sha256")):
                raise ValueError(f"{name} belongs to another candidate or source manifest")
    review, equivalent = optional["visual_review"], optional["equivalence"]
    review_pass = bool(review and review.get("full_sequence_reviewed") is True and review.get("passed") is True
        and review.get("reviewed_frame_count") == report["frame_count"])
    equivalence_pass = bool(equivalent and equivalent.get("raw_maps_equal") is True
        and equivalent.get("corrected_maps_equal") is True and equivalent.get("parent_artifacts_unchanged") is True)
    supported = [r for r in frames if r["supported"]]
    geometry_pass = bool(supported) and all(r["geometry"]["valid"] and not r["geometry"]["nonfinite_supported_projections"]
        and r["bounds"]["valid"] and r["bounds"]["max_px"] <= 12 for r in supported)
    gates = dict(required_paint_pass=paint_pass, full_sampled_support=len(supported) == len(frames),
        supported_geometry_pass=geometry_pass, transition_paint_pass=all(t["independent_neighbor_checks_pass"] for t in transitions),
        full_sequence_visual_review_pass=review_pass, resume_equivalence_pass=equivalence_pass,
        parent_artifacts_unchanged=report["parent_inputs_unchanged"] is True,
        boundary_mapping_unchanged=report["boundary_mapping_unchanged"] is True)
    baseline = None
    if baseline_path:
        paths.append(Path(baseline_path))
        baseline = json.loads(Path(baseline_path).read_text())
        if any(baseline[k] != scored[k] for k in ("source_sha256", "frames_sha256", "checks_sha256", "check_plan_sha256")):
            raise ValueError("Baseline and correction must use identical independent evidence")
        baseline = {k: baseline[k] for k in ("atlas_sha256", "passing_marking_frames", "marking_frame_count", "pooled_diagnostic")}
    receipt = dict(schema="field-recovery-reanchor-qualification-v1", pixel_qualified=all(gates.values()), gates=gates,
        metric_certified=False, full_game_accepted=False,
        **{k: report[k] for k in ("atlas_sha256", "frames_sha256", "source_sha256", "direction", "frame_count", "unsupported_intervals")},
        supported_frames=len(supported), reference_changes=transitions, baseline=baseline,
        checks={k: scored[k] for k in ("required_marking_frames", "passing_marking_frames", "missing_or_unmeasured_marking_frames", "pooled_diagnostic")},
        measured_runtime_seconds=dict(reanchoring=report["elapsed_seconds"], scoring=scored["elapsed_seconds"]),
        inputs={str(p.resolve()): sha256_file(p) for p in paths},
        limits=["Sampled pixel qualification is separate from physical metric accuracy and full-game acceptance.",
                "Source visibility, rejected samples and missing required groups remain in the frozen evidence."])
    with Path(output).open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--checks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--orientation-policy", choices=ORIENTATION_POLICIES, default="central_v1")
    parser.add_argument("--check-plan", type=Path, help="Frozen required paint groups; missing evidence fails")
    args = parser.parse_args()
    print(json.dumps(score(args.atlas, args.frames, args.checks, args.out, args.orientation_policy,
                           check_plan=args.check_plan), indent=2))


if __name__ == "__main__":
    main()
