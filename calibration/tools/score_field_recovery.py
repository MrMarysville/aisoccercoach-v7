"""Score a frozen atlas on separately selected source-only native paint evidence."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from calibration.alignment_trial import sha256_file
from calibration.field_atlas import FieldAtlas
from calibration.field_recovery import RecoveryAtlas, error_summary, feature_curve, curve_distance
from calibration.field_marking_check import (ORIENTATION_POLICIES, checked_marking_curve,
                                    helper_sha256, validate_orientation_policy)


def score(atlas_path, frames_path, checks_path, output, orientation_policy="central_v1"):
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
    grouped = {}
    for feature in checks["measurements"]:
        if feature.get("selected_count", 0):
            grouped.setdefault((feature["frame_index"], feature["label"]), []).append(feature)
    rows = []
    all_errors = []
    for (index, label), features in sorted(grouped.items()):
        frame = frames[index]
        mapping = atlas.frame(frame["source_pts"], frame["source_time_base"],
                              source_sha256=manifest["source"]["sha256"], native_size=frame["native_size"])
        pixels = np.concatenate([r["points_native"] for r in features])
        curve_world = feature_curve(dict(label=label, points_native=[[0, 0]]), samples=8001)
        curve, orientation = checked_marking_curve(mapping, curve_world, orientation_policy)
        errors = curve_distance(pixels, curve)
        summary = error_summary(errors)
        qualifies = (summary["finite_samples"] == summary["samples"]
                     and summary["median_px"] <= 3 and summary["p95_px"] <= 6)
        rows.append(dict(frame_index=index, source_pts=frame["source_pts"],
                         source_time_base=frame["source_time_base"], label=label,
                         source_seconds=frame["source_seconds"], **summary,
                         orientation=orientation,
                         experiment_pixel_target_pass=bool(qualifies),
                         measured_native_bounds=[pixels.min(axis=0).tolist(), pixels.max(axis=0).tolist()],
                         searched_samples=sum(r["searched_count"] for r in features),
                         feature_ids=[r["feature_id"] for r in features],
                         raw_samples=[dict(native_xy=p.tolist(), error_px=float(e) if np.isfinite(e) else None,
                                       projection_available=bool(np.isfinite(e))) for p, e in zip(pixels, errors)]))
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
    with Path(output).open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return {k: result[k] for k in ("check_frame_count", "marking_frame_count", "passing_marking_frames",
                                  "all_marking_frames_pass", "pooled_diagnostic", "elapsed_seconds")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--checks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--orientation-policy", choices=ORIENTATION_POLICIES, default="central_v1")
    args = parser.parse_args()
    print(json.dumps(score(args.atlas, args.frames, args.checks, args.out, args.orientation_policy), indent=2))


if __name__ == "__main__":
    main()
