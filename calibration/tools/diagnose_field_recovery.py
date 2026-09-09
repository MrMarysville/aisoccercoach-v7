"""Inspect signed paint residuals without changing a frozen score or calibration."""
import argparse
import json
from pathlib import Path

import numpy as np

from calibration.alignment_trial import sha256_file
from calibration.field_atlas import FieldAtlas
from calibration.field_recovery import RecoveryAtlas, feature_curve, points
from calibration.field_marking_check import checked_marking_curve, helper_sha256, validate_orientation_policy


def _nearest_residuals(pixels, curve):
    """Use exactly curve_distance's finite-segment, gap, and first-tie rules."""
    p = points(pixels)
    q = np.asarray(curve, dtype=np.float64)
    a, b = q[:-1], q[1:]
    d = b - a
    lengths = np.sum(d * d, axis=1)
    good = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1) & (lengths <= 100. ** 2)
    indices = np.flatnonzero(good)
    a, d, lengths = a[good], d[good], lengths[good]
    result = []
    for start in range(0, len(p), 64):
        chunk = p[start:start + 64]
        if len(a):
            v = chunk[:, None] - a
            raw_t = np.sum(v * d, axis=2) / np.maximum(lengths, 1e-20)
            t = np.clip(raw_t, 0., 1.)
            residual = v - t[:, :, None] * d
            squared = np.sum(residual ** 2, axis=2)
            nearest = np.argmin(squared, axis=1)
        for i in range(len(chunk)):
            row = dict(nearest_segment_index=None, nearest_native_xy=None,
                       segment_native_xy=None, segment_fraction=None,
                       unclamped_segment_fraction=None, clamped_to_start=None,
                       clamped_to_end=None, nearest_is_endpoint=None,
                       residual_native_dx_px=None, residual_native_dy_px=None,
                       distance_px=None, signed_normal_px=None, tangent_residual_px=None,
                       tangent_native_unit=None, normal_native_unit=None,
                       direction_status="unknown_no_scored_segment")
            if len(a):
                j = nearest[i]
                delta = residual[i, j]
                row.update(nearest_segment_index=int(indices[j]),
                           nearest_native_xy=(a[j] + t[i, j] * d[j]).tolist(),
                           segment_native_xy=[a[j].tolist(), (a[j] + d[j]).tolist()],
                           residual_native_dx_px=float(delta[0]), residual_native_dy_px=float(delta[1]),
                           distance_px=float(np.sqrt(squared[i, j])))
                if lengths[j] > 1e-20:
                    tangent = d[j] / np.sqrt(lengths[j])
                    normal = np.array([-tangent[1], tangent[0]])
                    row.update(segment_fraction=float(t[i, j]),
                               unclamped_segment_fraction=float(raw_t[i, j]),
                               clamped_to_start=bool(raw_t[i, j] < 0),
                               clamped_to_end=bool(raw_t[i, j] > 1),
                               nearest_is_endpoint=bool(t[i, j] == 0 or t[i, j] == 1),
                               signed_normal_px=float(normal @ delta),
                               tangent_residual_px=float(tangent @ delta),
                               tangent_native_unit=tangent.tolist(), normal_native_unit=normal.tolist(),
                               direction_status="known")
                else:
                    row["direction_status"] = "unknown_degenerate_segment"
            result.append(row)
    return result


def _projected_curve(mapping, label, orientation_policy="central_v1"):
    world = feature_curve(dict(label=label, points_native=[[0, 0]]), samples=8001)
    return checked_marking_curve(mapping, world, orientation_policy)[0]


def _summary(records, key):
    values = [r[key] for r in records if r[key] is not None]
    return dict(known_count=len(values), unknown_count=len(records) - len(values),
                median_px=float(np.median(values)) if values else None,
                min_px=float(np.min(values)) if values else None,
                max_px=float(np.max(values)) if values else None)


def diagnose(atlas_path, frames_path, score_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    paths = dict(atlas=Path(atlas_path), frames=Path(frames_path), score=Path(score_path))
    hashes = {key: sha256_file(path) for key, path in paths.items()}
    document, manifest, frozen = [json.loads(paths[key].read_text()) for key in ("atlas", "frames", "score")]
    if frozen.get("schema") != "field-recovery-independent-score-v1":
        raise ValueError("Expected an existing frozen field-recovery score")
    orientation_policy = frozen.get("orientation_policy", "central_v1")
    validate_orientation_policy(orientation_policy)
    pinned_helper = frozen.get("orientation_helper_sha256")
    if ((pinned_helper is not None and pinned_helper != helper_sha256())
            or (orientation_policy == "domain_boundary_v2" and pinned_helper is None)):
        raise ValueError("Frozen score/orientation helper hash mismatch or missing v2 helper hash")
    for key in ("atlas", "frames"):
        if frozen.get(key + "_sha256") != hashes[key]:
            raise ValueError(f"Frozen score/{key} hash mismatch")
    source_hash = manifest["source"]["sha256"]
    if frozen.get("source_sha256") != source_hash or document["payload"]["source"]["sha256"] != source_hash:
        raise ValueError("Frozen score/atlas/frames source mismatch")
    atlas = (RecoveryAtlas(document) if document["schema"] == "field-recovery-atlas-v1"
             else FieldAtlas(document))
    frames = {r["index"]: r for r in manifest["frames"]}
    if len(frames) != len(manifest["frames"]):
        raise ValueError("Duplicate frame indices")
    rows = []
    max_disagreement = 0.
    for original in frozen["rows"]:
        if type(original["frame_index"]) is not int or type(original["source_pts"]) is not int:
            raise ValueError("Frozen score requires integer frame index and exact source PTS")
        frame = frames[original["frame_index"]]
        if (original["source_pts"] != frame["source_pts"]
                or original["source_time_base"] != frame["source_time_base"]
                or original["source_seconds"] != frame["source_seconds"]):
            raise ValueError("Frozen score/frame exact PTS/time-base mismatch")
        if frame.get("source_sha256", source_hash) != source_hash:
            raise ValueError("Frame/source hash mismatch")
        mapping = atlas.frame(frame["source_pts"], frame["source_time_base"],
                              source_sha256=source_hash, native_size=frame["native_size"])
        samples = original["raw_samples"]
        if len(samples) != original["samples"] or not samples:
            raise ValueError("Frozen raw sample count mismatch or empty group")
        residuals = _nearest_residuals([r["native_xy"] for r in samples],
                                       _projected_curve(mapping, original["label"], orientation_policy))
        enriched = []
        for sample, residual in zip(samples, residuals):
            expected, actual = sample["error_px"], residual["distance_px"]
            available = actual is not None
            if (sample["projection_available"] != available
                    or (expected is not None) != available):
                raise ValueError("Unsigned distance availability disagrees with frozen score")
            if available:
                difference = abs(actual - expected)
                if not np.isfinite(expected) or difference > 1e-7:
                    raise ValueError("Unsigned distance disagrees with frozen score by more than 1e-7 px")
                max_disagreement = max(max_disagreement, difference)
            enriched.append(dict(sample, residual_diagnostic=residual))
        rows.append(dict(original, raw_samples=enriched,
                         signed_normal_summary=_summary(residuals, "signed_normal_px"),
                         tangent_residual_summary=_summary(residuals, "tangent_residual_px"),
                         frame_image_sha256=frame.get("image_sha256")))
    result = dict(schema="field-recovery-signed-residual-diagnostic-v1",
                  source_sha256=source_hash, **{key + "_sha256": value for key, value in hashes.items()},
                  frozen_score_metadata={key: value for key, value in frozen.items() if key != "rows"},
                  orientation_policy=orientation_policy, orientation_helper_sha256=helper_sha256(),
                  rows=rows, marking_frame_count=len(rows), sample_count=sum(len(r["raw_samples"]) for r in rows),
                  max_unsigned_distance_disagreement_px=max_disagreement,
                  convention=dict(residual="native paint minus nearest projected model point",
                                  native_axes="x right, y down",
                                  tangent="follows the named world curve's sample order",
                                  normal="[-tangent_y, tangent_x]; clockwise 90 degrees on screen",
                                  endpoint="clamped endpoint errors retain tangential and Euclidean components",
                                  unknown="missing segments and degenerate tangents have null signed direction"),
                  limits=["Read-only diagnostic; frozen acceptance and scores are preserved without reclassification.",
                          "Sign follows curve enumeration; reversing the curve reverses normal and tangent signs.",
                          "Same 8001 curve samples, positive Jacobian and 100 px segment-gap guards as frozen scorer."])
    if any(sha256_file(path) != hashes[key] for key, path in paths.items()):
        raise ValueError("Diagnostic inputs changed during analysis")
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return {key: result[key] for key in ("marking_frame_count", "sample_count", "max_unsigned_distance_disagreement_px")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.atlas, args.frames, args.score, args.out), indent=2))


if __name__ == "__main__":
    main()
