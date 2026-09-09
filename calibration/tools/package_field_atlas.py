"""Convert the frozen v7 experiment to a portable, data-only field atlas."""
import argparse
import hashlib
import json
from pathlib import Path

from scipy.interpolate import BSpline

from calibration.field_atlas import save


def _read(path):
    return json.loads(path.read_text())


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package(preview_dir, output):
    preview = Path(preview_dir)
    base = preview.parent
    bindings = _read(preview / "artifact-bindings.json")
    required = ("source-selection.json", "temporal-boundary-model.json",
                "all-transition-checks.json", "dense-boundary-check.json")
    for name in required:
        if _sha(preview / name) != bindings["files"][name]:
            raise ValueError(f"Frozen preview changed: {name}")
    v5_path = base / "rough-motion-v5/atlas.json"
    v6_path = base / "rough-motion-v6/boundary-model.json"
    for path in (v5_path, v6_path):
        if _sha(path) != bindings["external_inputs"][str(path)]:
            raise ValueError(f"Frozen input changed: {path.name}")
    v5, v6 = _read(v5_path), _read(v6_path)
    timing = _read(preview / "source-selection.json")
    model = _read(preview / "temporal-boundary-model.json")
    motion = _read(preview / "all-transition-checks.json")
    if v5["source"]["sha256"] != timing["source"]["sha256"]:
        raise ValueError("Source mismatch between geometry and decoded frames")
    if len(v5["frames"]) != len(timing["frames"]) or timing["count"] != len(timing["frames"]):
        raise ValueError("Geometry and decoded frame counts differ")
    spline = model["spline"]
    evaluate = BSpline(spline["t"], spline["c"], spline["k"], extrapolate=False)
    flags = {r["index"]: r for r in motion["rows"] if not r["diagnostic_pass"]}
    frames = []
    for index, (geometry, source) in enumerate(zip(v5["frames"], timing["frames"])):
        if (source["index"] != index or geometry["time_s"] != source["nominal_time_s"]
                or source["scaled_matches_v6_byte_exact"] is not True):
            raise ValueError("Frame alignment or source-pixel proof mismatch")
        frames.append(dict(
            source_pts=source["source_pts"], source_time_base=source["source_time_base"],
            native_size=[timing["crop"]["source_width"], timing["crop"]["source_height"]],
            reference_to_native=geometry["T_reference_pixels_to_native_pixels"],
            boundary_offset_native_y_px=float(evaluate(geometry["time_s"] - spline["domain_nominal_s"][0])),
            warnings=["scene_motion_diagnostic_failed"] if index in flags else [],
            decoded_yuv_checksum=source["decoded_yuv_checksum"],
            decoded_plane_checksums=source["decoded_plane_checksums"],
            preview_source=dict(index=index, size=[1280, 720], sha256=geometry["source_frame_sha256"])))
    payload = dict(
        status="approximate", metric_certified=False,
        source={k: timing["source"][k] for k in ("game_id", "sha256")},
        field=dict(length_m=109.728, width_m=64.008, axes="centre,+x_camera_right,+y_near",
                   dimensions_provenance="Owner supplied 120 x 70 yd outer dimensions",
                   interior_provenance="Standard-yard markings assumed; physical sizes unverified",
                   markings=dict(penalty_depth_m=16.4592, penalty_half_width_m=20.1168,
                                 goal_area_depth_m=5.4864, goal_area_half_width_m=9.144,
                                 penalty_spot_depth_m=10.9728, circle_radius_m=9.144)),
        spatial=dict(
            field_to_reference_models=v5["projection_models_field_to_reference_pixels"],
            blend=dict(end_x_start_m=10, end_x_span_m=28, near_y_start_m=15,
                       near_y_span_m=15, near_right_x_start_m=0, near_right_x_span_m=15),
            far_boundary=dict(reference_y_coefficients=v6["reference_boundary"]["coefficients"],
                              reference_x_origin_px=800, reference_x_scale_px=1000,
                              protected_y_m=-20.1168)),
        frames=frames,
        provenance=dict(
            converter="v7 experiment migration; estimates are not refitted",
            geometry_sha256=_sha(v5_path), boundary_sha256=_sha(v6_path),
            preview_bindings_sha256=_sha(preview / "artifact-bindings.json"),
            timing_sha256=_sha(preview / "source-selection.json"),
            temporal_fit_sha256=_sha(preview / "temporal-boundary-model.json"),
            status="Owner accepts small corner offsets for further approximate development"),
        limitations=["This is a fitted development clip, not a new-game calibrator.",
                     "Only the listed source PTS have a mapping; no time interpolation.",
                     "Metric accuracy and uncertainty are unverified.",
                     "Frame warnings withhold public positions; raw projection remains available for review."])
    save(output, payload)
    return dict(output=str(output), frames=len(frames), review_required_frames=len(flags),
                metric_certified=False, source=payload["source"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preview_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(package(args.preview_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
