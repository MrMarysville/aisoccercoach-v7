"""Optional near-boundary development refinement of an immutable source-bound map.

Only declared fit-frame observations estimate remap parameters. Charts, image motion, far
corrections and existing warnings are retained. Temporal gaps remain estimates.
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
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import field_recovery as recovery
from calibration.tools.fit_field_recovery import fitting_frame_bindings


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _near_branch(mapping, world, native_width, delta=0.):
    shifted = world+[0., delta]
    curve = mapping.project(shifted)
    candidates = np.flatnonzero(np.isfinite(curve).all(axis=1) & (curve[:, 0] >= 0) & (curve[:, 0] < native_width))
    p = shifted[candidates]
    epsilon = .001
    dx = (mapping.project(p+[epsilon, 0])-mapping.project(p-[epsilon, 0]))/(2*epsilon)
    dy = (mapping.project(p+[0, epsilon])-mapping.project(p-[0, epsilon]))/(2*epsilon)
    jacobian = dx[:, 0]*dy[:, 1]-dx[:, 1]*dy[:, 0]
    positive = np.isfinite(dx).all(axis=1) & np.isfinite(dy).all(axis=1) & np.isfinite(jacobian) & (jacobian > 0)
    positions = candidates[positive]
    curve = curve[positions]
    info = dict(finite_native_x_curve_samples=len(candidates), positive_orientation_curve_samples=len(positions))
    if (len(curve) < 10 or np.any(np.diff(positions) != 1) or
            np.any(np.diff(curve[:, 0]) <= 0) or np.max(np.diff(curve[:, 0]), initial=0) > 100):
        return None, info
    return curve, info


def fit_native_curves(parent, observations, frames, allowed, weight_profile=recovery.LINEAR_SPATIAL_WEIGHT):
    """Fit one field-y remap parameter against native-y curve residuals."""
    world = np.c_[np.linspace(-recovery.HALF_L, recovery.HALF_L, 8001),
                  np.full(8001, recovery.HALF_W)]
    gx, gy = np.meshgrid(np.linspace(-recovery.HALF_L, recovery.HALF_L, 181), np.linspace(-recovery.HALF_W, recovery.HALF_W, 107))
    parent_grid = np.c_[gx.ravel(), gy.ravel()]
    accepted, rejected = [], []
    for observation in observations:
        index = observation["frame_index"]
        if index not in allowed:
            raise ValueError("Near-boundary fit attempted on a check or undeclared frame")
        row = frames[index]
        mapping = parent.frame(row["source_pts"], row["source_time_base"],
                    source_sha256=parent.payload["source"]["sha256"], native_size=row["native_size"])
        native = recovery.points(observation["points_native"])
        record = dict(frame_index=index, source_pts=row["source_pts"], source_time_base=row["source_time_base"],
                      image_sha256=row["image_sha256"], time_s=float(row["source_pts"]*Fraction(row["source_time_base"])),
                      supplied_samples=len(native), points_native=native.tolist())
        geometry = recovery.visible_geometry_check(mapping, parent_grid)
        if not geometry["valid"]:
            rejected.append(dict(**record, reason="parent visible field has invalid geometry", geometry=geometry))
            continue
        # The predicted line may lie outside image height: recovering its y error
        # is the purpose of this fit. Keep native-x support and certify the local
        # orientation directly, including finite offscreen predictions.
        curve, branch_info = _near_branch(mapping, world, row["native_size"][0])
        record.update(branch_info)
        if curve is None:
            rejected.append(dict(**record, reason="no single finite increasing positive-orientation near branch in native x"))
            continue
        selected = (native[:, 0] >= curve[0, 0]) & (native[:, 0] <= curve[-1, 0])
        record.update(selected_native_indices=np.flatnonzero(selected).tolist(),
                      outside_projected_x_samples=int((~selected).sum()),
                      projected_native_x_range_px=[float(curve[0, 0]), float(curve[-1, 0])])
        selected_points = native[selected]
        predicted = np.interp(selected_points[:, 0], curve[:, 0], curve[:, 1])
        residuals = selected_points[:, 1]-predicted
        offscreen = (predicted < 0) | (predicted >= row["native_size"][1])
        record.update(predicted_native_y_px=predicted.tolist(), raw_native_y_residuals_px=residuals.tolist(),
                      predicted_offscreen_y_samples=int(offscreen.sum()),
                      fit_includes_offscreen_parent_predictions=bool(offscreen.any()))
        if selected.sum() < 10:
            rejected.append(dict(**record, reason="fewer than 10 observations inside projected visible x support"))
            continue
        def residual(parameter):
            shifted_curve, _ = _near_branch(mapping, world, row["native_size"][0], float(parameter[0]))
            if shifted_curve is None:
                return np.full(2*len(selected_points), 1e6)
            predicted_y = np.interp(selected_points[:, 0], shifted_curve[:, 0], shifted_curve[:, 1])
            outside_x = selected_points[:, 0]-np.clip(selected_points[:, 0], shifted_curve[0, 0], shifted_curve[-1, 0])
            return np.r_[selected_points[:, 1]-predicted_y, outside_x]
        # This lower bound follows from 1 + delta*d(weight)/dy > 0;
        # it is a remap-orientation constraint, not a measured distance prior.
        peak_derivative = (1. if weight_profile == recovery.LINEAR_SPATIAL_WEIGHT else 1.5)/(recovery.HALF_W-recovery.BOX18_HALF_W)
        lower = -1./peak_derivative+1e-6
        fit = least_squares(residual, [0.], bounds=([lower], [np.inf]), loss="soft_l1", f_scale=1.5,
                            max_nfev=100, xtol=1e-10, ftol=1e-10, gtol=1e-10)
        delta = float(fit.x[0])
        fitted_curve, _ = _near_branch(mapping, world, row["native_size"][0], delta)
        if (not fit.success or fitted_curve is None or np.any(selected_points[:, 0] < fitted_curve[0, 0]) or
                np.any(selected_points[:, 0] > fitted_curve[-1, 0])):
            rejected.append(dict(**record, reason="field-y fit failed convergence or retained x support", fitted_delta_field_y_m=delta))
            continue
        remaining = residual([delta])[:len(selected_points)]
        accepted.append(dict(**record, samples=len(residuals), raw_median_native_y_residual_px=float(np.median(residuals)),
            fitted_delta_field_y_m=delta, parameter_meaning="field-y remapping parameter, not measured field distance",
            optimizer_success=bool(fit.success), optimizer_nfev=fit.nfev, parameter_lower_bound_m=lower,
            mad_px=float(np.median(abs(remaining))),
            field_y_remap_residuals_px=remaining.tolist(),
            residual_p95_abs_px=float(np.percentile(abs(remaining), 95)),
            residual_max_abs_px=float(np.max(abs(remaining))),
            observed_x_range_px=[float(np.min(selected_points[:, 0])), float(np.max(selected_points[:, 0]))],
            metric="native-y residual to enumerated monotone near curve; fit evidence only"))
    return accepted, rejected


def _near_strip_visible(mapping):
    x, y = np.meshgrid(np.linspace(-recovery.HALF_L, recovery.HALF_L, 181),
                       np.linspace(recovery.BOX18_HALF_W, recovery.HALF_W, 25)[1:])
    q = mapping.project(np.c_[x.ravel(), y.ravel()])
    return bool(np.any(np.isfinite(q).all(axis=1) & (q >= 0).all(axis=1) & (q < mapping.record["native_size"]).all(axis=1)))


def refine(parent_path, frames_path, measurements_path, output, *, render=True,
           weight_profile=recovery.LINEAR_SPATIAL_WEIGHT):
    started = time.monotonic()
    parent_path, frames_path, measurements_path, output = map(Path, (parent_path, frames_path, measurements_path, output))
    manifest = json.loads(frames_path.read_text())
    measurements = json.loads(measurements_path.read_text())
    parent = recovery.RecoveryAtlas.load(parent_path)
    payload = deepcopy(parent.payload)
    if payload.get("near_temporal_boundary") is not None or any(f.get("near_boundary_delta_field_y_m", 0.) for f in payload["frames"]):
        raise ValueError("Near refinement requires a parent without an existing near correction")
    if weight_profile not in (recovery.LINEAR_SPATIAL_WEIGHT, recovery.SMOOTH_SPATIAL_WEIGHT):
        raise ValueError("Unknown near-boundary spatial profile")
    if manifest.get("schema") != "alignment-trial-frames-v1" or measurements.get("schema") not in (
            "field-recovery-near-boundary-measurements-v1", "field-recovery-measurements-v1"):
        raise ValueError("Invalid source or near-boundary measurement schema")
    source_hash = parent.payload["source"]["sha256"]
    if manifest["source"]["sha256"] != source_hash or measurements["source_sha256"] != source_hash:
        raise ValueError("Near-boundary measurement/source mismatch")
    if measurements.get("parent_atlas_sha256") != sha(parent_path):
        raise ValueError("Near refinement must bind the exact parent atlas")
    parent_inputs_hash = payload["provenance"]["measurements_sha256"]
    input_path = measurements.get("parent_measurements_path")
    if (measurements.get("parent_measurements_sha256") != parent_inputs_hash or
            not input_path or sha(input_path) != parent_inputs_hash):
        raise ValueError("Near refinement must bind unchanged parent input bytes")
    parent_inputs = json.loads(Path(input_path).read_text())
    if any(set(measurements[k]) != set(parent_inputs[k]) for k in ("fit_frame_indices", "check_frame_indices")):
        raise ValueError("Near refinement must preserve the frozen parent fit/check split")
    if "references" in measurements and measurements["references"] != parent_inputs.get("references"):
        raise ValueError("Near-only refinement cannot change parent chart inputs")
    saved = {f["index"]:f for f in payload["frames"]}
    rows = {}
    for row in manifest["frames"]:
        i = row["index"]
        if i in rows or i not in saved or row["source_sha256"] != source_hash:
            raise ValueError("Invalid or foreign source frame")
        for key in ("source_pts", "source_time_base", "native_size", "image_sha256"):
            if saved[i][key] != row[key]:
                raise ValueError("Near source frame differs from exact parent frame")
        rows[i] = row
    allowed = set(measurements["fit_frame_indices"])
    checks = set(measurements.get("check_frame_indices", []))
    if not allowed or allowed & checks or not allowed <= set(rows):
        raise ValueError("Near fitting needs declared existing fit frames disjoint from checks")
    raw = measurements.get("near_boundary_observations", measurements.get("boundary_observations", []))
    grouped = {}
    for obs in raw:
        i = obs["frame_index"]
        if i not in allowed or i in checks:
            raise ValueError("Near observation is on a check or undeclared frame")
        row = rows[i]
        if obs.get("source_sha256", source_hash) != source_hash:
            raise ValueError("Near observation source hash mismatch")
        for key in ("source_pts", "source_time_base", "image_sha256"):
            if obs.get(key) != row[key]:
                raise ValueError("Near observation exact source binding missing or mismatched")
        path = (frames_path.parent/row["file"]).resolve()
        if not path.is_relative_to(frames_path.parent.resolve()) or sha(path) != row["image_sha256"]:
            raise ValueError("Near source image path/hash mismatch")
        features = obs.get("features", [obs])
        for feature in features:
            if feature.get("label") != "touch_near":
                raise ValueError("Near-only observations must name touch_near")
            points = recovery.points(feature["points_native"])
            if np.any(points < 0) or np.any(points >= row["native_size"]):
                raise ValueError("Near observation lies outside native source pixels")
            grouped.setdefault(i, []).extend(points.tolist())
    observations = [dict(frame_index=i, points_native=points) for i, points in sorted(grouped.items())]
    fitted, rejected = fit_native_curves(parent, observations, rows, allowed, weight_profile)
    model = None
    if fitted:
        fitted.sort(key=lambda r:r["time_s"])
        model = dict(method=recovery.NEAR_TEMPORAL_METHOD, weight_profile=weight_profile,
                     protected_y_m=recovery.BOX18_HALF_W, time_s=[r["time_s"] for r in fitted],
                     delta_field_y_m=[r["fitted_delta_field_y_m"] for r in fitted],
                     parameter_meaning="field-y remapping parameter, not measured field distance",
                     domain_s=[fitted[0]["time_s"], fitted[-1]["time_s"]],
                     fit_frame_indices=[r["frame_index"] for r in fitted],
                     source_observation_counts=[r["samples"] for r in fitted],
                     interpolation_uncertainty="Between observations, remap parameters are linear estimates; gaps are not observed coverage.")
    payload["near_temporal_boundary"] = model
    support = []
    for frame in payload["frames"]:
        t = float(frame["source_pts"]*Fraction(frame["source_time_base"]))
        frame["near_boundary_delta_field_y_m"] = 0.
        record = dict(frame_index=frame["index"], time_s=t, status="outside_observed_span", bracketing_frame_indices=[], gap_seconds=None)
        if model and model["domain_s"][0] <= t <= model["domain_s"][1]:
            frame["near_boundary_delta_field_y_m"] = recovery.near_temporal_offset(model, t)
            anchor = next((i for i, value in enumerate(model["time_s"]) if value == t), None)
            if anchor is not None:
                record.update(status="parameter_fitted_to_source", bracketing_frame_indices=[model["fit_frame_indices"][anchor]], gap_seconds=0.)
            else:
                right = int(np.searchsorted(model["time_s"], t))
                record.update(status="estimated_linear_interpolation", bracketing_frame_indices=model["fit_frame_indices"][right-1:right+1],
                              gap_seconds=model["time_s"][right]-model["time_s"][right-1])
        else:
            mapping = parent.frame(frame["source_pts"], frame["source_time_base"], source_sha256=source_hash, native_size=frame["native_size"])
            if _near_strip_visible(mapping):
                frame["warnings"].append("visible_near_strip_outside_observed_correction_span")
                record["visible_near_strip_without_correction"] = True
        support.append(record)
    if output.exists():
        raise ValueError("Output must be a new immutable directory")
    output.mkdir(parents=True)
    def write(name, value):
        (output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    declaration = dict(source_sha256=source_hash, parent_atlas_sha256=sha(parent_path), parent_measurements_sha256=parent_inputs_hash,
                       frames_sha256=sha(frames_path), measurements_sha256=sha(measurements_path),
                       core_sha256=sha(recovery.__file__), helper_sha256=sha(__file__),
                       fit_frame_indices=sorted(allowed), weight_profile=weight_profile,
                       changed="optional near field-y remapping parameter and changed-map warnings only",
                       charts_motion_far_boundary="preserved exactly", status="check-informed development hypothesis")
    write("declaration.json", declaration)
    # Add provenance without replacing the parent fit-input binding.
    payload.setdefault("provenance", {})["near_boundary_refinement"] = declaration
    evidence = payload["provenance"].setdefault("fitting_frames", [])
    evidence.extend(row for row in fitting_frame_bindings(manifest, grouped) if row not in evidence)
    atlas = recovery.RecoveryAtlas.from_payload(payload)
    gx, gy = np.meshgrid(np.linspace(-recovery.HALF_L, recovery.HALF_L, 181), np.linspace(-recovery.HALF_W, recovery.HALF_W, 107))
    grid = np.c_[gx.ravel(), gy.ravel()]
    geometry = []
    for frame in payload["frames"]:
        mapping = atlas.frame(frame["source_pts"], frame["source_time_base"], source_sha256=source_hash, native_size=frame["native_size"])
        check = recovery.visible_geometry_check(mapping, grid)
        geometry.append(dict(frame_index=frame["index"], **check))
        if not check["valid"]:
            frame["warnings"].append("invalid_visible_geometry_after_near_boundary_refinement")
    atlas = recovery.RecoveryAtlas.from_payload(payload)
    atlas.save(output/"atlas.json")
    atlas = recovery.RecoveryAtlas.load(output/"atlas.json")
    if render:
        (output/"overlay").mkdir()
        for i, row in enumerate(manifest["frames"]):
            path = (frames_path.parent/row["file"]).resolve()
            if not path.is_relative_to(frames_path.parent.resolve()) or sha(path) != row["image_sha256"]:
                raise ValueError("Render source image path/hash mismatch")
            im = cv2.imread(str(path))
            mapping = atlas.frame(row["source_pts"], row["source_time_base"], source_sha256=source_hash, native_size=row["native_size"])
            cv2.imwrite(str(output/"overlay"/f"{row['index']:05d}.jpg"), recovery.render_frame(im, mapping), [cv2.IMWRITE_JPEG_QUALITY, 92])
    report = dict(accepted=False, status="approximate near-boundary development revision; independent checks pending",
                  declaration=declaration, total_frames=len(payload["frames"]), near_temporal_model=model,
                  source_fit_observations=fitted, rejected_observations=rejected, temporal_support=support, geometry=geometry,
                  frames_with_warnings=sum(bool(f["warnings"]) for f in payload["frames"]),
                  source_fit_parameter_frames=sum(s["status"] == "parameter_fitted_to_source" for s in support),
                  interpolated_frames=sum(s["status"] == "estimated_linear_interpolation" for s in support),
                  outside_observed_span_frames=sum(s["status"] == "outside_observed_span" for s in support),
                  rendered_frame_indices=[r["index"] for r in manifest["frames"]] if render else [],
                  atlas_sha256=sha(output/"atlas.json"), elapsed_seconds=time.monotonic()-started,
                  limitations=["The field-y shift is a mapping parameter, not measured field distance; raw and remaining fit residuals are retained.",
                               "Interpolation across missing observations is estimated, with its time gaps reported.",
                               "Preserved parent warnings cannot be cleared by this boundary-only refinement."])
    write("report.json", report)
    print(json.dumps({k:report[k] for k in ("total_frames", "source_fit_parameter_frames", "interpolated_frames", "frames_with_warnings", "elapsed_seconds")}), flush=True)
    print("NEAR_FIELD_RECOVERY_REFINEMENT_COMPLETE", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "frames", "measurements", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--weight-profile", choices=(recovery.LINEAR_SPATIAL_WEIGHT, recovery.SMOOTH_SPATIAL_WEIGHT), default=recovery.LINEAR_SPATIAL_WEIGHT)
    args = parser.parse_args()
    refine(args.parent, args.frames, args.measurements, args.out, render=not args.skip_render, weight_profile=args.weight_profile)


if __name__ == "__main__":
    main()
