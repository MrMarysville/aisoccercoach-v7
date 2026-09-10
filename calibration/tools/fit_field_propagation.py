"""Refine a fresh end reference using connected source-fit views, never check paint.

The parent is an immutable single-end run from fit_field_recovery. Every new
image connection is measured and checked. A shared polynomial residual retains
the end and midfield paint in one solve; saved maps remain approximate.
"""
import argparse
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.spatial import cKDTree

from calibration import field_propagation as propagation
from calibration import field_recovery as recovery
from calibration.tools.fit_field_recovery import (independent_drift_observations,
                                         independent_reference_connections, image_anchor_positions,
                                         fitting_frame_bindings, load_inputs)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(parent_path, parent_measurements_path, frames_path, measurements_path, output, *,
        turf_profile="green_v1", motion_width=1920, reference_stride=10,
        degree="auto", render=False):
    started = time.monotonic()
    parent_path, parent_measurements_path, frames_path, measurements_path, output = map(
        Path, (parent_path, parent_measurements_path, frames_path, measurements_path, output))
    manifest, measurements, by_index = load_inputs(frames_path, measurements_path)
    parent = recovery.RecoveryAtlas.load(parent_path)
    if (parent.projection_mode != recovery.SINGLE_REFERENCE_PROJECTION_MODE or
            parent.payload["source"]["sha256"] != manifest["source"]["sha256"]):
        raise ValueError("Propagation requires a same-source single-end parent")
    if sha(parent_measurements_path) != parent.payload["provenance"]["measurements_sha256"]:
        raise ValueError("Parent measurement checksum mismatch")
    original = json.loads(parent_measurements_path.read_text())
    if any(set(original[role]) != set(measurements[role])
           for role in ("fit_frame_indices", "check_frame_indices")):
        raise ValueError("Parent fit/check frame roles must remain unchanged")
    if (len(original["references"]) != 1 or original["references"][0] not in measurements["references"]
            or original["references"][0]["chart"] not in ("left", "right")):
        raise ValueError("The original observed end must remain unchanged in the shared fit")
    if degree not in ("auto", "2", "3") or reference_stride < 1 or motion_width < 1:
        raise ValueError("Invalid propagation configuration")
    rows = manifest["frames"]
    positions = {row["index"]: i for i, row in enumerate(rows)}
    source_times = {i: row["source_pts"]*Fraction(row["source_time_base"]) for i, row in enumerate(rows)}
    if len(parent.payload["frames"]) != len(rows):
        raise ValueError("Parent frame coverage differs from the source manifest")
    for a, b in zip(parent.payload["frames"], rows):
        if any(a[key] != b[key] for key in ("index", "source_pts", "source_time_base", "native_size", "image_sha256")):
            raise ValueError("Parent is bound to a different exact source frame")
    gauge = positions[original["references"][0]["frame_index"]]
    paint_indices = [positions[ref["frame_index"]] for ref in measurements["references"]]
    targets = image_anchor_positions(rows, measurements, reference_stride)
    seed = recovery.matrix(parent.charts[0]["field_to_reference"])
    if not np.allclose(parent.payload["frames"][gauge]["reference_to_native"], np.eye(3), atol=1e-8):
        raise ValueError("Single-end parent must retain its declared image gauge")
    output.mkdir(parents=True, exist_ok=False)

    def write(path, value):
        with Path(path).open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")

    helpers = [Path(recovery.__file__), Path(propagation.__file__), Path(__file__),
               Path(__file__).with_name("fit_field_recovery.py")]
    frozen = output/"frozen"
    frozen.mkdir()
    for path in helpers:
        (frozen/path.name).write_bytes(path.read_bytes())
    write(output/"declaration.json", dict(
        inputs={str(p.resolve()): sha(p) for p in (parent_path, parent_measurements_path, frames_path, measurements_path)},
        helpers={str(p.resolve()): sha(p) for p in helpers}, field=recovery.FIELD,
        fit_frame_indices=measurements["fit_frame_indices"], check_frame_indices=measurements["check_frame_indices"],
        source_fit_reference_indices=[rows[i]["index"] for i in paint_indices],
        image_anchor_indices=[rows[i]["index"] for i in targets], turf_profile=turf_profile,
        motion_width=motion_width, reference_stride=reference_stride, degree=degree,
        basis_profile="bounded_radial_v1", radial_scale_rule="twice maximum normalized observed source-fit radius in the reference image",
        regularization_sigma_px=40., selection_rule="lowest degree passing every source-fit portion p95<=6px and full-frame visible geometry; check paint is not read"))
    cv2.setNumThreads(4)

    def read(row):
        path = (frames_path.parent/row["file"]).resolve()
        if not path.is_relative_to(frames_path.parent.resolve()):
            raise ValueError("Source image escapes manifest directory")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["image_sha256"]:
            raise ValueError("Source image checksum mismatch")
        im = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if im is None or [im.shape[1], im.shape[0]] != row["native_size"]:
            raise ValueError("Source image native dimensions mismatch")
        return im

    cache = {i: propagation.reference_features(read(rows[i]), turf_profile=turf_profile,
                                               max_width=motion_width) for i in targets}

    def register(a, b):
        return recovery.register_reference(cache[a], cache[b], rows[b]["native_size"])

    registered, paths, attempts = independent_reference_connections(
        gauge, paint_indices, targets, register, times=source_times)
    write(output/"image-connections.json", dict(gauge_frame_index=rows[gauge]["index"], paths=paths,
          index_space="positions in the frozen source manifest", attempts=attempts,
          transforms={i: h.tolist() for i, h in registered.items()}))
    if any(i not in registered for i in paint_indices):
        raise ValueError("A source-fit reference has no independently checked image connection")
    observations, drift_paths, drift_attempts = independent_drift_observations(
        gauge, paint_indices, targets, registered, paths, register, times=source_times)
    transforms, drift = recovery.smooth_reference_drift(
        [np.array(f["reference_to_native"]) for f in parent.payload["frames"]],
        [observations.get(i) for i in range(len(rows))],
        [float(source_times[i]) for i in range(len(rows))], gauge)
    write(output/"image-motion.json", dict(reference_to_native=[h.tolist() for h in transforms],
          drift=drift, paths=drift_paths, attempts=drift_attempts,
          method="independent source-image drift refinement of immutable parent motion; failed adjacent-step warnings persist"))
    fit_views = [{**ref, "reference_to_native": transforms[positions[ref["frame_index"]]].tolist()}
                 for ref in measurements["references"]]
    native_size = rows[gauge]["native_size"]
    observed_reference = np.concatenate([recovery.project(np.linalg.inv(obs["reference_to_native"]),
        np.concatenate([f["points_native"] for f in obs["features"]])) for obs in fit_views])
    radial_scale = 2*float(np.max(np.linalg.norm(
        (observed_reference-np.array(native_size)/2)/max(native_size), axis=1)))
    if not np.isfinite(radial_scale) or radial_scale <= 0:
        raise ValueError("No finite observed reference radius")
    xx, yy = np.meshgrid(np.linspace(-recovery.HALF_L, recovery.HALF_L, 181),
                          np.linspace(-recovery.HALF_W, recovery.HALF_W, 107))
    grid = np.c_[xx.ravel(), yy.ravel()]
    chosen = None
    outcomes = []
    for candidate_degree in ([2, 3] if degree == "auto" else [int(degree)]):
        candidate_started = time.monotonic()
        out = output/f"degree{candidate_degree}"
        out.mkdir()
        print(f"Shared source fit: degree {candidate_degree}", flush=True)
        try:
            poly = propagation.fit_reference_polynomial(seed, fit_views, native_size,
                degree=candidate_degree, basis_profile="bounded_radial_v1", radial_scale=radial_scale)
        except ValueError as error:
            failed = dict(degree=candidate_degree, error=str(error), selected=False)
            write(out/"failure.json", failed)
            outcomes.append(failed)
            continue
        write(out/"fit.json", poly)
        p = deepcopy(parent.payload)
        p.update(projection_mode=recovery.POLYNOMIAL_REFERENCE_PROJECTION_MODE,
                 reference_polynomial=poly, spatial_boundary=None, temporal_boundary=None)
        p["provenance"].update(parent_atlas_sha256=sha(parent_path), measurements_sha256=sha(measurements_path),
            fit_frame_indices=measurements["fit_frame_indices"],
            fitting_frames=parent.payload["provenance"].get("fitting_frames", []) +
                fitting_frame_bindings(manifest, [rows[i]["index"] for i in targets]),
            source_fit_reference_indices=[ref["frame_index"] for ref in fit_views],
            image_motion_sha256=sha(output/"image-motion.json"), runner_sha256=sha(__file__),
            note="Shared end/mid source fit; check paint excluded. Old geometric and source-fit warnings are recomputed; original failed adjacent-motion warnings persist.")
        for i, row in enumerate(p["frames"]):
            row["reference_to_native"] = transforms[i].tolist()
            row["boundary_offset_native_y_px"] = 0.
            row["warnings"] = [w for w in row["warnings"] if w not in (
                "no_independent_slow_drift_support", "invalid_or_absent_visible_supported_geometry",
                "gauge_paint_fit_exceeds_6px_p95")]
            if i in drift["unsupported_indices"]:
                row["warnings"].append("no_independent_slow_drift_support")
        atlas = recovery.RecoveryAtlas.from_payload(p)
        training, support = [], []
        for obs in fit_views:
            row = by_index[obs["frame_index"]]
            mapping = atlas.frame(row["source_pts"], row["source_time_base"],
                source_sha256=manifest["source"]["sha256"], native_size=row["native_size"])
            for feature in obs["features"]:
                world = recovery.feature_curve(feature, samples=8001)
                pixels = mapping.project(world)
                if not np.isfinite(pixels).all():
                    raise ValueError("Nonfinite source-fit curve cannot establish observed support")
                _, nearest = cKDTree(pixels).query(feature["points_native"])
                support.extend(world[nearest].tolist())
                training.append(dict(frame_index=row["index"], label=feature["label"],
                    **recovery.error_summary(recovery.feature_errors(mapping.project, feature))))
        p["charts"][0]["support_world_polygon"] = cv2.convexHull(np.asarray(support, np.float32)).reshape(-1, 2).tolist()
        p["charts"][0]["support_provenance"] = "Hull of nearest points on named finite source-fit curves across the declared observed reference views; not an accuracy certificate."
        paint_ok = all(t["finite_samples"] == t["samples"] and t["p95_px"] <= 6 for t in training)
        if not paint_ok:
            for row in p["frames"]:
                row["warnings"].append("shared_reference_paint_fit_exceeds_6px_p95")
        atlas = recovery.RecoveryAtlas.from_payload(p)
        geometry = []
        for i, row in enumerate(rows):
            mapping = atlas.frame(row["source_pts"], row["source_time_base"],
                source_sha256=manifest["source"]["sha256"], native_size=row["native_size"])
            result = recovery.visible_geometry_check(mapping, grid)
            geometry.append(dict(frame_index=row["index"], **result))
            if not result["valid"]:
                p["frames"][i]["warnings"].append("invalid_or_absent_visible_supported_geometry")
        atlas = recovery.RecoveryAtlas.from_payload(p)
        atlas.save(out/"atlas.json")
        geometry_ok = all(g["valid"] for g in geometry)
        eligible = paint_ok and geometry_ok and not any(f["warnings"] for f in p["frames"])
        report = dict(degree=candidate_degree, source_fit_pass=paint_ok, geometry_pass=geometry_ok,
                      frames_with_warnings=sum(bool(f["warnings"]) for f in p["frames"]),
                      training=training, geometry=geometry, selected=eligible, accepted=False,
                      status="approximate; independent checks pending", atlas_sha256=sha(out/"atlas.json"),
                      elapsed_seconds=time.monotonic()-candidate_started)
        write(out/"report.json", report)
        outcomes.append({k: report[k] for k in ("degree", "source_fit_pass", "geometry_pass", "frames_with_warnings", "selected", "elapsed_seconds")})
        print(json.dumps(outcomes[-1]), flush=True)
        if eligible:
            chosen = out
            if render:
                atlas = recovery.RecoveryAtlas.load(out/"atlas.json")
                (out/"overlay").mkdir()
                for i, row in enumerate(rows):
                    mapping = atlas.frame(row["source_pts"], row["source_time_base"],
                        source_sha256=manifest["source"]["sha256"], native_size=row["native_size"])
                    if not cv2.imwrite(str(out/"overlay"/f"{i+1:05d}.jpg"),
                        recovery.render_frame(read(row), mapping), [cv2.IMWRITE_JPEG_QUALITY, 92]):
                        raise ValueError("Could not write source-bound overlay")
            break
    selection = dict(selected_atlas=str((chosen/"atlas.json").resolve()) if chosen else None,
                     accepted=False, status="approximate; independent checks pending" if chosen else "no candidate passed source fit, motion and geometry",
                     outcomes=outcomes, source_frames=len(rows), elapsed_seconds=time.monotonic()-started)
    write(output/"selection.json", selection)
    print(json.dumps(selection), flush=True)
    print("FIELD_PROPAGATION_COMPLETE", flush=True)
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "parent-measurements", "frames", "measurements", "out"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--turf-profile", choices=("green_v1", "warm_green_v1"), default="green_v1")
    parser.add_argument("--motion-width", type=int, default=1920)
    parser.add_argument("--reference-stride", type=int, default=10)
    parser.add_argument("--degree", choices=("auto", "2", "3"), default="auto")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    run(args.parent, args.parent_measurements, args.frames, args.measurements, args.out,
        turf_profile=args.turf_profile, motion_width=args.motion_width,
        reference_stride=args.reference_stride, degree=args.degree, render=args.render)


if __name__ == "__main__":
    main()
