"""Refine source-only named browser polylines against native paint ridges.

No fitted field projection is read. Every search region comes from the supplied
source-bound annotation. Rejected samples and review images remain inspectable.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from calibration.alignment_trial import sha256_file


TURF_PROFILES = ("green_v1", "warm_green_v1")


def turf_criterion(profile):
    if profile not in TURF_PROFILES:
        raise ValueError(f"Unknown turf support profile: {profile}")
    return dict(color_space="OpenCV HSV; H 0..179, S/V 0..255",
                hue_any_of=([dict(gt=24, lt=100)] if profile == "green_v1"
                            else [dict(lt=100), dict(gt=170)]),
                saturation_gt=22, value_gt=35)


def polyline_samples(points, spacing=8.):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("Paint refinement requires a visible polyline")
    length = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.r_[True, length > .01]
    points = points[keep]
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if distance[-1] < spacing:
        raise ValueError("Visible marking is too short to refine")
    at = np.arange(spacing / 2, distance[-1] - spacing / 2, spacing)
    out = np.c_[np.interp(at, distance, points[:, 0]), np.interp(at, distance, points[:, 1])]
    delta = np.diff(points, axis=0)
    segment = np.minimum(np.searchsorted(distance, at, side="right") - 1, len(delta) - 1)
    tangent = delta[segment]
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    return out, np.c_[-tangent[:, 1], tangent[:, 0]]


def paint_layers(image, turf_profile="green_v1"):
    """Per-image ridge and turf layers, computed once and shared by many polylines."""
    turf_criterion(turf_profile)
    # Cast before all channel arithmetic. Bright lines may otherwise wrap uint8.
    white = gaussian_filter(np.min(image.astype(float), axis=2), sigma=.65)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(float)
    hue = ((hsv[:, :, 0] > 24) & (hsv[:, :, 0] < 100) if turf_profile == "green_v1"
           else (hsv[:, :, 0] < 100) | (hsv[:, :, 0] > 170))
    grass = (hue & (hsv[:, :, 1] > 22) & (hsv[:, :, 2] > 35)).astype(float)
    return dict(white=white, grass=grass, turf_profile=turf_profile, shape=image.shape[:2])


def refine_polyline(image, points, radius=20., spacing=8., required_grass_sides=2,
                    turf_profile="green_v1", layers=None):
    turf_criterion(turf_profile)
    origin, normal = polyline_samples(points, spacing)
    if layers is None:
        layers = paint_layers(image, turf_profile)
    elif layers["turf_profile"] != turf_profile or tuple(layers["shape"]) != tuple(image.shape[:2]):
        raise ValueError("Precomputed paint layers do not match this image or turf profile")
    white, grass = layers["white"], layers["grass"]
    offsets = np.arange(-radius, radius + .01, .5)
    candidates = origin[:, None, :] + normal[:, None, :] * offsets[None, :, None]

    def sample(layer, xy):
        return map_coordinates(layer, [xy[..., 1], xy[..., 0]], order=1, mode="nearest")

    intensity = sample(white, candidates)
    strength = np.maximum.reduce([
        intensity - .5 * (sample(white, candidates + normal[:, None, :] * width)
                          + sample(white, candidates - normal[:, None, :] * width))
        for width in (4., 8., 14.)])
    sides = np.maximum.reduce([
        np.minimum(sample(grass, candidates + normal[:, None, :] * width),
                   sample(grass, candidates - normal[:, None, :] * width))
        for width in (5., 12., 18.)])
    score = np.clip(strength, 0, 70) * (.25 + .75 * sides)
    # Track a continuous ridge within the coarse strip. This regularises image
    # measurement only; no field geometry or downstream residual enters it.
    dp = score[0].copy()
    back = np.zeros(score.shape, np.int16)
    transition = .15 * (offsets[:, None] - offsets[None, :]) ** 2
    for index in range(1, len(origin)):
        proposal = dp[None, :] - transition
        back[index] = np.argmax(proposal, axis=1)
        dp = score[index] + proposal[np.arange(len(offsets)), back[index]]
    selected = np.zeros(len(origin), dtype=int)
    selected[-1] = np.argmax(dp)
    for index in range(len(origin) - 1, 0, -1):
        selected[index - 1] = back[index, selected[index]]
    picked = candidates[np.arange(len(origin)), selected].copy()
    peak = strength[np.arange(len(origin)), selected]
    # Thick nearby paint has a plateau, often with two edge maxima in a narrow
    # ridge filter. Measure both half-height crossings, then use their midpoint.
    # This supplies the same paint-centre convention for thin and thick lines.
    widths = np.zeros(len(origin))
    bounded = np.zeros(len(origin), dtype=bool)
    crossings = np.zeros(len(origin), dtype=bool)
    contrast = np.zeros(len(origin))
    for i, chosen in enumerate(selected):
        profile = intensity[i]
        background = np.percentile(profile, 20)
        threshold = background + .5 * (profile[chosen] - background)
        left = right = int(chosen)
        while left > 0 and profile[left - 1] >= threshold:
            left -= 1
        while right < len(offsets) - 1 and profile[right + 1] >= threshold:
            right += 1
        crossings[i] = left > 0 and right < len(offsets) - 1
        contrast[i] = profile[chosen] - background
        if left == 0 or right == len(offsets) - 1 or profile[chosen] - background < 4:
            continue
        lo = offsets[left - 1] + .5 * (threshold - profile[left - 1]) / max(
            profile[left] - profile[left - 1], 1e-9)
        hi = offsets[right] + .5 * (profile[right] - threshold) / max(
            profile[right] - profile[right + 1], 1e-9)
        widths[i] = hi - lo
        bounded[i] = 1 <= widths[i] <= 30
        picked[i] = origin[i] + normal[i] * ((lo + hi) / 2)
    clearance = widths / 2 + 4
    grass_a = sample(grass, picked + normal * clearance[:, None])
    grass_b = sample(grass, picked - normal * clearance[:, None])
    if required_grass_sides not in (1, 2):
        raise ValueError("Require one or two grass sides")
    side = np.minimum(grass_a, grass_b) if required_grass_sides == 2 else np.maximum(grass_a, grass_b)
    height, width = image.shape[:2]
    in_viewport = ((picked[:, 0] >= 6) & (picked[:, 0] < width - 6)
                   & (picked[:, 1] >= 6) & (picked[:, 1] < height - 6))
    valid = ((peak >= 4) & (side >= .25) & bounded
             & (picked[:, 0] >= 6) & (picked[:, 0] < width - 6)
             & (picked[:, 1] >= 6) & (picked[:, 1] < height - 6))
    samples = [dict(coarse_xy=a.tolist(), native_xy=p.tolist(), ridge_strength=float(s),
                    grass_support=float(g), accepted=bool(ok),
                    normal_offset_px=float(np.dot(p - a, n)), paint_width_px=float(width),
                    measurement_method="cross_section_half_height_midpoint")
               for a, n, p, s, g, ok, width in zip(origin, normal, picked, peak, side, valid, widths)]
    for i, row in enumerate(samples):
        conditions = dict(ridge_strength=bool(peak[i] >= 4),
                          turf_neighbor_support=bool(side[i] >= .25),
                          cross_section_contrast=bool(contrast[i] >= 4),
                          half_height_crossings=bool(crossings[i]),
                          paint_width=bool(bounded[i]), viewport=bool(in_viewport[i]))
        row.update(turf_profile=turf_profile, grass_support_positive_side=float(grass_a[i]),
                   grass_support_negative_side=float(grass_b[i]),
                   required_grass_sides=required_grass_sides,
                   cross_section_contrast=float(contrast[i]),
                   half_height_crossings_bounded=bool(crossings[i]),
                   conditions_passed=conditions,
                   rejection_reasons=[key for key, passed in conditions.items() if not passed])
    return picked[valid], samples


def measure(frames_path, annotation_path, output, chart_by_index, radius=6., turf_profile="green_v1"):
    criterion = turf_criterion(turf_profile)
    started = time.monotonic()
    manifest = json.loads(Path(frames_path).read_text())
    annotations = json.loads(Path(annotation_path).read_text())
    if annotations["source"]["sha256"] != manifest["source"]["sha256"]:
        raise ValueError("Annotation/source mismatch")
    if annotations.get("frames_manifest_sha256") != sha256_file(frames_path):
        raise ValueError("Annotation/frame-manifest mismatch")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    references, review, records = [], {}, []
    grouped = {}
    for feature in annotations["features"]:
        grouped.setdefault(feature["frame_index"], []).append(feature)
    for index, features in sorted(grouped.items()):
        frame = next(r for r in manifest["frames"] if r["index"] == index)
        path = Path(frames_path).parent / frame["file"]
        if sha256_file(path) != frame["image_sha256"]:
            raise ValueError("Changed source pixels")
        image = cv2.imread(str(path))
        overlay = image.copy()
        measured = []
        for feature_number, feature in enumerate(features, 1):
            binding = feature.get("frame", feature)
            # The viewer also performs these validations at save time. Recheck
            # when the fields are present; the manifest digest binds all rows.
            for key in ("source_pts", "source_time_base", "image_sha256", "native_size"):
                if key in binding and binding[key] != frame[key]:
                    raise ValueError(f"Wrong annotation frame binding: {key}")
            coarse = [p["native_xy"] for p in feature["points"]]
            if feature.get("kind", "polyline") != "polyline" or len(coarse) < 2:
                records.append(dict(feature_id=feature["id"], frame_index=index,
                                    label=feature["label"], status="unmeasured",
                                    reason="Point requires named intersection refinement"))
                continue
            # A named outer touchline can separate green playing turf from a
            # brown apron. Its semantic browser review supplies the boundary
            # identity; requiring grass on that exterior side discards real paint.
            near = feature["label"] == "touch_near"
            search_radius = (max(radius, 36.) if near else
                             max(radius, 20.) if feature["label"] == "halfway" else radius)
            points, raw = refine_polyline(image, coarse, radius=search_radius,
                                         required_grass_sides=1 if near else 2, turf_profile=turf_profile)
            record = dict(feature_id=feature["id"], frame_index=index, label=feature["label"],
                          points_native=points.tolist(), samples=raw,
                          selected_count=len(points), searched_count=len(raw),
                          review_status="needs_semantic_review", coarse_points_native=coarse,
                          search_radius_px=search_radius, required_grass_sides=1 if near else 2,
                          turf_profile=turf_profile)
            records.append(record)
            if len(points) >= 3:
                measured.append({k: record[k] for k in ("label", "points_native", "feature_id")})
            for row in raw:
                point = tuple(np.rint(row["native_xy"]).astype(int))
                cv2.circle(overlay, point, 2, (255, 0, 255) if row["accepted"] else (0, 0, 255), -1)
            if coarse:
                cv2.putText(overlay, str(feature_number) + ":" + feature["label"],
                            tuple(np.rint(coarse[0]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX,
                            .45, (0, 255, 255), 1, cv2.LINE_AA)
        references.append(dict(frame_index=index, chart=chart_by_index.get(index, "unassigned"),
                               features=measured, **{k: frame[k] for k in
                                   ("source_pts", "source_time_base", "image_sha256", "native_size")}))
        target = output / f"review-{index:05d}.png"
        cv2.imwrite(str(target), overlay)
        review[str(index)] = str(target.resolve())
    result = dict(schema="field-recovery-measurements-v1", source_sha256=manifest["source"]["sha256"],
                  frames_manifest_sha256=sha256_file(frames_path),
                  annotation_sha256=sha256_file(annotation_path),
                  created_at=datetime.now(timezone.utc).isoformat(), references=references,
                  review_status="needs_semantic_review", fit_frame_indices=[], check_frame_indices=[],
                  method="Native minimum-channel ridge within source-only annotated polylines",
                  turf_profile=turf_profile, turf_hsv_criterion=criterion,
                  paint_acceptance_criteria=dict(ridge_strength_min=4, turf_support_min=.25,
                                                 cross_section_contrast_min=4,
                                                 half_height_crossings="both inside search strip",
                                                 paint_width_px_inclusive=[1, 30],
                                                 native_viewport_margin_px=6),
                  radius_px=radius, measurements=records, elapsed_seconds=time.monotonic() - started)
    (output / "measurements.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (output / "review.json").write_text(json.dumps(review, indent=2) + "\n")
    return dict(output=str(output), features=len(records),
                measured_points=sum(r.get("selected_count", 0) for r in records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chart", action="append", default=[], help="frame_index:left|mid|right")
    parser.add_argument("--radius", type=float, default=6.)
    parser.add_argument("--turf-profile", choices=TURF_PROFILES, default="green_v1",
                        help="Explicit source-reviewed turf color assumption")
    args = parser.parse_args()
    charts = {int(value.split(":")[0]): value.split(":")[1] for value in args.chart}
    print(json.dumps(measure(args.frames, args.annotations, args.out, charts, args.radius,
                             turf_profile=args.turf_profile), indent=2))


if __name__ == "__main__":
    main()
