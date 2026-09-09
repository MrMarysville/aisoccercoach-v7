"""Fresh-paint fitting and exact-frame evaluation of the v3/v5/v6/v7 method.

No artifacts are read on import. Metres use centre,+x_camera_right,+y_near;
image coordinates address native pixel centres (the top-left centre is 0,0).
The outer size is owner supplied; interior dimensions are assumed standard yards.
All fits remain approximate. Numerical geometry checks do not certify accuracy.
"""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import BSpline, make_smoothing_spline
from scipy.optimize import least_squares

SCHEMA = "field-recovery-atlas-v1"
V7_PROJECTION_MODE = "v7_fixed_blend_v1"
LEGACY_PROJECTION_MODE = "legacy_support_clipped_blend_v1"
SINGLE_REFERENCE_PROJECTION_MODE = "single_reference_homography_v1"
END_MID_PROJECTION_MODE = "end_mid_x_blend_v1"
POLYNOMIAL_REFERENCE_PROJECTION_MODE = "polynomial_reference_residual_v1"
SMOOTH_SPATIAL_WEIGHT = "smoothstep_field_y_v1"
LINEAR_SPATIAL_WEIGHT = "linear_field_y_v1"
NEAR_TEMPORAL_METHOD = "linear_field_y_parameter_interpolation_v1"
HALF_L, HALF_W = 54.864, 32.004
CIRCLE_R, BOX18_DEPTH, BOX18_HALF_W = 9.144, 16.4592, 20.1168
BOX6_DEPTH, BOX6_HALF_W = 5.4864, 9.144
PENALTY_SPOT_DISTANCE = 10.9728
BLEND = dict(end_x_start_m=10., end_x_span_m=28., near_y_start_m=15.,
             near_y_span_m=15., near_right_x_start_m=0., near_right_x_span_m=15.)
FIELD = dict(length_m=109.728, width_m=64.008,
             dimensions_provenance="owner supplied 120 by 70 yards",
             axes="centre,+x_camera_right,+y_near",
             interior_assumptions=dict(provenance="assumed standard yard dimensions",
                                       circle_radius_m=CIRCLE_R,
                                       penalty_spot_distance_m=PENALTY_SPOT_DISTANCE,
                                       box18_depth_m=BOX18_DEPTH, box18_width_m=40.2336,
                                       box6_depth_m=BOX6_DEPTH, box6_width_m=18.288))


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def points(value):
    p = np.asarray(value, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
        raise ValueError("Expected finite N by 2 coordinates")
    return p


def matrix(value):
    h = np.asarray(value, dtype=float)
    if h.shape != (3, 3) or not np.isfinite(h).all() or np.linalg.matrix_rank(h) != 3:
        raise ValueError("Expected finite nonsingular 3 by 3 matrix")
    return h


def project(h, p):
    p = np.asarray(p, dtype=float)
    q = np.c_[p, np.ones(len(p))] @ np.asarray(h).T
    with np.errstate(divide="ignore", invalid="ignore"):
        return q[:, :2] / q[:, 2, None]


def normalize_h(h):
    h = matrix(h)
    divisor = h[2, 2] if abs(h[2, 2]) > 1e-12 else np.linalg.norm(h)
    return h / divisor


def resize_transform(native_size, resized_size):
    """Native pixel centres -> cv2.resize pixel centres, including half-pixel shift."""
    sx, sy = np.asarray(resized_size, float) / np.asarray(native_size, float)
    if min(sx, sy) <= 0:
        raise ValueError("Image dimensions must be positive")
    return np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1.]])


def native_motion(resized_motion, source_native, source_resized, target_native, target_resized):
    return normalize_h(np.linalg.inv(resize_transform(target_native, target_resized))
                       @ matrix(resized_motion) @ resize_transform(source_native, source_resized))


def smooth(value):
    z = np.clip(value, 0, 1)
    return z * z * (3 - 2 * z)


def polynomial_basis(pixels, origin_px, scale_px, degree, *, profile="plain_v1", radial_scale=1.):
    """Total-degree monomials in an explicitly normalized reference pixel plane."""
    xy = (np.asarray(pixels, float)-np.asarray(origin_px, float))/scale_px
    x, y = xy.T
    basis = np.stack([x**i*y**(total-i) for total in range(degree+1)
                      for i in range(total, -1, -1)], axis=1)
    if profile == "bounded_radial_v1":
        basis /= (1+((x*x+y*y)/radial_scale**2)**2)[:, None]
    elif profile != "plain_v1":
        raise ValueError("Unknown reference polynomial basis profile")
    return basis


def penalty_arc(side, samples=401):
    """Finite 10-yard arc outside the 18-yard area, toward midfield.

    The supporting circle is centred on the assumed 12-yard penalty spot.
    Endpoints lie on the penalty-area front; the rest of the circle is not paint.
    """
    if side not in ("left", "right"):
        raise ValueError("Unknown penalty arc side")
    sign = -1 if side == "left" else 1
    alpha = np.arccos((BOX18_DEPTH-PENALTY_SPOT_DISTANCE)/CIRCLE_R)
    theta = np.linspace(-alpha, alpha, samples)
    centre_x = sign*(HALF_L-PENALTY_SPOT_DISTANCE)
    return np.c_[centre_x-sign*CIRCLE_R*np.cos(theta), CIRCLE_R*np.sin(theta)]


def world_markings(samples=401):
    """Named finite regulation paint extents; no inferred offscreen click coordinates."""
    result = {
        "touch_far": np.array([[-HALF_L, -HALF_W], [HALF_L, -HALF_W]]),
        "touch_near": np.array([[-HALF_L, HALF_W], [HALF_L, HALF_W]]),
        "halfway": np.array([[0, -HALF_W], [0, HALF_W]])}
    for side, s in (("left", -1), ("right", 1)):
        x = s * HALF_L
        result[f"goal_{side}"] = np.array([[x, -HALF_W], [x, HALF_W]])
        result[f"pen_arc_{side}"] = penalty_arc(side, samples)
        for name, depth, hw in (("box18", BOX18_DEPTH, BOX18_HALF_W),
                                ("box6", BOX6_DEPTH, BOX6_HALF_W)):
            front = s * (HALF_L - depth)
            result[f"{name}_{side}_front"] = np.array([[front, -hw], [front, hw]])
            result[f"{name}_{side}_far"] = np.array([[x, -hw], [front, -hw]])
            result[f"{name}_{side}_near"] = np.array([[x, hw], [front, hw]])
    theta = np.linspace(0, 2 * np.pi, samples)
    result["centre_circle"] = CIRCLE_R * np.c_[np.cos(theta), np.sin(theta)]
    return result


def resolve_feature(feature):
    f = deepcopy(feature)
    f["points_native"] = points(f["points_native"])
    if not len(f["points_native"]):
        raise ValueError("Empty paint feature")
    if "world_points" in f:
        f["world_points"] = points(f["world_points"])
        if len(f["world_points"]) != len(f["points_native"]):
            raise ValueError("Point correspondences must have equal lengths")
        f["kind"] = "points"
    elif f["label"] in ("pen_arc_left", "pen_arc_right"):
        side = f["label"].removeprefix("pen_arc_")
        sign = -1 if side == "left" else 1
        centre = np.array([sign*(HALF_L-PENALTY_SPOT_DISTANCE), 0.])
        if ("circle_radius_m" in f and not np.isclose(f["circle_radius_m"], CIRCLE_R)) or (
                "circle_center_m" in f and not np.allclose(f["circle_center_m"], centre)):
            raise ValueError("Named penalty arc must preserve the assumed yard dimensions")
        f.update(kind="circle", circle_radius_m=CIRCLE_R, circle_center_m=centre,
                 penalty_arc_side=side)
    elif "circle_radius_m" in f or f["label"] in ("centre_circle", "center_circle"):
        f["circle_radius_m"] = float(f.get("circle_radius_m", CIRCLE_R))
        f["circle_center_m"] = np.asarray(f.get("circle_center_m", [0, 0]), float)
        if f["circle_radius_m"] <= 0 or f["circle_center_m"].shape != (2,):
            raise ValueError("Invalid circle")
        f["kind"] = "circle"
    else:
        if "world_line" not in f:
            marks = world_markings()
            if f["label"] not in marks or len(marks[f["label"]]) != 2:
                raise ValueError(f"Unknown named line: {f['label']}")
            ends = marks[f["label"]]
            f["world_line"] = np.cross(np.r_[ends[0], 1.], np.r_[ends[1], 1.])
            f["world_segment"] = ends
        line = np.asarray(f["world_line"], float)
        if line.shape != (3,) or not np.isfinite(line).all() or np.linalg.norm(line[:2]) < 1e-12:
            raise ValueError("Invalid world line")
        f["world_line"] = line / np.linalg.norm(line[:2])
        f["kind"] = "line"
    return f


def _tls_line(p):
    centre = p.mean(axis=0)
    _, _, vh = np.linalg.svd(p - centre, full_matrices=False)
    normal = vh[-1]
    return np.r_[normal, -normal @ centre]


def visible_line_intersections(features, native_size):
    """Measured-line intersections as a numerical seed only, never new annotations.

    Only intersections inside the native image and near both measured segments
    are used. Dual DLT below also uses the measured line incidences directly.
    """
    lines = [resolve_feature(f) for f in features]
    lines = [f for f in lines if f["kind"] == "line" and len(f["points_native"]) >= 2]
    pairs = []
    for i, a in enumerate(lines):
        for b in lines[i + 1:]:
            w = np.cross(a["world_line"], b["world_line"])
            q = np.cross(_tls_line(a["points_native"]), _tls_line(b["points_native"]))
            if min(abs(w[2]), abs(q[2])) < 1e-9:
                continue
            w, q = w[:2] / w[2], q[:2] / q[2]
            if not (np.all(q >= 0) and np.all(q < native_size)):
                continue
            if all(np.min(np.linalg.norm(f["points_native"] - q, axis=1)) <= 40 for f in (a, b)):
                pairs.append((w, q))
    return pairs


def _normalizers(native_size):
    scale = float(max(native_size))
    image = np.array([[1 / scale, 0, -native_size[0] / (2 * scale)],
                      [0, 1 / scale, -native_size[1] / (2 * scale)], [0, 0, 1.]])
    world = np.diag([1 / 50., 1 / 50., 1.])
    return image, world


def _linear_seed(features, image, world):
    rows = []
    for f in features:
        pixels = project(image, f["points_native"])
        if f["kind"] == "line":
            line = np.linalg.inv(world).T @ f["world_line"]
            # Each sample contributes l_world.T @ G_image_to_world @ p_image = 0.
            for pixel in pixels:
                row = np.outer(line, np.r_[pixel, 1.]).ravel()
                rows.append(row / np.linalg.norm(row) / np.sqrt(len(pixels)))
        elif f["kind"] == "points":
            for (u, v), (x, y) in zip(pixels, project(world, f["world_points"])):
                rows.extend(([u, v, 1, 0, 0, 0, -x*u, -x*v, -x],
                             [0, 0, 0, u, v, 1, -y*u, -y*v, -y]))
    if len(rows) < 8:
        return None
    a = np.asarray(rows)
    _, s, vh = np.linalg.svd(a, full_matrices=True)
    if (s > s[0] * 1e-8).sum() < 8:
        return None
    try:
        return normalize_h(np.linalg.inv(vh[-1].reshape(3, 3)))
    except (ValueError, np.linalg.LinAlgError):
        return None


def _circle_seeds(features, image, world):
    seeds = []
    circles = {}
    for f in features:
        if f["kind"] != "circle":
            continue
        key = (*f["circle_center_m"], f["circle_radius_m"])
        circles.setdefault(key, []).append(f["points_native"])
    for (world_cx, world_cy, radius), fragments in circles.items():
        native = np.concatenate(fragments)
        if len(native) < 5:
            continue
        p = project(image, native).astype(np.float32)
        (cx, cy), (a, b), angle = cv2.fitEllipse(p)
        angle = np.deg2rad(angle)
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        base = rot @ np.diag([a, b]) / (2 * radius / 50.)
        for theta in (0., np.pi / 2, np.pi, -np.pi / 2):
            r = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            h = np.eye(3)
            h[:2, :2] = base @ r
            h[:2, 2] = [cx, cy] - h[:2, :2] @ (np.array([world_cx, world_cy]) / 50.)
            seeds.append(h)
    return seeds


def finite_segment_extent_residual(h, segment, pixels):
    """Tangential distance beyond the finite projected paint extent, in pixels.

    A world segment crossing an offscreen projective pole maps to two rays.
    Its endpoint gap must not be mistaken for observed paint or drawn across.
    The normal residual is computed separately; the pair is Euclidean distance
    to the projected segment (or two rays), without dropping observations.
    """
    segment = np.asarray(segment, float)
    ends = project(h, segment)
    direction = ends[1]-ends[0]
    length = np.linalg.norm(direction)
    if not np.isfinite(ends).all() or not np.isfinite(length) or length < 1e-12:
        return np.full(len(pixels), 1e9)
    along = (np.asarray(pixels)-ends[0]) @ (direction/length)
    den = np.c_[segment, np.ones(2)] @ np.asarray(h)[2]
    if den[0]*den[1] > 0:
        return along-np.clip(along, 0, length)
    return np.where((along > 0) & (along < length),
                    np.where(along < length/2, along, along-length), 0.)


def _fit_residual(h, features, finite_segments=True):
    try:
        inv = np.linalg.inv(h)
    except np.linalg.LinAlgError:
        return np.full(sum(len(f["points_native"]) * (2 if f["kind"] == "points" or (
            finite_segments and f["kind"] == "line") else 1)
                           for f in features), 1e9)
    out = []
    for f in features:
        p = f["points_native"]
        if f["kind"] == "points":
            r = (project(h, f["world_points"]) - p).ravel()
        elif f["kind"] == "line":
            line = inv.T @ f["world_line"]
            r = np.c_[p, np.ones(len(p))] @ line / max(np.linalg.norm(line[:2]), 1e-12)
            if finite_segments:
                r = np.r_[r, finite_segment_extent_residual(h, f["world_segment"], p)]
        else:
            cx, cy = f["circle_center_m"]
            c = np.array([[1, 0, -cx], [0, 1, -cy],
                          [-cx, -cy, cx*cx + cy*cy - f["circle_radius_m"]**2]])
            conic = inv.T @ c @ inv
            z = np.c_[p, np.ones(len(p))]
            v = z @ conic
            r = np.sum(v * z, axis=1) / np.maximum(2 * np.linalg.norm(v[:, :2], axis=1), 1e-12)
        # One feature cannot dominate simply because its paint was densely sampled.
        out.extend(np.nan_to_num(r, nan=1e9, posinf=1e9, neginf=-1e9) / np.sqrt(len(p)))
    return np.asarray(out)


def _inside_polygon(p, polygon):
    if polygon is None:
        return np.ones(len(p), bool)
    polygon = points(polygon)
    edges = np.roll(polygon, -1, axis=0)-polygon
    delta = np.asarray(p)[:, None, :]-polygon[None]
    cross = edges[None, :, 0]*delta[:, :, 1]-edges[None, :, 1]*delta[:, :, 0]
    return (cross >= -1e-5).all(axis=1) | (cross <= 1e-5).all(axis=1)


def validate_homography(h, native_size, support_world_polygon=None):
    """Reject visible poles/branches and reversed orientation before transport.

    The analytic denominator test uses the image-clipped field polygon for each
    homogeneous branch. A pole outside the supported visible domain is allowed;
    two nonzero visible branch areas are never accepted. A dense Jacobian check
    provides an additional numerical check of the surviving visible domain.
    """
    h = matrix(h)
    field = np.array([[-HALF_L, -HALF_W], [HALF_L, -HALF_W],
                      [HALF_L, HALF_W], [-HALF_L, HALF_W]])
    polygon = points(support_world_polygon) if support_world_polygon is not None else field
    # Sutherland-Hodgman convex clipping against a linear homogeneous inequality.
    def clip(poly, line):
        if not len(poly):
            return poly
        out = []
        for a, b in zip(poly, np.roll(poly, -1, axis=0)):
            va, vb = np.r_[a, 1.] @ line, np.r_[b, 1.] @ line
            ina, inb = va >= 0, vb >= 0
            if ina:
                out.append(a)
            if ina != inb:
                out.append(a + va / (va - vb) * (b - a))
        return np.asarray(out).reshape(-1, 2)
    branch = []
    width, height = native_size
    for sign in (1, -1):
        poly = polygon.copy()
        for l in (sign*h[2], sign*h[0], sign*((width-1)*h[2]-h[0]),
                  sign*h[1], sign*((height-1)*h[2]-h[1])):
            poly = clip(poly, l)
        area = abs(cv2.contourArea(poly.astype(np.float32))) if len(poly) >= 3 else 0.
        branch.append(area)
    reasons = []
    if min(branch) > 1e-7:
        reasons.append("two_visible_projective_branches")
    if max(branch) < 1e-7:
        reasons.append("no_supported_visible_field")
    active_sign = 1 if branch[0] >= branch[1] else -1
    if np.linalg.det(h) * active_sign <= 0:
        reasons.append("reversed_visible_orientation")
    return dict(valid=not reasons, reasons=reasons, visible_branch_world_areas_m2=branch,
                scope="supported visible field, analytic homogeneous clipping")


def curve_distance(pixels, curve, max_segment_px=100.):
    """Exact Euclidean distance to finite sampled segments, without crossing poles."""
    p, q = points(pixels), np.asarray(curve, float)
    a, b = q[:-1], q[1:]
    d = b - a
    length = np.sum(d*d, axis=1)
    good = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1) & (length <= max_segment_px**2)
    a, d, length = a[good], d[good], length[good]
    if not len(a):
        return np.full(len(p), np.inf)
    result = []
    for start in range(0, len(p), 64):
        v = p[start:start+64, None] - a[None]
        t = np.clip(np.sum(v * d[None], axis=2) / np.maximum(length, 1e-20), 0, 1)
        result.extend(np.sqrt(np.min(np.sum((v - t[:, :, None]*d[None])**2, axis=2), axis=1)))
    return np.asarray(result)


def feature_curve(feature, samples=4001):
    f = resolve_feature(feature)
    if f["kind"] == "points":
        return f["world_points"]
    if f["kind"] == "circle":
        if "penalty_arc_side" in f:
            return penalty_arc(f["penalty_arc_side"], samples)
        theta = np.linspace(0, 2*np.pi, samples)
        return f["circle_center_m"] + f["circle_radius_m"] * np.c_[np.cos(theta), np.sin(theta)]
    if "world_segment" in f:
        return np.linspace(*np.asarray(f["world_segment"]), samples)
    # Clip an explicitly supplied infinite world line to the owner field rectangle.
    a, b, c = f["world_line"]
    ends = []
    if abs(b) > 1e-12:
        ends.extend([[x, -(a*x+c)/b] for x in (-HALF_L, HALF_L)
                     if abs((a*x+c)/b) <= HALF_W+1e-8])
    if abs(a) > 1e-12:
        ends.extend([[-(b*y+c)/a, y] for y in (-HALF_W, HALF_W)
                     if abs((b*y+c)/a) <= HALF_L+1e-8])
    if len(ends) < 2:
        raise ValueError("Line has no finite field extent")
    # Corner intersections can duplicate; choose farthest pair.
    ends = np.asarray(ends)
    i, j = np.unravel_index(np.argmax(np.linalg.norm(ends[:, None]-ends[None], axis=2)), (len(ends), len(ends)))
    return np.linspace(ends[i], ends[j], samples)


def feature_errors(projector, feature):
    f = resolve_feature(feature)
    if f["kind"] == "points":
        return np.linalg.norm(projector(f["world_points"]) - f["points_native"], axis=1)
    return curve_distance(f["points_native"], projector(feature_curve(feature)))


def error_summary(errors):
    errors = np.asarray(errors)
    finite = np.isfinite(errors)
    return dict(samples=len(errors), finite_samples=int(finite.sum()),
                median_px=float(np.median(errors[finite])) if finite.any() else None,
                p95_px=float(np.percentile(errors[finite], 95)) if finite.any() else None,
                max_px=float(np.max(errors[finite])) if finite.any() else None,
                metric="native Euclidean distance to sampled finite marking segments")


def fit_reference(features, native_size, support_world_polygon=None, max_nfev=1200):
    """Fit anew from measured points/lines/circle; never accepts an old H0."""
    fs = [resolve_feature(f) for f in features]
    for f in fs:
        if f["kind"] == "line" and "world_segment" not in f:
            f["world_segment"] = feature_curve(f, samples=2)
    image, world = _normalizers(native_size)
    seed = _linear_seed(fs, image, world)
    seed_method = "normalized dual line/point DLT" if seed is not None else "fresh measured ellipse multistart"
    seeds = [seed] if seed is not None else _circle_seeds(fs, image, world)
    if not seeds:
        raise ValueError("Insufficient independent named paint constraints for a fresh homography")
    candidates = []
    for seed in seeds:
        # Image-to-world parameters make the dominant line incidences linear
        # before their native-distance normalization. Forward-H parameters can
        # converge very slowly when the world origin lies near an offscreen pole.
        inverse_seed = np.linalg.inv(seed).ravel()
        pivot = int(np.argmax(abs(inverse_seed)))
        inverse_seed /= inverse_seed[pivot]
        def unpack(v):
            inverse = np.insert(v, pivot, 1.).reshape(3, 3)
            return np.linalg.inv(image) @ np.linalg.inv(inverse) @ world
        # A fixed warm start aligns the supporting lines/conic first. Only the
        # final fit, including every finite line extent, may become a candidate.
        warm = least_squares(lambda v: _fit_residual(unpack(v), fs, finite_segments=False),
                             np.delete(inverse_seed, pivot), loss="soft_l1", f_scale=1.,
                             x_scale="jac", max_nfev=min(300, max_nfev))
        fit = least_squares(lambda v: _fit_residual(unpack(v), fs), warm.x,
                            loss="soft_l1", f_scale=1., x_scale="jac", max_nfev=max_nfev)
        h = normalize_h(unpack(fit.x))
        candidate_support = support_world_polygon
        if candidate_support is None:
            measured = np.concatenate([f["world_points"] if f["kind"] == "points"
                                       else project(np.linalg.inv(h), f["points_native"]) for f in fs])
            measured = measured[np.isfinite(measured).all(axis=1)]
            candidate_support = cv2.convexHull(measured.astype(np.float32)).reshape(-1, 2).tolist()
        geometry = validate_homography(h, native_size, candidate_support)
        if fit.success and geometry["valid"]:
            candidates.append((fit.cost, h, fit, geometry, warm.nfev))
    if not candidates:
        raise ValueError("Fresh paint fit failed geometry validation or optimizer convergence")
    _, h, fit, geometry, warm_nfev = min(candidates, key=lambda c: c[0])
    if np.linalg.matrix_rank(fit.jac, tol=np.linalg.norm(fit.jac, 2)*1e-8) < 8:
        raise ValueError("Fresh paint constraints do not identify eight homography parameters")
    if support_world_polygon is None:
        measured = np.concatenate([f["world_points"] if f["kind"] == "points"
                                   else project(np.linalg.inv(h), f["points_native"]) for f in fs])
        measured = measured[np.isfinite(measured).all(axis=1)]
        support_world_polygon = cv2.convexHull(measured.astype(np.float32)).reshape(-1, 2).tolist()
    return dict(field_to_native=h.tolist(), support_world_polygon=support_world_polygon,
                geometry=geometry, optimizer_success=bool(fit.success), optimizer_cost=float(fit.cost),
                optimizer_nfev=fit.nfev, optimizer_warm_start_nfev=warm_nfev,
                optimizer_optimality=float(fit.optimality),
                optimizer_parameterization="normalized image-to-world inverse homography, largest-entry gauge",
                fit_residual="native finite-segment normal/extent distance / point distance; circle Sampson approximation (fit only)",
                features=[dict(label=f["label"], **error_summary(feature_errors(lambda p: project(h, p), f)))
                          for f in features],
                seed_method=seed_method)


def ground_mask(image, top_fraction=.12):
    """Ground candidates; widened arithmetic prevents uint8 green-channel wrap."""
    b, g, r = cv2.split(image.astype(np.int16))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green = (g > r + 3) & (g > b + 3)
    mask = cv2.inRange(hsv, (20, 15, 20), (105, 255, 250))
    mask[~green] = 0
    mask[:round(len(image)*top_fraction)] = 0
    return mask


def ground_cell_partition(pixel_points, cell_size=96):
    """Checks occupy different source-ground cells from all estimation tracks."""
    cells = np.floor(np.asarray(pixel_points) / cell_size).astype(int)
    return (cells[:, 0] + 2*cells[:, 1]) % 4 == 0


def fit_motion_tracks(source_points, target_points, native_size, *, min_fit=24, min_check=12,
                      max_check_median=2., min_hull_fraction=.025):
    p, q = points(source_points), points(target_points)
    if len(p) != len(q):
        raise ValueError("Motion track count mismatch")
    check = ground_cell_partition(p)
    fit = ~check
    stats = dict(fit_tracks=int(fit.sum()), check_tracks=int(check.sum()),
                 split="disjoint 96-native-pixel source ground cells", image_registration_pass=False)
    if fit.sum() < min_fit or check.sum() < min_check:
        return None, {**stats, "reason": "insufficient_independent_ground_tracks"}
    h, inliers = cv2.findHomography(p[fit], q[fit], cv2.USAC_MAGSAC, 2., maxIters=5000, confidence=.999)
    if h is None or inliers is None:
        return None, {**stats, "reason": "motion_homography_failed"}
    errors = np.linalg.norm(project(h, p[check]) - q[check], axis=1)
    inliers = inliers.ravel().astype(bool)
    area = cv2.contourArea(cv2.convexHull(q[fit][inliers].astype(np.float32))) / np.prod(native_size) if inliers.sum() >= 3 else 0.
    den = np.c_[p[fit][inliers], np.ones(inliers.sum())] @ h[2]
    geometry_ok = len(den) > 0 and np.isfinite(den).all() and np.min(den)*np.max(den) > 0 and np.all(np.linalg.det(h)/den**3 > 0)
    passed = geometry_ok and inliers.sum() >= min_fit and np.median(errors) <= max_check_median and np.mean(errors <= 6) >= .8 and area >= min_hull_fraction
    stats.update(fit_inliers=int(inliers.sum()), check_median_px=float(np.median(errors)),
                 check_p95_px=float(np.percentile(errors, 95)),
                 check_consistent_fraction=float(np.mean(errors <= 6)), inlier_hull_fraction=float(area),
                 image_registration_pass=bool(passed), supported_track_orientation_valid=bool(geometry_ok))
    return (normalize_h(h) if passed else None), stats


def adjacent_motion(previous, current, max_width=1280):
    native = [previous.shape[1], previous.shape[0]]
    if current.shape != previous.shape:
        raise ValueError("Adjacent frames require matching dimensions")
    width = min(max_width, native[0])
    size = (width, round(native[1]*width/native[0]))
    a, b = [cv2.resize(im, size) for im in (previous, current)]
    gray0, gray1 = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in (a, b)]
    p = cv2.goodFeaturesToTrack(gray0, 2500, .01, 7, mask=ground_mask(a))
    if p is None:
        return None, dict(image_registration_pass=False, reason="no_ground_features")
    q, ok, _ = cv2.calcOpticalFlowPyrLK(gray0, gray1, p, None, winSize=(25, 25), maxLevel=3)
    if q is None:
        return None, dict(image_registration_pass=False, reason="flow_failed")
    back, bok, _ = cv2.calcOpticalFlowPyrLK(gray1, gray0, q, None, winSize=(25, 25), maxLevel=3)
    good = (ok.ravel() > 0) & (bok.ravel() > 0) & (np.linalg.norm(back-p, axis=2).ravel() < .5)
    inv = np.linalg.inv(resize_transform(native, size))
    return fit_motion_tracks(project(inv, p[good, 0]), project(inv, q[good, 0]), native)


def reference_features(image, max_width=1280):
    native = [image.shape[1], image.shape[0]]
    width = min(max_width, native[0])
    size = (width, round(native[1]*width/native[0]))
    im = cv2.resize(image, size)
    sift = cv2.SIFT_create(nfeatures=7000, contrastThreshold=.012, edgeThreshold=12)
    kp, descriptors = sift.detectAndCompute(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), ground_mask(im))
    p = np.array([k.pt for k in kp]).reshape(-1, 2)
    return project(np.linalg.inv(resize_transform(native, size)), p), descriptors


def register_reference(source_features, target_features, native_size):
    p, dp = source_features
    q, dq = target_features
    if dp is None or dq is None or min(len(dp), len(dq)) < 2:
        return None, dict(image_registration_pass=False, reason="no_reference_features")
    pairs = cv2.BFMatcher().knnMatch(dp, dq, k=2)
    matches = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < .72*pair[1].distance]
    if not matches:
        return None, dict(image_registration_pass=False, reason="no_reference_matches")
    return fit_motion_tracks(np.array([p[m.queryIdx] for m in matches]),
                             np.array([q[m.trainIdx] for m in matches]), native_size)


def accumulate_motion(steps, reference_index=0, actual_paint_reference_indices=(), reference_transforms=None):
    """Reference -> native candidate motion; failed steps stay flagged until paint refit.

    steps[i] maps frame i-1 to i. Identity is a diagnostic continuation across a
    missing step, always warned. Direct image registration never clears the flag.
    A new actual paint reference clears only frames on its side of the failed step.
    """
    n = len(steps)
    transforms = [None]*n
    warnings = [[] for _ in range(n)]
    transforms[reference_index] = np.eye(3)
    actual = set(actual_paint_reference_indices) | {reference_index}
    reference_transforms = reference_transforms or {}
    for direction in (1, -1):
        active = []
        for i in range(reference_index+direction, n if direction > 0 else -1, direction):
            step_index = i if direction > 0 else i+1
            h = steps[step_index]
            if h is None:
                active.append(f"failed_adjacent_motion_step_{step_index}")
                h = np.eye(3)
            if i in actual:
                active = []
            warnings[i] = active.copy()
            transforms[i] = normalize_h((h if direction > 0 else np.linalg.inv(h)) @ transforms[i-direction])
            if i in actual and i in reference_transforms:
                transforms[i] = normalize_h(reference_transforms[i])
    return transforms, warnings


def smooth_reference_drift(accumulated, independent, times_s, reference_index, sigma_s=.75):
    """v5: smooth independent-registration drift, retaining fast adjacent motion.

    Independent entries are measured reference->frame matrices or None. Missing
    intervals longer than 3 sigma are left uncorrected and explicitly reported.
    """
    times = np.asarray(times_s)
    valid = [i for i, h in enumerate(independent) if h is not None]
    if not valid:
        return [h.copy() for h in accumulated], dict(observations=0, unsupported_indices=list(range(len(times))))
    corrections = np.stack([normalize_h(np.linalg.inv(accumulated[i]) @ independent[i]) for i in valid])
    smoothed, unsupported = [], []
    for i, t in enumerate(times):
        delta = times[valid]-t
        w = np.exp(-.5*(delta/sigma_s)**2)
        w[np.abs(delta) > 3*sigma_s] = 0
        if w.sum() <= 1e-12:
            smoothed.append(np.eye(3))
            unsupported.append(i)
        else:
            smoothed.append(normalize_h(np.einsum("n,nij->ij", w/w.sum(), corrections)))
    gauge = np.linalg.inv(smoothed[reference_index])
    result = [normalize_h(a @ c @ gauge) for a, c in zip(accumulated, smoothed)]
    return result, dict(observations=len(valid), sigma_s=sigma_s, unsupported_indices=unsupported,
                        correction_status="smoothed estimated slow reference drift")


def chart_weights(p, blend=None):
    b = {**BLEND, **(blend or {})}
    x, y = np.asarray(p).T
    wr = smooth((x-b["end_x_start_m"])/b["end_x_span_m"])
    wl = smooth((-x-b["end_x_start_m"])/b["end_x_span_m"])
    yn = smooth((y-b["near_y_start_m"])/b["near_y_span_m"])
    nr = smooth((x-b["near_right_x_start_m"])/b["near_right_x_span_m"])
    return np.c_[(1-yn)*wl+yn*(1-nr), (1-yn)*(1-wl-wr), (1-yn)*wr+yn*nr]


class RecoveryAtlas:
    """Portable v7 evaluator with optional charts and explicit measured support."""
    def __init__(self, document):
        if document.get("schema") != SCHEMA or digest(document["payload"]) != document.get("payload_sha256"):
            raise ValueError("Invalid recovery atlas schema or checksum")
        self.payload = deepcopy(document["payload"])
        p = self.payload
        if p.get("metric_certified") is not False or p.get("status") != "approximate":
            raise ValueError("Recovery output must remain approximate")
        if p["field"]["length_m"] != 109.728 or p["field"]["width_m"] != 64.008:
            raise ValueError("Owner outer field dimensions are fixed")
        # Missing mode must retain the exact semantics of already-saved trials.
        self.projection_mode = p.get("projection_mode", LEGACY_PROJECTION_MODE)
        if self.projection_mode not in (V7_PROJECTION_MODE, LEGACY_PROJECTION_MODE,
                                        SINGLE_REFERENCE_PROJECTION_MODE, END_MID_PROJECTION_MODE,
                                        POLYNOMIAL_REFERENCE_PROJECTION_MODE):
            raise ValueError("Unknown projection mode")
        self.reference_polynomial = p.get("reference_polynomial")
        if self.projection_mode == POLYNOMIAL_REFERENCE_PROJECTION_MODE:
            poly = self.reference_polynomial or {}
            degree = poly.get("degree")
            if degree not in (2, 3):
                raise ValueError("Polynomial reference needs declared degree two or three")
            coefficients = np.asarray(poly.get("coefficients_native_px"), float)
            origin = np.asarray(poly.get("origin_native_px"), float)
            scale = poly.get("scale_native_px", 0.)
            profile = poly.get("basis_profile", "plain_v1")
            radial_scale = poly.get("radial_scale", 1.)
            if (coefficients.shape != ((degree+1)*(degree+2)//2, 2) or
                    not np.isfinite(coefficients).all() or origin.shape != (2,) or
                    not np.isfinite(origin).all() or not np.isfinite(scale) or scale <= 0 or
                    profile not in ("plain_v1", "bounded_radial_v1") or
                    not np.isfinite(radial_scale) or radial_scale <= 0):
                raise ValueError("Invalid polynomial reference parameters")
            if p.get("spatial_boundary") is not None or p.get("temporal_boundary") is not None:
                raise ValueError("Polynomial reference has no implicit boundary correction")
        elif self.reference_polynomial is not None:
            raise ValueError("Polynomial correction requires its explicit projection mode")
        spatial_boundary = p.get("spatial_boundary")
        self.spatial_weight_profile = (spatial_boundary or {}).get("weight_profile", SMOOTH_SPATIAL_WEIGHT)
        if self.spatial_weight_profile not in (SMOOTH_SPATIAL_WEIGHT, LINEAR_SPATIAL_WEIGHT):
            raise ValueError("Unknown spatial boundary weight profile")
        self.near_boundary = p.get("near_temporal_boundary")
        if self.near_boundary is not None:
            near = self.near_boundary
            t = np.asarray(near["time_s"], float)
            offsets = np.asarray(near["delta_field_y_m"], float)
            if (near.get("method") != NEAR_TEMPORAL_METHOD or
                    near.get("weight_profile") not in (LINEAR_SPATIAL_WEIGHT, SMOOTH_SPATIAL_WEIGHT) or
                    near.get("protected_y_m") != BOX18_HALF_W or t.ndim != 1 or not len(t) or
                    offsets.shape != t.shape or not np.isfinite(t).all() or not np.isfinite(offsets).all() or
                    np.any(np.diff(t) <= 0) or list(near["domain_s"]) != [t[0], t[-1]]):
                raise ValueError("Invalid optional near-boundary model")
        self.charts = p["charts"]
        if not self.charts:
            raise ValueError("No measured charts")
        if self.projection_mode in (SINGLE_REFERENCE_PROJECTION_MODE, POLYNOMIAL_REFERENCE_PROJECTION_MODE) and len(self.charts) != 1:
            raise ValueError("Single-reference projection requires exactly one measured chart")
        if self.projection_mode == END_MID_PROJECTION_MODE and (
                len(self.charts) < 2 or "mid" not in {c["chart"] for c in self.charts}):
            raise ValueError("End/mid blending requires an actual midfield chart and an end chart")
        if len({c["chart"] for c in self.charts}) != len(self.charts):
            raise ValueError("Duplicate spatial chart")
        for c in self.charts:
            matrix(c["field_to_reference"])
            if c["chart"] not in ("left", "mid", "right"):
                raise ValueError("Unknown spatial chart")
            polygon = points(c["support_world_polygon"])
            if len(polygon) < 3 or not cv2.isContourConvex(polygon.astype(np.float32)):
                raise ValueError("Chart requires convex measured support polygon")
        self.frames = {}
        previous = None
        for f in p["frames"]:
            if type(f["source_pts"]) is not int or not isinstance(f["source_time_base"], str):
                raise ValueError("Require integer PTS and rational time base")
            tb = Fraction(f["source_time_base"])
            t = f["source_pts"]*tb
            if tb <= 0 or (previous is not None and t <= previous):
                raise ValueError("Frames must have positive time bases and strictly ordered PTS")
            matrix(f["reference_to_native"])
            if not np.isfinite(f["boundary_offset_native_y_px"]):
                raise ValueError("Nonfinite boundary correction")
            near_offset = f.get("near_boundary_delta_field_y_m", 0.)
            if not np.isfinite(near_offset):
                raise ValueError("Nonfinite near-boundary correction")
            expected_near = 0.
            if self.near_boundary is not None:
                try:
                    expected_near = near_temporal_offset(self.near_boundary, float(t))
                except ValueError:
                    pass  # Outside observed time support the correction is absent.
            if abs(near_offset-expected_near) > 1e-9:
                raise ValueError("Near-boundary frame offset does not match its observed temporal support")
            self.frames[t] = f
            previous = t
        self.document = deepcopy(document)

    @classmethod
    def from_payload(cls, payload):
        return cls(dict(schema=SCHEMA, payload=payload, payload_sha256=digest(payload)))

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def save(self, path):
        with Path(path).open("x") as f:
            json.dump(self.document, f, indent=2, allow_nan=False)
            f.write("\n")

    def edge_weight(self, p):
        protected = self.payload.get("protected_y_m", -BOX18_HALF_W)
        return smooth((protected-np.asarray(p)[:, 1])/(HALF_W+protected))

    def spatial_edge_weight(self, p):
        """Explicit spatial-only taper; existing maps retain their smoothstep.

        Linear interpolation avoids the smoothstep derivative's 1.5-times peak.
        It preserves both strip endpoints and the protected interior, with a
        derivative kink at the endpoints. The native temporal taper is unchanged.
        """
        if self.spatial_weight_profile == LINEAR_SPATIAL_WEIGHT:
            protected = self.payload.get("protected_y_m", -BOX18_HALF_W)
            return np.clip((protected-np.asarray(p)[:, 1])/(HALF_W+protected), 0, 1)
        return self.edge_weight(p)

    def near_edge_weight(self, p):
        if self.near_boundary is None:
            return np.zeros(len(p))
        z = np.clip((np.asarray(p)[:, 1]-BOX18_HALF_W)/(HALF_W-BOX18_HALF_W), 0, 1)
        return z if self.near_boundary["weight_profile"] == LINEAR_SPATIAL_WEIGHT else smooth(z)

    def weights(self, p):
        if self.projection_mode != END_MID_PROJECTION_MODE:
            return chart_weights(p, self.payload.get("blend"))
        # An observed midfield chart supplies the diagnostic continuation. Only
        # actually supplied end charts contribute residuals. No near-y override
        # replaces a measured halfway line with an extrapolated end-view line.
        b = {**BLEND, **self.payload.get("blend", {})}
        x = np.asarray(p)[:, 0]
        names = {c["chart"] for c in self.charts}
        left = smooth((-x-b["end_x_start_m"])/b["end_x_span_m"]) if "left" in names else np.zeros(len(x))
        right = smooth((x-b["end_x_start_m"])/b["end_x_span_m"]) if "right" in names else np.zeros(len(x))
        return np.c_[left, 1-left-right, right]

    def support_mask(self, p):
        """Public observed support; this never changes the diagnostic geometry.

        For fixed blending every contributing chart must cover the point. These
        are measured hulls, without dilation, and remain uncertified support.
        """
        p = np.asarray(p, float)
        if self.projection_mode in (SINGLE_REFERENCE_PROJECTION_MODE, POLYNOMIAL_REFERENCE_PROJECTION_MODE):
            return (np.isfinite(p).all(axis=1) & (abs(p) <= [HALF_L, HALF_W]).all(axis=1)
                    & _inside_polygon(p, self.charts[0]["support_world_polygon"]))
        weights = self.weights(p)
        if self.projection_mode in (V7_PROJECTION_MODE, END_MID_PROJECTION_MODE):
            supported = self.projection_domain_mask(p)
            supported &= np.isfinite(p).all(axis=1) & (abs(p) <= [HALF_L, HALF_W]).all(axis=1)
            for c in self.charts:
                w = weights[:, ("left", "mid", "right").index(c["chart"])]
                supported &= (w <= 0) | _inside_polygon(p, c["support_world_polygon"])
            return supported
        supported_weight = np.zeros(len(p), float)
        for c in self.charts:
            w = weights[:, ("left", "mid", "right").index(c["chart"])]
            supported_weight += w * _inside_polygon(p, c["support_world_polygon"])
        return supported_weight > 1e-12

    def projection_domain_mask(self, p):
        """Where the saved diagnostic model is defined, separate from observed support."""
        if self.projection_mode in (SINGLE_REFERENCE_PROJECTION_MODE, END_MID_PROJECTION_MODE,
                                    POLYNOMIAL_REFERENCE_PROJECTION_MODE):
            return np.isfinite(np.asarray(p, float)).all(axis=1)
        if self.projection_mode == LEGACY_PROJECTION_MODE:
            return self.support_mask(p)
        weights = self.weights(p)
        available = {c["chart"] for c in self.charts}
        defined = np.ones(len(p), bool)
        for i, name in enumerate(("left", "mid", "right")):
            if name not in available:
                defined &= weights[:, i] <= 0
        return defined

    def base(self, p, supported_only=None):
        p = np.asarray(p, float)
        if self.projection_mode in (SINGLE_REFERENCE_PROJECTION_MODE, POLYNOMIAL_REFERENCE_PROJECTION_MODE):
            # One independently fitted chart defines this diagnostic homography.
            # Observed support does not manufacture any additional chart fits.
            out = project(self.charts[0]["field_to_reference"], p)
            if supported_only:
                out[~self.support_mask(p)] = np.nan
            return out
        original = self.weights(p)
        if self.projection_mode in (V7_PROJECTION_MODE, END_MID_PROJECTION_MODE):
            # Exact v7: fixed smooth field weights, no hull clipping or reweighting.
            out = np.zeros((len(p), 2))
            for c in self.charts:
                w = original[:, ("left", "mid", "right").index(c["chart"])]
                take = w > 0  # Never multiply an inactive chart pole by zero.
                out[take] += project(c["field_to_reference"], p[take])*w[take, None]
            defined = self.projection_domain_mask(p)
            if supported_only:
                defined &= self.support_mask(p)
            out[~defined] = np.nan
            return out
        if supported_only is None:
            supported_only = True
        # Compatibility evaluator for immutable candidates lacking a mode field.
        weights, values = [], []
        for c in self.charts:
            w = original[:, ("left", "mid", "right").index(c["chart"])].copy()
            support = _inside_polygon(p, c["support_world_polygon"])
            if supported_only:
                w[~support] = 0
            weights.append(w)
            values.append(project(c["field_to_reference"], p))
        weights = np.asarray(weights).T
        total = weights.sum(axis=1)
        out = np.full((len(p), 2), np.nan)
        active = total > 1e-12
        out[active] = 0
        for w, q in zip(weights.T, values):
            take = active & (w > 0)
            out[take] += q[take]*(w[take]/total[take])[:, None]
        return out

    def reference(self, p, supported_only=None):
        p = np.asarray(p, float)
        out = self.base(p, supported_only)
        if self.reference_polynomial is not None:
            poly = self.reference_polynomial
            out += polynomial_basis(out, poly["origin_native_px"], poly["scale_native_px"],
                                    poly["degree"], profile=poly.get("basis_profile", "plain_v1"),
                                    radial_scale=poly.get("radial_scale", 1.)) @ np.asarray(poly["coefficients_native_px"])
        boundary = self.payload.get("spatial_boundary")
        if boundary is not None:
            weight = self.spatial_edge_weight(p)
            active = weight > 0
            far = self.base(np.c_[p[active, 0], np.full(active.sum(), -HALF_W)], supported_only=False)
            x = (far[:, 0]-boundary["x_origin_px"])/boundary["x_scale_px"]
            target = np.polynomial.polynomial.polyval(x, boundary["coefficients"])
            out[active, 1] += (target-far[:, 1])*weight[active]
        return out

    def frame(self, source_pts, source_time_base, *, source_sha256, native_size):
        if source_sha256 != self.payload["source"]["sha256"]:
            raise ValueError("Source hash mismatch")
        if type(source_pts) is not int or not isinstance(source_time_base, str):
            raise ValueError("Exact integer PTS and rational time base required")
        t = source_pts*Fraction(source_time_base)
        if t not in self.frames:
            raise ValueError("No mapping for exact source PTS")
        record = self.frames[t]
        if list(native_size) != record["native_size"]:
            raise ValueError("Native dimensions mismatch")
        return RecoveryFrame(self, record)


class RecoveryFrame:
    def __init__(self, atlas, record):
        self.atlas, self.record = atlas, deepcopy(record)

    def project(self, p, supported_only=None):
        """Saved diagnostic projection; observed support is a separate public gate.

        Explicit diagnostic modes default to their full approximate drawing.
        supported_only=True masks unsupported output without changing any weights.
        Legacy saved maps keep their original support-clipped default.
        """
        p = np.asarray(p, float)
        out = project(self.record["reference_to_native"], self.atlas.reference(p, supported_only))
        out[:, 1] += self.record["boundary_offset_native_y_px"]*self.atlas.edge_weight(p)
        near_delta = self.record.get("near_boundary_delta_field_y_m", 0.)
        if near_delta:
            weight = self.atlas.near_edge_weight(p)
            active = weight > 0
            canonical = p[active]
            mapped = canonical.copy()
            mapped[:, 1] += near_delta*weight[active]
            # Support belongs to canonical field coordinates, never to a shifted
            # parameter position. This preserves the original observed hull.
            changed = project(self.record["reference_to_native"], self.atlas.reference(mapped, supported_only=False))
            changed[:, 1] += self.record["boundary_offset_native_y_px"]*self.atlas.edge_weight(mapped)
            changed[~self.atlas.projection_domain_mask(canonical)] = np.nan
            if supported_only or (supported_only is None and self.atlas.projection_mode == LEGACY_PROJECTION_MODE):
                changed[~self.atlas.support_mask(canonical)] = np.nan
            out[active] = changed
        return out

    def public_projection(self, p):
        if self.record.get("warnings"):
            return dict(status="unavailable", native_pixels=None, warnings=self.record["warnings"], metric_certified=False)
        p = np.asarray(p, float)
        q = self.project(p, supported_only=True)
        good = (self.atlas.support_mask(p) & np.isfinite(q).all(axis=1)
                & (q >= 0).all(axis=1) & (q < self.record["native_size"]).all(axis=1))
        return dict(status="approximate" if good.any() else "unavailable",
                    native_pixels=[v.tolist() if ok else None for v, ok in zip(q, good)],
                    observed_support=good.tolist(), warnings=[], metric_certified=False)


def visible_geometry_check(mapping, grid, epsilon_m=.001):
    """Check the saved diagnostic domain and report observed support separately.

    A stencil that leaves the model's projection domain cannot establish a
    two-sided derivative. In fixed-blend mode the observation hull never clips
    this domain and cannot excuse a nonfinite diagnostic derivative. The older
    support-boundary count names refer to projection-domain boundaries.
    Nonfinite grid centres have unknown visibility and are counted separately;
    an offscreen projective pole alone is not a visible-domain failure.
    """
    grid = points(grid)
    if not np.isfinite(epsilon_m) or epsilon_m <= 0:
        raise ValueError("Geometry stencil step must be positive and finite")
    q = mapping.project(grid)
    finite = np.isfinite(q).all(axis=1)
    supported = mapping.atlas.support_mask(grid)
    nonfinite_supported = int((supported & ~finite).sum())
    visible = finite & (q >= 0).all(axis=1) & (q < mapping.record["native_size"]).all(axis=1)
    p = grid[visible]
    deltas = np.array([[epsilon_m, 0], [-epsilon_m, 0], [0, epsilon_m], [0, -epsilon_m]])
    stencil = p[:, None, :] + deltas[None]
    flat = stencil.reshape(-1, 2)
    observed_stencil_support = mapping.atlas.support_mask(flat).reshape(-1, 4)
    stencil_support = mapping.atlas.projection_domain_mask(flat).reshape(-1, 4)
    values = mapping.project(flat).reshape(-1, 4, 2)
    finite_values = np.isfinite(values).all(axis=2)
    with np.errstate(invalid="ignore", over="ignore"):
        dx = (values[:, 0]-values[:, 1])/(2*epsilon_m)
        dy = (values[:, 2]-values[:, 3])/(2*epsilon_m)
        det = dx[:, 0]*dy[:, 1]-dx[:, 1]*dy[:, 0]
    derivative_finite = np.isfinite(dx).all(axis=1) & np.isfinite(dy).all(axis=1) & np.isfinite(det)
    boundary_crossing = ~stencil_support.all(axis=1)
    interior_nonfinite = ~derivative_finite & (
        ~boundary_crossing | (stencil_support & ~finite_values).any(axis=1))
    boundary_only_nonfinite = ~derivative_finite & boundary_crossing & ~interior_nonfinite
    nonpositive = derivative_finite & (det <= 0)
    observed_crossing = (supported[visible] | observed_stencil_support.any(axis=1)) & ~observed_stencil_support.all(axis=1)
    return dict(visible_grid_points=len(p), nonpositive_jacobians=int(nonpositive.sum()),
                nonfinite_jacobians=int((~derivative_finite).sum()),
                interior_nonfinite_jacobians=int(interior_nonfinite.sum()),
                support_boundary_stencil_crossings=int(boundary_crossing.sum()),
                support_boundary_only_nonfinite_jacobians=int(boundary_only_nonfinite.sum()),
                observed_support_stencil_crossings=int(observed_crossing.sum()),
                visible_grid_points_outside_observed_support=int((~supported[visible]).sum()),
                nonfinite_supported_projections=nonfinite_supported,
                projection_mode=mapping.atlas.projection_mode,
                valid=bool(len(p) and not (nonpositive.any() or interior_nonfinite.any())),
                stencil_step_m=epsilon_m)


def fit_spatial_boundary(observations, fit_frame_indices):
    """Fit v6 quadratic in reference pixels using only declared fit-frame paint.

    observations: frame_index, points_native, reference_to_native. The image
    registration has been checked before this fitter is called.
    """
    allowed = set(fit_frame_indices)
    rows = [r for r in observations if r["frame_index"] in allowed]
    if not rows:
        return None
    ref, native, transforms = [], [], []
    for row in rows:
        p = points(row["points_native"])
        h = matrix(row["reference_to_native"])
        q = project(np.linalg.inv(h), p)
        # Spatial bin medians limit imbalance from dense samples.
        bins = np.floor(q[:, 0]/24).astype(int)
        for b in np.unique(bins):
            take = bins == b
            ref.append(np.median(q[take], axis=0))
            native.append(np.median(p[take], axis=0))
            transforms.append(h)
    ref, native, transforms = np.asarray(ref), np.asarray(native), np.asarray(transforms)
    if len(ref) < 6 or np.ptp(ref[:, 0]) < 100:
        return None
    origin, scale = float(np.median(ref[:, 0])), float(max(np.ptp(ref[:, 0])/2, 100))
    x = (ref[:, 0]-origin)/scale
    initial = np.polynomial.polynomial.polyfit(x, ref[:, 1], 2)
    def residual(coeff):
        q = np.c_[ref[:, 0], np.polynomial.polynomial.polyval(x, coeff), np.ones(len(x))]
        pred = np.einsum("nij,nj->ni", transforms, q)
        return pred[:, 1]/pred[:, 2]-native[:, 1]
    fit = least_squares(residual, initial, loss="soft_l1", f_scale=1.5)
    if not fit.success:
        return None
    return dict(coefficients=fit.x.tolist(), x_origin_px=origin, x_scale_px=scale,
                fit_frame_indices=sorted({r["frame_index"] for r in rows}),
                fit_residual="native y-only boundary residual (fit only)",
                fit_bins=len(ref), fit_success=True)


def fit_temporal_boundary(observations, fit_frame_indices, smoothing_lambda=.5):
    """v7 spline from fit-frame native-y offsets; never uses check-frame scores."""
    allowed = set(fit_frame_indices)
    rows = sorted([r for r in observations if r["frame_index"] in allowed], key=lambda r:r["time_s"])
    if len(rows) < 5:
        return None
    t = np.array([r["time_s"] for r in rows])
    if np.any(np.diff(t) <= 0):
        raise ValueError("Temporal fit anchors must have unique increasing times")
    y = np.array([r["median_offset_px"] for r in rows])
    w = np.array([min(r["samples"]/100, 1)/max(1., r["mad_px"])**2 for r in rows])
    spline = make_smoothing_spline(t-t[0], y, w=w, lam=smoothing_lambda)
    return dict(t=spline.t.tolist(), c=spline.c.tolist(), k=spline.k,
                origin_s=float(t[0]), domain_s=[float(t[0]), float(t[-1])],
                fit_frame_indices=[r["frame_index"] for r in rows], smoothing_lambda=smoothing_lambda)


def temporal_offset(model, time_s):
    if model is None:
        return 0.
    if not model["domain_s"][0] <= time_s <= model["domain_s"][1]:
        raise ValueError("Temporal correction cannot extrapolate beyond fit anchors")
    return float(BSpline(model["t"], model["c"], model["k"], extrapolate=False)(time_s-model["origin_s"]))


def near_temporal_offset(model, time_s):
    """Interpolate fitted field-y remap parameters, not measured field distances."""
    if not model["domain_s"][0] <= time_s <= model["domain_s"][1]:
        raise ValueError("Near-boundary correction cannot extrapolate beyond observations")
    return float(np.interp(time_s, model["time_s"], model["delta_field_y_m"]))


def render_frame(image, mapping, *, thickness=2):
    """Render the exact saved mapping used by numerical evaluation at native size."""
    if [image.shape[1], image.shape[0]] != mapping.record["native_size"]:
        raise ValueError("Render image is not native frame size")
    out = image.copy()
    width, height = image.shape[1], image.shape[0]
    warning = bool(mapping.record.get("warnings"))
    colour = (0, 140, 255) if warning else (0, 190, 255)
    for name in world_markings():
        f = dict(label=name, points_native=[[0, 0]])
        q = mapping.project(feature_curve(f, samples=1501))
        if getattr(mapping.atlas, "projection_mode", LEGACY_PROJECTION_MODE) in (V7_PROJECTION_MODE, SINGLE_REFERENCE_PROJECTION_MODE):
            # Batch only connected runs satisfying the same native branch guards.
            # OpenCV clips these runs. Joint AA can differ from separate segments;
            # the projected vertices, rounding, colour and thickness are identical.
            finite = np.isfinite(q).all(axis=1) & (np.max(abs(q), axis=1) <= 1e6)
            with np.errstate(invalid="ignore", over="ignore"):
                connected = finite[:-1] & finite[1:] & (np.linalg.norm(np.diff(q, axis=0), axis=1) <= 100)
            edges = np.diff(np.r_[False, connected, False].astype(np.int8))
            runs = [np.rint(q[start:stop+1]).astype(np.int32)
                    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]
            if runs:
                cv2.polylines(out, runs, False, colour, thickness, cv2.LINE_AA)
            continue
        for a, b in zip(q[:-1], q[1:]):
            if not np.isfinite([a, b]).all() or np.linalg.norm(b-a) > 100 or np.max(np.abs([a, b])) > 1e6:
                continue
            ok, u, v = cv2.clipLine((0, 0, width, height), tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)))
            if ok:
                cv2.line(out, u, v, colour, thickness, cv2.LINE_AA)
    label = "REVIEW REQUIRED | motion/geometry warning" if warning else "APPROXIMATE | metric accuracy unverified"
    cv2.putText(out, label, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, colour, 2, cv2.LINE_AA)
    return out
