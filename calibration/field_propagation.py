"""Constrained paint updates conditioned on an independently transported map.

These are local development hypotheses, not independent eight-parameter paint
solutions or metric certificates. The caller must retain the seed's source,
image-registration path, support and failure evidence.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from calibration import field_recovery as recovery


def reference_features(image, *, turf_profile="green_v1", max_width=1280):
    """Source-only SIFT proposals with an explicit turf-color assumption.

    Warm turf uses the already reviewed paint sampler's HSV criterion. People
    and other warm objects may enter the mask; disjoint-cell motion checks still
    have to reject inconsistent correspondences. This mask certifies no ground.
    """
    if turf_profile == "green_v1":
        return recovery.reference_features(image, max_width=max_width)
    if turf_profile != "warm_green_v1":
        raise ValueError("Unknown motion turf profile")
    native = [image.shape[1], image.shape[0]]
    width = min(max_width, native[0])
    size = (width, round(native[1]*width/native[0]))
    im = cv2.resize(image, size)
    h, s, v = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
    mask = (((h < 100) | (h > 170)) & (s > 22) & (v > 35)).astype(np.uint8)*255
    mask[:round(len(im)*.12)] = 0
    sift = cv2.SIFT_create(nfeatures=7000, contrastThreshold=.012, edgeThreshold=12)
    kp, descriptors = sift.detectAndCompute(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), mask)
    pixels = np.array([k.pt for k in kp]).reshape(-1, 2)
    return recovery.project(np.linalg.inv(recovery.resize_transform(native, size)), pixels), descriptors


def fit_constrained_reference(features, native_size, transported_h, *, model="affine_v1",
                              max_nfev=1200):
    """Fit a bounded 4/6-parameter native-image update to the transported H.

    The projective denominator remains supplied by image motion and the end
    reference. No regularizer or duplicated chart pretends to observe its two
    remaining degrees of freedom in the new paint.
    """
    if model not in ("similarity_v1", "affine_v1"):
        raise ValueError("Unknown constrained reference model")
    seed = recovery.normalize_h(transported_h)
    if not recovery.validate_homography(seed, native_size)["valid"]:
        raise ValueError("Transported seed has invalid visible geometry")
    fs = [recovery.resolve_feature(f) for f in features]
    for f in fs:
        if f["kind"] == "line" and "world_segment" not in f:
            f["world_segment"] = recovery.feature_curve(f, samples=2)
    width, height = native_size
    scale = max(width, height)
    normal = np.array([[1/scale, 0, -width/(2*scale)],
                       [0, 1/scale, -height/(2*scale)], [0, 0, 1.]])

    def unpack(v):
        if model == "similarity_v1":
            c, s = np.exp(v[2])*np.array([np.cos(v[3]), np.sin(v[3])])
            delta = np.array([[c, -s, v[0]], [s, c, v[1]], [0, 0, 1.]])
        else:
            delta = np.array([[1+v[2], v[3], v[0]],
                              [v[4], 1+v[5], v[1]], [0, 0, 1.]])
        return recovery.normalize_h(np.linalg.inv(normal) @ delta @ normal @ seed)

    count = 4 if model == "similarity_v1" else 6
    bounds = np.array([.1, .1, .2, .15] if count == 4 else [.1, .1, .2, .2, .2, .2])
    fit = least_squares(lambda v: recovery._fit_residual(unpack(v), fs), np.zeros(count),
                        bounds=(-bounds, bounds), loss="soft_l1", f_scale=1.,
                        x_scale="jac", max_nfev=max_nfev)
    rank = np.linalg.matrix_rank(fit.jac, tol=np.linalg.norm(fit.jac, 2)*1e-8)
    if not fit.success or rank < count:
        raise ValueError("Paint does not identify the constrained update")
    h = unpack(fit.x)
    measured = np.concatenate([f["world_points"] if f["kind"] == "points" else
                               recovery.project(np.linalg.inv(h), f["points_native"]) for f in fs])
    if not np.isfinite(measured).all():
        raise ValueError("Nonfinite constrained paint support")
    support = cv2.convexHull(measured.astype(np.float32)).reshape(-1, 2).tolist()
    geometry = recovery.validate_homography(h, native_size, support)
    if not geometry["valid"]:
        raise ValueError("Constrained fit has invalid visible geometry")
    return dict(field_to_native=h.tolist(), transported_field_to_native=seed.tolist(),
                support_world_polygon=support, geometry=geometry, model=model,
                parameter_count=count, constrained_jacobian_rank=int(rank),
                parameters=fit.x.tolist(), parameter_bounds=bounds.tolist(),
                active_parameter_bounds=fit.active_mask.tolist(),
                optimizer_success=bool(fit.success), optimizer_nfev=fit.nfev,
                optimizer_cost=float(fit.cost), metric_certified=False,
                limitation="Projective denominator inherited from source-bound image transport; not identified by this paint fit.",
                features=[dict(label=f["label"], **recovery.error_summary(
                    recovery.feature_errors(lambda p: recovery.project(h, p), f))) for f in features])


def nearest_curve_residual(pixels, curve):
    """Native vector distance to nearby finite polyline segments, with endpoints.

    Nearest vertices propose adjacent segments, never a line beyond its finite
    ends. Dense source curves are required; the same 100 px gap limit used by
    the marking scorer prevents connections through projective poles.
    """
    curve = np.asarray(curve, float)
    valid = np.isfinite(curve).all(axis=1)
    indices = np.flatnonzero(valid)
    if len(indices) < 2:
        return np.full_like(pixels, 1e6)
    _, nearest = cKDTree(curve[indices]).query(pixels, k=min(3, len(indices)))
    nearest = indices[np.asarray(nearest).reshape(len(pixels), -1)]
    segments = np.concatenate([nearest-1, nearest], axis=1)
    available = (segments >= 0) & (segments < len(curve)-1)
    segments = np.clip(segments, 0, len(curve)-2)
    a, b = curve[segments], curve[segments+1]
    delta = b-a
    squared = np.sum(delta**2, axis=2)
    available &= np.isfinite(squared) & (squared > 1e-20) & (squared <= 100**2)
    amount = np.clip(np.sum((pixels[:, None]-a)*delta, axis=2)/np.maximum(squared, 1e-20), 0, 1)
    residuals = pixels[:, None]-(a+amount[:, :, None]*delta)
    distances = np.sum(residuals**2, axis=2)
    distances[~available] = np.inf
    chosen = np.argmin(distances, axis=1)
    result = residuals[np.arange(len(pixels)), chosen]
    result[~available.any(axis=1)] = 1e6
    return result


def fit_reference_polynomial(seed_h, observations, native_size, *, degree=2,
                             regularization_sigma_px=40., max_nfev=400,
                             basis_profile="plain_v1", radial_scale=1.):
    """Fit one smooth reference-plane residual using all declared fit views.

    Observations contain a measured reference-to-native transform and named
    source paint. No check paint is accepted here. The weak view contributes
    constraints to the shared residual, while the end observations keep their
    constraints in the same solve. A broad displacement penalty is explicit;
    paint alone must still identify all correction coefficients.
    """
    if (degree not in (2, 3) or regularization_sigma_px <= 0 or
            basis_profile not in ("plain_v1", "bounded_radial_v1") or radial_scale <= 0):
        raise ValueError("Invalid polynomial fit configuration")
    seed = recovery.normalize_h(seed_h)
    origin = np.asarray(native_size, float)/2
    scale = float(max(native_size))
    count = (degree+1)*(degree+2)//2
    prepared = []
    for obs in observations:
        transform = recovery.matrix(obs["reference_to_native"])
        for feature in obs["features"]:
            pixels = recovery.points(feature["points_native"])
            world = recovery.feature_curve(feature, samples=1001)
            base = recovery.project(seed, world)
            basis = recovery.polynomial_basis(base, origin, scale, degree,
                                              profile=basis_profile, radial_scale=radial_scale)
            prepared.append((pixels, base, basis, transform))
    if not prepared:
        raise ValueError("Polynomial fit requires source paint")
    # Penalize displacement on a uniform gauge-image lattice. This supplies no
    # paint observations and is reported separately from the paint Jacobian.
    gx, gy = np.meshgrid(np.linspace(0, native_size[0]-1, 11),
                         np.linspace(0, native_size[1]-1, 7))
    controls = np.c_[gx.ravel(), gy.ravel()]
    control_basis = recovery.polynomial_basis(controls, origin, scale, degree,
                                              profile=basis_profile, radial_scale=radial_scale)

    def paint_residual(v):
        coefficients = v.reshape(count, 2)
        return np.concatenate([nearest_curve_residual(pixels, recovery.project(transform,
            base+basis@coefficients)).ravel() for pixels, base, basis, transform in prepared])

    def residual(v):
        return np.r_[paint_residual(v),
                     (control_basis@v.reshape(count, 2)).ravel()/regularization_sigma_px]

    fit = least_squares(residual, np.zeros(count*2), loss="soft_l1", f_scale=1.,
                        x_scale="jac", max_nfev=max_nfev)
    baseline = paint_residual(fit.x)
    step = 1e-3
    jac = np.column_stack([(paint_residual(fit.x+np.eye(count*2)[i]*step)-baseline)/step
                           for i in range(count*2)])
    rank = np.linalg.matrix_rank(jac, tol=np.linalg.norm(jac, 2)*1e-8)
    if not fit.success or rank < count*2:
        raise ValueError(f"Shared polynomial fit unavailable: success={fit.success}, paint_rank={rank}/{count*2}")
    coefficients = fit.x.reshape(count, 2)
    return dict(degree=degree, origin_native_px=origin.tolist(), scale_native_px=scale,
                basis_profile=basis_profile, radial_scale=radial_scale,
                coefficients_native_px=coefficients.tolist(), parameter_count=count*2,
                paint_jacobian_rank=int(rank), optimizer_nfev=fit.nfev,
                optimizer_cost=float(fit.cost), optimizer_success=bool(fit.success),
                regularization_sigma_px=regularization_sigma_px,
                regularization_lattice_native_px=controls.tolist(),
                regularization_displacements_native_px=(control_basis@coefficients).tolist(),
                fit_native_vector_residual_px=baseline.reshape(-1, 2).tolist(),
                fit_metric="native Euclidean distance to 1001-sample finite curves; source fit evidence only",
                metric_certified=False)
