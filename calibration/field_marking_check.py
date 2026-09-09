"""Versioned orientation checks for sampled markings; never change atlas geometry."""
import hashlib
from pathlib import Path

import numpy as np

from calibration.field_recovery import RecoveryAtlas, V7_PROJECTION_MODE


ORIENTATION_POLICIES = ("central_v1", "domain_boundary_v2")


def validate_orientation_policy(policy):
    if policy not in ORIENTATION_POLICIES:
        raise ValueError(f"Unknown marking orientation policy: {policy}")


def helper_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def check_marking_orientation(mapping, world, policy="central_v1"):
    """Return the original curve and derivatives, with explicit stencil evidence.

    V2 only resolves a missing central probe at the explicit fixed-chart domain
    edge of a RecoveryAtlas. In-domain poles, support guesses and legacy atlas
    domains get no fallback. A finite central determinant, including a fold,
    always keeps its original value.
    """
    validate_orientation_policy(policy)
    world = np.asarray(world, float)
    curve = mapping.project(world)
    plus = [mapping.project(world + [.001, 0]), mapping.project(world + [0, .001])]
    minus = [mapping.project(world - [.001, 0]), mapping.project(world - [0, .001])]
    derivatives = [(plus[i] - minus[i]) / .002 for i in range(2)]
    dx, dy = derivatives
    central = dx[:, 0] * dy[:, 1] - dx[:, 1] * dy[:, 0]
    one_sided = np.zeros((len(world), 2), dtype=bool)
    atlas = getattr(mapping, "atlas", None)
    if (policy == "domain_boundary_v2" and isinstance(atlas, RecoveryAtlas)
            and atlas.projection_mode == V7_PROJECTION_MODE):
        domain = atlas.projection_domain_mask(world)
        for axis in range(2):
            step = np.zeros(2)
            step[axis] = .001
            inside_plus = atlas.projection_domain_mask(world + step)
            inside_minus = atlas.projection_domain_mask(world - step)
            missing = (domain & np.isfinite(curve).all(axis=1) & ~np.isfinite(central)
                       & ~np.isfinite(derivatives[axis]).all(axis=1))
            for direction, inside, outside, first in ((-1, inside_minus, inside_plus, minus[axis]),
                                                       (1, inside_plus, inside_minus, plus[axis])):
                chosen = np.flatnonzero(missing & inside & ~outside & np.isfinite(first).all(axis=1))
                if not len(chosen):
                    continue
                second_world = world[chosen] + direction * 2 * step
                second = mapping.project(second_world)
                valid = (atlas.projection_domain_mask(second_world) & np.isfinite(second).all(axis=1))
                selected = chosen[valid]
                derivative = direction * (-3 * curve[selected] + 4 * first[selected] - second[valid]) / .002
                finite = np.isfinite(derivative).all(axis=1)
                derivatives[axis][selected[finite]] = derivative[finite]
                one_sided[selected[finite], axis] = True
    dx, dy = derivatives
    determinant = dx[:, 0] * dy[:, 1] - dx[:, 1] * dy[:, 0]
    positive = np.isfinite(determinant) & (determinant > 0)
    return dict(curve=curve, determinant=determinant, positive=positive,
                central_determinant=central, one_sided_axes=one_sided,
                derivatives=np.stack(derivatives, axis=2))


def checked_marking_curve(mapping, world, policy="central_v1"):
    checked = check_marking_orientation(mapping, world, policy)
    curve = checked["curve"].copy()
    positive = checked["positive"]
    determinant = checked["determinant"]
    sided = checked["one_sided_axes"]
    stats = dict(curve_samples=len(curve), finite_projection_points=int(np.isfinite(curve).all(axis=1).sum()),
                 finite_jacobian_points=int(np.isfinite(determinant).sum()),
                 positive_orientation_points=int(positive.sum()),
                 nonpositive_finite_jacobian_points=int((np.isfinite(determinant) & (determinant <= 0)).sum()),
                 unknown_jacobian_points=int((~np.isfinite(determinant)).sum()),
                 one_sided_points=int(sided.any(axis=1).sum()),
                 one_sided_x_points=int(sided[:, 0].sum()), one_sided_y_points=int(sided[:, 1].sum()),
                 one_sided_positive_points=int((sided.any(axis=1) & positive).sum()))
    curve[~positive] = np.nan
    return curve, stats
