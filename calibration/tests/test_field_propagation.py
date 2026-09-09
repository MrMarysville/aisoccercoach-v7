"""Synthetic checks of conditioned updates; no footage or metric acceptance."""
from copy import deepcopy

import numpy as np
import pytest

from calibration import field_recovery as r
from calibration.field_adjustment import AdjustmentAtlas
from calibration.field_propagation import (fit_constrained_reference, fit_reference_polynomial,
                                  nearest_curve_residual)


H = np.array([[12., 1., 960.], [.2, 5., 400.], [.0005, .006, 1.]])


def midfield_paint(h):
    return [dict(label=name, points_native=r.project(h, r.feature_curve(
        dict(label=name, points_native=[[0, 0]]), 101)).tolist())
        for name in ("centre_circle", "halfway")]


@pytest.mark.parametrize("model,delta", [
    ("similarity_v1", np.array([[1.02, -.015, 14.], [.015, 1.02, -11.], [0, 0, 1.]])),
    ("affine_v1", np.array([[1.03, .02, -12.], [-.01, .98, 15.], [0, 0, 1.]])),
])
def test_weak_paint_recovers_known_conditioned_update(model, delta):
    target = delta @ H
    fit = fit_constrained_reference(midfield_paint(target), [1920, 1080], H, model=model)
    probes = np.array([[-8., -8.], [0, 0], [8, 8], [0, 25]])
    assert np.max(np.linalg.norm(r.project(fit["field_to_native"], probes)
                                 - r.project(target, probes), axis=1)) < .01
    assert fit["constrained_jacobian_rank"] == fit["parameter_count"]
    assert not fit["metric_certified"]
    np.testing.assert_allclose(np.array(fit["field_to_native"])[2], H[2], atol=1e-12)


def test_single_line_cannot_identify_affine_update():
    with pytest.raises(ValueError, match="does not identify"):
        fit_constrained_reference(midfield_paint(H)[1:], [1920, 1080], H)


def test_reversed_seed_is_rejected_before_fitting():
    bad = H.copy()
    bad[:, 0] *= -1
    with pytest.raises(ValueError, match="invalid visible geometry"):
        fit_constrained_reference(midfield_paint(H), [1920, 1080], bad)


def test_end_mid_blend_retains_observed_charts_and_halfway_reference():
    full = [[-r.HALF_L, -r.HALF_W], [r.HALF_L, -r.HALF_W],
            [r.HALF_L, r.HALF_W], [-r.HALF_L, r.HALF_W]]
    shifted = np.array([[1., 0, 8.], [0, 1, -3.], [0, 0, 1.]]) @ H
    p = dict(status="approximate", metric_certified=False,
             projection_mode=r.END_MID_PROJECTION_MODE, field=r.FIELD,
             source=dict(game_id="synthetic", sha256="a"*64),
             charts=[dict(chart="left", field_to_reference=H.tolist(), support_world_polygon=full),
                     dict(chart="mid", field_to_reference=shifted.tolist(),
                          support_world_polygon=[[-10, -32], [10, -32], [10, 32], [-10, 32]])],
             frames=[dict(index=0, source_pts=0, source_time_base="1/90000",
                          native_size=[1920, 1080], reference_to_native=np.eye(3).tolist(),
                          boundary_offset_native_y_px=0., warnings=[])])
    atlas = r.RecoveryAtlas.from_payload(p)
    halfway = np.c_[np.zeros(21), np.linspace(-32, 32, 21)]
    np.testing.assert_allclose(atlas.base(halfway), r.project(shifted, halfway))
    np.testing.assert_allclose(atlas.base([[-45, 0]]), r.project(H, [[-45, 0]]))
    assert len(atlas.charts) == 2
    assert not atlas.support_mask([[45, 0]])[0]
    assert np.isfinite(atlas.base([[45, 0]])).all()  # Declared diagnostic extrapolation.
    no_mid = deepcopy(p)
    no_mid["charts"] = no_mid["charts"][:1]
    with pytest.raises(ValueError, match="actual midfield"):
        r.RecoveryAtlas.from_payload(no_mid)


def test_native_curve_residual_keeps_finite_ends_and_pole_gaps():
    pixels = np.array([[3., 2.], [13, 0], [6, 4]])
    curve = np.array([[0., 0.], [5, 0], [10, 0]])
    np.testing.assert_allclose(nearest_curve_residual(pixels, curve), [[0, 2], [3, 0], [0, 4]])
    gap = np.array([[0., 0.], [2, 0], [np.nan, np.nan], [8, 0], [10, 0]])
    assert np.linalg.norm(nearest_curve_residual(np.array([[5., 0.]]), gap)) == 3


def test_shared_polynomial_recovers_bowed_paint_across_measured_views():
    coefficients = np.zeros((6, 2))
    coefficients[3, 1] = 40.  # y displacement from reference x squared.
    coefficients[5, 0] = 60.  # x displacement from reference y squared.
    origin, scale = [960., 540.], 1920.

    def reference(world):
        base = r.project(H, world)
        return base+r.polynomial_basis(base, origin, scale, 2)@coefficients

    observations = []
    for transform in (np.eye(3), np.array([[1.02, .01, 35.], [-.01, 1.02, -9.], [0, 0, 1.]])):
        features = []
        for name in ("touch_near", "touch_far", "halfway", "goal_left", "goal_right", "centre_circle"):
            world = r.feature_curve(dict(label=name, points_native=[[0, 0]]), samples=31)
            features.append(dict(label=name, points_native=r.project(transform, reference(world)).tolist()))
        observations.append(dict(reference_to_native=transform.tolist(), features=features))
    fit = fit_reference_polynomial(H, observations, [1920, 1080], degree=2)
    world = np.array([[-35., -20.], [0, 20], [30, 10], [-10, -10]])
    base = r.project(H, world)
    actual = base+r.polynomial_basis(base, fit["origin_native_px"], fit["scale_native_px"],
                                     fit["degree"])@np.array(fit["coefficients_native_px"])
    assert np.max(np.linalg.norm(actual-reference(world), axis=1)) < .1
    assert fit["paint_jacobian_rank"] == 12
    assert not fit["metric_certified"]


def test_bounded_polynomial_tail_cannot_grow_like_cubic_extrapolation():
    pixels = np.array([[1e6, 1e6], [1e9, 1e9]])
    plain = r.polynomial_basis(pixels, [0, 0], 1., 3)
    bounded = r.polynomial_basis(pixels, [0, 0], 1., 3, profile="bounded_radial_v1", radial_scale=2.)
    assert np.max(abs(plain[1])) > np.max(abs(plain[0]))*1e6
    assert np.max(abs(bounded[1])) < np.max(abs(bounded[0]))/900


def test_saved_polynomial_is_used_by_supported_inverse_and_warning_gate():
    full = [[-r.HALF_L, -r.HALF_W], [r.HALF_L, -r.HALF_W],
            [r.HALF_L, r.HALF_W], [-r.HALF_L, r.HALF_W]]
    coefficients = np.zeros((6, 2))
    coefficients[3, 1], coefficients[5, 0] = 40., 60.
    p = dict(status="approximate", metric_certified=False,
             projection_mode=r.POLYNOMIAL_REFERENCE_PROJECTION_MODE, field=r.FIELD,
             source=dict(game_id="synthetic", sha256="a"*64),
             reference_polynomial=dict(degree=2, origin_native_px=[960, 540], scale_native_px=1920.,
                coefficients_native_px=coefficients.tolist(), basis_profile="bounded_radial_v1", radial_scale=2.),
             charts=[dict(chart="left", field_to_reference=H.tolist(), support_world_polygon=full)],
             frames=[dict(index=0, source_pts=0, source_time_base="1/90000",
                          native_size=[1920, 1080], reference_to_native=np.eye(3).tolist(),
                          boundary_offset_native_y_px=0., warnings=[])])
    atlas = r.RecoveryAtlas(r.RecoveryAtlas.from_payload(p).document)
    mapping = AdjustmentAtlas.create(atlas, "b"*64, []).frame(
        0, "1/90000", source_sha256="a"*64, native_size=[1920, 1080])
    world = [[-20., -10.], [0, 20]]
    pixels = mapping.project(world)
    assert np.max(abs(pixels-r.project(H, world))) > .1
    located = mapping.locate(pixels.tolist())
    np.testing.assert_allclose([row["field_position_m"] for row in located], world, atol=1e-6)
    assert all(not row["metric_certified"] for row in located)
    p["frames"][0]["warnings"] = ["failed_adjacent_motion_step_1"]
    warned = r.RecoveryAtlas.from_payload(p).frame(0, "1/90000", source_sha256="a"*64, native_size=[1920, 1080])
    assert warned.public_projection(world)["status"] == "unavailable"
