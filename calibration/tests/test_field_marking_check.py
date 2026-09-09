"""Analytic boundary checks and frozen-score compatibility; synthetic data only."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from calibration import field_recovery as r
from calibration.field_atlas import FieldAtlas
from calibration.field_marking_check import check_marking_orientation, checked_marking_curve, helper_sha256
from calibration.tools.diagnose_field_recovery import diagnose
from calibration.tools.score_field_recovery import score


def restricted(function, domain, known=True):
    if known:
        atlas = object.__new__(r.RecoveryAtlas)
        atlas.projection_mode = r.V7_PROJECTION_MODE
    else:
        atlas = object.__new__(FieldAtlas)
    atlas.projection_domain_mask = domain
    def project(world):
        world = np.asarray(world, float)
        with np.errstate(all="ignore"):
            pixels = np.array(function(world), dtype=float)
        pixels[~domain(world)] = np.nan
        return pixels
    return SimpleNamespace(atlas=atlas, project=project)


def test_linear_boundary_uses_inward_positive_derivative_and_preserves_interior():
    mapping = restricted(lambda p: p @ np.array([[2., .3], [.4, 3.]]), lambda p: p[:, 0] <= 0)
    world = np.array([[0., 1.], [-2., 1.]])
    old = check_marking_orientation(mapping, world)
    new = check_marking_orientation(mapping, world, "domain_boundary_v2")
    assert old["positive"].tolist() == [False, True]
    assert new["positive"].tolist() == [True, True]
    np.testing.assert_allclose(new["determinant"], 5.88, atol=1e-10)
    assert new["one_sided_axes"].tolist() == [[True, False], [False, False]]
    assert new["determinant"][1:].tobytes() == old["determinant"][1:].tobytes()


def test_projective_boundary_agrees_with_analytic_determinant():
    h = np.array([[2., .1, 4.], [.2, 3., 5.], [.03, .02, 1.]])
    mapping = restricted(lambda p: r.project(h, p), lambda p: p[:, 0] <= 0)
    world = np.c_[np.zeros(21), np.linspace(-2, 2, 21)]
    checked = check_marking_orientation(mapping, world, "domain_boundary_v2")
    expected = np.linalg.det(h) / (np.c_[world, np.ones(len(world))] @ h[2]) ** 3
    np.testing.assert_allclose(checked["determinant"], expected, rtol=1e-7)
    assert checked["positive"].all()


def test_forward_inward_stencil_matches_analytic_linear_orientation():
    mapping = restricted(lambda p: np.c_[2*p[:, 0], 3*p[:, 1]], lambda p: p[:, 0] >= 0)
    checked = check_marking_orientation(mapping, [[0., 1.]], "domain_boundary_v2")
    assert checked["positive"].tolist() == [True]
    assert checked["one_sided_axes"].tolist() == [[True, False]]
    np.testing.assert_allclose(checked["determinant"], [6.], atol=1e-10)


def test_second_order_stencil_matches_smooth_quadratic_at_both_domain_edges():
    def function(p):
        x, y = p.T
        return np.c_[x + .1*x*x + .01*y*y, y + .2*y*y + .05*x*y]
    mapping = restricted(function, lambda p: (p <= 0).all(axis=1))
    world = np.array([[0., -.2], [-.2, 0.], [0., 0.], [-.2, -.2]])
    checked = check_marking_orientation(mapping, world, "domain_boundary_v2")
    x, y = world.T
    expected = (1 + .2*x)*(1 + .4*y + .05*x) - .02*y*.05*y
    np.testing.assert_allclose(checked["determinant"], expected, atol=1e-10)
    assert checked["one_sided_axes"].tolist() == [[True, False], [False, True], [True, True], [False, False]]


@pytest.mark.parametrize("function", [lambda p: np.c_[-p[:, 0], p[:, 1]],
                                      lambda p: np.c_[p[:, 0], p[:, 1]**2]])
def test_boundary_reversal_and_fold_remain_rejected(function):
    mapping = restricted(function, lambda p: p[:, 0] <= 0)
    world = np.array([[0., -1.], [-.5, -1.]])
    old = check_marking_orientation(mapping, world)
    new = check_marking_orientation(mapping, world, "domain_boundary_v2")
    assert (new["determinant"] < 0).all()
    assert not new["positive"].any()
    assert new["determinant"][1:].tobytes() == old["determinant"][1:].tobytes()


def test_real_poles_inside_domain_are_not_rescued():
    mapping = restricted(lambda p: p / p[:, :1], lambda p: np.ones(len(p), bool))
    world = np.array([[0., 1.], [.001, 1.]])
    with np.errstate(all="ignore"):
        checked = check_marking_orientation(mapping, world, "domain_boundary_v2")
    assert not checked["positive"].any()
    assert not checked["one_sided_axes"].any()


@pytest.mark.parametrize("domain", [lambda p: (p[:, 0] <= 0) & (p[:, 0] >= -.0015),
                                     lambda p: p[:, 0] == 0])
def test_insufficient_inward_stencil_remains_unknown(domain):
    checked = check_marking_orientation(restricted(lambda p: p, domain), [[0., 1.]], "domain_boundary_v2")
    assert not checked["positive"].any()
    assert not np.isfinite(checked["determinant"]).any()
    assert not checked["one_sided_axes"].any()


def test_nonfinite_inward_probe_and_legacy_unknown_domains_have_no_fallback():
    def pole(p):
        result = p.copy()
        result[np.isclose(p[:, 0], -.001, rtol=0, atol=1e-12)] = np.nan
        return result
    known = restricted(pole, lambda p: p[:, 0] <= 0)
    legacy = restricted(lambda p: p, lambda p: p[:, 0] <= 0, known=False)
    for mapping in (known, legacy):
        checked = check_marking_orientation(mapping, [[0., 1.]], "domain_boundary_v2")
        assert not checked["positive"].any()
        assert not checked["one_sided_axes"].any()


def two_chart_atlas():
    h = np.array([[5., 0., 960.], [0., 5., 500.], [0., 0., 1.]])
    hull = [[-r.HALF_L, -r.HALF_W], [r.HALF_L, -r.HALF_W],
            [r.HALF_L, r.HALF_W], [-r.HALF_L, r.HALF_W]]
    payload = dict(status="approximate", metric_certified=False, source=dict(game_id="synthetic", sha256="a"*64),
                   field=r.FIELD, projection_mode=r.V7_PROJECTION_MODE, blend=r.BLEND, spatial_boundary=None,
                   protected_y_m=-r.BOX18_HALF_W,
                   charts=[dict(chart=name, reference_frame_index=0, field_to_reference=h.tolist(),
                                support_world_polygon=hull) for name in ("left", "mid")],
                   frames=[dict(index=0, source_pts=9000, source_time_base="1/90000", native_size=[1920, 1080],
                                reference_to_native=np.eye(3).tolist(), boundary_offset_native_y_px=0., warnings=[])])
    return r.RecoveryAtlas.from_payload(payload)


def test_actual_two_chart_halfway_is_complete_only_under_explicit_v2():
    atlas = two_chart_atlas()
    frame = atlas.frame(9000, "1/90000", source_sha256="a"*64, native_size=[1920, 1080])
    world = r.feature_curve(dict(label="halfway", points_native=[[0, 0]]), samples=8001)
    old, old_stats = checked_marking_curve(frame, world)
    new, new_stats = checked_marking_curve(frame, world, "domain_boundary_v2")
    assert old_stats["positive_orientation_points"] == 5875
    assert new_stats["positive_orientation_points"] == 8001
    assert new_stats["one_sided_points"] == 2126
    assert np.isfinite(new).all()
    np.testing.assert_array_equal(old[np.isfinite(old).all(axis=1)], new[np.isfinite(old).all(axis=1)])
    # Independent literal old algorithm establishes byte-identical central output.
    expected = frame.project(world)
    dx = (frame.project(world + [.001, 0]) - frame.project(world - [.001, 0])) / .002
    dy = (frame.project(world + [0, .001]) - frame.project(world - [0, .001])) / .002
    det = dx[:, 0]*dy[:, 1] - dx[:, 1]*dy[:, 0]
    expected[~(np.isfinite(det) & (det > 0))] = np.nan
    assert old.tobytes() == expected.tobytes()
    assert [c["chart"] for c in atlas.charts] == ["left", "mid"]


def make_score_inputs(tmp_path):
    two_chart_atlas().save(tmp_path / "atlas.json")
    source = dict(game_id="synthetic", sha256="a"*64)
    frame = dict(index=0, source_pts=9000, source_time_base="1/90000", source_seconds=.1,
                 native_size=[1920, 1080], source_sha256="a"*64, image_sha256="b"*64)
    (tmp_path / "frames.json").write_text(json.dumps(dict(source=source, frames=[frame])))
    pixels = [[963., 600.], [963., 625.], [963., 650.]]
    checks = dict(source_sha256="a"*64, measurements=[dict(frame_index=0, label="halfway",
                  selected_count=3, searched_count=3, feature_id="near-halfway-fixture", points_native=pixels)])
    (tmp_path / "checks.json").write_text(json.dumps(checks))


@pytest.mark.parametrize("policy", ["central_v1", "domain_boundary_v2"])
def test_saved_score_and_signed_diagnostic_agree_with_frozen_policy(tmp_path, policy):
    make_score_inputs(tmp_path)
    paths = [tmp_path / name for name in ("atlas.json", "frames.json", "checks.json")]
    score(*paths, tmp_path / "score.json", orientation_policy=policy)
    frozen = json.loads((tmp_path / "score.json").read_text())
    assert frozen["orientation_policy"] == policy
    assert frozen["orientation_helper_sha256"] == helper_sha256()
    result = diagnose(*paths[:2], tmp_path / "score.json", tmp_path / "diagnostic.json")
    assert result["max_unsigned_distance_disagreement_px"] == 0
    if policy == "domain_boundary_v2":
        assert frozen["rows"][0]["orientation"]["one_sided_points"] == 2126
        assert frozen["rows"][0]["max_px"] == pytest.approx(3.)
    else:
        assert frozen["rows"][0]["median_px"] > 40
        del frozen["orientation_policy"]
        del frozen["orientation_helper_sha256"]
        del frozen["rows"][0]["orientation"]
        (tmp_path / "old-score.json").write_text(json.dumps(frozen))
        result = diagnose(*paths[:2], tmp_path / "old-score.json", tmp_path / "old-diagnostic.json")
        assert result["max_unsigned_distance_disagreement_px"] == 0


def test_invalid_policy_and_changed_helper_fail_before_output(tmp_path):
    with pytest.raises(ValueError, match="Unknown"):
        score("missing", "missing", "missing", tmp_path / "out", orientation_policy="guess")
    make_score_inputs(tmp_path)
    score(tmp_path / "atlas.json", tmp_path / "frames.json", tmp_path / "checks.json", tmp_path / "score.json")
    frozen = json.loads((tmp_path / "score.json").read_text())
    frozen["orientation_helper_sha256"] = "0"*64
    (tmp_path / "changed.json").write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="helper hash"):
        diagnose(tmp_path / "atlas.json", tmp_path / "frames.json", tmp_path / "changed.json", tmp_path / "out")
    assert not (tmp_path / "out").exists()
