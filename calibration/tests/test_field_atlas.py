"""Synthetic geometry and binding tests; no real footage or accuracy claims."""
from copy import deepcopy

import numpy as np
import pytest

from calibration.field_atlas import FieldAtlas, SCHEMA, digest, save


def document():
    payload = dict(
        status="approximate", metric_certified=False,
        source=dict(game_id="synthetic", sha256="a" * 64),
        field=dict(length_m=20., width_m=10., axes="centre,+x_camera_right,+y_near",
                   dimensions_provenance="Synthetic test dimensions"),
        spatial=dict(
            field_to_reference_models=[[[30., 0, 400.], [0, 30., 300.], [0, 0, 1]]] * 3,
            blend=dict(end_x_start_m=2., end_x_span_m=4., near_y_start_m=2., near_y_span_m=2.,
                       near_right_x_start_m=0., near_right_x_span_m=3.),
            far_boundary=dict(reference_y_coefficients=[152., 1., -.5], reference_x_origin_px=400.,
                              reference_x_scale_px=300., protected_y_m=-3.)),
        frames=[dict(source_pts=3001, source_time_base="1/30", native_size=[800, 600],
                     reference_to_native=[[1., 0, 10.], [0, 1., 20.], [0, 0, 1.]],
                     boundary_offset_native_y_px=-2., warnings=[])])
    return dict(schema=SCHEMA, payload=payload, payload_sha256=digest(payload))


def frame(doc=None):
    atlas = FieldAtlas(doc or document())
    return atlas.frame(3001, "1/30", source_sha256="a" * 64, native_size=[800, 600])


def reseal(doc):
    doc["payload_sha256"] = digest(doc["payload"])
    return doc


def test_numeric_sign_boundary_and_protected_interior():
    view = frame()
    pts = np.array([[0., -5.], [0., -3.], [0., 0.], [0., 5.]])
    # The boundary moves +2 reference y then -2 native y; the interior only translates.
    assert np.allclose(view.project(pts), [[410, 170], [410, 230], [410, 320], [410, 470]])
    assert view.locate([[410, 320]])[0]["field_position_m"] == pytest.approx([0, 0], abs=1e-7)


def test_inverse_uses_nonlinear_map_with_native_after_transform_correction():
    doc = document()
    doc["payload"]["frames"][0]["reference_to_native"] = [[1.05, .06, -12.], [.02, .96, 4.], [.0001, -.00008, 1.]]
    doc["payload"]["frames"][0]["boundary_offset_native_y_px"] = -4.
    view = frame(reseal(doc))
    truth = np.random.default_rng(42).uniform([-9.5, -4.9], [9.5, 4.9], (300, 2))
    results = view.locate(view.project(truth))
    assert all(r["status"] == "approximate" and not r["metric_certified"] for r in results)
    assert np.max(np.linalg.norm(np.array([r["field_position_m"] for r in results]) - truth, axis=1)) < 1e-5
    assert all(r["uncertainty_m"] is None for r in results)


@pytest.mark.parametrize("pixels", [[[float("nan"), 1]], [[-1, 100]], [[800, 100]], [[100, 600]], [[799, 599]]])
def test_nonfinite_off_image_and_outside_field_do_not_extrapolate(pixels):
    result = frame().locate(pixels)[0]
    assert result["status"] == "unavailable" and result["field_position_m"] is None


def test_exact_pts_domain_and_source_are_required():
    atlas = FieldAtlas(document())
    for pts, tb, sha, size in [(3000, "1/30", "a" * 64, [800, 600]),
                               (3001, "1/30", "b" * 64, [800, 600]),
                               (3001, "1/30", "a" * 64, [400, 300]),
                               (3001., "1/30", "a" * 64, [800, 600]),
                               (3001, "0/30", "a" * 64, [800, 600])]:
        with pytest.raises(ValueError):
            atlas.frame(pts, tb, source_sha256=sha, native_size=size)
    equivalent = atlas.frame(6002, "1/60", source_sha256="a" * 64, native_size=[800, 600])
    assert equivalent.locate([[410, 320]])[0]["status"] == "approximate"


def test_review_flag_withholds_public_position_but_keeps_candidate_for_review():
    doc = document()
    doc["payload"]["frames"][0]["warnings"] = ["scene_motion_diagnostic_failed"]
    result = frame(reseal(doc)).locate([[410, 320]])[0]
    assert result["status"] == "unavailable" and result["field_position_m"] is None
    assert result["reason"] == "frame_requires_review"
    assert result["candidate_field_position_m"] == pytest.approx([0, 0], abs=1e-7)


def test_reflection_or_fold_does_not_become_valid_ground():
    doc = document()
    doc["payload"]["frames"][0]["reference_to_native"] = [[-1., 0, 800.], [0, 1., 0], [0, 0, 1.]]
    assert frame(reseal(doc)).locate([[400, 300]])[0]["status"] == "unavailable"


def test_checksum_certification_and_duplicate_frame_rejected():
    doc = document()
    doc["payload"]["field"]["width_m"] = 11
    with pytest.raises(ValueError, match="checksum"):
        FieldAtlas(doc)
    doc = document()
    doc["payload"]["metric_certified"] = True
    with pytest.raises(ValueError, match="uncertified"):
        FieldAtlas(reseal(doc))
    doc = document()
    doc["payload"]["frames"].append(deepcopy(doc["payload"]["frames"][0]))
    with pytest.raises(ValueError, match="unique"):
        FieldAtlas(reseal(doc))


def test_portable_package_and_immutable_output(tmp_path):
    path = tmp_path / "renamed" / "atlas.json"
    save(path, document()["payload"])
    view = FieldAtlas.load(path).frame(3001, "1/30", source_sha256="a" * 64, native_size=[800, 600])
    assert view.locate([[410, 320]])[0]["status"] == "approximate"
    with pytest.raises(FileExistsError):
        save(path, document()["payload"])
