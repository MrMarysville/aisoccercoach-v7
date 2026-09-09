"""Explicit turf color assumptions preserve every other source-paint condition."""
import hashlib
import json
import sys

import cv2
import numpy as np
import pytest

from calibration.alignment_trial import sha256_file
from calibration.tools import measure_field_annotations as m


OLD_FIELDS = ("coarse_xy", "native_xy", "ridge_strength", "grass_support", "accepted",
              "normal_offset_px", "paint_width_px", "measurement_method")
COARSE = [[20, 106], [480, 106]]


def stripe(hue=60, saturation=100, value=110, width=14, paint=210):
    color = cv2.cvtColor(np.uint8([[[hue, saturation, value]]]), cv2.COLOR_HSV2BGR)[0, 0]
    image = np.full((200, 500, 3), color, dtype=np.uint8)
    cv2.line(image, (10, 100), (490, 100), (paint, paint, paint), width, cv2.LINE_AA)
    return image


def original_fields(points, raw):
    return dict(points=points.tolist(), samples=[{k: row[k] for k in OLD_FIELDS} for row in raw])


def test_default_numerical_outputs_match_frozen_original_sampler():
    # Recorded from the unchanged seed sampler on these synthetic inputs.
    cases = [stripe(), stripe(hue=15, saturation=69, value=109), stripe(width=2), stripe(width=22)]
    values = []
    for image in cases:
        implicit = m.refine_polyline(image, COARSE)
        explicit = m.refine_polyline(image, COARSE, turf_profile="green_v1")
        assert original_fields(*implicit) == original_fields(*explicit)
        values.append(original_fields(*implicit))
    digest = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
    assert digest == "5c8928809f6395ed137521d1f87f02f4e38f67e8e0567a40eb71a4b16017080a"


def test_green_strip_both_profiles_agree_exactly():
    image = stripe()
    a = m.refine_polyline(image, COARSE, turf_profile="green_v1")
    b = m.refine_polyline(image, COARSE, turf_profile="warm_green_v1")
    assert original_fields(*a) == original_fields(*b)
    assert len(a[0]) >= 50


def test_brown_turf_requires_explicit_profile_and_recovers_subpixel_center():
    image = stripe(hue=15, saturation=69, value=109, width=16)
    old, rejected = m.refine_polyline(image, COARSE)
    found, accepted = m.refine_polyline(image, COARSE, turf_profile="warm_green_v1")
    assert len(old) == 0 and len(found) >= 50
    assert np.max(abs(found[:, 1] - 100)) < .5
    assert all(row["rejection_reasons"] == ["turf_neighbor_support"] for row in rejected)
    assert all(row["rejection_reasons"] == [] for row in accepted)
    assert all(row["grass_support_positive_side"] == row["grass_support_negative_side"] == 1 for row in accepted)


@pytest.mark.parametrize("hue,expected", [(0, True), (24, True), (99, True), (100, False),
                                        (170, False), (171, True), (179, True)])
def test_warm_hue_bounds_remain_explicit(hue, expected):
    found, _ = m.refine_polyline(stripe(hue=hue), COARSE, turf_profile="warm_green_v1")
    assert bool(len(found)) == expected


@pytest.mark.parametrize("image", [stripe(saturation=0), stripe(hue=120), stripe(saturation=22), stripe(value=35)])
def test_grey_blue_and_low_saturation_value_still_lack_support(image):
    found, raw = m.refine_polyline(image, COARSE, turf_profile="warm_green_v1")
    assert len(found) == 0
    assert all("turf_neighbor_support" in row["rejection_reasons"] for row in raw)


@pytest.mark.parametrize("image,reason", [(stripe(hue=15, width=70), "half_height_crossings"),
                                         (stripe(hue=15, saturation=69, value=109, paint=82), "ridge_strength")])
def test_unbounded_and_low_contrast_paint_still_rejected(image, reason):
    found, raw = m.refine_polyline(image, COARSE, turf_profile="warm_green_v1")
    assert len(found) == 0
    assert all(reason in row["rejection_reasons"] for row in raw)


def test_one_or_two_neighbor_sides_rule_is_preserved():
    image = stripe(hue=15)
    image[110:] = [90, 90, 90]
    two, raw = m.refine_polyline(image, COARSE, turf_profile="warm_green_v1")
    one, _ = m.refine_polyline(image, COARSE, required_grass_sides=1, turf_profile="warm_green_v1")
    assert len(two) == 0 and len(one) >= 50
    assert all("turf_neighbor_support" in row["rejection_reasons"] for row in raw)


def test_invalid_profile_fails_before_output_or_input_reads(tmp_path):
    with pytest.raises(ValueError, match="Unknown turf"):
        m.refine_polyline(None, None, turf_profile="automatic")
    with pytest.raises(ValueError, match="Unknown turf"):
        m.measure(tmp_path / "missing", tmp_path / "missing2", tmp_path / "out", {}, turf_profile="automatic")
    assert not (tmp_path / "out").exists()


def test_cli_records_explicit_numeric_profile_and_raw_conditions(tmp_path, monkeypatch, capsys):
    image_path = tmp_path / "frame.png"
    cv2.imwrite(str(image_path), stripe(hue=15, saturation=69, value=109))
    source = dict(sha256="a" * 64)
    frame = dict(index=0, file="frame.png", image_sha256=sha256_file(image_path),
                 source_pts=9000, source_time_base="1/90000", native_size=[500, 200])
    frames_path = tmp_path / "frames.json"
    frames_path.write_text(json.dumps(dict(source=source, frames=[frame])))
    annotation = dict(source=source, frames_manifest_sha256=sha256_file(frames_path),
                      features=[dict(id="synthetic-halfway", frame_index=0, label="halfway", frame=frame,
                                     points=[dict(native_xy=p) for p in COARSE])])
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(json.dumps(annotation))
    output = tmp_path / "measured"
    monkeypatch.setattr(sys, "argv", ["measure", "--frames", str(frames_path), "--annotations", str(annotation_path),
                                     "--out", str(output), "--turf-profile", "warm_green_v1"])
    m.main()
    assert json.loads(capsys.readouterr().out)["measured_points"] >= 50
    data = json.loads((output / "measurements.json").read_text())
    assert data["turf_profile"] == "warm_green_v1"
    assert data["turf_hsv_criterion"]["hue_any_of"] == [dict(lt=100), dict(gt=170)]
    assert data["turf_hsv_criterion"]["saturation_gt"] == 22
    assert data["turf_hsv_criterion"]["value_gt"] == 35
    assert data["paint_acceptance_criteria"]["ridge_strength_min"] == 4
    row = data["measurements"][0]
    assert row["required_grass_sides"] == 2
    assert all(sample["turf_profile"] == "warm_green_v1" for sample in row["samples"])
    assert all(sample["accepted"] == all(sample["conditions_passed"].values()) for sample in row["samples"])
    default_output = tmp_path / "default-measured"
    monkeypatch.setattr(sys, "argv", ["measure", "--frames", str(frames_path), "--annotations", str(annotation_path),
                                     "--out", str(default_output)])
    m.main()
    assert json.loads(capsys.readouterr().out)["measured_points"] == 0
    default = json.loads((default_output / "measurements.json").read_text())
    assert default["turf_profile"] == "green_v1"
    assert default["turf_hsv_criterion"]["hue_any_of"] == [dict(gt=24, lt=100)]
