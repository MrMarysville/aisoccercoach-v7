"""Paint-centre behavior, including the thick-line failure seen in browser review."""
import cv2
import numpy as np
import pytest

from calibration.tools.measure_field_annotations import refine_polyline


@pytest.mark.parametrize("width", [2, 6, 14, 22])
def test_same_centre_for_thin_and_thick_paint_from_biased_proposal(width):
    image = np.full((200, 500, 3), [50, 100, 55], dtype=np.uint8)
    cv2.line(image, (10, 100), (490, 100), (210, 210, 210), width, cv2.LINE_AA)
    found, raw = refine_polyline(image, [[20, 106], [480, 106]])
    assert len(found) >= 50
    assert np.max(abs(found[:, 1] - 100)) < .5
    assert all(row["measurement_method"] == "cross_section_half_height_midpoint" for row in raw)


def test_absent_paint_does_not_become_an_observation():
    image = np.full((200, 500, 3), [50, 100, 55], dtype=np.uint8)
    found, raw = refine_polyline(image, [[20, 106], [480, 106]])
    assert len(found) == 0
    assert raw and all(not row["accepted"] for row in raw)


def test_named_outer_boundary_can_have_a_brown_exterior_apron():
    image = np.full((200, 500, 3), [50, 100, 55], dtype=np.uint8)
    image[100:] = [90, 100, 110]
    cv2.line(image, (10, 100), (490, 100), (210, 210, 210), 14, cv2.LINE_AA)
    found, _ = refine_polyline(image, [[20, 106], [480, 106]], required_grass_sides=1)
    assert len(found) >= 50
    assert np.max(abs(found[:, 1] - 100)) < .5


def test_thin_named_strip_does_not_jump_to_brighter_neighboring_marking():
    image = np.full((200, 500, 3), [50, 100, 55], dtype=np.uint8)
    cv2.line(image, (10, 100), (490, 100), (165, 165, 165), 2, cv2.LINE_AA)
    cv2.line(image, (10, 114), (490, 114), (245, 245, 245), 2, cv2.LINE_AA)
    found, _ = refine_polyline(image, [[20, 99], [480, 99]], radius=6.)
    assert len(found) >= 50
    assert np.max(abs(found[:, 1] - 100)) < .5
