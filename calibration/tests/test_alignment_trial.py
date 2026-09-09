"""Evidence-boundary regressions for the real-game visual trial helpers."""
from copy import deepcopy
import json
import subprocess

import numpy as np
import pytest

from calibration.alignment_trial import (decode_window, guard_interval, load_plan,
                                preview_to_native, sha256_file, source_time,
                                validate_boundary, validate_frame, verify_source)
from calibration.field_atlas import digest


def game():
    return dict(game_id="test", source_sha256="a" * 64, source_bytes=1,
                clips=[dict(id="closed", half=1, start_s=10, end_s=12, role="reserved_evaluation")])


@pytest.mark.parametrize("start,end", [(9, 11), (10, 12), (11, 13), (9, 13)])
def test_reserved_window_overlap_is_rejected(start, end):
    with pytest.raises(ValueError, match="Closed reserved"):
        guard_interval(game(), start, end)


def test_reserved_endpoints_are_half_open():
    guard_interval(game(), 9, 10)
    guard_interval(game(), 12, 13)


def test_native_crop_conversion_uses_pixel_centres():
    actual = preview_to_native([[0, 0], [99, 49]], crop_xywh=[100, 200, 200, 100],
                               preview_size=[100, 50], native_size=[1920, 1080])
    np.testing.assert_allclose(actual, [[100.5, 200.5], [298.5, 298.5]])
    with pytest.raises(ValueError, match="outside preview"):
        preview_to_native([[100, 0]], crop_xywh=[100, 200, 200, 100],
                          preview_size=[100, 50], native_size=[1920, 1080])


def test_source_and_exact_timestamp_rejected_independently():
    frame = dict(game_id="test", source_sha256="a" * 64, source_pts=1500,
                 source_time_base="1/1000", source_seconds=1.5)
    assert validate_frame(frame, game(), 1, 2) == source_time(135000, "1/90000")
    for update in (dict(game_id="wrong"), dict(source_sha256="b" * 64),
                   dict(source_pts=2000), dict(source_seconds=1.6)):
        with pytest.raises(ValueError):
            validate_frame(frame | update, game(), 1, 2)


def test_frozen_plan_rejects_game_hash_swap_even_with_recomputed_checksum(tmp_path):
    catalog = dict(games=[dict(game_id="test", video_sha256="a" * 64, video_bytes=1,
                              halves=[dict(half=1, start_pts_s=0, end_pts_s=20)])])
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog))
    payload = dict(games=[game()])
    plan = dict(schema="field-atlas-clips-v1", payload=payload, payload_sha256=digest(payload),
                catalog_sha256=sha256_file(catalog_path))
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    load_plan(path, catalog_path)
    changed = deepcopy(plan)
    changed["payload"]["games"][0]["source_sha256"] = "b" * 64
    changed["payload_sha256"] = digest(changed["payload"])
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Game/source"):
        load_plan(path, catalog_path)


def test_missing_opening_stays_missing():
    boundary = dict(half=1, boundary="start", status="not_observed", source_pts=None,
                    source_time_base=None, evidence="Play already underway at recording start.",
                    review_clip="opening.webm")
    validate_boundary(boundary)
    with pytest.raises(ValueError, match="Missing event"):
        validate_boundary(boundary | dict(source_pts=0, source_time_base="1/90000"))
    with pytest.raises(ValueError, match="exact source"):
        validate_boundary(boundary | dict(status="user_confirmed"))


def test_absent_boundary_preview_needs_explicit_reason():
    boundary = dict(half=2, boundary="end", status="uncertain", source_pts=None,
                    source_time_base=None, evidence="Source interrupted.")
    with pytest.raises(ValueError, match="review clip"):
        validate_boundary(boundary)
    validate_boundary(boundary | dict(review_unavailable_reason="No frames after interruption."))


def test_real_decoder_preserves_nonzero_source_pts_and_rejects_changed_source(tmp_path):
    source = tmp_path / "synthetic.mkv"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=64x48:rate=30:duration=2", "-vf", "setpts=PTS+5/TB",
                    "-c:v", "ffv1", str(source)], check=True, timeout=30)
    synthetic = game() | dict(video_path=str(source), source_sha256=sha256_file(source),
                              source_bytes=source.stat().st_size)
    receipt = verify_source(synthetic)
    result = decode_window(synthetic, receipt, 5.2, 5.7, 10, tmp_path / "decoded")
    assert len(result["frames"]) == 5
    assert result["frames"][0]["source_pts"] == 5200
    assert result["frames"][0]["source_time_base"] == "1/1000"
    assert result["frames"][0]["native_size"] == [64, 48]
    assert [f["source_seconds"] for f in result["frames"]] == [5.2, 5.3, 5.4, 5.5, 5.6]
    with pytest.raises(FileExistsError):
        decode_window(synthetic, receipt, 5.2, 5.7, 10, tmp_path / "decoded")
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Source changed"):
        decode_window(synthetic, receipt, 5.2, 5.7, 10, tmp_path / "changed")
