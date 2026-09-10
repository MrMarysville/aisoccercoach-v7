"""Interrupted propagation must retain exact maps and fail closed on changed evidence."""
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

from calibration import field_recovery as recovery
from calibration.alignment_trial import sha256_file
from calibration.tools import propagate_field_recovery as runner


def inputs(root):
    source = dict(game_id="synthetic", sha256="a" * 64)
    game = dict(game_id="synthetic", source_sha256=source["sha256"], source_bytes=1,
                clips=[dict(id="closed", half=1, start_s=10, end_s=12, role="reserved_evaluation")])
    catalog = dict(games=[dict(game_id="synthetic", video_sha256=source["sha256"], video_bytes=1,
                              halves=[dict(half=1, start_pts_s=0, end_pts_s=20)])])
    (root / "catalog.json").write_text(json.dumps(catalog))
    payload = dict(games=[game])
    (root / "plan.json").write_text(json.dumps(dict(schema="field-atlas-clips-v1", payload=payload,
        payload_sha256=recovery.digest(payload), catalog_sha256=sha256_file(root / "catalog.json"))))
    anchor = None
    for name, stamps, interval in (("clip", [5000], [5, 5.1]),
                                    ("frames-forward", [5100, 5200, 5300], [5.05, 5.35]),
                                    ("frames-backward", [4700, 4800, 4900], [4.7, 5])):
        directory = root / name
        directory.mkdir()
        rows = []
        for index, pts in enumerate(stamps):
            image = directory / f"{index}.png"
            cv2.imwrite(str(image), np.full((180, 320, 3), index + 1, np.uint8))
            rows.append(dict(index=index, file=image.name, game_id=source["game_id"],
                source_sha256=source["sha256"], source_pts=pts, source_time_base="1/1000",
                source_seconds=pts / 1000, native_size=[320, 180], image_sha256=sha256_file(image)))
        manifest = dict(schema="alignment-trial-frames-v1", source=source, frames=rows,
                        sampling_fps=10, requested_interval_s=interval)
        (directory / "frames.json").write_text(json.dumps(manifest))
        if name == "clip":
            anchor = rows[0]
    payload = dict(status="approximate", metric_certified=False, source=source, field=recovery.FIELD,
        projection_mode=recovery.SINGLE_REFERENCE_PROJECTION_MODE, blend=recovery.BLEND,
        provenance=dict(fit_frame_indices=[0], check_frame_indices=[]),
        charts=[dict(chart="mid", field_to_reference=[[2, 0, 160], [0, 2, 90], [0, 0, 1]],
                     support_world_polygon=[[-20, -20], [20, -20], [20, 20], [-20, 20]])],
        frames=[anchor | dict(reference_to_native=np.eye(3).tolist(), boundary_offset_native_y_px=0.,
                               warnings=["anchor_geometry_warning", "propagated_without_paint_reference"])])
    recovery.RecoveryAtlas.from_payload(payload).save(root / "atlas.json")
    return ["--atlas", str(root / "atlas.json"), "--frames", str(root / "clip/frames.json"),
            "--plan", str(root / "plan.json"), "--catalog", str(root / "catalog.json"),
            "--reuse-frames", str(root), "--seconds", ".3", "--score-every", "1",
            "--independent-stride", "1", "--render-every", "2"]


def invoke(monkeypatch, args, out, *extra):
    monkeypatch.setattr(sys, "argv", ["propagate", *args, "--out", str(out), *extra])
    runner.main()


@pytest.mark.parametrize("paint_lock", [False, True])
def test_pause_crash_resume_matches_uninterrupted_maps_and_retains_failures(tmp_path, monkeypatch, paint_lock):
    args = inputs(tmp_path)
    if paint_lock:
        args.append("--paint-lock")
        def correction(image, atlas, record, layers, **kwargs):
            transform = np.array(record["reference_to_native"])
            transform[1, 2] += .25
            return transform, [dict(observations=50)]
        monkeypatch.setattr(runner, "paint_lock", correction)
    def motion(previous, current):
        failed = current[0, 0, 0] == 2
        return (None if failed else np.array([[1., 0., 2.], [0., 1., -1.], [0., 0., 1.]]),
                dict(image_registration_pass=not failed, reason="synthetic missing motion"))
    monkeypatch.setattr(recovery, "adjacent_motion", motion)
    monkeypatch.setattr(recovery, "reference_features", lambda im: int(im[0, 0, 0]))
    monkeypatch.setattr(recovery, "register_reference", lambda a, b, size:
                        (None, dict(reason="synthetic missing registration")) if b == 2
                        else (np.eye(3), dict(image_registration_pass=True)))
    monkeypatch.setattr(runner, "gauge_frame", lambda im, *a, **kw:
        (dict(measured=bool(im[0, 0, 0] != 2), samples=50, median_px=1., p95_px=2.), {}))
    baseline, resumed = tmp_path / "baseline", tmp_path / "resumed"
    invoke(monkeypatch, args, baseline)
    invoke(monkeypatch, args, resumed, "--max-frames", "1")
    assert not json.loads((resumed / "report.json").read_text())["complete"]
    assert not list(resumed.glob("atlas-*.json"))
    first = resumed / "progress-forward/000001.json"
    original_bytes, original_mtime = first.read_bytes(), first.stat().st_mtime_ns
    read_image = runner.read_image
    def crash(directory, row):
        if Path(directory).name == "frames-forward" and row["index"] == 2:
            raise OSError("simulated interruption")
        return read_image(directory, row)
    monkeypatch.setattr(runner, "read_image", crash)
    with pytest.raises(OSError, match="interruption"):
        invoke(monkeypatch, args, resumed, "--resume")
    assert (resumed / "progress-forward/000002.json").exists()
    monkeypatch.setattr(runner, "read_image", read_image)
    invoke(monkeypatch, args, resumed, "--resume")
    for direction in ("forward", "backward"):
        a = json.loads((baseline / f"result-{direction}.json").read_text())
        b = json.loads((resumed / f"result-{direction}.json").read_text())
        assert a["records"] == b["records"] and a["rows"] == b["rows"]
        assert b["complete"] and b["failed_adjacent_steps"]
        assert not b["survival_gauge_p95"]["survived_whole_window"]
        assert b["survival_gauge_p95"]["unmeasured_frames"] == 1
        assert b["survival_independent_p95"]["unmeasured_frames"] == 1
        atlas_file = resumed / f"atlas-{direction}.json"
        assert atlas_file.read_bytes() == (baseline / atlas_file.name).read_bytes()
        atlas = recovery.RecoveryAtlas.load(atlas_file)
        assert "fit_frame_indices" not in atlas.payload["provenance"]
        assert atlas.payload["provenance"]["parent_provenance"]["fit_frame_indices"] == [0]
        for record in atlas.payload["frames"]:
            assert "anchor_geometry_warning" in record["warnings"]
            assert "propagated_without_paint_reference" in record["warnings"]
            mapping = atlas.frame(record["source_pts"], record["source_time_base"],
                                 source_sha256="a" * 64, native_size=[320, 180])
            assert mapping.public_projection([[0, 0]])["status"] == "unavailable"
    invoke(monkeypatch, args, resumed, "--resume")
    assert first.read_bytes() == original_bytes and first.stat().st_mtime_ns == original_mtime
    with pytest.raises(ValueError, match="settings differ"):
        invoke(monkeypatch, args, resumed, "--resume", "--lock-radius", "11")
    with (resumed / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="already in use"):
            invoke(monkeypatch, args, resumed, "--resume")
    changed = json.loads(first.read_text())
    changed["payload"]["record"]["reference_to_native"][0][2] += 1
    first.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="checkpoint"):
        invoke(monkeypatch, args, resumed, "--resume")


def test_reuse_rejects_reserved_times_and_changed_frames_before_processing(tmp_path, monkeypatch):
    args = inputs(tmp_path)
    manifest = json.loads((tmp_path / "frames-forward/frames.json").read_text())
    plan = json.loads((tmp_path / "plan.json").read_text())
    game = plan["payload"]["games"][0]
    changed = deepcopy(manifest)
    changed["requested_interval_s"] = [10, 10.3]
    with pytest.raises(ValueError, match="Closed reserved"):
        runner.validate_manifest(changed, game, [10, 10.3], 10)
    for update in (dict(source_seconds=5.2), dict(source_sha256="b" * 64),
                   dict(source_pts=10000), dict(source_time_base="1/90000")):
        changed = deepcopy(manifest)
        changed["frames"][0].update(update)
        with pytest.raises(ValueError):
            runner.validate_manifest(changed, game, [5.05, 5.35], 10)
    out = tmp_path / "run"
    invoke(monkeypatch, args, out, "--max-frames", "1")
    manifest_path = tmp_path / "frames-forward/frames.json"
    original = manifest_path.read_bytes()
    manifest_path.write_text(json.dumps(manifest | dict(elapsed_seconds=99)))
    with pytest.raises(ValueError, match="Extension frames differ"):
        invoke(monkeypatch, args, out, "--resume")
    manifest_path.write_bytes(original)
    image = tmp_path / "frames-forward/0.png"
    image.write_bytes(b"changed completed frame")
    with pytest.raises(ValueError, match="frame hash mismatch"):
        invoke(monkeypatch, args, out, "--resume")
    monkeypatch.setattr(runner, "read_image", lambda *a: pytest.fail("Reserved run accessed images"))
    with pytest.raises(ValueError, match="Closed reserved"):
        invoke(monkeypatch, args, tmp_path / "reserved-run", "--seconds", "5")
