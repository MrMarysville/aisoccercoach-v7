"""Authoritative manual edits, inverse lookup and immutable session history."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from calibration import field_recovery as r
from calibration.field_adjustment import (AdjustmentAtlas, AdjustmentError, AdjustmentFrame, AdjustmentStore,
                                 file_hash, prepare, validate_geometry)


def parent(warnings=False, support=None):
    h = np.array([[8., .3, 960.], [.1, 5., 450.], [.001, .002, 1.]])
    hull = support or [[-r.HALF_L, -r.HALF_W], [r.HALF_L, -r.HALF_W],
                       [r.HALF_L, r.HALF_W], [-r.HALF_L, r.HALF_W]]
    frames = [dict(index=i, source_pts=9000+i*9000, source_time_base="1/90000", native_size=[1920, 1080],
                   reference_to_native=np.array([[1., .01*i, 12.*i], [-.01*i, 1., -4.*i],
                                                  [.00001*i, -.00002*i, 1.]]).tolist(),
                   boundary_offset_native_y_px=2.*i, warnings=["unverified_motion"] if warnings else []) for i in range(3)]
    return r.RecoveryAtlas.from_payload(dict(source=dict(game_id="synthetic", sha256="a"*64),
            status="approximate", metric_certified=False, field=r.FIELD, projection_mode=r.SINGLE_REFERENCE_PROJECTION_MODE,
            charts=[dict(chart="mid", reference_frame_index=0, field_to_reference=h.tolist(), support_world_polygon=hull)],
            blend=r.BLEND, spatial_boundary=None, protected_y_m=-r.BOX18_HALF_W, frames=frames))


def get_frame(atlas, index=0):
    return atlas.frame(9000+index*9000, "1/90000", source_sha256="a"*64, native_size=[1920, 1080])


def operation(atlas, mode="local", center=(0., 0.), delta_native=(2., 3.), index=0):
    frame = get_frame(atlas, index)
    start = frame.project([center])[0]
    end = start + delta_native
    inv = np.linalg.inv(np.array(frame.record["reference_to_native"]))
    delta = r.project(inv, [end])[0]-r.project(inv, [start])[0]
    return dict(mode=mode, center_world=list(center), delta_reference=delta.tolist(),
                anchor=dict(frame_index=index, source_pts=9000+index*9000, source_time_base="1/90000",
                            image_sha256=frame.record.get("image_sha256", f"{index+1:064x}"),
                            from_native=start.tolist(), to_native=end.tolist()))


def session(tmp_path, warnings=False, support=None):
    original = parent(warnings=warnings, support=support)
    for i, row in enumerate(original.payload["frames"]):
        image = np.full((1080, 1920, 3), 40+i, np.uint8)
        cv2.imwrite(str(tmp_path / f"{i}.png"), image)
        row["image_sha256"] = file_hash(tmp_path / f"{i}.png")
    original = r.RecoveryAtlas.from_payload(original.payload)
    original.save(tmp_path / "parent.json")
    records = [dict(index=i, source_pts=9000+i*9000, source_time_base="1/90000", source_seconds=.1+i*.1,
                    image_sha256=file_hash(tmp_path / f"{i}.png"), native_size=[1920, 1080], source_sha256="a"*64,
                    file=f"{i}.png") for i in range(3)]
    (tmp_path / "frames.json").write_text(json.dumps(dict(source=dict(sha256="a"*64), frames=records)))
    prepare(tmp_path / "parent.json", tmp_path / "frames.json", tmp_path / "session", "fixture", "Synthetic")
    return AdjustmentStore(tmp_path / "session")


def request(store, operations, confirmed=False):
    state = store.state()
    return dict(expected_revision=state["revision"], source_sha256=state["source_sha256"],
                base_sha256=state["base_sha256"], operations=operations, confirmed=confirmed)


def test_noop_and_kernel_outside_are_exact():
    original = parent()
    empty = AdjustmentAtlas.create(original, "b"*64, [])
    world = np.array([[-45., 0.], [0., 0.], [45., 0.], [0., 20.], [30., -25.]])
    for index in range(3):
        assert get_frame(empty, index).project(world).tobytes() == get_frame(original, index).project(world).tobytes()
    adjusted = AdjustmentAtlas.create(original, "b"*64, [operation(original)])
    np.testing.assert_array_equal(get_frame(adjusted).project(world[[0, 2, 3, 4]]), get_frame(original).project(world[[0, 2, 3, 4]]))


@pytest.mark.parametrize("mode", ["local", "global"])
def test_anchor_drag_hits_target_and_global_transport_follows_pan(mode):
    original = parent()
    op = operation(original, mode=mode, index=1)
    adjusted = AdjustmentAtlas.create(original, "b"*64, [op])
    np.testing.assert_allclose(get_frame(adjusted, 1).project([op["center_world"]])[0], op["anchor"]["to_native"], atol=1e-6)
    if mode == "global":
        world = [[-20., -29.], [0., 0.], [30., 20.]]
        for index in range(3):
            frame = get_frame(original, index)
            m = np.array(frame.record["reference_to_native"])
            expected = r.project(m, r.project(np.linalg.inv(m), frame.project(world)) + op["delta_reference"])
            np.testing.assert_allclose(get_frame(adjusted, index).project(world), expected, atol=1e-6)


def test_inverse_uses_adjusted_mapping_and_withholds_warned_or_unsupported():
    original = parent()
    adjusted = AdjustmentAtlas.create(original, "b"*64, [operation(original)])
    frame = get_frame(adjusted, 2)
    world = [[0., 0.], [12., 4.], [-25., -4.]]
    pixels = frame.project(world)
    result = frame.locate(pixels.tolist())
    assert all(row["field_position_m"] is not None for row in result)
    recovered = np.array([row["field_position_m"] for row in result])
    assert np.max(np.linalg.norm(frame.project(recovered)-pixels, axis=1)) < .01
    assert np.max(np.linalg.norm(recovered-world, axis=1)) < 1e-5
    warned = AdjustmentAtlas.create(parent(warnings=True), "b"*64, [])
    assert get_frame(warned).locate([[960., 450.]])[0]["field_position_m"] is None
    assert get_frame(warned).public_projection([[0., 0.]])["warnings"] == ["unverified_motion"]
    narrow = parent(support=[[-2., -2.], [2., -2.], [2., 2.], [-2., 2.]])
    wrapped = AdjustmentAtlas.create(narrow, "b"*64, [operation(narrow, mode="global")])
    target = get_frame(wrapped).project([[30., 0.]])[0]
    assert get_frame(wrapped).locate([target.tolist()])[0]["field_position_m"] is None
    assert get_frame(wrapped).public_projection([[30., 0.]])["native_pixels"] == [None]


def test_prepare_packet_and_save_undo_restore_exactly(tmp_path):
    store = session(tmp_path)
    packet = json.loads((store.root / "packet.json").read_text())
    assert packet["fps"] == 10 and len(packet["frames"]) == 3
    assert packet["frames"][0]["image_sha256"] == file_hash(tmp_path / "0.png")
    assert len(next(m for m in packet["markings"] if m["id"] == "centre_circle")["world"]) == 121
    for marking in packet["markings"]:
        assert np.max(np.linalg.norm(np.diff(marking["world"], axis=0), axis=1)) <= 1.000001
    baseline = store.state()
    op = operation(store.baseline)
    first = store.save(request(store, [op], confirmed=True))
    assert first["confirmed"] is True and first["metric_certified"] is False
    assert first["validation"]["frames_checked"] == 3
    saved = AdjustmentAtlas(store._active()["atlas"])
    np.testing.assert_allclose(get_frame(saved).project([[0., 0.]])[0], op["anchor"]["to_native"], atol=1e-6)
    second = store.save(request(store, []))
    restored = AdjustmentAtlas(store._active()["atlas"])
    world = [[0., 0.], [15., 3.], [-40., -30.]]
    assert get_frame(restored).project(world).tobytes() == get_frame(store.baseline).project(world).tobytes()
    assert second["previous_revision"] == first["revision"]
    assert (store.root / "revisions" / f"{baseline['revision']}.json").exists()
    assert len(list((store.root / "revisions").glob("*.json"))) == 3


def test_sequential_operation_validation_and_tampering_rejection(tmp_path):
    store = session(tmp_path)
    op = operation(store.baseline)
    first_atlas = AdjustmentAtlas.create(store.baseline.parent, store.manifest["base_sha256"], [op])
    second = operation(first_atlas, center=(4., 2.), index=1)
    assert len(store.save(request(store, [op, second]))["operations"]) == 2
    baseline_request = request(store, [op, second])
    for mutation in ("revision", "source", "pts", "hash", "delta", "nan", "from"):
        bad = deepcopy(baseline_request)
        if mutation == "revision": bad["expected_revision"] = "0"*64
        elif mutation == "source": bad["source_sha256"] = "0"*64
        elif mutation == "pts": bad["operations"][0]["anchor"]["source_pts"] = 9001
        elif mutation == "hash": bad["operations"][0]["anchor"]["image_sha256"] = "0"*64
        elif mutation == "delta": bad["operations"][0]["delta_reference"][0] += 5
        elif mutation == "nan": bad["operations"][0]["center_world"][0] = float("nan")
        else: bad["operations"][0]["anchor"]["from_native"][0] += 1
        before = store.state()["revision"]
        with pytest.raises(AdjustmentError): store.save(bad)
        assert store.state()["revision"] == before


def test_huge_local_warp_and_overlong_list_are_rejected(tmp_path):
    store = session(tmp_path)
    op = operation(store.baseline, delta_native=(700., 0.))
    with pytest.raises(AdjustmentError, match="invalid visible geometry"):
        store.save(request(store, [op]))
    with pytest.raises(AdjustmentError):
        store.save(request(store, [operation(store.baseline)]*101))


def test_cli_errors_are_json(tmp_path):
    result = subprocess.run([sys.executable, "-m", "calibration.tools.field_adjustment", "state", "--root", str(tmp_path / "missing")],
                            text=True, capture_output=True, check=False)
    assert result.returncode == 1
    assert set(json.loads(result.stdout)) == {"error", "code"}
    malformed = subprocess.run([sys.executable, "-m", "calibration.tools.field_adjustment", "state"],
                               text=True, capture_output=True, check=False)
    assert malformed.returncode == 1
    assert json.loads(malformed.stdout)["code"] == "invalid_arguments"


def test_prepare_preserves_existing_video_and_direct_cli(tmp_path, monkeypatch):
    store = session(tmp_path)
    destination = tmp_path / "video-first"
    destination.mkdir()
    video = destination / "source-clean.webm"
    video.write_bytes(b"synthetic video sentinel, never decoded")
    monkeypatch.setattr("calibration.field_adjustment._verify_video", lambda *args: dict(scope="unit filename-selection stub"))
    prepare(tmp_path / "parent.json", tmp_path / "frames.json", destination, "video", "Video")
    assert video.read_bytes() == b"synthetic video sentinel, never decoded"
    assert json.loads((destination / "manifest.json").read_text())["video_path"] == video.name
    before = (destination / "packet.json").read_bytes()
    with pytest.raises(AdjustmentError, match="already exists"):
        prepare(tmp_path / "parent.json", tmp_path / "frames.json", destination, "video", "Video")
    assert (destination / "packet.json").read_bytes() == before
    script = Path(__file__).resolve().parents[1] / "tools" / "field_adjustment.py"
    result = subprocess.run([sys.executable, str(script), "state", "--root", str(store.root)], cwd=tmp_path,
                            text=True, capture_output=True, check=False)
    assert result.returncode == 0 and json.loads(result.stdout)["revision"] == store.state()["revision"]


def test_multiple_supported_positive_inverse_roots_are_withheld():
    original = parent()
    frame = get_frame(original)
    def folded(world, supported_only=None):
        p = np.asarray(world)
        return np.c_[960+100*(p[:, 0]**3-p[:, 0]), 450+5*p[:, 1]]
    frame.project = folded
    wrapped = AdjustmentFrame(frame, [])
    result = wrapped.locate([[960., 450.]])[0]
    assert result["field_position_m"] is None
    assert result["reason"] == "ambiguous_inverse"


def test_source_image_hash_and_native_dimensions_are_verified(tmp_path):
    from calibration.field_adjustment import _verify_frame_image
    cv2.imwrite(str(tmp_path / "frame.png"), np.zeros((24, 32, 3), np.uint8))
    record = dict(file="frame.png", image_sha256=file_hash(tmp_path / "frame.png"), native_size=[32, 24])
    _verify_frame_image(tmp_path / "frames.json", record, record)
    with pytest.raises(AdjustmentError, match="hash"):
        _verify_frame_image(tmp_path / "frames.json", record, dict(record, image_sha256="0"*64))
    with pytest.raises(AdjustmentError, match="dimensions"):
        _verify_frame_image(tmp_path / "frames.json", dict(record, native_size=[1920, 1080]), dict(native_size=[1920, 1080]))
    with (tmp_path / "frame.png").open("ab") as stream:
        stream.write(b"changed after measurement")
    with pytest.raises(AdjustmentError, match="hash"):
        _verify_frame_image(tmp_path / "frames.json", record, record)


def test_multiple_videos_require_explicit_basename_and_probe_must_match(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from calibration.field_adjustment import _select_video, _verify_video
    for name in ("source.webm", "source-full.webm"):
        (tmp_path / name).write_bytes(b"synthetic probe fixture")
    with pytest.raises(AdjustmentError, match="Multiple"):
        _select_video(tmp_path, None)
    selected = _select_video(tmp_path, "source-full.webm")
    assert selected.name == "source-full.webm"
    with pytest.raises(AdjustmentError, match="basename"):
        _select_video(tmp_path, "../source-full.webm")
    probe = dict(streams=[dict(width=1920, height=1080, avg_frame_rate="10/1", r_frame_rate="10/1", nb_read_frames="351")],
                 format=dict(duration="35.1"))
    monkeypatch.setattr("calibration.field_adjustment.subprocess.run", lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(probe)))
    assert _verify_video(selected, [1920, 1080], 351)["frames"] == 351
    with pytest.raises(AdjustmentError, match="mismatch"):
        _verify_video(selected, [1920, 1080], 300)


def test_packet_observed_support_preserves_diagnostic_pixels(tmp_path):
    store = session(tmp_path, support=[[-10., -10.], [10., -10.], [10., 10.], [-10., 10.]])
    packet = json.loads((tmp_path / "session" / "packet.json").read_text())
    marks = {mark["id"]: mark for mark in packet["markings"]}
    assert not any(marks["touch_near"]["observed_support"])
    assert all(marks["centre_circle"]["observed_support"])
    assert any(marks["halfway"]["observed_support"])
    assert not all(marks["halfway"]["observed_support"])
    # The display mask does not rewrite the original projection or saved map.
    near = next(i for i, mark in enumerate(packet["markings"]) if mark["id"] == "touch_near")
    assert all(p is not None for p in packet["frames"][0]["base_pixels"][near])
    assert store.state()["operations"] == []


@pytest.mark.parametrize("case,code", [("warned", "warned_anchor"), ("unsupported", "unsupported_anchor")])
def test_save_rejects_warned_and_unsupported_anchors_without_mutation(tmp_path, case, code):
    store = session(tmp_path, warnings=case == "warned", support=(
        [[-10., -10.], [10., -10.], [10., 10.], [-10., 10.]] if case == "unsupported" else None))
    before = store.state()
    revisions = sorted(p.name for p in (store.root / "revisions").iterdir())
    op = operation(store.baseline.parent, center=(20., 0.) if case == "unsupported" else (0., 0.))
    with pytest.raises(AdjustmentError) as error:
        store.save(request(store, [op]))
    assert error.value.code == code
    assert store.state() == before
    assert sorted(p.name for p in (store.root / "revisions").iterdir()) == revisions
