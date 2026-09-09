"""Frozen-development preparation checks using only mocked source decoding."""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from calibration.alignment_trial import sha256_file
from calibration.field_atlas import digest
from calibration.tools import prepare_field_recovery as prep


@pytest.fixture
def frozen(tmp_path):
    games, sources = [], []
    for gid, sha, windows in [
        ("13232938", "a"*64, [300, 700, 1500, 2300, 2820, 3400]),
        ("13217031", "b"*64, [400, 980, 1600, 2370, 2970, 3500]),
    ]:
        halves = [dict(half=1, start_pts_s=0, end_pts_s=1900), dict(half=2, start_pts_s=1900, end_pts_s=4000)]
        sources.append(dict(game_id=gid, video_sha256=sha, video_bytes=123,
                            video_filename=gid+".mp4", source_identity_verified=True, halves=halves))
        clips = [dict(id=f"{gid}-h{i//3+1}-slot{i%3+1}", half=i//3+1,
                      start_s=start, end_s=start+30, role="reserved_evaluation" if i%3 == 2 else "development")
                 for i, start in enumerate(windows)]
        games.append(dict(game_id=gid, source_sha256=sha, source_bytes=123, video_path="/synthetic/"+gid+".mp4", clips=clips))
    catalog = tmp_path/"catalog.json"
    catalog.write_text(json.dumps(dict(schema="verified-game-sources-v1", games=sources)))
    plan = dict(schema="field-atlas-clips-v1", payload=dict(games=games), catalog_sha256=sha256_file(catalog))
    plan["payload_sha256"] = digest(plan["payload"])
    path = tmp_path/"clips.json"
    path.write_text(json.dumps(plan))
    return path, catalog, plan


def mocked_sources(monkeypatch, *, phase=Fraction(0), corrupt=None):
    verify = Mock(side_effect=lambda g:dict(game_id=g["game_id"], sha256=g["source_sha256"], bytes=g["source_bytes"],
                                         path=g["video_path"], mtime_ns=1, hash_seconds=0.))
    def decode(game, source, start, end, fps, output):
        output = Path(output)
        split = json.loads((output.parent/"split.json").read_text())
        count = 351 if start == 620.03 else 300
        assert split["fit_frame_indices"] == list(range(0, count, 10)) + ([] if count == 351 else [299])
        assert len(split["check_frame_indices"]) == count-len(split["fit_frame_indices"])
        first = 55803000 if count == 351 else int((Fraction(str(start))+phase)*90000)
        rows = [dict(index=i, file=f"{i+1:05d}.png", game_id=game["game_id"], source_sha256=game["source_sha256"],
                     source_pts=first+9000*i, source_time_base="1/90000", source_seconds=(first+9000*i)/90000,
                     native_size=[1920, 1080], image_sha256="c"*64) for i in range(count)]
        manifest = dict(schema="alignment-trial-frames-v1", source=deepcopy(source), requested_interval_s=[start, end],
                        sampling_fps=fps, crop_xywh=[0, 0, 1920, 1080], frames=rows)
        if corrupt:
            corrupt(manifest)
        output.mkdir()
        (output/"frames.json").write_text(json.dumps(manifest))
        return manifest
    decoder = Mock(side_effect=decode)
    monkeypatch.setattr(prep, "verify_source", verify)
    monkeypatch.setattr(prep, "decode_window", decoder)
    return verify, decoder


@pytest.mark.parametrize("episode,phase,first", [
    ("13232938-h2-slot1", Fraction(1, 250), 207000360),
    ("13232938-h2-slot2", Fraction(1, 250), 253800360),
    ("13217031-h1-slot2", Fraction(0), 88200000),
])
def test_exact_development_ids_keep_source_phase_and_prefrozen_split(frozen, tmp_path, monkeypatch, episode, phase, first):
    plan, catalog, _ = frozen
    verify, decode = mocked_sources(monkeypatch, phase=phase)
    out = tmp_path/"episode"
    receipt = prep.prepare(plan, episode, out, catalog_path=catalog)
    assert receipt["first_pts"] == first
    assert receipt["last_pts"] == first+299*9000
    assert receipt["first_sample_phase_s"] == float(phase)
    assert receipt["clip_id"] == episode and receipt["role"] == "development"
    assert receipt["count"] == 300
    assert receipt["plan_sha256"] == sha256_file(plan)
    assert receipt["catalog_sha256"] == sha256_file(catalog)
    assert receipt["measured_first_s"] == first/90000
    split = json.loads((out/"split.json").read_text())
    assert len(split["fit_frame_indices"]) == 31
    assert len(split["check_frame_indices"]) == 269
    assert split["broad_annotation_check_indices"] == [5, 55, 105, 155, 205, 255]
    verify.assert_called_once()
    decode.assert_called_once()


@pytest.mark.parametrize("episode,count,first", [("granite-control", 351, 55803000), ("butte-h2-slot2", 300, 267300000)])
def test_historical_aliases_keep_exact_original_pts(frozen, tmp_path, monkeypatch, episode, count, first):
    plan, catalog, _ = frozen
    mocked_sources(monkeypatch)
    receipt = prep.prepare(plan, episode, tmp_path/episode, catalog_path=catalog)
    assert (receipt["count"], receipt["first_pts"], receipt["last_pts"], receipt["source_time_base"]) == (
        count, first, first+9000*(count-1), "1/90000")


@pytest.mark.parametrize("case", ["unknown", "reserved", "duplicate", "catalog", "plan", "game", "clip_game", "padded_reserved"])
def test_invalid_selection_and_bindings_fail_before_output_or_decode(frozen, tmp_path, monkeypatch, case):
    path, catalog, plan = frozen
    episode = "13232938-h2-slot1"
    if case == "unknown":
        episode = "not-in-the-frozen-plan"
    elif case == "reserved":
        episode = "13232938-h2-slot3"
    elif case == "catalog":
        catalog.write_text(catalog.read_text()+"\n")
    else:
        game = plan["payload"]["games"][0]
        if case == "duplicate":
            game["clips"].append(dict(game["clips"][0], start_s=1000, end_s=1030))
        elif case == "plan":
            game["clips"][0]["end_s"] += 1
        elif case == "game":
            game["source_sha256"] = "d"*64
        elif case == "clip_game":
            game["clips"][0]["id"] = "13217031-h1-wrong-game"
        elif case == "padded_reserved":
            game["clips"][3].update(start_s=3369.95, end_s=3399.95)
        if case != "plan":
            plan["payload_sha256"] = digest(plan["payload"])
        path.write_text(json.dumps(plan))
    verify, decode = mocked_sources(monkeypatch)
    out = tmp_path/"rejected"
    with pytest.raises(ValueError):
        prep.prepare(path, episode, out, catalog_path=catalog)
    assert not out.exists()
    verify.assert_not_called()
    decode.assert_not_called()


def test_existing_directory_is_never_overwritten(frozen, tmp_path, monkeypatch):
    path, catalog, _ = frozen
    out = tmp_path/"existing"
    out.mkdir()
    (out/"keep.txt").write_text("unchanged")
    verify, decode = mocked_sources(monkeypatch)
    with pytest.raises(FileExistsError):
        prep.prepare(path, "13232938-h2-slot1", out, catalog_path=catalog)
    assert (out/"keep.txt").read_text() == "unchanged"
    verify.assert_not_called()
    decode.assert_not_called()


@pytest.mark.parametrize("corrupt", [
    lambda m:m["frames"].pop(),
    lambda m:m["frames"][10].update(source_pts=m["frames"][10]["source_pts"]+1, source_seconds=m["frames"][10]["source_seconds"]+1/90000),
    lambda m:m["frames"][20].update(game_id="wrong-game"),
    lambda m:m["frames"][30].update(source_seconds=0.),
    lambda m:m.update(crop_xywh=[10, 0, 1910, 1080]),
])
def test_decoded_count_cadence_bindings_and_native_extent_are_checked(frozen, tmp_path, monkeypatch, corrupt):
    path, catalog, _ = frozen
    mocked_sources(monkeypatch, phase=Fraction(1, 250), corrupt=corrupt)
    with pytest.raises(ValueError):
        prep.prepare(path, "13232938-h2-slot1", tmp_path/"invalid-decoding", catalog_path=catalog)
    assert not (tmp_path/"invalid-decoding"/"preparation.json").exists()


def test_historical_butte_alias_rejects_changed_phase(frozen, tmp_path, monkeypatch):
    path, catalog, _ = frozen
    mocked_sources(monkeypatch, phase=Fraction(1, 250))
    with pytest.raises(ValueError, match="historical PTS"):
        prep.prepare(path, "butte-h2-slot2", tmp_path/"wrong-phase", catalog_path=catalog)


def test_plan_changed_after_verification_stops_before_decoding(frozen, tmp_path, monkeypatch):
    path, catalog, _ = frozen
    verify, decode = mocked_sources(monkeypatch)
    original = verify.side_effect
    def mutate_plan(game):
        receipt = original(game)
        path.write_text(path.read_text()+"\n")
        return receipt
    verify.side_effect = mutate_plan
    with pytest.raises(ValueError, match="changed during preparation"):
        prep.prepare(path, "13232938-h2-slot1", tmp_path/"changed-input", catalog_path=catalog)
    decode.assert_not_called()
