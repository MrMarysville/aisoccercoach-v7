"""Prepare named frozen development episodes while retaining exact source PTS."""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path

from calibration.alignment_trial import (decode_window, guard_interval, load_plan,
                                sha256_file, validate_frame, verify_source)
from calibration.tools.game_sources import CATALOG, read_catalog


EPISODES = {
    "granite-control": ("13232938", 620.03, 655.04, 351, 55803000),
    "butte-h2-slot2": ("13217031", 2970, 3000, 300, 267300000),
}


def select_episode(plan_path, episode, catalog_path):
    try:
        plan = load_plan(plan_path, catalog_path)
    except (KeyError, TypeError, StopIteration) as exc:
        raise ValueError("Invalid frozen plan structure or source binding") from exc
    catalog = read_catalog(catalog_path)
    clips = {}
    games = {game["game_id"]:game for game in plan["payload"]["games"]}
    for game in games.values():
        for clip in game["clips"]:
            clip_id = clip.get("id")
            if not isinstance(clip_id, str) or not clip_id or clip_id in clips:
                raise ValueError("Duplicate or invalid frozen clip ID")
            if not clip_id.startswith(f"{game['game_id']}-h{clip['half']}-"):
                raise ValueError("Clip ID/game/half binding mismatch")
            clips[clip_id] = (game, clip)
    if episode in EPISODES:
        gid, start, end, count, first_pts = EPISODES[episode]
        if gid not in games:
            raise ValueError("Historical episode game is absent from frozen plan")
        game = games[gid]
        if episode == "butte-h2-slot2":
            clip_id = "13217031-h2-slot2"
            if clip_id not in clips:
                raise ValueError("Historical Butte episode is absent from frozen plan")
            _, clip = clips[clip_id]
            if (clip["start_s"], clip["end_s"], clip["role"]) != (start, end, "development"):
                raise ValueError("Historical Butte episode differs from frozen plan")
        else:
            clip_id = episode
            clip = dict(id=episode, half=1, role="development", start_s=start, end_s=end)
    else:
        if episode not in clips:
            raise ValueError("Unknown frozen development clip ID")
        game, clip = clips[episode]
        clip_id = episode
        start, end = clip["start_s"], clip["end_s"]
        count, first_pts = 300, None
        if Fraction(str(end))-Fraction(str(start)) != 30:
            raise ValueError("Development preparation requires a predeclared 30-second clip")
    if clip["role"] != "development":
        raise ValueError("Reserved evaluation clips remain closed")
    half = next((h for h in catalog[game["game_id"]]["halves"] if h["half"] == clip["half"]), None)
    if (half is None or not Fraction(str(half["start_pts_s"])) <= Fraction(str(start)) < Fraction(str(end)) <= Fraction(str(half["end_pts_s"]))):
        raise ValueError("Selected episode is outside its verified source half")
    guard_interval(game, start, end)
    # Decode_window reads this padded interval; reject a reserved-window overlap
    # before making an output directory or opening the video.
    guard_interval(game, max(Fraction(0), Fraction(str(start))-Fraction(1, 10)), Fraction(str(end))+Fraction(1, 10))
    return game, dict(clip_id=clip_id, half=clip["half"], role=clip["role"], start_s=start,
                      end_s=end, count=count, historical_first_pts=first_pts,
                      selection_kind="historical_alias" if episode in EPISODES else "frozen_development_clip")


def validate_decoded(manifest, game, source, selection):
    frames = manifest.get("frames", [])
    if (manifest.get("schema") != "alignment-trial-frames-v1" or manifest.get("source") != source
            or source.get("game_id") != game["game_id"] or source.get("sha256") != game["source_sha256"]
            or source.get("bytes") != game["source_bytes"] or manifest.get("sampling_fps") != 10
            or manifest.get("requested_interval_s") != [selection["start_s"], selection["end_s"]]
            or len(frames) != selection["count"]):
        raise ValueError("Decoded episode source, interval or frame count mismatch")
    stamps = []
    size = frames[0]["native_size"]
    if (len(size) != 2 or any(type(value) is not int or value <= 0 for value in size)
            or manifest.get("crop_xywh") != [0, 0, *size]):
        raise ValueError("Decoded recovery frames must retain the native full image")
    for index, frame in enumerate(frames):
        if (type(frame["index"]) is not int or frame["index"] != index or frame["native_size"] != size
                or frame["source_time_base"] != frames[0]["source_time_base"]):
            raise ValueError("Decoded frame index, native size or time base mismatch")
        stamps.append(validate_frame(frame, game, selection["start_s"], selection["end_s"]))
    if not Fraction(str(selection["start_s"])) <= stamps[0] < Fraction(str(selection["start_s"]))+Fraction(1, 10):
        raise ValueError("First selected frame is outside the first sample interval")
    if any(b-a != Fraction(1, 10) for a, b in zip(stamps, stamps[1:])):
        raise ValueError("Selected source timestamps do not have exact rational 10 fps cadence")
    first_pts = selection["historical_first_pts"]
    if first_pts is not None and (frames[0]["source_time_base"] != "1/90000"
            or [f["source_pts"] for f in frames] != [first_pts+9000*i for i in range(selection["count"])]):
        raise ValueError("Recovery frame selection differs from exact historical PTS")
    return stamps


def prepare(plan_path, episode, output, *, catalog_path=CATALOG):
    plan_hash, catalog_hash = sha256_file(plan_path), sha256_file(catalog_path)
    game, selection = select_episode(plan_path, episode, catalog_path)
    def unchanged_inputs():
        if sha256_file(plan_path) != plan_hash or sha256_file(catalog_path) != catalog_hash:
            raise ValueError("Frozen plan or catalog changed during preparation")
    unchanged_inputs()
    start, end, count = (selection[k] for k in ("start_s", "end_s", "count"))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    # Freeze split before source decoding or new annotation. This same-episode
    # experiment is development evidence, including its designated check frames.
    fit = list(range(0, count, 10))
    if count - 1 not in fit:
        fit.append(count - 1)
    checks = [i for i in range(count) if i not in fit]
    broad = [i for i in range(5, count, 50) if i in checks]
    protocol = dict(schema="field-recovery-split-v1", episode=episode,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    source_sha256=game["source_sha256"],
                    original_split_sha256=plan_hash,
                    plan_sha256=plan_hash, catalog_sha256=catalog_hash,
                    clip_id=selection["clip_id"], half=selection["half"], role=selection["role"],
                    selected_interval_s=[start, end], selection_kind=selection["selection_kind"],
                    fit_frame_indices=fit, check_frame_indices=checks,
                    broad_annotation_check_indices=broad,
                    reserved_evaluation_policy="All four original intervals remain closed",
                    annotation_policy="Fresh source-only proposals; no old fitted maps or coordinates",
                    status="development experiment; not metric certification")
    (output / "split.json").write_text(json.dumps(protocol, indent=2) + "\n")
    source = verify_source(game)
    if (source.get("game_id"), source.get("sha256"), source.get("bytes")) != (
            game["game_id"], game["source_sha256"], game["source_bytes"]):
        raise ValueError("Verified source receipt does not match the selected game")
    (output / "source-verification.json").write_text(json.dumps(source, indent=2) + "\n")
    unchanged_inputs()
    manifest = decode_window(game, source, start, end, 10, output / "source")
    unchanged_inputs()
    stamps = validate_decoded(manifest, game, source, selection)
    receipt = dict(episode=episode, count=count, first_pts=manifest["frames"][0]["source_pts"],
                   last_pts=manifest["frames"][-1]["source_pts"], source_time_base=manifest["frames"][0]["source_time_base"],
                   measured_first_s=float(stamps[0]), measured_last_s=float(stamps[-1]),
                   first_sample_phase_s=float(stamps[0]-Fraction(str(start))),
                   source_sha256=source["sha256"], game_id=game["game_id"],
                   plan_sha256=plan_hash, catalog_sha256=catalog_hash,
                   clip_id=selection["clip_id"], half=selection["half"], role=selection["role"],
                   selected_interval_s=[start, end], selection_kind=selection["selection_kind"],
                   frames_sha256=sha256_file(output / "source/frames.json"),
                   split_sha256=sha256_file(output / "split.json"))
    (output / "preparation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--episode", required=True, help="Exact frozen development clip ID or historical control alias")
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.plan, args.episode, args.out, catalog_path=args.catalog), indent=2), flush=True)
    print("RECOVERY PREPARATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
