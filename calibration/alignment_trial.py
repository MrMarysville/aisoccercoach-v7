"""Local trial evidence guards; no calibration, source mutation or remote work.

Times are absolute rational source times. Intervals are half-open. Native pixel
coordinates use OpenCV's pixel-centre convention, including preview resizing.
"""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import time


def sha256_file(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_plan(path, catalog_path):
    """Reject changed plans and cross-game sources before any image decoding."""
    raw = Path(path).read_bytes()
    plan = json.loads(raw)
    payload = json.dumps(plan["payload"], sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    if plan.get("schema") != "field-atlas-clips-v1" or hashlib.sha256(payload).hexdigest() != plan.get("payload_sha256"):
        raise ValueError("Invalid frozen clip plan")
    catalog_raw = Path(catalog_path).read_bytes()
    if hashlib.sha256(catalog_raw).hexdigest() != plan.get("catalog_sha256"):
        raise ValueError("Source catalog differs from frozen plan")
    catalog = {g["game_id"]: g for g in json.loads(catalog_raw)["games"]}
    seen = set()
    for game in plan["payload"]["games"]:
        gid = game["game_id"]
        if gid in seen or gid not in catalog:
            raise ValueError("Duplicate or unknown source game")
        seen.add(gid)
        source = catalog[gid]
        if (game["source_sha256"], game["source_bytes"]) != (source["video_sha256"], source["video_bytes"]):
            raise ValueError("Game/source binding mismatch")
        previous_end = None
        for clip in sorted(game["clips"], key=lambda c: c["start_s"]):
            start, end = Fraction(str(clip["start_s"])), Fraction(str(clip["end_s"]))
            half = next(h for h in source["halves"] if h["half"] == clip["half"])
            if (clip["role"] not in ("development", "reserved_evaluation")
                    or not Fraction(str(half["start_pts_s"])) <= start < end <= Fraction(str(half["end_pts_s"]))
                    or (previous_end is not None and start < previous_end)):
                raise ValueError("Invalid or overlapping clip intervals")
            previous_end = end
    return plan


def verify_source(game, path=None):
    path = Path(path or game["video_path"])
    start = time.monotonic()
    if path.stat().st_size != game["source_bytes"]:
        raise ValueError("Source byte count mismatch")
    actual = sha256_file(path)
    if actual != game["source_sha256"]:
        raise ValueError("Source SHA-256 mismatch")
    return dict(game_id=game["game_id"], path=str(path.resolve()), sha256=actual,
                bytes=path.stat().st_size, hash_seconds=time.monotonic() - start,
                mtime_ns=path.stat().st_mtime_ns)


def source_time(pts, time_base):
    if type(pts) is not int or not isinstance(time_base, str):
        raise ValueError("Expected integer PTS and rational time-base string")
    try:
        scale = Fraction(time_base)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("Invalid time base") from exc
    if scale <= 0:
        raise ValueError("Nonpositive time base")
    return pts * scale


def guard_interval(game, start_s, end_s):
    start, end = Fraction(str(start_s)), Fraction(str(end_s))
    if not 0 <= start < end:
        raise ValueError("Invalid source interval")
    for clip in game["clips"]:
        if (clip["role"] == "reserved_evaluation"
                and start < Fraction(str(clip["end_s"]))
                and end > Fraction(str(clip["start_s"]))):
            raise ValueError(f"Closed reserved window: {clip['id']}")


def validate_frame(frame, game, start_s, end_s):
    if frame["source_sha256"] != game["source_sha256"] or frame["game_id"] != game["game_id"]:
        raise ValueError("Frame source mismatch")
    stamp = source_time(frame["source_pts"], frame["source_time_base"])
    if not Fraction(str(start_s)) <= stamp < Fraction(str(end_s)):
        raise ValueError("Frame timestamp outside requested interval")
    guard_interval(game, stamp, stamp + Fraction(frame["source_time_base"]))
    if "source_seconds" in frame and abs(float(stamp) - frame["source_seconds"]) > 1e-7:
        raise ValueError("Frame seconds disagree with PTS")
    return stamp


def preview_to_native(points, *, crop_xywh, preview_size, native_size):
    """Undo OpenCV resize then crop; all coordinates address pixel centres."""
    import numpy as np
    values = np.asarray(points, dtype=float)
    x, y, width, height = crop_xywh
    pw, ph = preview_size
    nw, nh = native_size
    if (values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all()
            or min(width, height, pw, ph, nw, nh) <= 0
            or min(x, y) < 0 or x + width > nw or y + height > nh):
        raise ValueError("Invalid crop, resize or coordinate array")
    if np.any(values < -.5) or np.any(values >= np.array([pw, ph]) - .5):
        raise ValueError("Point outside preview")
    return (values + .5) * [width / pw, height / ph] - .5 + [x, y]


def validate_boundary(boundary):
    """Missing events keep null timestamps; recording cuts are not match ends."""
    if boundary["half"] not in (1, 2) or boundary["boundary"] not in ("start", "end"):
        raise ValueError("Invalid period boundary")
    status = boundary["status"]
    if status not in ("proposed", "user_confirmed", "uncertain", "not_observed"):
        raise ValueError("Unknown boundary status")
    if not boundary.get("evidence"):
        raise ValueError("Boundary evidence or missing-evidence reason required")
    pts, tb = boundary.get("source_pts"), boundary.get("source_time_base")
    if status == "not_observed":
        if pts is not None or tb is not None:
            raise ValueError("Missing event cannot borrow a recording timestamp")
    elif pts is not None:
        source_time(pts, tb)
    elif status in ("proposed", "user_confirmed") or tb is not None:
        raise ValueError("Proposed/confirmed boundary needs an exact source frame")
    if not boundary.get("review_clip") and not boundary.get("review_unavailable_reason"):
        raise ValueError("Boundary needs a review clip or explicit unavailable reason")
    return boundary


def decode_window(game, source_receipt, start_s, end_s, fps, out):
    """Save native PNGs selected on original PTS; refuses an existing directory.

    select+showinfo keeps source timestamps, unlike an fps filter's output clock.
    The source must have been fully verified in this run; stat checks detect a
    subsequently changed file. The receipt is local evidence, not authentication.
    """
    guard_interval(game, start_s, end_s)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid sampling frequency")
    source = Path(source_receipt["path"])
    if (source_receipt["sha256"] != game["source_sha256"]
            or source_receipt["game_id"] != game["game_id"]
            or source.stat().st_size != source_receipt["bytes"]
            or source.stat().st_mtime_ns != source_receipt["mtime_ns"]):
        raise ValueError("Source changed after verification")
    seek = max(0, float(start_s) - .1)
    decode_end = float(end_s) + .1
    guard_interval(game, seek, decode_end)
    probe = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=start_time", "-of", "json", str(source)],
        text=True, timeout=30))
    source_start = float(probe["format"].get("start_time", 0))
    relative_seek = max(0, seek - source_start)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    elapsed = time.monotonic()
    select = (f"select=gte(t\\,{float(start_s):.9f})*lt(t\\,{float(end_s):.9f})*"
              f"(isnan(prev_selected_t)+gte(t-prev_selected_t\\,{1 / fps - 1e-7:.9f})),showinfo")
    command = ["ffmpeg", "-hide_banner", "-nostdin", "-copyts", "-threads", "2", "-ss", str(relative_seek),
               "-t", str(decode_end - seek), "-i", str(source), "-an", "-vf", select,
               "-fps_mode", "passthrough", "-c:v", "png", "-compression_level", "1",
               "-threads", "2", str(out / "%05d.png")]
    (out / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (out / "decode.log").open("w") as log:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=log, check=True, timeout=900)
    log = (out / "decode.log").read_text()
    time_base = re.search(r"config in time_base: (\d+/\d+)", log).group(1)
    pattern = r"\bn:\s*(\d+)\s+pts:\s*(-?\d+).*?\bs:(\d+)x(\d+).*?\bchecksum:([0-9A-F]+)"
    frames = []
    for n, pts, width, height, checksum in re.findall(pattern, log):
        image = out / f"{int(n) + 1:05d}.png"
        row = dict(index=int(n), file=image.name, game_id=game["game_id"], source_sha256=game["source_sha256"],
                   source_pts=int(pts), source_time_base=time_base,
                   source_seconds=float(source_time(int(pts), time_base)),
                   native_size=[int(width), int(height)], decoded_checksum=checksum,
                   image_sha256=sha256_file(image))
        validate_frame(row, game, start_s, end_s)
        if frames and source_time(row["source_pts"], time_base) <= source_time(frames[-1]["source_pts"], time_base):
            raise ValueError("Nonmonotonic selected source timestamps")
        frames.append(row)
    if not frames or len(frames) != len(list(out.glob("*.png"))):
        raise ValueError("Decoded frame/PTS count mismatch")
    receipt = dict(schema="alignment-trial-frames-v1", source=source_receipt,
                   requested_interval_s=[start_s, end_s], sampling_fps=fps,
                   crop_xywh=[0, 0, *frames[0]["native_size"]], frames=frames,
                   elapsed_seconds=time.monotonic() - elapsed)
    (out / "frames.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt
