"""Local, allowlisted source-frame annotation with immutable revision history.

Run with --frames frames.json --out ignored/directory [--review-manifest review.json].
Review JSON is a mapping from frame index to an explicitly approved image path.
No video is opened, decoded or served. Coordinates address native pixel centres.
"""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import struct
import tempfile
import threading
from urllib.parse import urlsplit

SCHEMA = "field-browser-annotations-v1"
BINDING = ("game_id", "source_sha256", "source_pts", "source_time_base", "native_size", "image_sha256")
LABELS = ["touch_far", "touch_near", "halfway", "goal_left", "goal_right"] + [
    f"box{size}_{side}_{edge}" for size in (18, 6)
    for side in ("left", "right") for edge in ("front", "near", "far")
] + ["centre_circle", "pen_arc_left", "pen_arc_right"]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def image_size(data):
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return list(struct.unpack(">II", data[16:24]))
    # Existing, licensed OpenCV environment; no new dependencies.
    import cv2
    import numpy as np
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Invalid frame image")
    return [image.shape[1], image.shape[0]]


def pointer_to_native(client_xy, display_rect, native_size):
    """DOM rectangle is viewport-relative, so scrolling is already accounted for."""
    return [(client_xy[i] - display_rect[key]) * native_size[i] / display_rect[dimension] - .5
            for i, key, dimension in ((0, "left", "width"), (1, "top", "height"))]


class AnnotationStore:
    def __init__(self, frames_path, out, review_manifest=None):
        self.frames_path = Path(frames_path).resolve(strict=True)
        raw = self.frames_path.read_bytes()
        self.manifest_hash = digest(raw)
        self.manifest = json.loads(raw)
        if self.manifest.get("schema") != "alignment-trial-frames-v1":
            raise ValueError("Wrong frames manifest schema")
        self.source = self.manifest["source"]
        if (not isinstance(self.source.get("game_id"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", self.source.get("sha256", ""))
                or not isinstance(self.source.get("path"), str)):
            raise ValueError("Invalid source binding")
        self.frames = {}
        self.images = {}
        for frame in self.manifest["frames"]:
            index = frame["index"]
            size = frame["native_size"]
            if (type(index) is not int or index < 0 or index in self.frames
                    or frame["game_id"] != self.source["game_id"]
                    or frame["source_sha256"] != self.source["sha256"]
                    or type(frame["source_pts"]) is not int
                    or not isinstance(frame["source_time_base"], str)
                    or len(size) != 2 or any(type(x) is not int or x <= 0 for x in size)):
                raise ValueError("Invalid frame/source binding")
            try:
                scale = Fraction(frame["source_time_base"])
            except (ValueError, ZeroDivisionError) as exc:
                raise ValueError("Invalid rational time base") from exc
            if scale <= 0 or not finite(frame["source_seconds"]) or abs(
                    float(frame["source_pts"] * scale) - frame["source_seconds"]) > 1e-7:
                raise ValueError("Frame seconds disagree with PTS")
            relative = Path(frame["file"])
            path = (self.frames_path.parent / relative).resolve(strict=True)
            if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(self.frames_path.parent):
                raise ValueError("Frame file outside manifest directory")
            data = path.read_bytes()
            if digest(data) != frame["image_sha256"] or image_size(data) != size:
                raise ValueError("Frame image/hash/dimensions mismatch")
            self.frames[index] = frame
            self.images[index] = path
        if not self.frames:
            raise ValueError("No permitted frames")
        self.out = Path(out).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.review_path = Path(review_manifest).resolve(strict=True) if review_manifest else None
        self.reviews = {}
        self.reload_reviews()
        self.latest()  # Fail closed on foreign, incomplete or corrupted histories.

    def reload_reviews(self):
        reviews = {}
        if self.review_path:
            for key, value in json.loads(self.review_path.read_text()).items():
                index = int(key)
                if index not in self.frames:
                    raise ValueError("Review image references an unknown frame")
                path = Path(value)
                if not path.is_absolute():
                    raise ValueError("Review image path must be absolute")
                path = path.resolve(strict=True)
                data = path.read_bytes()
                if image_size(data) != self.frames[index]["native_size"]:
                    raise ValueError("Review image must have source native dimensions")
                reviews[index] = (path, digest(data))
        self.reviews = reviews

    def blank(self):
        return dict(schema=SCHEMA, source=self.source, frames_manifest_sha256=self.manifest_hash,
                    revision=0, previous_revision_sha256=None, created_at=None,
                    status="proposed", features=[], action_log=[])

    def latest(self):
        latest, previous_hash = self.blank(), None
        paths = sorted(self.out.glob("annotations-*.json"))
        for revision, path in enumerate(paths, 1):
            if path.name != f"annotations-{revision:06d}.json" or path.is_symlink():
                raise ValueError("Invalid revision filename/history")
            raw = path.read_bytes()
            value = json.loads(raw)
            if (value.get("revision") != revision or value.get("previous_revision_sha256") != previous_hash):
                raise ValueError("Broken revision chain")
            self.validate(value, check_images=False)
            latest, previous_hash = value, digest(raw)
        return latest, previous_hash

    @staticmethod
    def browser_annotation(value):
        """Keep source nanoseconds exact through every JavaScript response."""
        result = dict(value, source=dict(value["source"]))
        if "mtime_ns" in result["source"]:
            result["source"]["mtime_ns"] = str(result["source"]["mtime_ns"])
        return result

    def state(self):
        with self.lock:
            self.reload_reviews()
            latest, latest_hash = self.latest()
            browser_annotation = self.browser_annotation(dict(latest, source=dict(self.source)))
            return dict(annotation=browser_annotation, revision_sha256=latest_hash, labels=LABELS,
                        frames=[dict(frame, image_url=f"/frames/{index}",
                                     review_url=f"/reviews/{index}" if index in self.reviews else None,
                                     review_image_sha256=self.reviews[index][1] if index in self.reviews else None)
                                for index, frame in self.frames.items()])

    def validate(self, value, check_images=True):
        supplied_source = value.get("source")
        source_matches = supplied_source == self.source
        if isinstance(supplied_source, dict) and not source_matches:
            expected = dict(self.source)
            supplied = dict(supplied_source)
            expected_mtime = expected.pop("mtime_ns", None)
            supplied_mtime = supplied.pop("mtime_ns", None)
            # Browser metadata uses a decimal string; source identity remains exact.
            source_matches = (expected == supplied and type(expected_mtime) is int
                              and supplied_mtime == str(expected_mtime))
            # Read-only compatibility for immutable revisions saved before this fix.
            # New saves cannot use this rounded numeric representation.
            if not check_images and not source_matches:
                source_matches = (expected == supplied and type(expected_mtime) is int
                                  and type(supplied_mtime) in (int, float)
                                  and float(expected_mtime) == float(supplied_mtime))
        if (value.get("schema") != SCHEMA or not source_matches
                or value.get("frames_manifest_sha256") != self.manifest_hash
                or value.get("status") != "proposed"):
            raise ValueError("Annotation source/manifest/status mismatch")
        features = value.get("features")
        if not isinstance(features, list) or len(features) > 10000:
            raise ValueError("Invalid features")
        ids, checked = set(), set()
        for feature in features:
            identifier = feature.get("id")
            index = feature.get("frame_index")
            if (not isinstance(identifier, str) or not 1 <= len(identifier) <= 128 or identifier in ids
                    or type(index) is not int or index not in self.frames):
                raise ValueError("Invalid feature id/frame")
            ids.add(identifier)
            frame = self.frames[index]
            if any(feature.get(key) != frame[key] for key in BINDING):
                raise ValueError("Feature frame identity mismatch")
            if check_images and index not in checked:
                if digest(self.images[index].read_bytes()) != frame["image_sha256"]:
                    raise ValueError("Source frame image changed")
                checked.add(index)
            label, kind = feature.get("label"), feature.get("kind")
            if (kind not in ("polyline", "point") or not isinstance(label, str) or not 1 <= len(label.strip()) <= 200
                    or (kind == "polyline" and label not in LABELS)
                    or not isinstance(feature.get("reason"), str) or len(feature["reason"]) > 4000
                    or feature.get("review_status") != "proposed"):
                raise ValueError("Invalid feature label/kind/review status")
            points = feature.get("points")
            if (not isinstance(points, list) or not 1 <= len(points) <= 10000
                    or (kind == "point" and len(points) != 1)):
                raise ValueError("Invalid feature points")
            for point in points:
                rect = point.get("display_rect", {})
                if (point.get("method") != "browser_pointer" or type(point.get("frame_index")) is not int
                        or point["frame_index"] != index
                        or not finite(point.get("event_timestamp")) or point["event_timestamp"] < 0
                        or point.get("zoom") not in (.5, 1, 2)
                        or any(not finite(rect.get(key)) for key in ("left", "top", "width", "height"))
                        or min(rect["width"], rect["height"]) <= 0):
                    raise ValueError("Invalid pointer provenance")
                for key in ("native_xy", "client_xy", "scroll_xy"):
                    if (not isinstance(point.get(key), list) or len(point[key]) != 2
                            or any(not finite(x) for x in point[key])):
                        raise ValueError("Invalid pointer coordinates")
                if any(abs(rect[key] - frame["native_size"][i] * point["zoom"]) > 1e-6
                       for i, key in enumerate(("width", "height"))):
                    raise ValueError("Pointer zoom/rectangle mismatch")
                expected = pointer_to_native(point["client_xy"], rect, frame["native_size"])
                if any(abs(a - b) > 1e-9 for a, b in zip(expected, point["native_xy"])):
                    raise ValueError("Native pixel-centre conversion mismatch")
                if any(not -.5 <= coordinate < size - .5
                       for coordinate, size in zip(point["native_xy"], frame["native_size"])):
                    raise ValueError("Point outside source frame")
        actions = value.get("action_log")
        if not isinstance(actions, list) or len(actions) > 100000:
            raise ValueError("Invalid action log")
        if any(not isinstance(action, dict) or not isinstance(action.get("action"), str)
               or not finite(action.get("event_timestamp")) for action in actions):
            raise ValueError("Invalid action log entry")

    def save(self, value):
        with self.lock:
            latest, latest_hash = self.latest()
            if (type(value.get("revision")) is not int or value["revision"] != latest["revision"] + 1
                    or value.get("previous_revision_sha256") != latest_hash):
                raise ValueError("Stale revision: reload before saving")
            self.validate(value)
            if value["action_log"][:len(latest["action_log"])] != latest["action_log"]:
                raise ValueError("Prior action log cannot be rewritten")
            result = dict(value, source=dict(self.source), created_at=datetime.now(timezone.utc).isoformat())
            data = encoded(result)
            destination = self.out / f"annotations-{result['revision']:06d}.json"
            descriptor, temporary = tempfile.mkstemp(prefix=".annotation-", dir=self.out)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.link(temporary, destination)  # Atomic publication, refusing overwrite.
                directory = os.open(self.out, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                os.unlink(temporary)
            return dict(annotation=result, revision_sha256=digest(data))


def make_server(store, port=8767):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, data, content_type="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def same_origin(self):
            expected = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            return self.headers.get("Host") in expected and (
                not self.headers.get("Origin") or self.headers["Origin"] in {f"http://{host}" for host in expected})

        def do_GET(self):
            if not self.same_origin():
                self.reply(403, encoded({"error": "Local origin required"}))
                return
            path = urlsplit(self.path).path
            try:
                if path == "/":
                    self.reply(200, Path(__file__).with_name("field_annotation_viewer.html").read_bytes(), "text/html; charset=utf-8")
                elif path == "/api/state":
                    self.reply(200, encoded(store.state()))
                elif path == "/favicon.ico":
                    self.reply(204, b"")
                elif re.fullmatch(r"/(frames|reviews)/\d+", path):
                    category, key = path.strip("/").split("/")
                    index = int(key)
                    if category == "frames":
                        data = store.images[index].read_bytes()
                        expected_hash = store.frames[index]["image_sha256"]
                    else:
                        image, expected_hash = store.reviews[index]
                        data = image.read_bytes()
                    if digest(data) != expected_hash:
                        raise ValueError("Image changed after manifest validation")
                    self.reply(200, data, "image/png" if data.startswith(b"\x89PNG") else "image/jpeg")
                else:
                    self.reply(404, encoded({"error": "Not found"}))
            except (ValueError, KeyError, OSError) as exc:
                self.reply(400, encoded({"error": str(exc)}))

        def do_POST(self):
            if not self.same_origin():
                self.reply(403, encoded({"error": "Local origin required"}))
                return
            if self.path != "/api/save":
                self.reply(404, encoded({"error": "Not found"}))
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16 * 1024 * 1024:
                    raise ValueError("Invalid request size")
                value = json.loads(self.rfile.read(length))
                if not isinstance(value, dict):
                    raise ValueError("Expected annotation object")
                saved = store.save(value)
                saved["annotation"] = store.browser_annotation(saved["annotation"])
                self.reply(200, encoded(saved))
            except (ValueError, KeyError, TypeError, OSError) as exc:
                self.reply(400, encoded({"error": str(exc)}))

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--review-manifest", type=Path)
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    server = make_server(AnnotationStore(args.frames, args.out, args.review_manifest), args.port)
    print(f"Field annotation ready: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
