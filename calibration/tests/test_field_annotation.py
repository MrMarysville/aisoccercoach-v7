"""Synthetic source-frame proof; never real-match calibration acceptance."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import zlib

SPEC = importlib.util.spec_from_file_location("field_annotation_server", Path(__file__).parents[1] / "tools/field_annotation_server.py")
server = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(server)


def png(width=1600, height=1200):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    rows = b"".join(b"\0" + bytes((40, 75 + (y // 100) % 2 * 20, 55)) * width for y in range(height))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


def fixture(root):
    data = png()
    (root / "frame.png").write_bytes(data)
    source = dict(game_id="synthetic", sha256="a" * 64, path="/synthetic/fixture.mp4")
    frame = dict(index=0, file="frame.png", game_id="synthetic", source_sha256=source["sha256"],
                 source_pts=90000, source_time_base="1/90000", source_seconds=1.,
                 native_size=[1600, 1200], image_sha256=hashlib.sha256(data).hexdigest())
    (root / "frames.json").write_text(json.dumps(dict(schema="alignment-trial-frames-v1", source=source, frames=[frame])))
    return server.AnnotationStore(root / "frames.json", root / "saved")


def payload(store, zoom=.5):
    value = store.blank()
    value["revision"] = 1
    frame = store.frames[0]
    rect = dict(left=-137.25, top=-251.75, width=1600 * zoom, height=1200 * zoom)
    native = [432.25, 678.125]
    client = [rect["left"] + (native[0] + .5) * zoom, rect["top"] + (native[1] + .5) * zoom]
    point = dict(native_xy=native, client_xy=client, display_rect=rect, scroll_xy=[153, 489],
                 zoom=zoom, method="browser_pointer", event_timestamp=125.25, frame_index=0)
    feature = dict(id="fixture-feature", frame_index=0, label="touch_near", kind="polyline", reason="Synthetic fixture",
                   points=[point], review_status="proposed", **{key: frame[key] for key in server.BINDING})
    value["features"] = [feature]
    value["action_log"] = [dict(action="pointer", event_timestamp=125.25, frame_index=0, point=point)]
    return value


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_pixel_centres_three_zooms_and_nonzero_scroll(self):
        errors = []
        for zoom in (.5, 1, 2):
            value = payload(self.store, zoom)
            point = value["features"][0]["points"][0]
            measured = server.pointer_to_native(point["client_xy"], point["display_rect"], [1600, 1200])
            errors.extend(abs(a - b) for a, b in zip(measured, point["native_xy"]))
            self.store.validate(value)
        self.assertLess(max(errors), 1e-9)

    def test_immutable_revisions_and_reopen(self):
        first = self.store.save(payload(self.store))
        original = (self.root / "saved/annotations-000001.json").read_bytes()
        second = copy.deepcopy(first["annotation"])
        second.update(revision=2, previous_revision_sha256=first["revision_sha256"], features=[])
        second["action_log"].append(dict(action="undo", event_timestamp=200))
        self.store.save(second)
        with self.assertRaisesRegex(ValueError, "Stale revision"):
            self.store.save(second)
        self.assertEqual((self.root / "saved/annotations-000001.json").read_bytes(), original)
        reopened = server.AnnotationStore(self.root / "frames.json", self.root / "saved")
        self.assertEqual(reopened.state()["annotation"]["features"], [])
        self.assertEqual(reopened.state()["annotation"]["revision"], 2)

    def test_large_mtime_roundtrips_as_string_without_weakening_source_identity(self):
        document = json.loads((self.root / "frames.json").read_text())
        document["source"]["mtime_ns"] = 1788227559339036900
        (self.root / "frames.json").write_text(json.dumps(document))
        store = server.AnnotationStore(self.root / "frames.json", self.root / "mtime-saved")
        value = payload(store)
        value["source"] = store.state()["annotation"]["source"]
        self.assertEqual(value["source"]["mtime_ns"], "1788227559339036900")
        for field, replacement in (("mtime_ns", 1788227559339037000),
                                   ("mtime_ns", "1788227559339036901"),
                                   ("sha256", "b" * 64), ("path", "/changed.mp4")):
            changed = copy.deepcopy(value)
            changed["source"][field] = replacement
            with self.subTest(field=field, replacement=replacement), self.assertRaises(ValueError):
                store.validate(changed)
        saved = store.save(value)["annotation"]
        self.assertEqual(saved["source"], document["source"])
        self.assertEqual(saved["features"], value["features"])
        self.assertEqual(saved["action_log"], value["action_log"])

    def test_reject_wrong_source_frame_hash_pts_dimensions_bounds_and_conversion(self):
        for field, replacement in (("source_sha256", "b" * 64), ("frame_index", 1), ("source_pts", 1),
                                   ("source_time_base", "1/30"), ("native_size", [800, 600]), ("image_sha256", "c" * 64)):
            value = payload(self.store)
            value["features"][0][field] = replacement
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.store.save(value)
        value = copy.deepcopy(payload(self.store))
        value["source"]["sha256"] = "b" * 64
        with self.assertRaises(ValueError):
            self.store.save(value)
        for native in ([1600, 1200], [10, 10]):
            value = payload(self.store)
            value["features"][0]["points"][0]["native_xy"] = native
            with self.assertRaises(ValueError):
                self.store.save(value)

    def test_reject_changed_image(self):
        (self.root / "frame.png").write_bytes(png(10, 10))
        with self.assertRaisesRegex(ValueError, "changed"):
            self.store.save(payload(self.store))

    def test_reject_manifest_traversal(self):
        manifest = json.loads((self.root / "frames.json").read_text())
        manifest["frames"][0]["file"] = "../" + self.root.name + "/frame.png"
        (self.root / "frames.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "outside"):
            server.AnnotationStore(self.root / "frames.json", self.root / "other")

    def test_reject_manifest_source_and_image_mismatch(self):
        for field, value in (("source_sha256", "b" * 64), ("image_sha256", "b" * 64)):
            manifest = json.loads((self.root / "frames.json").read_text())
            manifest["frames"][0][field] = value
            path = self.root / f"bad-{field}.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                server.AnnotationStore(path, self.root / "other")

    def test_review_allowlist_and_dimensions(self):
        review = self.root / "review.png"
        review.write_bytes(png())
        path = self.root / "review.json"
        path.write_text(json.dumps({"0": str(review)}))
        store = server.AnnotationStore(self.root / "frames.json", self.root / "saved", path)
        self.assertEqual(store.state()["frames"][0]["review_url"], "/reviews/0")
        review.write_bytes(png(10, 10))
        with self.assertRaisesRegex(ValueError, "dimensions"):
            store.reload_reviews()

    def test_http_source_rejection_and_path_allowlist(self):
        http = server.make_server(self.store, 0)
        thread = threading.Thread(target=http.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{http.server_port}"
        try:
            self.assertEqual(urllib.request.urlopen(base + "/").status, 200)
            self.assertEqual(urllib.request.urlopen(base + "/frames/0").status, 200)
            for path in ("/../../etc/passwd", "/frames/../frames.json", "/frames/%2e%2e/etc/passwd", "/video", "/frames/999"):
                with self.subTest(path=path), self.assertRaises(urllib.error.HTTPError):
                    urllib.request.urlopen(base + path)
            value = copy.deepcopy(payload(self.store))
            value["source"]["sha256"] = "b" * 64
            request = urllib.request.Request(base + "/api/save", data=json.dumps(value).encode(), headers={"Content-Type": "application/json"})
            with self.assertRaises(urllib.error.HTTPError) as result:
                urllib.request.urlopen(request)
            self.assertEqual(result.exception.code, 400)
        finally:
            http.shutdown()
            http.server_close()
            thread.join()

    def test_http_repeated_save_keeps_exact_nanoseconds_without_reload(self):
        document = json.loads((self.root / "frames.json").read_text())
        document["source"]["mtime_ns"] = 1788227559339036900
        (self.root / "frames.json").write_text(json.dumps(document))
        store = server.AnnotationStore(self.root / "frames.json", self.root / "http-mtime")
        http = server.make_server(store, 0)
        thread = threading.Thread(target=http.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{http.server_port}"
        try:
            state = json.load(urllib.request.urlopen(base + "/api/state"))
            value = payload(store)
            value["source"] = state["annotation"]["source"]
            first_bytes = None
            for revision in (1, 2):
                request = urllib.request.Request(base + "/api/save", data=json.dumps(value).encode(),
                                                 headers={"Content-Type": "application/json"})
                response = json.load(urllib.request.urlopen(request))
                self.assertEqual(response["annotation"]["source"]["mtime_ns"], "1788227559339036900")
                persisted = json.loads((store.out / f"annotations-{revision:06d}.json").read_text())
                self.assertEqual(persisted["source"], document["source"])
                self.assertEqual(persisted["features"], value["features"])
                if revision == 1:
                    first_bytes = (store.out / "annotations-000001.json").read_bytes()
                value = response["annotation"]
                value.update(revision=revision + 1, previous_revision_sha256=response["revision_sha256"])
            self.assertEqual((store.out / "annotations-000001.json").read_bytes(), first_bytes)
            value["source"]["mtime_ns"] = 1788227559339037000
            bad = urllib.request.Request(base + "/api/save", data=json.dumps(value).encode(),
                                         headers={"Content-Type": "application/json"})
            with self.assertRaises(urllib.error.HTTPError) as rejected:
                urllib.request.urlopen(bad)
            self.assertEqual(rejected.exception.code, 400)
            self.assertFalse((store.out / "annotations-000003.json").exists())
        finally:
            http.shutdown()
            http.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
