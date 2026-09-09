"""Source-bound immutable manual adjustments shared by preview and ground lookup."""
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import time

import numpy as np
from scipy.spatial import cKDTree

from calibration import field_recovery as recovery
from calibration.field_marking_check import check_marking_orientation


SCHEMA = "field-adjustment-atlas-v1"
RADIUS_M = 20.
MAX_OPERATIONS = 100


class AdjustmentError(ValueError):
    def __init__(self, message, code="invalid_adjustment"):
        super().__init__(message)
        self.code = code


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pair(value, name):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in value)):
        raise AdjustmentError(f"{name} must contain two finite numbers")
    return np.asarray(value, float)


def atomic_json(path, value, exclusive=False):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=".adjustment-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def kernel(world, center):
    radius = np.minimum(np.linalg.norm(np.asarray(world) - center, axis=1) / RADIUS_M, 1.)
    return (1 - radius) ** 4 * (4 * radius + 1)


class AdjustmentAtlas:
    def __init__(self, document):
        if document.get("schema") != SCHEMA:
            raise AdjustmentError("Wrong adjusted atlas schema")
        payload = document["payload"]
        if document.get("payload_sha256") != recovery.digest(payload):
            raise AdjustmentError("Adjusted atlas checksum mismatch")
        self.parent = recovery.RecoveryAtlas(payload["parent"])
        if payload["parent_document_sha256"] != recovery.digest(payload["parent"]):
            raise AdjustmentError("Embedded parent checksum mismatch")
        if payload.get("metric_certified") is not False or payload.get("status") != "approximate":
            raise AdjustmentError("Manual adjustment cannot certify geometry")
        if (payload.get("method") != "reference_wendland_translation_v1" or payload.get("radius_m") != RADIUS_M
                or payload.get("source") != self.parent.payload["source"]
                or payload.get("field") != self.parent.payload["field"]):
            raise AdjustmentError("Adjusted atlas method/source/field mismatch")
        self.operations = deepcopy(payload["operations"])
        if not isinstance(self.operations, list) or len(self.operations) > MAX_OPERATIONS:
            raise AdjustmentError("Too many adjustment operations")
        for operation in self.operations:
            if operation.get("mode") not in ("local", "global"):
                raise AdjustmentError("Unknown adjustment mode")
            center = pair(operation["center_world"], "center_world")
            pair(operation["delta_reference"], "delta_reference")
            if (abs(center) > [recovery.HALF_L, recovery.HALF_W]).any():
                raise AdjustmentError("Adjustment center lies outside the field")
        self.document = deepcopy(document)
        self.payload = deepcopy(payload)

    @classmethod
    def create(cls, parent, base_sha256, operations):
        payload = dict(parent=deepcopy(parent.document), parent_document_sha256=recovery.digest(parent.document),
                       base_sha256=base_sha256, source=deepcopy(parent.payload["source"]),
                       field=deepcopy(parent.payload["field"]), status="approximate", metric_certified=False,
                       method="reference_wendland_translation_v1", radius_m=RADIUS_M,
                       scope="existing_frames_in_this_prepared_clip", operations=deepcopy(operations))
        return cls(dict(schema=SCHEMA, payload=payload, payload_sha256=recovery.digest(payload)))

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def frame(self, source_pts, source_time_base, *, source_sha256, native_size):
        parent = self.parent.frame(source_pts, source_time_base, source_sha256=source_sha256, native_size=native_size)
        return AdjustmentFrame(parent, self.operations)


class AdjustmentFrame:
    def __init__(self, parent, operations):
        self.parent, self.operations = parent, operations
        self.atlas, self.record = parent.atlas, deepcopy(parent.record)

    def project(self, world, supported_only=None):
        world = np.asarray(world, float)
        native = self.parent.project(world, supported_only=supported_only)
        if not self.operations:
            return native  # Exact identity, including all nonfinite patterns.
        shift = np.zeros((len(world), 2))
        for operation in self.operations:
            delta = np.asarray(operation["delta_reference"])
            if operation["mode"] == "global":
                shift += delta
            else:
                shift += kernel(world, operation["center_world"])[:, None] * delta
        active = np.any(shift != 0, axis=1) & np.isfinite(native).all(axis=1)
        if active.any():
            motion = np.asarray(self.record["reference_to_native"])
            reference = recovery.project(np.linalg.inv(motion), native[active]) + shift[active]
            native[active] = recovery.project(motion, reference)
        return native

    def public_projection(self, world):
        warnings = list(self.record.get("warnings", []))
        if warnings:
            return dict(status="unavailable", native_pixels=None, warnings=warnings, metric_certified=False)
        world = np.asarray(world, float)
        native = self.project(world, supported_only=True)
        good = (self.atlas.support_mask(world) & np.isfinite(native).all(axis=1)
                & (native >= 0).all(axis=1) & (native < self.record["native_size"]).all(axis=1))
        return dict(status="approximate" if good.any() else "unavailable", metric_certified=False,
                    native_pixels=[p.tolist() if ok else None for p, ok in zip(native, good)],
                    observed_support=good.tolist(), warnings=[])

    def locate(self, pixels):
        pixels = np.asarray([pair(p, "native pixel") for p in pixels])
        warnings = list(self.record.get("warnings", []))
        results = [dict(status="unavailable", field_position_m=None, uncertainty_m=None,
                        metric_certified=False, warnings=warnings, reason="no_valid_supported_inverse") for _ in pixels]
        if warnings or not len(pixels):
            for result in results:
                result["reason"] = "frame_requires_review"
            return results
        x, y = np.meshgrid(np.linspace(-recovery.HALF_L, recovery.HALF_L, 181),
                           np.linspace(-recovery.HALF_W, recovery.HALF_W, 107))
        grid = np.c_[x.ravel(), y.ravel()]
        projected = self.project(grid)
        orientation = check_marking_orientation(self, grid, "domain_boundary_v2")
        available = self.atlas.support_mask(grid) & np.isfinite(projected).all(axis=1) & orientation["positive"]
        if not available.any():
            return results
        grid, projected = grid[available], projected[available]
        tree = cKDTree(projected)
        for index, target in enumerate(pixels):
            if (target < 0).any() or (target >= self.record["native_size"]).any():
                results[index]["reason"] = "outside_native_image"
                continue
            _, nearest = tree.query(target, k=min(48, len(grid)))
            seeds = []
            for candidate in grid[np.atleast_1d(nearest)]:
                if not seeds or min(np.linalg.norm(candidate - s) for s in seeds) >= .5:
                    seeds.append(candidate.copy())
                if len(seeds) == 12:
                    break
            guess = np.asarray(seeds)
            for _ in range(30):
                prediction = self.project(guess)
                error = np.linalg.norm(prediction - target, axis=1)
                checked = check_marking_orientation(self, guess, "domain_boundary_v2")
                jac = checked["derivatives"]
                active = checked["positive"] & np.isfinite(error) & (error > 1e-7)
                if not active.any():
                    break
                step = np.zeros_like(guess)
                step[active] = np.linalg.solve(jac[active], (prediction - target)[active, :, None])[:, :, 0]
                step *= np.minimum(1, 10 / np.maximum(np.linalg.norm(step, axis=1), 1e-12))[:, None]
                for scale in (1., .5, .25, .125, .0625):
                    proposal = np.clip(guess - scale*step, [-recovery.HALF_L, -recovery.HALF_W],
                                       [recovery.HALF_L, recovery.HALF_W])
                    new_error = np.linalg.norm(self.project(proposal) - target, axis=1)
                    improve = active & np.isfinite(new_error) & (new_error < error)
                    guess[improve] = proposal[improve]
                    active[improve] = False
            error = np.linalg.norm(self.project(guess) - target, axis=1)
            checked = check_marking_orientation(self, guess, "domain_boundary_v2")
            valid = self.atlas.support_mask(guess) & checked["positive"] & np.isfinite(error) & (error <= .01)
            roots = guess[valid]
            if not len(roots):
                continue
            if np.max(np.linalg.norm(roots - roots[0], axis=1)) > .001:
                results[index]["reason"] = "ambiguous_inverse"
                continue
            best = np.flatnonzero(valid)[np.argmin(error[valid])]
            results[index].update(status="approximate", field_position_m=guess[best].tolist(),
                                  numeric_residual_px=float(error[best]), reason="metric_accuracy_unverified")
        return results


def validate_geometry(atlas, timeout_s=75.):
    started = time.monotonic()
    x, y = np.meshgrid(np.linspace(-recovery.HALF_L, recovery.HALF_L, 181),
                       np.linspace(-recovery.HALF_W, recovery.HALF_W, 107))
    grid = np.c_[x.ravel(), y.ravel()]
    rows = []
    for record in atlas.parent.payload["frames"]:
        if time.monotonic() - started > timeout_s:
            raise AdjustmentError("Geometry validation exceeded its local time budget", "validation_timeout")
        frame = atlas.frame(record["source_pts"], record["source_time_base"],
                            source_sha256=atlas.parent.payload["source"]["sha256"], native_size=record["native_size"])
        result = recovery.visible_geometry_check(frame, grid)
        if result["nonfinite_supported_projections"]:
            before = frame.parent.project(grid)
            after = frame.project(grid)
            new_pole = (frame.atlas.support_mask(grid) & np.isfinite(before).all(axis=1)
                        & ~np.isfinite(after).all(axis=1))
            if new_pole.any():
                raise AdjustmentError("Adjustment creates a new pole in observed support", "invalid_geometry")
        if not result["valid"]:
            raise AdjustmentError(f"Adjustment creates invalid visible geometry at PTS {record['source_pts']}", "invalid_geometry")
        rows.append(dict(source_pts=record["source_pts"], **result))
    return dict(status="numerical_geometry_checked", metric_certified=False, frames_checked=len(rows),
                grid_points_per_frame=len(grid), elapsed_seconds=time.monotonic() - started, frames=rows)


def _markings():
    labels = ["touch_far", "touch_near", "halfway", "goal_left", "goal_right"]
    labels += [f"box{size}_{side}_{edge}" for size in (18, 6) for side in ("left", "right")
               for edge in ("front", "near", "far")]
    labels += ["centre_circle", "pen_arc_left", "pen_arc_right"]
    markings = []
    for label in labels:
        feature = dict(label=label, points_native=[[0, 0]])
        if "circle" in label or "arc" in label:
            world = recovery.feature_curve(feature, samples=121)
        else:
            ends = recovery.feature_curve(feature, samples=2)
            world = recovery.feature_curve(feature, samples=int(np.ceil(np.linalg.norm(ends[1]-ends[0])))+1)
        markings.append(dict(id=label, label=label, world=world.tolist()))
    return markings


def _verify_frame_image(frames_path, record, parent_record):
    base = Path(frames_path).resolve().parent
    relative = Path(record["file"])
    path = (base / relative).resolve(strict=True)
    if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(base):
        raise AdjustmentError("Frame file lies outside its source manifest", "frame_mismatch")
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != record["image_sha256"] or parent_record.get("image_sha256", actual) != actual:
        raise AdjustmentError("Source frame image hash mismatch", "frame_mismatch")
    if len(data) >= 24 and data.startswith(b"\x89PNG\r\n\x1a\n") and data[12:16] == b"IHDR":
        size = list(struct.unpack(">II", data[16:24]))
    else:
        import cv2
        decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        size = [decoded.shape[1], decoded.shape[0]] if decoded is not None else None
    if size != record["native_size"] or size != parent_record["native_size"]:
        raise AdjustmentError("Source frame native dimensions mismatch", "frame_mismatch")


def _select_video(output, video_name):
    if video_name is not None:
        if not isinstance(video_name, str) or Path(video_name).name != video_name:
            raise AdjustmentError("Video name must be a basename", "invalid_video_name")
        path = output / video_name
        if path.suffix.lower() != ".webm" or not path.is_file() or path.is_symlink():
            raise AdjustmentError("Selected video must be an existing local WebM file", "invalid_video_name")
        return path
    videos = sorted(output.glob("source*.webm"))
    if len(videos) > 1:
        raise AdjustmentError("Multiple retained videos: choose --video-name explicitly", "ambiguous_video")
    if videos and (not videos[0].is_file() or videos[0].is_symlink()):
        raise AdjustmentError("Invalid prepared video file", "invalid_video_name")
    return videos[0] if videos else None


def _verify_video(path, native_size, count):
    from fractions import Fraction
    try:
        completed = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                                    "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_read_frames:format=duration",
                                    "-of", "json", str(path)], capture_output=True, text=True, check=True, timeout=45)
        probe = json.loads(completed.stdout)
        stream = probe["streams"][0]
        duration = float(probe["format"]["duration"])
        valid = ([stream["width"], stream["height"]] == native_size
                 and Fraction(stream["avg_frame_rate"]) == 10
                 and Fraction(stream["r_frame_rate"]) == 10
                 and int(stream["nb_read_frames"]) == count and abs(duration-count/10) <= .05)
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError, TypeError, ZeroDivisionError) as exc:
        raise AdjustmentError("Prepared video could not be verified", "video_mismatch") from exc
    if not valid:
        raise AdjustmentError("Prepared video dimensions, cadence, frame count or duration mismatch", "video_mismatch")
    return dict(sha256=file_hash(path), native_size=native_size, frames=count, fps=10, duration_s=duration,
                method="ffprobe decoded frame count and stream metadata")


def prepare(atlas_path, frames_path, out, identifier, title, video_name=None):
    from fractions import Fraction
    output = Path(out)
    for name in ("manifest.json", "packet.json", "base-map.json", "revisions", "active-map.json"):
        if (output / name).exists():
            raise AdjustmentError("An adjustment session already exists in this directory", "session_exists")
    video = _select_video(output, video_name)
    input_hashes = {str(path): file_hash(path) for path in (atlas_path, frames_path)}
    parent = recovery.RecoveryAtlas.load(atlas_path)
    frames = json.loads(Path(frames_path).read_text())
    source_hash = parent.payload["source"]["sha256"]
    if frames["source"]["sha256"] != source_hash:
        raise AdjustmentError("Atlas/source manifest mismatch", "source_mismatch")
    records = frames["frames"]
    if len(records) != len(parent.frames) or not records:
        raise AdjustmentError("Prepared clip must contain exactly the saved atlas frames")
    native_size = records[0]["native_size"]
    video_validation = _verify_video(video, native_size, len(records)) if video else None
    previous = None
    packet_frames = []
    markings = _markings()
    for marking in markings:
        marking["observed_support"] = parent.support_mask(marking["world"]).tolist()
    for index, record in enumerate(records):
        stamp = record["source_pts"] * Fraction(record["source_time_base"])
        if previous is not None and stamp-previous != Fraction(1, 10):
            raise AdjustmentError("Prepared clip requires the saved 10 fps frame cadence")
        previous = stamp
        if record["native_size"] != native_size or record.get("source_sha256", source_hash) != source_hash:
            raise AdjustmentError("Prepared frame source/dimensions mismatch")
        frame = parent.frame(record["source_pts"], record["source_time_base"], source_sha256=source_hash, native_size=native_size)
        _verify_frame_image(frames_path, record, frame.record)
        pixels = []
        for marking in markings:
            projected = frame.project(marking["world"])
            pixels.append([np.round(p, 6).tolist() if np.isfinite(p).all() else None for p in projected])
        packet_frames.append(dict(frame_index=index, source_pts=record["source_pts"],
                                   source_time_base=record["source_time_base"], source_seconds=float(stamp),
                                   image_sha256=record["image_sha256"], reference_to_native=frame.record["reference_to_native"],
                                   base_pixels=pixels, warnings=list(frame.record.get("warnings", []))))
    if any(file_hash(path) != expected for path, expected in input_hashes.items()):
        raise AdjustmentError("Atlas or source manifest changed during preparation", "input_changed")
    base_hash = input_hashes[str(atlas_path)]
    packet = dict(id=identifier, title=title, source_sha256=source_hash, native_size=native_size,
                  fps=10, duration_s=len(records)/10, source_start_s=packet_frames[0]["source_seconds"],
                  base_sha256=base_hash, markings=markings, frames=packet_frames, metric_certified=False)
    manifest = dict(schema="field-adjustment-session-v1", id=identifier, title=title,
                    source_sha256=source_hash, base_sha256=base_hash, native_size=native_size,
                    frames_manifest_sha256=input_hashes[str(frames_path)], source_frames_path=str(Path(frames_path).resolve()),
                    parent_atlas_path=str(Path(atlas_path).resolve()), frames=records,
                    packet_path="packet.json", active_map_path="active-map.json",
                    video_path=video.name if video else None, video_validation=video_validation,
                    source_frame_verification="image bytes, SHA-256 and native dimensions verified at preparation")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "packet.json", "base-map.json", "revisions", "active-map.json"):
        if (output / name).exists():
            raise AdjustmentError("An adjustment session already exists in this directory", "session_exists")
    (output / "revisions").mkdir()
    baseline = AdjustmentAtlas.create(parent, base_hash, [])
    state = dict(revision=base_hash, operations=[], confirmed=False, updated_at=None,
                 source_sha256=source_hash, base_sha256=base_hash, metric_certified=False)
    atomic_json(output / "manifest.json", manifest, exclusive=True)
    atomic_json(output / "packet.json", packet, exclusive=True)
    atomic_json(output / "base-map.json", baseline.document, exclusive=True)
    atomic_json(output / "revisions" / f"{base_hash}.json", dict(state=state, atlas=baseline.document), exclusive=True)
    atomic_json(output / "active-map.json", dict(revision=base_hash, path=f"revisions/{base_hash}.json"), exclusive=True)
    return dict(root=str(output.resolve()), **state, frames=len(records))


class AdjustmentStore:
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.baseline = AdjustmentAtlas.load(self.root / "base-map.json")

    def _active(self):
        pointer = json.loads((self.root / "active-map.json").read_text())
        revision = pointer["revision"]
        if (not isinstance(revision, str) or len(revision) != 64
                or any(c not in "0123456789abcdef" for c in revision)
                or pointer["path"] != f"revisions/{revision}.json"):
            raise AdjustmentError("Invalid active revision pointer")
        record = json.loads((self.root / pointer["path"]).read_text())
        if record["state"]["revision"] != revision:
            raise AdjustmentError("Active revision mismatch")
        atlas = AdjustmentAtlas(record["atlas"])
        if (atlas.payload["parent_document_sha256"] != self.baseline.payload["parent_document_sha256"]
                or atlas.payload["base_sha256"] != self.manifest["base_sha256"]
                or atlas.operations != record["state"]["operations"]):
            raise AdjustmentError("Active map does not match its immutable parent/state")
        if revision != self.manifest["base_sha256"]:
            state = {k: v for k, v in record["state"].items() if k != "revision"}
            if revision != recovery.digest(dict(state=state, atlas_sha256=atlas.document["payload_sha256"])):
                raise AdjustmentError("Active revision checksum mismatch")
        return record

    def state(self):
        return self._active()["state"]

    def _binding(self, request, expected_revision, active):
        if not isinstance(request, dict):
            raise AdjustmentError("Expected a source-bound JSON object")
        if request.get(expected_revision) != active["state"]["revision"]:
            raise AdjustmentError("The adjustment changed; reload before continuing", "revision_conflict")
        for key in ("source_sha256", "base_sha256"):
            if request.get(key) != self.manifest[key]:
                raise AdjustmentError(f"Wrong {key}", "source_mismatch")

    def _frame_record(self, anchor):
        if not isinstance(anchor, dict):
            raise AdjustmentError("Expected an anchor object")
        index = anchor.get("frame_index")
        records = self.manifest["frames"]
        if type(index) is not int or not 0 <= index < len(records):
            raise AdjustmentError("Unknown anchor frame")
        record = records[index]
        for key in ("source_pts", "source_time_base", "image_sha256"):
            if anchor.get(key) != record[key] or (key == "source_pts" and type(anchor[key]) is not int):
                raise AdjustmentError(f"Anchor {key} mismatch", "frame_mismatch")
        return record

    def _operations(self, requested):
        if not isinstance(requested, list) or len(requested) > MAX_OPERATIONS:
            raise AdjustmentError("At most 100 operations are supported")
        operations = []
        for operation in requested:
            if not isinstance(operation, dict) or operation.get("mode") not in ("local", "global"):
                raise AdjustmentError("Unknown adjustment operation")
            center = pair(operation.get("center_world"), "center_world")
            supplied_delta = pair(operation.get("delta_reference"), "delta_reference")
            if (abs(center) > [recovery.HALF_L, recovery.HALF_W]).any():
                raise AdjustmentError("Anchor lies outside the field")
            anchor = operation["anchor"]
            record = self._frame_record(anchor)
            previous = AdjustmentAtlas.create(self.baseline.parent, self.manifest["base_sha256"], operations)
            frame = previous.frame(record["source_pts"], record["source_time_base"],
                                   source_sha256=self.manifest["source_sha256"], native_size=record["native_size"])
            if frame.record.get("warnings"):
                raise AdjustmentError("Choose an anchor frame without mapping warnings", "warned_anchor")
            if not frame.atlas.support_mask([center])[0]:
                raise AdjustmentError("Anchor lies outside observed field support", "unsupported_anchor")
            expected_from = frame.project([center])[0]
            supplied_from = pair(anchor.get("from_native"), "from_native")
            target = pair(anchor.get("to_native"), "to_native")
            if (not np.isfinite(expected_from).all() or np.linalg.norm(expected_from-supplied_from) > 1e-4
                    or (supplied_from < 0).any() or (supplied_from >= record["native_size"]).any()
                    or (target < 0).any() or (target >= record["native_size"]).any()):
                raise AdjustmentError("Drag does not match a visible authoritative anchor", "anchor_mismatch")
            inverse = np.linalg.inv(np.asarray(frame.record["reference_to_native"]))
            delta = recovery.project(inverse, [target])[0] - recovery.project(inverse, [expected_from])[0]
            if not np.isfinite(delta).all() or not np.allclose(delta, supplied_delta, rtol=1e-9, atol=1e-4):
                raise AdjustmentError("Drag delta disagrees with source-bound anchor", "delta_mismatch")
            canonical = dict(mode=operation["mode"], center_world=center.tolist(), delta_reference=delta.tolist(),
                             anchor=dict(frame_index=anchor["frame_index"], source_pts=record["source_pts"],
                                         source_time_base=record["source_time_base"], image_sha256=record["image_sha256"],
                                         from_native=expected_from.tolist(), to_native=target.tolist()))
            operations.append(canonical)
        return operations

    def save(self, request):
        with (self.root / ".lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            active = self._active()
            self._binding(request, "expected_revision", active)
            if type(request.get("confirmed")) is not bool:
                raise AdjustmentError("confirmed must be a boolean")
            operations = self._operations(request.get("operations"))
            atlas = AdjustmentAtlas.create(self.baseline.parent, self.manifest["base_sha256"], operations)
            geometry = validate_geometry(atlas)
            state = dict(operations=operations, confirmed=request["confirmed"], updated_at=datetime.now(timezone.utc).isoformat(),
                         source_sha256=self.manifest["source_sha256"], base_sha256=self.manifest["base_sha256"],
                         previous_revision=active["state"]["revision"], metric_certified=False,
                         validation=dict(status=geometry["status"], frames_checked=geometry["frames_checked"],
                                         alignment_status="manually_adjusted_unverified", previous_paint_checks="parent_revision_only"))
            revision = recovery.digest(dict(state=state, atlas_sha256=atlas.document["payload_sha256"]))
            state["revision"] = revision
            atomic_json(self.root / "revisions" / f"{revision}.json", dict(state=state, atlas=atlas.document, geometry=geometry), exclusive=True)
            atomic_json(self.root / "active-map.json", dict(revision=revision, path=f"revisions/{revision}.json"))
            return state

    def locate(self, request):
        active = self._active()
        self._binding(request, "revision", active)
        record = self._frame_record(request)
        atlas = AdjustmentAtlas(active["atlas"])
        frame = atlas.frame(record["source_pts"], record["source_time_base"],
                            source_sha256=self.manifest["source_sha256"], native_size=record["native_size"])
        pixels = request.get("native_pixels")
        if pixels is None and "native_pixel" in request:
            pixels = [request["native_pixel"]]
        if not isinstance(pixels, list) or not 1 <= len(pixels) <= 100:
            raise AdjustmentError("Provide 1 to 100 native pixels")
        return dict(revision=active["state"]["revision"], metric_certified=False, results=frame.locate(pixels))
