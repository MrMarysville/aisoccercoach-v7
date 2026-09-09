"""Portable approximate field atlas, bound to exact native source frames.

This module evaluates an existing fit. It neither calibrates a new game nor
certifies metric accuracy. Importing it reads no artifacts and starts no work.
"""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


SCHEMA = "native-field-atlas-v1"


def digest(data):
    """Canonical content checksum; provenance integrity, not an authenticity claim."""
    raw = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def save(path, payload):
    path = Path(path)
    document = {"schema": SCHEMA, "payload": payload, "payload_sha256": digest(payload)}
    # Validate before creating any output. Existing packages are immutable.
    FieldAtlas(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _matrix(value, name):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid {name}")
    if np.linalg.matrix_rank(matrix) != 3:
        raise ValueError(f"Singular {name}")
    return matrix


def _time(pts, time_base):
    if type(pts) is not int or not isinstance(time_base, str):
        raise ValueError("Use integer source PTS and a rational time-base string")
    try:
        scale = Fraction(time_base)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("Invalid source time base") from exc
    if scale <= 0:
        raise ValueError("Source time base must be positive")
    return pts * scale


def _points(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Expected an N by 2 coordinate array")
    return points


def _project(matrix, points):
    points = _points(points)
    q = np.c_[points, np.ones(len(points))] @ matrix.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return q[:, :2] / q[:, 2:]


def _smooth(value):
    value = np.clip(value, 0, 1)
    return value * value * (3 - 2 * value)


class FieldAtlas:
    """An approximate spatial chart plus measured-frame motion packets."""

    def __init__(self, document):
        if document.get("schema") != SCHEMA:
            raise ValueError("Unknown field-atlas schema")
        data = deepcopy(document["payload"])
        if digest(data) != document.get("payload_sha256"):
            raise ValueError("Field-atlas content checksum mismatch")
        if data.get("status") != "approximate" or data.get("metric_certified") is not False:
            raise ValueError("This evaluator only supports approximate, uncertified maps")
        source = data["source"]
        if (not isinstance(source.get("game_id"), str)
                or not isinstance(source.get("sha256"), str)
                or len(source["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in source["sha256"])):
            raise ValueError("Missing source game/hash binding")
        self.source = deepcopy(source)
        field = data["field"]
        self.half = np.array([_number(field[k], k) / 2 for k in ("length_m", "width_m")])
        if np.any(self.half <= 0) or not field.get("dimensions_provenance"):
            raise ValueError("Positive field dimensions and provenance are required")
        if field.get("axes") != "centre,+x_camera_right,+y_near":
            raise ValueError("Unsupported field axes")
        spatial = data["spatial"]
        if len(spatial["field_to_reference_models"]) != 3:
            raise ValueError("This chart requires left, midfield and right models")
        self.models = [_matrix(m, "local chart") for m in spatial["field_to_reference_models"]]
        self.blend = {k: _number(spatial["blend"][k], k) for k in
                      ("end_x_start_m", "end_x_span_m", "near_y_start_m",
                       "near_y_span_m", "near_right_x_start_m", "near_right_x_span_m")}
        if self.blend["end_x_start_m"] < 0 or any(
                self.blend[k] <= 0 for k in self.blend if k.endswith("span_m")):
            raise ValueError("Invalid spatial blend intervals")
        edge = spatial["far_boundary"]
        self.coefficients = np.asarray(edge["reference_y_coefficients"], dtype=float)
        if self.coefficients.shape != (3,) or not np.isfinite(self.coefficients).all():
            raise ValueError("Boundary must have three finite polynomial coefficients")
        self.edge_origin = _number(edge["reference_x_origin_px"], "boundary origin")
        self.edge_scale = _number(edge["reference_x_scale_px"], "boundary scale")
        self.protected_y = _number(edge["protected_y_m"], "protected boundary")
        if self.edge_scale <= 0 or not -self.half[1] < self.protected_y < self.half[1]:
            raise ValueError("Invalid boundary correction domain")
        self.frames = {}
        previous = None
        for record in data["frames"]:
            time = _time(record["source_pts"], record["source_time_base"])
            if time in self.frames or (previous is not None and time <= previous):
                raise ValueError("Source frames must be unique and strictly ordered")
            size = record["native_size"]
            if len(size) != 2 or any(type(x) is not int or x <= 0 for x in size):
                raise ValueError("Invalid native frame dimensions")
            matrix = _matrix(record["reference_to_native"], "frame transform")
            offset = _number(record["boundary_offset_native_y_px"], "frame correction")
            warnings = record.get("warnings", [])
            if not isinstance(warnings, list) or not all(isinstance(x, str) for x in warnings):
                raise ValueError("Invalid frame warnings")
            self.frames[time] = (record, matrix, offset)
            previous = time
        if not self.frames:
            raise ValueError("Atlas has no exact source frames")
        self.document = deepcopy(document)
        self._lookup = None

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def _base(self, points):
        points = _points(points)
        b = self.blend
        x, y = points.T
        wr = _smooth((x - b["end_x_start_m"]) / b["end_x_span_m"])
        wl = _smooth((-x - b["end_x_start_m"]) / b["end_x_span_m"])
        yn = _smooth((y - b["near_y_start_m"]) / b["near_y_span_m"])
        nr = _smooth((x - b["near_right_x_start_m"]) / b["near_right_x_span_m"])
        weights = [(1 - yn) * wl + yn * (1 - nr), (1 - yn) * (1 - wl - wr),
                   (1 - yn) * wr + yn * nr]
        out = np.zeros((len(points), 2))
        for matrix, weight in zip(self.models, weights):
            use = weight > 0  # Avoid zero times an inactive chart's pole.
            out[use] += _project(matrix, points[use]) * weight[use, None]
        return out

    def edge_weight(self, points):
        return _smooth((self.protected_y - points[:, 1]) / (self.half[1] + self.protected_y))

    def reference(self, points):
        """Field metres to the nonlinear reference chart; raw mathematical projection."""
        points = _points(points)
        out = self._base(points)
        far = self._base(np.c_[points[:, 0], np.full(len(points), -self.half[1])])
        target = np.polynomial.polynomial.polyval(
            (far[:, 0] - self.edge_origin) / self.edge_scale, self.coefficients)
        out[:, 1] += (target - far[:, 1]) * self.edge_weight(points)
        return out

    def frame(self, source_pts, source_time_base, *, source_sha256, native_size):
        """Bind a consumer's decoded frame. No timestamp rounding or interpolation."""
        if source_sha256 != self.source["sha256"]:
            raise ValueError("Source video hash mismatch")
        time = _time(source_pts, source_time_base)
        if time not in self.frames:
            raise ValueError("No mapping for this exact source PTS")
        record, matrix, offset = self.frames[time]
        if list(native_size) != record["native_size"]:
            raise ValueError("Pixel-domain mismatch: native frame dimensions required")
        return FrameMapping(self, record, matrix, offset)

    def _initializer(self, reference_pixels):
        if self._lookup is None:
            x, y = np.meshgrid(np.linspace(-self.half[0], self.half[0], 221),
                               np.linspace(-self.half[1], self.half[1], 129))
            grid = np.c_[x.ravel(), y.ravel()]
            pixels = self.reference(grid)
            finite = np.isfinite(pixels).all(axis=1)
            if not finite.any():
                raise ValueError("No finite reference chart")
            self._lookup = (grid[finite], cKDTree(pixels[finite]))
        grid, tree = self._lookup
        return grid[tree.query(reference_pixels)[1]].copy()


class FrameMapping:
    """One exact native frame. Query results remain approximate or unavailable."""

    def __init__(self, atlas, record, matrix, offset):
        self.atlas = atlas
        self.record = deepcopy(record)
        self.matrix = matrix.copy()
        self.offset = offset

    def project(self, field_points):
        """Raw field-to-native projection for diagnostics; clip before drawing.

        Offscreen points and projective poles are possible. A finite output by
        itself does not establish visibility, support or metric accuracy.
        """
        points = _points(field_points)
        out = _project(self.matrix, self.atlas.reference(points))
        out[:, 1] += self.offset * self.atlas.edge_weight(points)
        return out

    def _jacobian(self, points):
        eps = 1e-4
        dx = (self.project(points + [eps, 0]) - self.project(points - [eps, 0])) / (2 * eps)
        dy = (self.project(points + [0, eps]) - self.project(points - [0, eps])) / (2 * eps)
        return np.stack([dx, dy], axis=2)

    def locate(self, native_pixels):
        """Invert within the field and visible image; never return a certified point."""
        pixels = _points(native_pixels)
        width, height = self.record["native_size"]
        valid = (np.isfinite(pixels).all(axis=1) & (pixels[:, 0] >= 0)
                 & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height))
        results = [dict(status="unavailable", field_position_m=None,
                        reason="outside_native_image_or_nonfinite", metric_certified=False,
                        uncertainty_m=None, warnings=list(self.record.get("warnings", [])))
                   for _ in pixels]
        if not valid.any():
            return results
        indices = np.flatnonzero(valid)
        target = pixels[valid]
        ref = _project(np.linalg.inv(self.matrix), target)
        finite = np.isfinite(ref).all(axis=1)
        indices, target, ref = indices[finite], target[finite], ref[finite]
        if not len(indices):
            return results
        guess = self.atlas._initializer(ref)
        for _ in range(20):
            pred = self.project(guess)
            error = np.linalg.norm(pred - target, axis=1)
            jac = self._jacobian(guess)
            det = np.linalg.det(jac)
            good = np.isfinite(jac).all(axis=(1, 2)) & np.isfinite(error) & (abs(det) > 1e-10)
            active = good & (error > 1e-6)
            if not active.any():
                break
            step = np.zeros_like(guess)
            step[active] = np.linalg.solve(jac[active], (pred - target)[active, :, None])[:, :, 0]
            step *= np.minimum(1, 10 / np.maximum(1e-12, np.linalg.norm(step, axis=1)))[:, None]
            # Bounded descent prevents an initializer from crossing a distant pole.
            for scale in (1, .5, .25, .125, .0625):
                proposal = np.clip(guess - scale * step, -self.atlas.half, self.atlas.half)
                new_error = np.linalg.norm(self.project(proposal) - target, axis=1)
                improve = active & np.isfinite(new_error) & (new_error < error)
                guess[improve] = proposal[improve]
                active[improve] = False
        error = np.linalg.norm(self.project(guess) - target, axis=1)
        det = np.linalg.det(self._jacobian(guess))
        good = (np.isfinite(guess).all(axis=1) & np.isfinite(error)
                & (error <= .01) & (det > 1e-8)
                & (abs(guess) <= self.atlas.half + 1e-8).all(axis=1))
        for j, index in enumerate(indices):
            result = results[index]
            if not good[j]:
                result["reason"] = "no_valid_visible_ground_inverse"
            elif self.record.get("warnings"):
                result["reason"] = "frame_requires_review"
                result["candidate_field_position_m"] = guess[j].tolist()
            else:
                result.update(status="approximate", field_position_m=guess[j].tolist(),
                              reason="metric_accuracy_unverified", numeric_residual_px=float(error[j]))
        return results
