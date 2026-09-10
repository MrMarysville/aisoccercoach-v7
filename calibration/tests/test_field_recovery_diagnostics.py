"""Signed residual direction and immutable frozen-score agreement; synthetic only."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from calibration import field_recovery as r
from calibration.alignment_trial import sha256_file
from calibration.tools.diagnose_field_recovery import _nearest_residuals, diagnose
from calibration.tools.score_field_recovery import score


class ResidualTests(unittest.TestCase):
    def test_horizontal_and_vertical_signs(self):
        horizontal = _nearest_residuals([[5, 3], [5, -2]], [[0, 0], [10, 0]])
        self.assertEqual([p["signed_normal_px"] for p in horizontal], [3., -2.])
        vertical = _nearest_residuals([[3, 5], [-2, 5]], [[0, 0], [0, 10]])
        self.assertEqual([p["signed_normal_px"] for p in vertical], [-3., 2.])
        self.assertEqual(horizontal[0]["residual_native_dy_px"], 3.)

    def test_reversal_flips_signs_preserving_unsigned_distance(self):
        p = [[5, 3], [30, 2]]
        curve = np.array([[0., 0.], [10., 0.]])
        a, b = _nearest_residuals(p, curve), _nearest_residuals(p, curve[::-1])
        for first, second in zip(a, b):
            self.assertEqual(first["distance_px"], second["distance_px"])
            self.assertEqual(first["signed_normal_px"], -second["signed_normal_px"])
            self.assertEqual(first["tangent_residual_px"], -second["tangent_residual_px"])

    def test_endpoint_retains_large_tangent_error_and_clamp(self):
        row = _nearest_residuals([[40, 0]], [[0, 0], [10, 0]])[0]
        self.assertEqual(row["signed_normal_px"], 0.)
        self.assertEqual(row["distance_px"], 30.)
        self.assertEqual(row["tangent_residual_px"], 30.)
        self.assertEqual(row["segment_fraction"], 1.)
        self.assertEqual(row["unclamped_segment_fraction"], 4.)
        self.assertTrue(row["clamped_to_end"])
        self.assertFalse(row["clamped_to_start"])

    def test_missing_gapped_isolated_and_degenerate_curves(self):
        for curve in ([[np.nan, np.nan], [1, 2]], [[0, 0], [101, 0]], [[1, 2]]):
            row = _nearest_residuals([[3, 4]], curve)[0]
            self.assertIsNone(row["distance_px"])
            self.assertIsNone(row["signed_normal_px"])
            self.assertEqual(row["direction_status"], "unknown_no_scored_segment")
        row = _nearest_residuals([[4, 6]], [[1, 2], [1, 2]])[0]
        self.assertEqual(row["distance_px"], 5.)
        self.assertEqual(row["nearest_native_xy"], [1., 2.])
        self.assertIsNone(row["signed_normal_px"])
        self.assertIsNone(row["segment_fraction"])
        self.assertEqual(row["direction_status"], "unknown_degenerate_segment")

    def test_gap_indices_and_chunked_distances_match_core(self):
        rng = np.random.default_rng(4321)
        curve = np.c_[np.arange(200.), np.sin(np.arange(200.) / 10)]
        curve[20:30] = np.nan
        curve[100:] += [1000, 0]
        pixels = rng.uniform([-20, -20], [1300, 20], size=(151, 2))
        records = _nearest_residuals(pixels, curve)
        np.testing.assert_array_equal([x["distance_px"] for x in records], r.curve_distance(pixels, curve))
        row = _nearest_residuals([[35, 5]], [[np.nan, np.nan], [30, 0], [40, 0]])[0]
        self.assertEqual(row["nearest_segment_index"], 1)


def fixture(root, reflected=False):
    matrix = np.array([[5., 0., 960.], [0., 5., 500.], [0., 0., 1.]])
    if reflected:
        matrix[0, 0] *= -1
    source = dict(game_id="synthetic", sha256="a" * 64)
    frame = dict(index=0, source_pts=9000, source_time_base="1/90000", source_seconds=.1,
                 source_sha256=source["sha256"], game_id=source["game_id"],
                 native_size=[1920, 1080], file="unread-synthetic.png", image_sha256="b" * 64)
    payload = dict(status="approximate", metric_certified=False, source=source, field=r.FIELD,
                   projection_mode=r.SINGLE_REFERENCE_PROJECTION_MODE,
                   charts=[dict(chart="right", reference_frame_index=0, field_to_reference=matrix.tolist(),
                                support_world_polygon=[[-r.HALF_L, -r.HALF_W], [r.HALF_L, -r.HALF_W],
                                                       [r.HALF_L, r.HALF_W], [-r.HALF_L, r.HALF_W]])],
                   blend=r.BLEND, spatial_boundary=None, protected_y_m=-r.BOX18_HALF_W,
                   frames=[dict(frame, reference_to_native=np.eye(3).tolist(),
                                boundary_offset_native_y_px=0., warnings=[])])
    r.RecoveryAtlas.from_payload(payload).save(root / "atlas.json")
    manifest = dict(schema="alignment-trial-frames-v1", source=source, frames=[frame])
    (root / "frames.json").write_text(json.dumps(manifest))
    measurements = []
    for label in ("halfway", "touch_near"):
        curve = r.project(matrix, r.feature_curve(dict(label=label, points_native=[[0, 0]]), samples=5))
        pixels = (curve + [2., 3.]).tolist()
        measurements.append(dict(frame_index=0, label=label, feature_id=label + "_fixture",
                                 selected_count=5, searched_count=6, points_native=pixels))
    (root / "checks.json").write_text(json.dumps(dict(source_sha256=source["sha256"], measurements=measurements)))
    score(root / "atlas.json", root / "frames.json", root / "checks.json", root / "score.json")


class IntegrationTests(unittest.TestCase):
    def test_frozen_plan_keeps_empty_groups_and_missing_maps(self):
        for missing_map in (False, True):
            with self.subTest(missing_map=missing_map), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                fixture(root)
                manifest = json.loads((root / "frames.json").read_text())
                frame = manifest["frames"][0]
                checks = json.loads((root / "checks.json").read_text())
                checks["frames_manifest_sha256"] = sha256_file(root / "frames.json")
                for feature in checks["measurements"]:
                    feature["review_status"] = "source_reviewed"
                (root / "checks.json").write_text(json.dumps(checks))
                plan = dict(source_sha256=manifest["source"]["sha256"],
                    frames_manifest_sha256=checks["frames_manifest_sha256"],
                    fit_frame_indices=[], check_frame_indices=[0], groups=[
                        dict(frame_index=0, label=label, feature_ids=[label + "_fixture"],
                             observation_status="visible", **{k: frame[k] for k in
                             ("source_pts", "source_time_base", "image_sha256", "native_size")})
                        for label in ("halfway", "touch_near", "touch_far")])
                (root / "plan.json").write_text(json.dumps(dict(
                    schema="field-recovery-check-plan-v1", payload=plan, payload_sha256=r.digest(plan))))
                atlas_path = root / "atlas.json"
                if missing_map:
                    payload = r.RecoveryAtlas.load(atlas_path).payload
                    payload["frames"] = []
                    atlas_path = root / "missing-map.json"
                    r.RecoveryAtlas.from_payload(payload).save(atlas_path)
                score_path = root / "planned-score.json"
                score(atlas_path, root / "frames.json", root / "checks.json", score_path,
                      check_plan=root / "plan.json")
                before = score_path.read_bytes()
                diagnose(atlas_path, root / "frames.json", score_path, root / "diagnostic.json")
                data = json.loads((root / "diagnostic.json").read_text())
                self.assertEqual(score_path.read_bytes(), before)
                self.assertEqual(data["marking_frame_count"], 3)
                self.assertEqual(data["sample_count"], 10)
                self.assertEqual(data["max_unsigned_distance_disagreement_px"], 0.)
                empty = next(row for row in data["rows"] if row["label"] == "touch_far")
                self.assertEqual(empty["raw_samples"], [])
                self.assertFalse(empty["evidence_complete"])
                self.assertFalse(empty["experiment_pixel_target_pass"])
                self.assertIsNone(empty["signed_normal_summary"]["median_px"])
                if missing_map:
                    self.assertTrue(all(sample["residual_diagnostic"]["distance_px"] is None
                                        for row in data["rows"] for sample in row["raw_samples"]))

    def test_saved_map_frozen_scorer_and_diagnostic_agree_without_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            paths = [root / name for name in ("atlas.json", "frames.json", "score.json")]
            before = [p.read_bytes() for p in paths]
            report = diagnose(*paths, root / "diagnostic.json")
            self.assertEqual(report["sample_count"], 10)
            self.assertEqual(report["marking_frame_count"], 2)
            self.assertEqual(report["max_unsigned_distance_disagreement_px"], 0.)
            self.assertEqual(before, [p.read_bytes() for p in paths])
            data = json.loads((root / "diagnostic.json").read_text())
            frozen = json.loads(paths[2].read_text())
            self.assertEqual(data["score_sha256"], sha256_file(paths[2]))
            self.assertFalse(data["frozen_score_metadata"]["accepted"])
            for old, new in zip(frozen["rows"], data["rows"]):
                self.assertEqual(old["experiment_pixel_target_pass"], new["experiment_pixel_target_pass"])
                self.assertEqual(new["signed_normal_summary"]["known_count"], 5)
                for original, enriched in zip(old["raw_samples"], new["raw_samples"]):
                    self.assertEqual(original, {key: enriched[key] for key in original})
            with self.assertRaises(FileExistsError):
                diagnose(*paths, root / "diagnostic.json")

    def test_reflection_guard_preserves_missing_scores_and_null_direction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root, reflected=True)
            diagnose(root / "atlas.json", root / "frames.json", root / "score.json", root / "diagnostic.json")
            data = json.loads((root / "diagnostic.json").read_text())
            for group in data["rows"]:
                self.assertEqual(group["signed_normal_summary"]["unknown_count"], 5)
                for sample in group["raw_samples"]:
                    self.assertIsNone(sample["error_px"])
                    self.assertIsNone(sample["residual_diagnostic"]["signed_normal_px"])

    def test_hash_pts_time_base_and_unsigned_score_tampering_fail_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            baseline = json.loads((root / "score.json").read_text())
            mutations = [("atlas_sha256", "0" * 64), ("frames_sha256", "0" * 64),
                         ("source_sha256", "0" * 64), ("source_pts", 18000),
                         ("source_pts", 9000.),
                         ("source_time_base", "2/180000"), ("error_px", 100.),
                         ("projection_available", False)]
            for key, value in mutations:
                changed = deepcopy(baseline)
                if key in ("source_pts", "source_time_base"):
                    changed["rows"][0][key] = value
                elif key in ("error_px", "projection_available"):
                    changed["rows"][0]["raw_samples"][0][key] = value
                else:
                    changed[key] = value
                (root / "tampered.json").write_text(json.dumps(changed))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    diagnose(root / "atlas.json", root / "frames.json", root / "tampered.json", root / "bad.json")
                self.assertFalse((root / "bad.json").exists())


if __name__ == "__main__":
    unittest.main()
