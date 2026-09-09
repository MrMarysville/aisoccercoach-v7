"""Synthetic near-boundary correction evidence; no match footage is read."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import field_recovery as r


class NearBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1]/"tools"/"refine_near_field_recovery.py"
        spec = importlib.util.spec_from_file_location("near_refiner", path)
        cls.helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.helper)

    def fixture(self, root):
        h = np.array([[10., .2, 960.], [.1, 5., 400.], [0., .003, 1.]])
        rows, saved = [], []
        for i in range(5):
            path = root/f"{i}.png"
            cv2.imwrite(str(path), np.full((1080, 1920, 3), [35, 90, 40], np.uint8))
            row = dict(index=i, file=path.name, source_pts=9000+i*9000, source_time_base="1/90000",
                       source_sha256="a"*64, native_size=[1920, 1080], image_sha256=self.helper.sha(path))
            rows.append(row)
            saved.append(dict(index=i, source_pts=row["source_pts"], source_time_base=row["source_time_base"],
                              native_size=row["native_size"], image_sha256=row["image_sha256"],
                              reference_to_native=np.eye(3).tolist(), boundary_offset_native_y_px=2.,
                              warnings=["fixture_parent_warning"] if i == 2 else []))
        inputs = root/"parent-inputs.json"
        inputs.write_text(json.dumps(dict(references=[])))
        payload = dict(status="approximate", metric_certified=False, projection_mode=r.SINGLE_REFERENCE_PROJECTION_MODE,
                       source=dict(game_id="synthetic", sha256="a"*64), field=r.FIELD, blend=r.BLEND,
                       spatial_boundary=None, temporal_boundary=None, protected_y_m=-r.BOX18_HALF_W,
                       charts=[dict(chart="right", field_to_reference=h.tolist(),
                                    support_world_polygon=[[20., -15.], [50., -15.], [50., 15.], [20., 15.]])],
                       frames=saved, provenance=dict(measurements_sha256=self.helper.sha(inputs)))
        parent = r.RecoveryAtlas.from_payload(payload)
        parent.save(root/"parent.json")
        manifest = dict(schema="alignment-trial-frames-v1", source=dict(game_id="synthetic", sha256="a"*64), frames=rows)
        (root/"frames.json").write_text(json.dumps(manifest))
        observations = []
        for i, delta in [(1, -1.), (3, -2.)]:
            world = np.c_[np.linspace(-35., 35., 25), np.full(25, r.HALF_W+delta)]
            native = r.project(h, world)
            observations.append(dict(frame_index=i, source_pts=rows[i]["source_pts"], source_time_base=rows[i]["source_time_base"],
                                     image_sha256=rows[i]["image_sha256"], label="touch_near", points_native=native.tolist()))
        measurements = dict(schema="field-recovery-near-boundary-measurements-v1", source_sha256="a"*64,
                            parent_atlas_sha256=self.helper.sha(root/"parent.json"),
                            parent_measurements_path=str(inputs), parent_measurements_sha256=self.helper.sha(inputs),
                            fit_frame_indices=[1, 3], check_frame_indices=[0, 2, 4], near_boundary_observations=observations)
        (root/"near.json").write_text(json.dumps(measurements))
        return parent, manifest, measurements

    def test_refiner_preserves_parent_and_protects_interior_with_full_frame_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent, manifest, _ = self.fixture(root)
            report = self.helper.refine(root/"parent.json", root/"frames.json", root/"near.json", root/"new", render=True)
            new = r.RecoveryAtlas.load(root/"new"/"atlas.json")
            self.assertEqual(new.payload["charts"], parent.payload["charts"])
            for key in ("spatial_boundary", "temporal_boundary", "blend", "field", "source", "projection_mode"):
                self.assertEqual(new.payload[key], parent.payload[key])
            self.assertEqual(len(report["geometry"]), 5)
            self.assertEqual(report["rendered_frame_indices"], list(range(5)))
            self.assertEqual(len(list((root/"new"/"overlay").glob("*.jpg"))), 5)
            self.assertTrue(all(x["valid"] for x in report["geometry"]))
            self.assertEqual(report["source_fit_parameter_frames"], 2)
            self.assertEqual(report["interpolated_frames"], 1)
            self.assertEqual(report["outside_observed_span_frames"], 2)
            self.assertAlmostEqual(report["source_fit_observations"][0]["fitted_delta_field_y_m"], -1., places=7)
            self.assertAlmostEqual(report["source_fit_observations"][1]["fitted_delta_field_y_m"], -2., places=7)
            self.assertEqual(len(report["source_fit_observations"][0]["raw_native_y_residuals_px"]), 25)
            self.assertLess(report["source_fit_observations"][0]["residual_p95_abs_px"], 1e-7)
            x, y = np.meshgrid(np.linspace(-50., 50., 15), [-r.HALF_W, -20., 0., r.BOX18_HALF_W])
            protected = np.c_[x.ravel(), y.ravel()]
            for old_row, new_row in zip(parent.payload["frames"], new.payload["frames"]):
                self.assertEqual(new_row["reference_to_native"], old_row["reference_to_native"])
                self.assertEqual(new_row["boundary_offset_native_y_px"], old_row["boundary_offset_native_y_px"])
                self.assertEqual(new_row["warnings"][:len(old_row["warnings"])], old_row["warnings"])
                bind = dict(source_sha256="a"*64, native_size=[1920, 1080])
                a = parent.frame(old_row["source_pts"], old_row["source_time_base"], **bind)
                b = new.frame(new_row["source_pts"], new_row["source_time_base"], **bind)
                np.testing.assert_array_equal(a.project(protected), b.project(protected))
            self.assertIn("visible_near_strip_outside_observed_correction_span", new.payload["frames"][0]["warnings"])
            self.assertEqual(new.payload["frames"][0]["near_boundary_delta_field_y_m"], 0.)
            corrected = new.frame(27000, "1/90000", source_sha256="a"*64, native_size=[1920, 1080])
            point = np.array([25., 28.])
            native = corrected.project([point])[0]
            inverse = least_squares(lambda p:corrected.project([p])[0]-native, [24., 27.], xtol=1e-12, ftol=1e-12, gtol=1e-12)
            self.assertLess(float(np.linalg.norm(inverse.x-point)), 1e-7)
            self.assertLess(float(np.linalg.norm(corrected.project([inverse.x])[0]-native)), 1e-7)

    def test_check_frame_and_mismatched_source_observations_fail_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, measurements = self.fixture(root)
            for label, change in [("check", lambda m:m["near_boundary_observations"][0].update(frame_index=2)),
                                  ("source", lambda m:m.update(source_sha256="b"*64)),
                                  ("still", lambda m:m["near_boundary_observations"][0].update(image_sha256="c"*64))]:
                wrong = deepcopy(measurements)
                change(wrong)
                path = root/f"{label}.json"
                path.write_text(json.dumps(wrong))
                with self.subTest(label=label), self.assertRaises(ValueError):
                    self.helper.refine(root/"parent.json", root/"frames.json", path, root/label, render=False)
                self.assertFalse((root/label).exists())

    def test_no_extrapolation_or_invented_zero_anchor(self):
        model = dict(domain_s=[1., 4.], time_s=[1., 4.], delta_field_y_m=[5., 11.])
        self.assertEqual(r.near_temporal_offset(model, 2.), 7.)
        for t in (0., 5.):
            with self.assertRaises(ValueError):
                r.near_temporal_offset(model, t)

    def test_reflected_near_curve_is_rejected_and_observation_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent, manifest, measurements = self.fixture(root)
            data = deepcopy(parent.payload)
            h = np.asarray(data["charts"][0]["field_to_reference"])
            data["charts"][0]["field_to_reference"] = (h @ np.diag([-1., 1., 1.])).tolist()
            reflected = r.RecoveryAtlas.from_payload(data)
            obs = measurements["near_boundary_observations"]
            fitted, rejected = self.helper.fit_native_curves(reflected, obs, {r["index"]:r for r in manifest["frames"]}, {1, 3})
            self.assertFalse(fitted)
            self.assertEqual(len(rejected), 2)
            self.assertEqual(rejected[0]["points_native"], obs[0]["points_native"])

    def test_visible_source_paint_recovers_parent_near_line_twenty_pixels_below_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent, manifest, _ = self.fixture(root)
            data = deepcopy(parent.payload)
            h = np.array([[10., 0., 960.], [0., 5., 1100.-5.*r.HALF_W], [0., 0., 1.]])
            data["charts"][0]["field_to_reference"] = h.tolist()
            below_image = r.RecoveryAtlas.from_payload(data)
            source = np.c_[np.linspace(700., 1200., 25), np.full(25, 1070.)]
            observations = [dict(frame_index=1, points_native=source.tolist())]
            fitted, rejected = self.helper.fit_native_curves(below_image, observations,
                {row["index"]:row for row in manifest["frames"]}, {1})
            self.assertFalse(rejected)
            self.assertEqual(len(fitted), 1)
            self.assertAlmostEqual(fitted[0]["raw_median_native_y_residual_px"], -30., places=8)
            self.assertAlmostEqual(fitted[0]["fitted_delta_field_y_m"], -6., places=8)
            self.assertEqual(fitted[0]["predicted_offscreen_y_samples"], 25)
            self.assertTrue(fitted[0]["fit_includes_offscreen_parent_predictions"])
            np.testing.assert_allclose(fitted[0]["predicted_native_y_px"], 1100., atol=1e-9)
            np.testing.assert_allclose(fitted[0]["raw_native_y_residuals_px"], -30., atol=1e-9)
            data["near_temporal_boundary"] = dict(method=r.NEAR_TEMPORAL_METHOD, weight_profile=r.LINEAR_SPATIAL_WEIGHT,
                protected_y_m=r.BOX18_HALF_W, time_s=[.2], delta_field_y_m=[-6.], domain_s=[.2, .2])
            data["frames"][1]["near_boundary_delta_field_y_m"] = -6.
            corrected = r.RecoveryAtlas.from_payload(data).frame(18000, "1/90000", source_sha256="a"*64, native_size=[1920, 1080])
            x, y = np.meshgrid(np.linspace(-50., 50., 19), np.linspace(-r.HALF_W, r.HALF_W, 13))
            self.assertTrue(r.visible_geometry_check(corrected, np.c_[x.ravel(), y.ravel()])["valid"])
            np.testing.assert_allclose(corrected.project([[0., r.HALF_W]])[0, 1], 1070., atol=1e-9)

    def test_halfway_and_goal_line_sets_stay_exact_under_field_y_remap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent, _, _ = self.fixture(root)
            data = deepcopy(parent.payload)
            # No far correction in this independent homography line-set check.
            for f in data["frames"]:
                f["boundary_offset_native_y_px"] = 0.
            original = r.RecoveryAtlas.from_payload(data).frame(18000, "1/90000", source_sha256="a"*64, native_size=[1920, 1080])
            data["near_temporal_boundary"] = dict(method=r.NEAR_TEMPORAL_METHOD, weight_profile=r.LINEAR_SPATIAL_WEIGHT,
                protected_y_m=r.BOX18_HALF_W, time_s=[.2], delta_field_y_m=[-3.], domain_s=[.2, .2])
            data["frames"][1]["near_boundary_delta_field_y_m"] = -3.
            changed = r.RecoveryAtlas.from_payload(data).frame(18000, "1/90000", source_sha256="a"*64, native_size=[1920, 1080])
            for x in (-r.HALF_L, 0., r.HALF_L):
                world = np.c_[np.full(401, x), np.linspace(-r.HALF_W, r.HALF_W, 401)]
                original_segment = original.project(world)
                distance = r.curve_distance(changed.project(world), original_segment)
                self.assertLess(float(np.max(distance)), 1e-7)
                vertical = original.project(world)
                vertical[:, 1] -= 30.*changed.atlas.near_edge_weight(world)
                self.assertGreater(float(np.max(r.curve_distance(vertical, original_segment))), 1.)
            protected = np.concatenate([r.feature_curve(dict(label=label, points_native=[[0, 0]]), 101)
                                        for label in ("centre_circle", "box18_left_front", "box18_right_front")])
            np.testing.assert_array_equal(changed.project(protected), original.project(protected))


if __name__ == "__main__":
    unittest.main()
