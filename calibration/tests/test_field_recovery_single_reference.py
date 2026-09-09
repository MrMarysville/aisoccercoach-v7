"""Synthetic acceptance for explicit single-chart diagnostics; no footage access."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import field_recovery as r


H = np.array([[12., 1., 960.], [.2, 5., 400.], [.0005, .006, 1.]])
SUPPORT = [[20., -15.], [50., -15.], [50., 15.], [20., 15.]]


def payload():
    return dict(status="approximate", metric_certified=False,
                projection_mode=r.SINGLE_REFERENCE_PROJECTION_MODE,
                source=dict(game_id="synthetic", sha256="a"*64), field=r.FIELD,
                charts=[dict(chart="right", reference_frame_index=0,
                             field_to_reference=H.tolist(), support_world_polygon=SUPPORT)],
                blend=r.BLEND, spatial_boundary=None, protected_y_m=-r.BOX18_HALF_W,
                frames=[dict(index=0, source_pts=9000, source_time_base="1/90000", native_size=[1920, 1080],
                             reference_to_native=np.eye(3).tolist(), boundary_offset_native_y_px=0., warnings=[])])


def mapping(data=None):
    atlas = r.RecoveryAtlas.from_payload(payload() if data is None else data)
    return atlas.frame(9000, "1/90000", source_sha256="a"*64, native_size=[1920, 1080])


class SingleReferenceTests(unittest.TestCase):
    def test_projection_and_numerical_inverse_match_known_homography(self):
        data = payload()
        motion = np.array([[1.03, .02, 11.], [-.01, 1.01, -4.], [.00001, -.00002, 1.]])
        data["frames"][0]["reference_to_native"] = motion.tolist()
        frame = mapping(data)
        x, y = np.meshgrid(np.linspace(-r.HALF_L, r.HALF_L, 31), np.linspace(-r.HALF_W, r.HALF_W, 19))
        world = np.c_[x.ravel(), y.ravel()]
        expected = r.project(motion @ H, world)
        actual = frame.project(world)
        self.assertLess(float(np.max(np.linalg.norm(actual-expected, axis=1))), 1e-7)
        inverse_world = r.project(np.linalg.inv(motion @ H), actual)
        self.assertLess(float(np.max(np.linalg.norm(inverse_world-world, axis=1))), 1e-7)
        self.assertLess(float(np.max(np.linalg.norm(frame.project(inverse_world)-expected, axis=1))), 1e-7)
        self.assertEqual(len(frame.atlas.charts), 1)
        self.assertEqual(frame.atlas.charts[0]["chart"], "right")

    def test_public_lookup_withholds_outside_measured_hull_and_warned_frames(self):
        frame = mapping()
        world = np.array([[-30., 0.], [0., 0.], [30., 0.], [60., 0.]])
        self.assertTrue(np.isfinite(frame.project(world)).all())
        public = frame.public_projection(world)
        self.assertEqual(public["observed_support"], [False, False, True, False])
        self.assertIsNone(public["native_pixels"][0])
        self.assertIsNone(public["native_pixels"][1])
        self.assertIsNotNone(public["native_pixels"][2])
        self.assertIsNone(public["native_pixels"][3])
        data = payload()
        data["frames"][0]["warnings"] = ["fixture_motion_failure"]
        self.assertIsNone(mapping(data).public_projection([[30., 0.]])["native_pixels"])

    def test_mode_requires_exactly_one_real_chart(self):
        for count in (0, 2, 3):
            data = payload()
            data["charts"] = [dict(deepcopy(data["charts"][0]), chart=name)
                              for name in ("left", "mid", "right")[:count]]
            with self.subTest(count=count), self.assertRaises(ValueError):
                r.RecoveryAtlas.from_payload(data)

    def test_existing_absent_chart_modes_keep_their_previous_domains(self):
        world = [[-30., 0.], [0., 0.], [30., 0.], [45., 0.]]
        legacy = payload()
        del legacy["projection_mode"]
        fixed = payload()
        fixed["projection_mode"] = r.V7_PROJECTION_MODE
        self.assertEqual(np.isfinite(mapping(legacy).project(world)).all(axis=1).tolist(), [False, False, True, True])
        self.assertEqual(np.isfinite(mapping(fixed).project(world)).all(axis=1).tolist(), [False, False, False, True])
        self.assertTrue(np.isfinite(mapping().project(world)).all())

    def test_geometry_checks_full_visible_domain_even_outside_observed_hull(self):
        x, y = np.meshgrid(np.linspace(-40, 40, 11), np.linspace(-20, 20, 9))
        grid = np.c_[x.ravel(), y.ravel()]
        good = r.visible_geometry_check(mapping(), grid)
        self.assertTrue(good["valid"])
        self.assertGreater(good["visible_grid_points_outside_observed_support"], 0)
        self.assertEqual(good["interior_nonfinite_jacobians"], 0)
        bad = payload()
        bad["charts"][0]["field_to_reference"] = (H @ np.diag([-1., 1., 1.])).tolist()
        rejected = r.visible_geometry_check(mapping(bad), grid)
        self.assertFalse(rejected["valid"])
        self.assertEqual(rejected["nonpositive_jacobians"], rejected["visible_grid_points"])

    def test_saved_mode_keeps_projection_and_real_chart_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame = mapping()
            path = Path(tmp)/"atlas.json"
            frame.atlas.save(path)
            loaded = r.RecoveryAtlas.load(path)
            saved = loaded.frame(9000, "1/90000", source_sha256="a"*64, native_size=[1920, 1080])
            np.testing.assert_array_equal(frame.project([[-30., 0.], [30., 0.]]), saved.project([[-30., 0.], [30., 0.]]))
            self.assertEqual(loaded.projection_mode, r.SINGLE_REFERENCE_PROJECTION_MODE)
            self.assertEqual([c["chart"] for c in loaded.charts], ["right"])

    def test_actual_fitter_uses_sole_right_reference_as_gauge_and_mode(self):
        spec = importlib.util.spec_from_file_location("single_reference_runner", Path(__file__).resolve().parents[1]/"tools"/"fit_field_recovery.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        synthetic_h = np.array([[2., 0., 192.], [0., 2., 108.], [0., 0., 1.]])
        x, y = np.meshgrid([20., 35., 50.], [-10., 0., 10.])
        world = np.c_[x.ravel(), y.ravel()]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for i in range(3):
                path = root/f"{i}.png"
                cv2.imwrite(str(path), np.full((216, 384, 3), [35, 90, 40], np.uint8))
                rows.append(dict(index=i, file=path.name, source_pts=9000+i*9000,
                                 source_time_base="1/90000", source_sha256="a"*64, native_size=[384, 216],
                                 image_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            manifest = dict(schema="alignment-trial-frames-v1", source=dict(game_id="synthetic", sha256="a"*64), frames=rows)
            measurements = dict(schema="field-recovery-measurements-v1", source_sha256="a"*64,
                                fit_frame_indices=[1], check_frame_indices=[0, 2],
                                references=[dict(frame_index=1, chart="right", support_world_polygon=SUPPORT,
                                                 features=[dict(label="synthetic named landmarks", world_points=world.tolist(),
                                                                points_native=r.project(synthetic_h, world).tolist())])])
            (root/"frames.json").write_text(json.dumps(manifest))
            (root/"measurements.json").write_text(json.dumps(measurements))
            with patch.object(r, "reference_features", return_value=(np.zeros((0, 2)), None)), \
                    patch.object(r, "register_reference", return_value=(np.eye(3), dict(image_registration_pass=True))), \
                    patch.object(r, "adjacent_motion", return_value=(np.eye(3), dict(image_registration_pass=True))):
                result = runner.run(root/"frames.json", root/"measurements.json", root/"candidate", render=False, reference_stride=1)
            loaded = r.RecoveryAtlas.load(root/"candidate"/"atlas.json")
            self.assertEqual(loaded.projection_mode, r.SINGLE_REFERENCE_PROJECTION_MODE)
            self.assertEqual([c["chart"] for c in loaded.charts], ["right"])
            self.assertEqual(loaded.charts[0]["reference_frame_index"], 1)
            self.assertEqual([x["frame_index"] for x in result["reference_connections"] if x["kind"] == "gauge"], [1])
            frame = loaded.frame(18000, "1/90000", source_sha256="a"*64, native_size=[384, 216])
            self.assertLess(float(np.max(np.linalg.norm(frame.project(world)-r.project(synthetic_h, world), axis=1))), 1e-7)


if __name__ == "__main__":
    unittest.main()
