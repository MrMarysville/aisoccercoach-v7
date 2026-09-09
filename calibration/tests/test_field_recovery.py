"""Synthetic evidence for reusable v7 geometry, not real-match acceptance."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import hashlib
import json
import importlib.util

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import field_recovery as r


H = np.array([[12., 1., 960.], [.2, 5., 400.], [.0005, .006, 1.]])
FULL = [[-r.HALF_L, -r.HALF_W], [r.HALF_L, -r.HALF_W],
        [r.HALF_L, r.HALF_W], [-r.HALF_L, r.HALF_W]]


def samples(h=H):
    features = []
    for label in ("touch_far", "touch_near", "halfway", "goal_left", "goal_right"):
        curve = r.feature_curve(dict(label=label, points_native=[[0, 0]]), 31)
        features.append(dict(label=label, points_native=r.project(h, curve).tolist()))
    return features


def atlas_payload(h=H):
    return dict(status="approximate", metric_certified=False,
                source=dict(game_id="synthetic", sha256="a"*64), field=r.FIELD,
                charts=[dict(chart=name, field_to_reference=h.tolist(), support_world_polygon=FULL)
                        for name in ("left", "mid", "right")],
                blend=r.BLEND, spatial_boundary=None, protected_y_m=-r.BOX18_HALF_W,
                frames=[dict(index=0, source_pts=9000, source_time_base="1/90000", native_size=[1920, 1080],
                             reference_to_native=np.eye(3).tolist(), boundary_offset_native_y_px=0., warnings=[])])


class RecoveryGeometryTests(unittest.TestCase):
    def test_resize_rotation_zoom_native_centres(self):
        theta = .12
        native = np.array([[1.08*np.cos(theta), -1.08*np.sin(theta), 23.],
                           [1.08*np.sin(theta), 1.08*np.cos(theta), -9.], [0, 0, 1.]])
        sa = r.resize_transform([1920, 1080], [1280, 720])
        sb = r.resize_transform([2048, 1152], [1000, 563])
        resized = sb @ native @ np.linalg.inv(sa)
        actual = r.native_motion(resized, [1920, 1080], [1280, 720], [2048, 1152], [1000, 563])
        p = np.array([[0., 0.], [1919, 1079], [650.2, 335.8]])
        self.assertLess(np.max(np.abs(r.project(actual, p)-r.project(native, p))), 1e-8)
        self.assertAlmostEqual(r.project(sa, [[0, 0]])[0, 0], -1/6)

    def test_ground_mask_channel_arithmetic_does_not_overflow(self):
        im = np.zeros((20, 20, 3), np.uint8)
        im[:] = [254, 255, 254]
        self.assertEqual(int(r.ground_mask(im, top_fraction=0).sum()), 0)

    def test_coherent_lines_recover_subpixel_geometry(self):
        fit = r.fit_reference(samples(), [1920, 1080], FULL)
        p = np.array([[-25., -20.], [0, 0], [25, 20]])
        self.assertLess(np.max(np.linalg.norm(r.project(fit["field_to_native"], p)-r.project(H, p), axis=1)), .1)
        self.assertTrue(fit["geometry"]["valid"])
        self.assertTrue(all(f["p95_px"] < .1 for f in fit["features"]))

    def test_fresh_later_reference_really_refits_new_paint(self):
        first = r.fit_reference(samples(), [1920, 1080], FULL)
        shift = np.array([[1., 0, 9.], [0, 1, -2.], [0, 0, 1.]])
        later = r.fit_reference(samples(shift @ H), [1920, 1080], FULL)
        old = r.project(first["field_to_native"], [[0, 0]])
        new = r.project(later["field_to_native"], [[0, 0]])
        self.assertGreater(float(np.linalg.norm(new-old)), 5.)
        self.assertLess(float(np.linalg.norm(new-r.project(shift @ H, [[0, 0]]))), .1)

    def test_failed_middle_motion_remains_warning_until_actual_paint(self):
        step = np.array([[1., 0, 3.], [0, 1, 0], [0, 0, 1.]])
        transforms, warnings = r.accumulate_motion([None, step, None, step, step, step], 0, [4])
        self.assertFalse(warnings[1])
        self.assertTrue(warnings[2])
        self.assertEqual(warnings[2], warnings[3])
        self.assertFalse(warnings[4])
        self.assertFalse(warnings[5])
        self.assertEqual(len(transforms), 6)

    def test_reject_two_visible_projective_branches(self):
        # u=960+100/x, v=540+2*y/x: both denominator signs show on screen.
        pole = np.array([[960., 0, 100.], [540., 2., 0], [1., 0, 0.]])
        check = r.validate_homography(pole, [1920, 1080], FULL)
        self.assertFalse(check["valid"])
        self.assertIn("two_visible_projective_branches", check["reasons"])

    def test_offsupport_pole_not_automatic_full_field_rejection(self):
        # Restrict to a visible positively oriented branch, beyond the pole.
        pole = np.array([[960., 0, -100.], [540., 2., 0], [1., 0, 0.]])
        check = r.validate_homography(pole, [1920, 1080], [[5, -10], [30, -10], [30, 10], [5, 10]])
        self.assertTrue(check["valid"], check)

    def test_euclidean_curve_distance_is_not_vertical_residual(self):
        distance = r.curve_distance([[5, 7]], np.array([[0, 0], [10, 10]]))
        self.assertAlmostEqual(distance[0], np.sqrt(2), places=10)

    def test_motion_checks_are_disjoint_ground_cells(self):
        x, y = np.meshgrid(np.arange(30, 1900, 24), np.arange(200, 1000, 24))
        p = np.c_[x.ravel(), y.ravel()]
        check = r.ground_cell_partition(p)
        fit_cells = set(map(tuple, np.floor(p[~check]/96).astype(int)))
        check_cells = set(map(tuple, np.floor(p[check]/96).astype(int)))
        self.assertFalse(fit_cells & check_cells)
        q = p+[4., 2.]
        q[check] += [14., 0.]  # Good fit tracks must not hide bad independent tracks.
        h, stats = r.fit_motion_tracks(p, q, [1920, 1080])
        self.assertIsNone(h)
        self.assertGreater(stats["check_median_px"], 10)

    def test_save_load_and_render_share_identical_projection(self):
        atlas = r.RecoveryAtlas.from_payload(atlas_payload())
        p = np.array([[-22., -25.], [0, 0], [30, 20.]])
        bind = dict(source_sha256="a"*64, native_size=[1920, 1080])
        before = atlas.frame(9000, "1/90000", **bind)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"atlas.json"
            atlas.save(path)
            saved = r.RecoveryAtlas.load(path).frame(1, "1/10", **bind)
            np.testing.assert_array_equal(before.project(p), saved.project(p))
            image = np.zeros((1080, 1920, 3), np.uint8)
            np.testing.assert_array_equal(r.render_frame(image, before), r.render_frame(image, saved))
        with self.assertRaises(ValueError):
            atlas.frame(9001, "1/90000", **bind)
        with self.assertRaises(ValueError):
            atlas.frame(9000, "1/90000", source_sha256="b"*64, native_size=[1920, 1080])

    def test_boundary_corrections_protect_entire_interior_exactly(self):
        base = atlas_payload()
        other = deepcopy(base)
        other["spatial_boundary"] = dict(x_origin_px=960., x_scale_px=1000., coefficients=[250., 5., 1.])
        other["frames"][0]["boundary_offset_native_y_px"] = 17.
        p = np.array([[0., -r.BOX18_HALF_W], [20, -10], [-30, 20]])
        a, b = [r.RecoveryAtlas.from_payload(v).frame(9000, "1/90000", source_sha256="a"*64, native_size=[1920, 1080]) for v in (base, other)]
        np.testing.assert_array_equal(a.project(p), b.project(p))

    def test_check_frame_boundary_samples_do_not_change_fit(self):
        obs = [dict(frame_index=i, time_s=float(i), samples=100, median_offset_px=float(i), mad_px=1.) for i in range(6)]
        model = r.fit_temporal_boundary(obs, [0, 1, 2, 3, 4])
        obs[-1]["median_offset_px"] = 10000.
        other = r.fit_temporal_boundary(obs, [0, 1, 2, 3, 4])
        self.assertEqual(model, other)
        with self.assertRaises(ValueError):
            r.temporal_offset(model, 5.)

    def test_warned_frame_withholds_public_projection(self):
        payload = atlas_payload()
        payload["frames"][0]["warnings"] = ["failed_adjacent_motion_step_2"]
        atlas = r.RecoveryAtlas.from_payload(payload)
        frame = atlas.frame(9000, "1/90000", source_sha256="a"*64, native_size=[1920, 1080])
        self.assertIsNone(frame.public_projection([[0, 0]])["native_pixels"])
        self.assertTrue(np.isfinite(frame.project([[0, 0]])).all())

    def test_absent_end_chart_never_hallucinates_unmeasured_region(self):
        payload = atlas_payload()
        payload["charts"] = [payload["charts"][1]]
        payload["charts"][0]["support_world_polygon"] = [[-15, -15], [15, -15], [15, 15], [-15, 15]]
        atlas = r.RecoveryAtlas.from_payload(payload)
        self.assertTrue(np.isnan(atlas.base([[50, 0]])).all())
        self.assertTrue(np.isfinite(atlas.base([[0, 0]])).all())


    def test_v7_order_matches_portable_evaluator_with_all_charts(self):
        import field_atlas as portable
        payload = atlas_payload()
        payload["projection_mode"] = r.V7_PROJECTION_MODE
        for c, offset in zip(payload["charts"], (-7., 0., 11.)):
            c["field_to_reference"][0][2] += offset
        payload["spatial_boundary"] = dict(x_origin_px=960., x_scale_px=1000., coefficients=[275., 9., -2.])
        payload["frames"][0]["reference_to_native"] = [[1.02, .03, 10.], [.01, .97, 20.], [.0001, -.00007, 1.]]
        payload["frames"][0]["boundary_offset_native_y_px"] = -3.5
        old = dict(status="approximate", metric_certified=False, source=payload["source"], field=payload["field"],
                   spatial=dict(field_to_reference_models=[c["field_to_reference"] for c in payload["charts"]],
                                blend=payload["blend"], far_boundary=dict(reference_y_coefficients=[275., 9., -2.],
                                reference_x_origin_px=960., reference_x_scale_px=1000., protected_y_m=-r.BOX18_HALF_W)),
                   frames=payload["frames"])
        reference = portable.FieldAtlas(dict(schema=portable.SCHEMA, payload=old, payload_sha256=portable.digest(old)))
        actual = r.RecoveryAtlas.from_payload(payload)
        bind = dict(source_sha256="a"*64, native_size=[1920, 1080])
        p = np.random.default_rng(123).uniform([-r.HALF_L, -r.HALF_W], [r.HALF_L, r.HALF_W], (1000, 2))
        expected = reference.frame(9000, "1/90000", **bind).project(p)
        observed = actual.frame(9000, "1/90000", **bind).project(p)
        self.assertLess(np.max(np.abs(expected-observed)), 1e-8)

    def test_fresh_circle_and_two_lines_initialize_without_an_old_map(self):
        fs = []
        for label in ("centre_circle", "halfway", "touch_far"):
            p = r.feature_curve(dict(label=label, points_native=[[0, 0]]), 81)
            fs.append(dict(label=label, points_native=r.project(H, p).tolist()))
        fit = r.fit_reference(fs, [1920, 1080])
        self.assertEqual(fit["seed_method"], "fresh measured ellipse multistart")
        self.assertTrue(all(f["p95_px"] < .1 for f in fit["features"]))

    def test_connected_paint_reacquisition_resets_transform_and_warning(self):
        reset = np.array([[1., 0, 27.], [0, 1, -3.], [0, 0, 1.]])
        transforms, warnings = r.accumulate_motion([None, None, np.eye(3), np.eye(3)], 0, [2], {2:reset})
        self.assertTrue(warnings[1])
        self.assertFalse(warnings[2])
        np.testing.assert_array_equal(transforms[2], reset)
        np.testing.assert_array_equal(transforms[3], reset)


    def test_penalty_arc_labels_fit_correct_centres_and_score_only_painted_arc(self):
        fs = samples()
        for side, sign in (("left", -1), ("right", 1)):
            label = f"pen_arc_{side}"
            centre = np.array([sign*(r.HALF_L-10.9728), 0.])
            feature = dict(label=label, points_native=[[0., 0.]])
            resolved = r.resolve_feature(feature)
            np.testing.assert_allclose(resolved["circle_center_m"], centre, atol=1e-12)
            self.assertEqual(resolved["circle_radius_m"], 9.144)
            world = r.feature_curve(feature, 121)
            np.testing.assert_allclose(np.linalg.norm(world-centre, axis=1), 9.144, atol=1e-12)
            self.assertTrue((sign*world[:, 0] <= r.HALF_L-r.BOX18_DEPTH+1e-12).all())
            np.testing.assert_allclose(world[[0, -1], 0], sign*(r.HALF_L-r.BOX18_DEPTH), atol=1e-12)
            np.testing.assert_allclose(world[[0, -1], 1], [-7.3152, 7.3152], atol=1e-12)
            np.testing.assert_array_equal(r.world_markings(121)[label], world)
            observed = dict(label=label, points_native=r.project(H, world).tolist())
            self.assertLess(float(np.max(r.feature_errors(lambda p:r.project(H, p), observed))), .01)
            fs.append(observed)
            # The unpainted back of the same supporting circle must not score zero.
            unpainted = centre+[sign*9.144, 0.]
            wrong = dict(label=label, points_native=r.project(H, [unpainted]).tolist())
            self.assertGreater(float(r.feature_errors(lambda p:r.project(H, p), wrong)[0]), 20.)
        fit = r.fit_reference(fs, [1920, 1080], FULL)
        self.assertTrue(all(f["p95_px"] < .1 for f in fit["features"]))
        with self.assertRaises(ValueError):
            r.resolve_feature(dict(label="pen_arc_left", points_native=[[0, 0]], circle_center_m=[0, 0]))

    def test_visible_geometry_counts_support_boundary_stencil_separately(self):
        payload = atlas_payload()
        for c in payload["charts"]:
            c["support_world_polygon"] = [[-5, -5], [5, -5], [5, 5], [-5, 5]]
        mapping = r.RecoveryAtlas.from_payload(payload).frame(9000, "1/90000", source_sha256="a"*64, native_size=[1920, 1080])
        check = r.visible_geometry_check(mapping, [[5., 0.], [0., 0.]])
        self.assertEqual(check["nonfinite_jacobians"], 1)
        self.assertEqual(check["support_boundary_stencil_crossings"], 1)
        self.assertEqual(check["support_boundary_only_nonfinite_jacobians"], 1)
        self.assertEqual(check["interior_nonfinite_jacobians"], 0)
        self.assertTrue(check["valid"])

    def test_visible_geometry_flags_interior_nonfinite_derivative(self):
        mapping = r.RecoveryAtlas.from_payload(atlas_payload()).frame(9000, "1/90000", source_sha256="a"*64, native_size=[1920, 1080])
        original = mapping.project
        def injected_projection(p):
            p = np.asarray(p)
            q = original(p)
            q[np.isclose(p[:, 0], .001, atol=1e-12) & np.isclose(p[:, 1], 0., atol=1e-12)] = np.nan
            return q
        with patch.object(mapping, "project", side_effect=injected_projection):
            check = r.visible_geometry_check(mapping, [[0., 0.], [1., 1.]])
        self.assertEqual(check["nonfinite_jacobians"], 1)
        self.assertEqual(check["interior_nonfinite_jacobians"], 1)
        self.assertEqual(check["support_boundary_only_nonfinite_jacobians"], 0)
        self.assertFalse(check["valid"])


    def test_finite_marking_extent_penalizes_points_on_unpainted_extension(self):
        # The old infinite-line objective returned zero for this point.
        x = r.HALF_L-r.BOX18_DEPTH
        f = r.resolve_feature(dict(label="box18_right_front", points_native=[[x, r.BOX18_HALF_W+9.]]))
        infinite = r._fit_residual(np.eye(3), [f], finite_segments=False)
        finite = r._fit_residual(np.eye(3), [f])
        self.assertLess(float(np.linalg.norm(infinite)), 1e-12)
        self.assertAlmostEqual(float(np.linalg.norm(finite)), 9., places=10)
        projected = r.project(np.eye(3), f["world_segment"])
        self.assertAlmostEqual(r.curve_distance(f["points_native"], projected)[0], 9., places=10)

    def test_finite_extent_handles_a_projective_pole_without_filling_its_gap(self):
        h = np.array([[0., 0, 1.], [0, 1., 0], [1., 0, 0]])
        # x in [-1,1] projects to two rays u<=-1 or u>=1, never the central gap.
        extent = r.finite_segment_extent_residual(h, [[-1., 0], [1., 0]], [[0., 0], [2., 0], [-2., 0]])
        np.testing.assert_allclose(abs(extent), [1., 0., 0.], atol=1e-12)

    def test_inverse_parameterization_handles_world_origin_at_offscreen_pole(self):
        truth = np.array([[20., 0, -100.], [10., 3., 0], [.1, 0, 0]])
        x, y = np.meshgrid([20., 35., 50.], [-10., 0, 10.])
        world = np.c_[x.ravel(), y.ravel()]
        native = r.project(truth, world)
        feature = dict(label="synthetic visible corner grid", world_points=world.tolist(), points_native=native.tolist())
        support = [[20., -10.], [50., -10.], [50., 10.], [20., 10.]]
        fit = r.fit_reference([feature], [384, 216], support)
        error = np.linalg.norm(r.project(fit["field_to_native"], world)-native, axis=1)
        self.assertLess(float(error.max()), .1)
        self.assertTrue(fit["geometry"]["valid"])
        self.assertIn("inverse homography", fit["optimizer_parameterization"])

    def test_circle_seed_combines_reviewed_fragments_of_the_same_marking(self):
        theta = np.linspace(0, 2*np.pi, 120, endpoint=False)
        native = r.project(H, r.CIRCLE_R*np.c_[np.cos(theta), np.sin(theta)])
        fs = [dict(label="centre_circle", points_native=p.tolist()) for p in np.array_split(native, 4)]
        for label in ("halfway", "touch_far"):
            p = r.feature_curve(dict(label=label, points_native=[[0, 0]]), 31)
            fs.append(dict(label=label, points_native=r.project(H, p).tolist()))
        image, world = r._normalizers([1920, 1080])
        self.assertEqual(len(r._circle_seeds([r.resolve_feature(f) for f in fs], image, world)), 4)
        fit = r.fit_reference(fs, [1920, 1080])
        self.assertTrue(all(f["p95_px"] < .1 for f in fit["features"]))


    def test_fixed_blend_is_continuous_at_measured_chart_hull_boundary(self):
        payload = atlas_payload()
        payload["charts"][0]["field_to_reference"] = (np.array([[1., 0, 20.], [0, 1, 0], [0, 0, 1.]]) @ H).tolist()
        payload["charts"][0]["support_world_polygon"] = [[-r.HALF_L, -r.HALF_W], [-24., -r.HALF_W], [-24., r.HALF_W], [-r.HALF_L, r.HALF_W]]
        sample = np.array([[-24.00001, 0.], [-23.99999, 0.]])
        legacy = r.RecoveryAtlas.from_payload(payload)
        legacy_pixels = legacy.base(sample)
        self.assertGreater(float(np.linalg.norm(legacy_pixels[1]-legacy_pixels[0])), 9.)
        payload["projection_mode"] = r.V7_PROJECTION_MODE
        fixed = r.RecoveryAtlas.from_payload(payload)
        fixed_pixels = fixed.base(sample)
        expected = r.project(H, sample) + r.chart_weights(sample)[:, 0, None]*[20., 0.]
        np.testing.assert_allclose(fixed_pixels, expected, atol=1e-9)
        self.assertLess(float(np.linalg.norm(fixed_pixels[1]-fixed_pixels[0])), .001)
        # Public output still rejects the side requiring extrapolation of the left chart.
        frame = fixed.frame(9000, "1/90000", source_sha256="a"*64, native_size=[1920, 1080])
        public = frame.public_projection(sample)
        self.assertIsNotNone(public["native_pixels"][0])
        self.assertIsNone(public["native_pixels"][1])
        self.assertEqual(public["observed_support"], [True, False])
        self.assertFalse(public["metric_certified"])

    def test_fixed_diagnostic_keeps_touchline_without_expanding_observed_hull(self):
        payload = atlas_payload()
        payload["projection_mode"] = r.V7_PROJECTION_MODE
        payload["charts"][2]["support_world_polygon"] = [[30., -20.8], [r.HALF_L, -20.8], [r.HALF_L, r.HALF_W-.01], [30., r.HALF_W-.01]]
        hull = deepcopy(payload["charts"][2]["support_world_polygon"])
        atlas = r.RecoveryAtlas.from_payload(payload)
        frame = atlas.frame(9000, "1/90000", source_sha256="a"*64, native_size=[1920, 1080])
        sample = [[40., r.HALF_W]]
        self.assertTrue(np.isfinite(frame.project(sample)).all())
        self.assertTrue(np.isnan(frame.project(sample, supported_only=True)).all())
        self.assertEqual(frame.public_projection(sample)["status"], "unavailable")
        self.assertIsNone(frame.public_projection(sample)["native_pixels"][0])
        self.assertEqual(atlas.payload["charts"][2]["support_world_polygon"], hull)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"fixed.json"
            atlas.save(path)
            loaded = r.RecoveryAtlas.load(path).frame(9000, "1/90000", source_sha256="a"*64, native_size=[1920, 1080])
            np.testing.assert_array_equal(frame.project(sample), loaded.project(sample))
            image = np.zeros((1080, 1920, 3), np.uint8)
            np.testing.assert_array_equal(r.render_frame(image, frame), r.render_frame(image, loaded))

    def test_saved_legacy_map_without_mode_retains_clipped_projection(self):
        payload = atlas_payload()
        payload["charts"][0]["field_to_reference"] = (np.array([[1., 0, 20.], [0, 1, 0], [0, 0, 1.]]) @ H).tolist()
        payload["charts"][0]["support_world_polygon"] = [[-r.HALF_L, -r.HALF_W], [-24., -r.HALF_W], [-24., r.HALF_W], [-r.HALF_L, r.HALF_W]]
        sample = np.array([[-25., 0.], [-23., 0.]])
        expected = r.project(H, sample)
        expected[0, 0] += 20*r.chart_weights(sample)[0, 0]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"old.json"
            r.RecoveryAtlas.from_payload(payload).save(path)
            loaded = r.RecoveryAtlas.load(path)
            self.assertEqual(loaded.projection_mode, r.LEGACY_PROJECTION_MODE)
            np.testing.assert_allclose(loaded.base(sample), expected, atol=1e-9)
            self.assertNotIn("projection_mode", loaded.payload)

    def test_fixed_blend_does_not_renormalize_missing_chart_or_hide_motion_warning(self):
        payload = atlas_payload()
        payload["projection_mode"] = r.V7_PROJECTION_MODE
        payload["charts"] = [payload["charts"][1]]
        atlas = r.RecoveryAtlas.from_payload(payload)
        self.assertTrue(np.isfinite(atlas.base([[0., 0.]])).all())
        self.assertTrue(np.isnan(atlas.base([[25., 0.]])).all())
        payload["frames"][0]["warnings"] = ["failed_adjacent_motion_step_2"]
        warned = r.RecoveryAtlas.from_payload(payload).frame(9000, "1/90000", source_sha256="a"*64, native_size=[1920, 1080])
        self.assertTrue(np.isfinite(warned.project([[0., 0.]])).all())
        self.assertIsNone(warned.public_projection([[0., 0.]])["native_pixels"])

    def test_fixed_diagnostic_geometry_failure_outside_observed_hull_is_not_excused(self):
        payload = atlas_payload()
        payload["projection_mode"] = r.V7_PROJECTION_MODE
        for c in payload["charts"]:
            c["support_world_polygon"] = [[-5., -5.], [-1., -5.], [-1., 5.], [-5., 5.]]
        mapping = r.RecoveryAtlas.from_payload(payload).frame(9000, "1/90000", source_sha256="a"*64, native_size=[1920, 1080])
        original = mapping.project
        def injected_projection(p):
            p = np.asarray(p)
            q = original(p)
            q[np.isclose(p[:, 0], .001, atol=1e-12) & np.isclose(p[:, 1], 0., atol=1e-12)] = np.nan
            return q
        with patch.object(mapping, "project", side_effect=injected_projection):
            check = r.visible_geometry_check(mapping, [[0., 0.]])
        self.assertEqual(check["visible_grid_points_outside_observed_support"], 1)
        self.assertEqual(check["interior_nonfinite_jacobians"], 1)
        self.assertFalse(check["valid"])


    def test_batched_renderer_preserves_nonfinite_and_large_jump_gaps(self):
        payload = atlas_payload()
        payload["projection_mode"] = r.V7_PROJECTION_MODE
        payload["frames"][0]["native_size"] = [384, 216]
        mapping = r.RecoveryAtlas.from_payload(payload).frame(9000, "1/90000", source_sha256="a"*64, native_size=[384, 216])
        projected = np.array([[20., 100.], [30., 100.], [150., 100.], [160., 100.],
                              [np.nan, np.nan], [250., 100.], [260., 100.]])
        with patch.object(mapping, "project", return_value=projected), patch.object(r, "world_markings", return_value={"halfway": np.array([[0., -1.], [0., 1.]])}):
            rendered = r.render_frame(np.zeros((216, 384, 3), np.uint8), mapping)
        for x in (25, 155, 255):
            self.assertGreater(int(rendered[100, x].sum()), 0)
        for x in (80, 210):
            self.assertEqual(int(rendered[100, x].sum()), 0)


    def test_explicit_linear_spatial_profile_removes_taper_fold_preserving_endpoints(self):
        truth = np.array([[10., 0, 960.], [0, 2., 400.], [0, 0, 1.]])
        payload = atlas_payload(truth)
        payload["projection_mode"] = r.V7_PROJECTION_MODE
        payload["spatial_boundary"] = dict(x_origin_px=960., x_scale_px=1000.,
                                           coefficients=[400.-2*r.HALF_W+18., 0., 0.])
        old = r.RecoveryAtlas.from_payload(payload)
        payload["spatial_boundary"]["weight_profile"] = r.LINEAR_SPATIAL_WEIGHT
        new = r.RecoveryAtlas.from_payload(payload)
        bind = dict(source_sha256="a"*64, native_size=[1920, 1080])
        band_middle = (-r.HALF_W-r.BOX18_HALF_W)/2
        grid = [[-10., band_middle], [0., band_middle], [10., band_middle]]
        before = r.visible_geometry_check(old.frame(9000, "1/90000", **bind), grid)
        after = r.visible_geometry_check(new.frame(9000, "1/90000", **bind), grid)
        self.assertEqual(before["nonpositive_jacobians"], 3)
        self.assertFalse(before["valid"])
        self.assertEqual(after["nonpositive_jacobians"], 0)
        self.assertTrue(after["valid"])
        endpoints = [[-20., -r.HALF_W], [0., -r.HALF_W], [20., -r.HALF_W],
                     [-20., -r.BOX18_HALF_W], [0., 0.], [20., r.HALF_W]]
        np.testing.assert_array_equal(old.reference(endpoints), new.reference(endpoints))
        # The temporal factor must remain the original smoothstep everywhere.
        p = np.array([[0., -29.], [0., -24.], [0., -r.BOX18_HALF_W], [0., 0.]])
        np.testing.assert_array_equal(old.edge_weight(p), new.edge_weight(p))
        self.assertNotEqual(float(old.spatial_edge_weight(p)[0]), float(new.spatial_edge_weight(p)[0]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"linear.json"
            new.save(path)
            loaded = r.RecoveryAtlas.load(path)
            self.assertEqual(loaded.spatial_weight_profile, r.LINEAR_SPATIAL_WEIGHT)
            np.testing.assert_array_equal(new.reference(p), loaded.reference(p))
        self.assertEqual(old.spatial_weight_profile, r.SMOOTH_SPATIAL_WEIGHT)
        self.assertNotIn("weight_profile", old.payload["spatial_boundary"])
        payload["spatial_boundary"]["weight_profile"] = "unrecognized"
        with self.assertRaises(ValueError):
            r.RecoveryAtlas.from_payload(payload)


class RecoveryRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1]/"tools"/"fit_field_recovery.py"
        spec = importlib.util.spec_from_file_location("recovery_runner", path)
        cls.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.runner)

    def test_failed_direct_paint_reference_uses_measured_image_anchor_bridge(self):
        first = np.array([[1., 0, 7.], [0, 1, 0.], [0, 0, 1.]])
        second = np.array([[1., 0, 0.], [0, 1, 4.], [0, 0, 1.]])
        stats = dict(image_registration_pass=True, inlier_hull_fraction=.4,
                     check_consistent_fraction=.95, check_median_px=.5)
        calls = []
        def register(a, b):
            calls.append((a, b))
            h = {(0, 1):first, (1, 3):second}.get((a, b))
            return h, stats if h is not None else dict(image_registration_pass=False, reason="fixture_no_overlap")
        connected, paths, attempts = self.runner.independent_reference_connections(0, [0, 3], [0, 1, 3], register)
        np.testing.assert_array_equal(connected[3], second @ first)
        self.assertEqual(paths[3]["path"], [0, 1, 3])
        self.assertEqual(paths[3]["cost"]["total_check_median_px"], 1.)
        self.assertIn((1, 3), calls)
        failed = [a for a in attempts if (a["from_index"], a["to_index"]) == (0, 3)]
        self.assertEqual(len(failed), 1)
        self.assertFalse(failed[0]["image_registration_pass"])

    def test_failed_bridge_never_uses_accumulation_or_another_bridge_as_first_leg(self):
        stats = dict(image_registration_pass=True, inlier_hull_fraction=.4,
                     check_consistent_fraction=.95, check_median_px=.5)
        calls = []
        def register(a, b):
            calls.append((a, b))
            passed = (a, b) in {(0, 1), (1, 3), (3, 4)}
            return (np.eye(3), stats) if passed else (None, dict(image_registration_pass=False))
        connected, paths, _ = self.runner.independent_reference_connections(0, [0, 3, 4], [0, 1, 2, 3, 4], register)
        self.assertIn(3, connected)
        self.assertNotIn(4, connected)
        self.assertNotIn(4, paths)
        self.assertNotIn((3, 4), calls)  # A bridged paint anchor cannot extend the one-hop search.
        self.assertNotIn((2, 4), calls)  # A failed first leg cannot seed a bridge.

    def test_drift_uses_nearest_paint_before_cached_gauge_and_preserves_failed_attempt(self):
        shift = lambda x: np.array([[1., 0, x], [0, 1, 0.], [0, 0, 1.]])
        registered = {120:np.eye(3), 30:shift(20), 50:shift(100), 60:shift(110)}
        paths = {i:dict(path=[120] if i == 120 else [120, i], kind="gauge_direct", cost=None) for i in registered}
        calls = []
        def register(a, b):
            calls.append((a, b))
            return (shift(2), dict(image_registration_pass=True)) if b == 50 else (None, dict(image_registration_pass=False, reason="fixture_overlap_failure"))
        observed, selected, attempts = self.runner.independent_drift_observations(120, [30, 120], [30, 50, 60, 120], registered, paths, register)
        np.testing.assert_array_equal(observed[50], shift(22))
        np.testing.assert_array_equal(observed[60], shift(110))
        np.testing.assert_array_equal(observed[30], registered[30])
        self.assertEqual(selected[50]["path"], [120, 30, 50])
        self.assertEqual(selected[60]["path"], [120, 60])
        self.assertEqual(calls, [(30, 50), (30, 60)])  # Gauge matches are reused, never recomputed.
        self.assertFalse(attempts[-1]["image_registration_pass"])

    def test_drift_keeps_end_bridge_but_never_loops_back_through_its_anchor(self):
        registered = {120:np.eye(3), 310:np.eye(3), 350:np.eye(3)}
        paths = {120:dict(path=[120], kind="gauge", cost=None),
                 310:dict(path=[120, 310], kind="gauge_direct", cost=None),
                 350:dict(path=[120, 310, 350], kind="gauge_single_bridge", cost={"fixture":True})}
        with patch.object(r, "register_reference") as register:
            observed, selected, attempts = self.runner.independent_drift_observations(120, [120, 350], [120, 310, 350], registered, paths, register)
            register.assert_not_called()
        self.assertEqual(selected[350], paths[350])
        self.assertEqual(selected[310]["path"], [120, 310])
        self.assertEqual(set(observed), {120, 310, 350})
        self.assertEqual(attempts[0]["reason"], "target_already_on_reference_connection_path")
        self.assertEqual(paths[350]["path"], [120, 310, 350])

    def fixture(self, directory, h=None, features=None):
        h = np.diag([.2, .2, 1.]) @ H if h is None else h
        rows = []
        for i in range(3):
            name = f"{i:03d}.png"
            image = np.zeros((216, 384, 3), np.uint8)
            image[:] = [35, 90, 40]
            cv2.imwrite(str(directory/name), image)
            rows.append(dict(index=i, file=name, source_pts=9000+i*9000,
                             source_time_base="1/90000", source_sha256="a"*64,
                             native_size=[384, 216], image_sha256=hashlib.sha256((directory/name).read_bytes()).hexdigest()))
        manifest = dict(schema="alignment-trial-frames-v1", source=dict(game_id="synthetic", sha256="a"*64), frames=rows)
        measurements = dict(schema="field-recovery-measurements-v1", source_sha256="a"*64,
                            fit_frame_indices=[0], check_frame_indices=[1, 2],
                            references=[dict(frame_index=0, chart="mid", features=features or samples(h),
                                             support_world_polygon=FULL)])
        (directory/"frames.json").write_text(json.dumps(manifest))
        (directory/"measurements.json").write_text(json.dumps(measurements))
        return manifest, measurements

    def test_actual_runner_preserves_failure_and_recomputes_changed_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, measurement = self.fixture(root)
            fit_state = dict(image_registration_pass=True)
            with patch.object(r, "reference_features", return_value=(np.zeros((0, 2)), None)), \
                    patch.object(r, "register_reference", return_value=(np.eye(3), fit_state)), \
                    patch.object(r, "adjacent_motion", side_effect=[(None, dict(image_registration_pass=False)), (np.eye(3), fit_state)]):
                result = self.runner.run(root/"frames.json", root/"measurements.json", root/"first", render=False, reference_stride=1)
            self.assertEqual(result["frames_with_warnings"], 2)
            a = r.RecoveryAtlas.load(root/"first"/"atlas.json")
            self.assertTrue(a.payload["frames"][1]["warnings"])
            self.assertTrue(a.payload["frames"][2]["warnings"])
            shift = np.array([[1., 0, -9.], [0, 1, 0.], [0, 0, 1.]])
            measurement["references"][0]["features"] = samples(shift @ np.diag([.2, .2, 1.]) @ H)
            (root/"measurements-later.json").write_text(json.dumps(measurement))
            with patch.object(r, "reference_features", return_value=(np.zeros((0, 2)), None)), \
                    patch.object(r, "register_reference", return_value=(np.eye(3), fit_state)), \
                    patch.object(r, "adjacent_motion", return_value=(np.eye(3), fit_state)):
                self.runner.run(root/"frames.json", root/"measurements-later.json", root/"later", render=False, reference_stride=1)
            b = r.RecoveryAtlas.load(root/"later"/"atlas.json")
            bind = dict(source_sha256="a"*64, native_size=[384, 216])
            old = a.frame(9000, "1/90000", **bind).project([[0, 0]])
            new = b.frame(9000, "1/90000", **bind).project([[0, 0]])
            self.assertAlmostEqual(float(new[0, 0]-old[0, 0]), -9., places=4)
            self.assertGreater(float(np.linalg.norm(new-old)), 5.)

    def test_actual_runner_adjacent_chart_fallback_is_never_independent_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, measurements = self.fixture(root)
            measurements["fit_frame_indices"] = [0, 2]
            measurements["check_frame_indices"] = [1]
            later = deepcopy(measurements["references"][0])
            later.update(frame_index=2, chart="left")
            measurements["references"].append(later)
            (root/"measurements.json").write_text(json.dumps(measurements))
            with patch.object(r, "reference_features", return_value=(np.zeros((0, 2)), None)), \
                    patch.object(r, "register_reference", return_value=(None, dict(image_registration_pass=False))), \
                    patch.object(r, "adjacent_motion", return_value=(np.eye(3), dict(image_registration_pass=True))), \
                    patch.object(r, "accumulate_motion", wraps=r.accumulate_motion) as accumulate:
                result = self.runner.run(root/"frames.json", root/"measurements.json", root/"candidate", render=False, reference_stride=1)
            self.assertEqual(result["adjacent_only_chart_frame_indices"], [2])
            self.assertEqual(result["drift"]["observations"], 1)
            self.assertEqual(accumulate.call_args.args[2], [0])
            self.assertEqual([x["frame_index"] for x in result["reference_connections"]], [0])
            saved = r.RecoveryAtlas.load(root/"candidate"/"atlas.json")
            self.assertIn("paint_reference_not_independently_connected_to_gauge", saved.payload["frames"][2]["warnings"])
            self.assertEqual({c["chart"] for c in saved.charts}, {"mid", "left"})

    def test_actual_runner_rejects_invalid_anchor_before_motion(self):
        pole = np.diag([.2, .2, 1.]) @ np.array([[960., 0, 100.], [540., 2., 0], [1., 0, 0.]])
        x, y = np.meshgrid([-10., -5., 5., 10.], [-5., 5.])
        world = np.c_[x.ravel(), y.ravel()]
        feature = dict(label="synthetic landmarks", world_points=world.tolist(), points_native=r.project(pole, world).tolist())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root, features=[feature])
            with patch.object(r, "adjacent_motion") as motion:
                with self.assertRaises(ValueError):
                    self.runner.run(root/"frames.json", root/"measurements.json", root/"invalid", render=False)
                motion.assert_not_called()

    def test_runner_split_rejects_reference_in_check_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, measurements = self.fixture(root)
            measurements["references"][0]["frame_index"] = 1
            (root/"bad.json").write_text(json.dumps(measurements))
            with self.assertRaises(ValueError):
                self.runner.load_inputs(root/"frames.json", root/"bad.json")


if __name__ == "__main__":
    unittest.main()
