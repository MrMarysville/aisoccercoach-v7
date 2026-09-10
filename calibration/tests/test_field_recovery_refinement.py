"""Boundary revision/source-binding evidence; no real-footage acceptance claims."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from calibration import field_recovery as r
from calibration.tools.refine_field_recovery import refine
from calibration.tools.score_field_recovery import score


class RefinementTests(unittest.TestCase):
    def fixture(self, root):
        h = np.array([[8., 0, 800.], [0, 5., 350.], [0, 0, 1.]])
        full = [[-r.HALF_L, -r.HALF_W], [r.HALF_L, -r.HALF_W],
                [r.HALF_L, r.HALF_W], [-r.HALF_L, r.HALF_W]]
        frames = [dict(index=i, source_pts=9000*i, source_time_base='1/90000',
                       source_seconds=.1*i, native_size=[1920, 1080], file=f'{i}.png',
                       source_sha256='a'*64, image_sha256=str(i)*64) for i in range(7)]
        manifest = dict(schema='alignment-trial-frames-v1', source=dict(game_id='synthetic', sha256='a'*64), frames=frames)
        parent_input = dict(schema='field-recovery-measurements-v1', source_sha256='a'*64,
            references=[], fit_frame_indices=list(range(6)), check_frame_indices=[6])
        (root/'parent-input.json').write_text(json.dumps(parent_input))
        parent_sha = hashlib.sha256((root/'parent-input.json').read_bytes()).hexdigest()
        payload = dict(status='approximate', metric_certified=False, source=manifest['source'], field=r.FIELD,
            charts=[dict(chart=name, reference_frame_index=0, field_to_reference=h.tolist(), support_world_polygon=full)
                    for name in ('left', 'mid', 'right')], blend=r.BLEND,
            spatial_boundary=None, temporal_boundary=None, protected_y_m=-r.BOX18_HALF_W,
            provenance=dict(measurements_sha256=parent_sha), frames=[])
        revised = deepcopy(parent_input)
        revised.update(parent_measurements_sha256=parent_sha, parent_measurements_path=str(root/'parent-input.json'),
                       boundary_observations=[])
        world = np.c_[np.linspace(-35, 35, 40), np.full(40, -r.HALF_W)]
        for row in frames:
            motion = np.array([[1., 0, 2.*row['index']], [0, 1, .5*row['index']], [0, 0, 1.]])
            saved = {k:row[k] for k in ('index', 'source_pts', 'source_time_base', 'native_size', 'image_sha256')}
            saved.update(reference_to_native=motion.tolist(), boundary_offset_native_y_px=0.,
                         warnings=['invalid_or_absent_visible_supported_geometry'])
            if row['index'] == 3:
                saved['warnings'].append('independent_motion_warning')
            payload['frames'].append(saved)
            if row['index'] < 6:
                pixels = r.project(motion@h, world) + [0., .2*np.sin(row['index'])]
                ref = {k:row[k] for k in ('source_pts', 'source_time_base', 'image_sha256')}
                ref.update(frame_index=row['index'], features=[dict(label='touch_far', points_native=pixels.tolist())])
                revised['boundary_observations'].append(ref)
        r.RecoveryAtlas.from_payload(payload).save(root/'parent.json')
        (root/'frames.json').write_text(json.dumps(manifest))
        (root/'revised.json').write_text(json.dumps(revised))
        return manifest, revised

    def test_revision_preserves_motion_and_uncertainty_and_rechecks_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            result = refine(root/'parent.json', root/'frames.json', root/'revised.json', root/'candidate', render=False)
            parent = r.RecoveryAtlas.load(root/'parent.json')
            new = r.RecoveryAtlas.load(root/'candidate/atlas.json')
            self.assertEqual(len(result['temporal_observations']), 6)
            self.assertEqual([f['source_pts'] for f in new.payload['provenance']['fitting_frames']],
                             [9000*i for i in range(6)])
            self.assertEqual(new.payload['charts'], parent.payload['charts'])
            for a, b in zip(parent.payload['frames'], new.payload['frames']):
                self.assertEqual(a['reference_to_native'], b['reference_to_native'])
                self.assertNotIn('invalid_or_absent_visible_supported_geometry', b['warnings'])
            self.assertIn('independent_motion_warning', new.payload['frames'][3]['warnings'])
            self.assertIn('outside_temporal_boundary_fit_support', new.payload['frames'][6]['warnings'])

    def test_reject_undeclared_check_paint_and_changed_chart_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, revised = self.fixture(root)
            revised['boundary_observations'][0]['frame_index'] = 6
            (root/'revised.json').write_text(json.dumps(revised))
            with self.assertRaisesRegex(ValueError, 'outside declared fit'):
                refine(root/'parent.json', root/'frames.json', root/'revised.json', root/'rejected', render=False)
            revised['boundary_observations'][0]['frame_index'] = 0
            row = manifest['frames'][0]
            revised['references'] = [dict(frame_index=0, chart='mid', source_pts=row['source_pts'],
                source_time_base=row['source_time_base'], image_sha256=row['image_sha256'],
                features=[dict(label='halfway', points_native=[[800., 300.], [800., 400.]])])]
            (root/'revised.json').write_text(json.dumps(revised))
            with self.assertRaisesRegex(ValueError, 'change fitted chart'):
                refine(root/'parent.json', root/'frames.json', root/'revised.json', root/'rejected2', render=False)
            revised['references'] = []
            revised['fit_frame_indices'].append(6)
            revised['check_frame_indices'] = []
            (root/'revised.json').write_text(json.dumps(revised))
            with self.assertRaisesRegex(ValueError, 'frozen parent fit/check split'):
                refine(root/'parent.json', root/'frames.json', root/'revised.json', root/'relabelled', render=False)
            self.assertFalse((root/'relabelled').exists())
            revised.update(fit_frame_indices=list(range(6)), check_frame_indices=[6], boundary_observations=[])
            manifest['frames'][0]['index'], manifest['frames'][6]['index'] = 6, 0
            (root/'reindexed-frames.json').write_text(json.dumps(manifest))
            (root/'revised.json').write_text(json.dumps(revised))
            with self.assertRaisesRegex(ValueError, 'Parent exact frame mismatch: index'):
                refine(root/'parent.json', root/'reindexed-frames.json', root/'revised.json', root/'reindexed', render=False)
            self.assertFalse((root/'reindexed').exists())

    def test_boundary_revision_preserves_explicit_single_chart_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            parent = r.RecoveryAtlas.load(root/'parent.json').payload
            parent['projection_mode'] = r.SINGLE_REFERENCE_PROJECTION_MODE
            parent['charts'] = [parent['charts'][-1]]
            r.RecoveryAtlas.from_payload(parent).save(root/'single-parent.json')
            refine(root/'single-parent.json', root/'frames.json', root/'revised.json', root/'candidate', render=False)
            atlas = r.RecoveryAtlas.load(root/'candidate/atlas.json')
            self.assertEqual(atlas.projection_mode, r.SINGLE_REFERENCE_PROJECTION_MODE)
            self.assertEqual(atlas.payload['charts'], parent['charts'])
            frame = atlas.frame(0, '1/90000', source_sha256='a'*64, native_size=[1920, 1080])
            self.assertTrue(np.isfinite(frame.project([[-30., 0.], [0., 0.]])).all())

    def test_score_retains_count_and_raw_two_pixel_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            checks = dict(source_sha256='a'*64, measurements=[dict(frame_index=0, label='halfway', feature_id='check',
                selected_count=2, searched_count=3, points_native=[[802., 300.], [802., 400.]])])
            (root/'checks.json').write_text(json.dumps(checks))
            score(root/'parent.json', root/'frames.json', root/'checks.json', root/'score.json')
            result = json.loads((root/'score.json').read_text())
            row = result['rows'][0]
            self.assertEqual(row['samples'], 2)
            self.assertEqual(row['searched_samples'], 3)
            self.assertEqual(len(row['raw_samples']), 2)
            self.assertAlmostEqual(row['median_px'], 2.)
            self.assertTrue(all(abs(s['error_px']-2.) < 1e-8 for s in row['raw_samples']))


if __name__ == '__main__':
    unittest.main()
