"""Check explicit reference selection without discarding source-reviewed evidence."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from calibration.tools.assemble_field_recovery import assemble


class AssemblyTests(unittest.TestCase):
    def fixture(self, root):
        source = 'a' * 64
        rows = [dict(index=i, source_pts=i*9000, source_time_base='1/90000',
                     image_sha256=str(i)*64, native_size=[1920, 1080]) for i in range(3)]
        features = [dict(feature_id=str(i), label='touch_near', points_native=[[10, 20], [20, 20], [30, 20]])
                    for i in range(2)]
        paint = dict(source_sha256=source, annotation_sha256='b'*64, frames_manifest_sha256='c'*64,
            references=[dict(frame_index=i, chart='right', features=[f]) for i, f in enumerate(features)],
            measurements=[dict(f, selected_count=3) for f in features])
        objects = dict(frames=dict(source=dict(sha256=source), frames=rows), paint=paint,
            split=dict(source_sha256=source, fit_frame_indices=[0, 1], check_frame_indices=[2]))
        for name, value in objects.items():
            (root/f'{name}.json').write_text(json.dumps(value))
        review = dict(source_sha256=source, measurement_sha256=hashlib.sha256((root/'paint.json').read_bytes()).hexdigest(),
                      annotation_sha256='b'*64, accepted_feature_ids=['0', '1'])
        (root/'review.json').write_text(json.dumps(review))
        return [root/f'{name}.json' for name in ('frames', 'paint', 'review', 'split', 'out')]

    def test_selects_one_real_chart_and_preserves_other_reviewed_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            result = assemble(*args, reference_indices=[1])
            data = json.loads(args[-1].read_text())
            self.assertEqual([r['frame_index'] for r in data['references']], [1])
            self.assertEqual([r['frame_index'] for r in data['additional_reviewed_reference_observations']], [0])
            self.assertEqual(data['semantic_review']['accepted_feature_ids'], ['0', '1'])
            self.assertEqual(result['fit_features'], 1)
            self.assertEqual(result['deferred_reference_frames'], [0])

    def test_rejects_implicit_duplicate_chart_or_unreviewed_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            with self.assertRaisesRegex(ValueError, 'one fresh reference per chart'):
                assemble(*args)
            with self.assertRaisesRegex(ValueError, 'lacks source-reviewed'):
                assemble(*args, reference_indices=[2])
            self.assertFalse(args[-1].exists())


if __name__ == '__main__':
    unittest.main()
