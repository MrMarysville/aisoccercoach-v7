"""Check explicit reference selection without discarding source-reviewed evidence."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from copy import deepcopy

import cv2
import numpy as np

from calibration.alignment_trial import sha256_file
from calibration.tools.assemble_field_recovery import assemble
from calibration.tools.measure_field_annotations import measure


class AssemblyTests(unittest.TestCase):
    def fixture(self, root):
        source = 'a' * 64
        rows = [dict(index=i, source_pts=i*9000, source_time_base='1/90000',
                     image_sha256=str(i)*64, native_size=[1920, 1080]) for i in range(3)]
        features = [dict(feature_id=str(i), label='touch_near', points_native=[[10, 20], [20, 20], [30, 20]])
                    for i in range(2)]
        (root/'frames.json').write_text(json.dumps(dict(source=dict(sha256=source), frames=rows)))
        manifest_hash = sha256_file(root/'frames.json')
        paint = dict(source_sha256=source, annotation_sha256='b'*64, frames_manifest_sha256=manifest_hash,
            references=[dict(frame_index=i, chart='right', features=[f]) for i, f in enumerate(features)],
            measurements=[dict(f, selected_count=3) for f in features])
        objects = dict(frames=dict(source=dict(sha256=source), frames=rows), paint=paint,
            split=dict(source_sha256=source, frames_manifest_sha256=manifest_hash,
                       fit_frame_indices=[0, 1], check_frame_indices=[2]))
        for name, value in objects.items():
            (root/f'{name}.json').write_text(json.dumps(value))
        review = dict(source_sha256=source, measurement_sha256=hashlib.sha256((root/'paint.json').read_bytes()).hexdigest(),
                      annotation_sha256='b'*64, accepted_feature_ids=['0', '1'])
        (root/'review.json').write_text(json.dumps(review))
        return [root/f'{name}.json' for name in ('frames', 'paint', 'review', 'split', 'out')]

    def write_paint(self, args, paint, accepted=None):
        args[1].write_text(json.dumps(paint))
        review = json.loads(args[2].read_text())
        review['measurement_sha256'] = sha256_file(args[1])
        if accepted is not None:
            review['accepted_feature_ids'] = accepted
        args[2].write_text(json.dumps(review))

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

    def test_conflicting_inline_identity_and_stale_split_reject_before_output(self):
        changes = dict(source_pts=0.0, source_time_base='1/1000', image_sha256='d'*64,
                       native_size=[1280, 720])
        for key, value in changes.items():
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                args = self.fixture(Path(tmp))
                paint = json.loads(args[1].read_text())
                paint['references'][0][key] = value
                self.write_paint(args, paint)
                with self.assertRaisesRegex(ValueError, 'exact frame binding'):
                    assemble(*args, reference_indices=[1])
                self.assertFalse(args[-1].exists())
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            split = json.loads(args[3].read_text())
            split['frames_manifest_sha256'] = 'e'*64
            args[3].write_text(json.dumps(split))
            with self.assertRaisesRegex(ValueError, 'split/full frame-manifest mismatch'):
                assemble(*args, reference_indices=[1])
            self.assertFalse(args[-1].exists())

    def test_old_paint_cannot_be_rebound_to_different_frame_at_same_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.fixture(root)
            frames = json.loads(args[0].read_text())
            original = dict(index=0, source_pts=9000, source_time_base='1/90000',
                            image_sha256='1'*64, native_size=[1920, 1080])
            frames['frames'] = [original]
            annotation = root/'annotation-frames.json'
            annotation.write_text(json.dumps(frames))
            paint = json.loads(args[1].read_text())
            paint['frames_manifest_sha256'] = sha256_file(annotation)
            paint['references'] = [dict(paint['references'][0], **{k:v for k,v in original.items() if k != 'index'})]
            self.write_paint(args, paint, ['0'])
            frames['frames'] = [dict(original, source_pts=18000, image_sha256='2'*64)]
            args[0].write_text(json.dumps(frames))
            split = dict(source_sha256=frames['source']['sha256'], frames_manifest_sha256=sha256_file(annotation),
                         fit_frame_indices=[0], check_frame_indices=[])
            for stale in (True, False):
                if not stale:
                    split['frames_manifest_sha256'] = sha256_file(args[0])
                args[3].write_text(json.dumps(split))
                with self.subTest(stale_split=stale), self.assertRaises(ValueError):
                    assemble(*args)
                self.assertFalse(args[-1].exists())

    def test_subset_inline_and_legacy_manifest_preserve_verified_identity_and_reindex(self):
        for legacy, reindex in ((False, False), (False, True), (True, False), (True, True)):
            with self.subTest(legacy=legacy, reindex=reindex), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                args = self.fixture(root)
                frames = json.loads(args[0].read_text())
                old_index = 70 if reindex else 1
                original = dict(frames['frames'][1], index=old_index)
                annotation = root/'annotation-frames.json'
                annotation.write_text(json.dumps(dict(source=frames['source'], frames=[original])))
                paint = json.loads(args[1].read_text())
                feature = deepcopy(paint['references'][1]['features'])
                paint['references'] = [dict(frame_index=old_index, chart='right', features=feature)]
                if not legacy:
                    paint['references'][0].update({k:v for k,v in original.items() if k != 'index'})
                paint['frames_manifest_sha256'] = sha256_file(annotation)
                self.write_paint(args, paint, ['1'])
                if legacy:
                    with self.assertRaisesRegex(ValueError, '--annotation-frames'):
                        assemble(*args)
                    self.assertFalse(args[-1].exists())
                assemble(*args, annotation_frames_path=annotation if legacy else None)
                data = json.loads(args[-1].read_text())
                self.assertEqual(data['references'][0]['frame_index'], 1)
                self.assertEqual(data['references'][0]['features'], feature)
                self.assertEqual(data['references'][0]['source_pts'], original['source_pts'])
                remaps = data['assembly_provenance']['reference_frame_index_remaps']
                self.assertEqual([(r['annotation_frame_index'], r['frame_index']) for r in remaps],
                                 [(70, 1)] if reindex else [])

    def test_legacy_manifest_binding_and_reindexed_check_role_cannot_be_bypassed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.fixture(root)
            frames = json.loads(args[0].read_text())
            annotation = root/'annotation-frames.json'
            source = dict(source=frames['source'], frames=[dict(frames['frames'][2], index=70)])
            annotation.write_text(json.dumps(source))
            paint = json.loads(args[1].read_text())
            paint['references'] = [dict(paint['references'][0], frame_index=70)]
            paint['frames_manifest_sha256'] = sha256_file(annotation)
            self.write_paint(args, paint, ['0'])
            with self.assertRaisesRegex(ValueError, 'reserved check frame'):
                assemble(*args, annotation_frames_path=annotation)
            annotation.write_text(annotation.read_text()+'\n')
            with self.assertRaisesRegex(ValueError, 'Paint/annotation frame-manifest mismatch'):
                assemble(*args, annotation_frames_path=annotation)
            source['source'] = dict(sha256='f'*64)
            annotation.write_text(json.dumps(source))
            paint['frames_manifest_sha256'] = sha256_file(annotation)
            self.write_paint(args, paint, ['0'])
            with self.assertRaisesRegex(ValueError, 'Annotation manifest/source mismatch'):
                assemble(*args, annotation_frames_path=annotation)
            self.assertFalse(args[-1].exists())

    def test_sampler_retains_exact_reference_frame_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = np.full((100, 200, 3), [20, 100, 20], np.uint8)
            cv2.line(image, (10, 50), (190, 50), (220, 220, 220), 3)
            cv2.imwrite(str(root/'frame.png'), image)
            frame = dict(index=70, file='frame.png', source_pts=9000, source_time_base='1/90000',
                         image_sha256=sha256_file(root/'frame.png'), native_size=[200, 100])
            source = dict(sha256='a'*64)
            (root/'frames.json').write_text(json.dumps(dict(source=source, frames=[frame])))
            annotation = dict(source=source, frames_manifest_sha256=sha256_file(root/'frames.json'),
                features=[dict(id='line', frame_index=70, label='halfway',
                               points=[dict(native_xy=p) for p in ([10, 50], [190, 50])])])
            (root/'annotation.json').write_text(json.dumps(annotation))
            measure(root/'frames.json', root/'annotation.json', root/'measured', {70:'mid'})
            reference = json.loads((root/'measured/measurements.json').read_text())['references'][0]
            self.assertGreaterEqual(len(reference['features'][0]['points_native']), 3)
            self.assertEqual(reference['frame_index'], 70)
            for key in ('source_pts', 'source_time_base', 'image_sha256', 'native_size'):
                self.assertEqual(reference[key], frame[key])


if __name__ == '__main__':
    unittest.main()
