"""Assemble fit inputs from hash-bound paint, semantic review and a frozen split."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from calibration.alignment_trial import sha256_file


def assemble(frames_path, paint_path, review_path, split_path, output, parent_path=None, reference_indices=None,
             boundary_label='touch_far', parent_atlas_path=None):
    paths = list(map(Path, (frames_path, paint_path, review_path, split_path, output)))
    frames_path, paint_path, review_path, split_path, output = paths
    frames, paint, review, split = [json.loads(p.read_text()) for p in paths[:4]]
    if boundary_label not in ('touch_far', 'touch_near'):
        raise ValueError('Unknown boundary label')
    if boundary_label == 'touch_near' and (parent_path is None or parent_atlas_path is None):
        raise ValueError('Near-boundary assembly requires parent measurements and atlas')
    source = frames['source']['sha256']
    if any(d['source_sha256'] != source for d in (paint, review, split)):
        raise ValueError('Source identity mismatch')
    if review['measurement_sha256'] != sha256_file(paint_path):
        raise ValueError('Reviewed measurement hash mismatch')
    if review['annotation_sha256'] != paint['annotation_sha256']:
        raise ValueError('Reviewed annotation hash mismatch')
    fit, check = map(set, (split['fit_frame_indices'], split['check_frame_indices']))
    by_index = {r['index']: r for r in frames['frames']}
    if not fit or fit & check or (fit | check) != set(by_index):
        raise ValueError('Split must partition every source frame without overlap')
    accepted = set(review['accepted_feature_ids'])
    all_measured = {f['feature_id']: f for f in paint['measurements']}
    if not accepted or not accepted <= set(all_measured):
        raise ValueError('Reviewed feature missing from measurement record')
    references = []
    used = set()
    for ref in paint['references']:
        features = [f for f in ref['features'] if f['feature_id'] in accepted]
        if not features:
            continue
        index = ref['frame_index']
        if index not in fit:
            raise ValueError('Fit feature is on a reserved check frame')
        row = by_index[index]
        bound = {k: row[k] for k in ('source_pts', 'source_time_base', 'native_size', 'image_sha256')}
        bound.update(frame_index=index, features=features)
        if parent_path is None:
            if ref['chart'] not in ('left', 'mid', 'right'):
                raise ValueError('Assign every fitted reference to an explicit chart')
            bound['chart'] = ref['chart']
        elif any(f['label'] != boundary_label for f in features):
            raise ValueError('Boundary refinement review must contain only the selected boundary label')
        references.append(bound)
        used.update(f['feature_id'] for f in features)
    missing_fit_features = sorted(accepted-used)
    # Semantic approval of a marking cannot promote rejected cross sections.
    if any(all_measured[f]['selected_count'] >= 3 for f in missing_fit_features):
        raise ValueError('Usable accepted feature unexpectedly missing from fit references')
    deferred = []
    if reference_indices is not None:
        if parent_path is not None:
            raise ValueError('Reference selection is for fresh chart fits, not boundary refinement')
        selected = set(reference_indices)
        if not selected or not selected <= {r['frame_index'] for r in references}:
            raise ValueError('Selected reference lacks source-reviewed fit evidence')
        deferred = [r for r in references if r['frame_index'] not in selected]
        references = [r for r in references if r['frame_index'] in selected]
    if parent_path is None and len({r['chart'] for r in references}) != len(references):
        raise ValueError('Select one fresh reference per chart with --reference-frame; other reviewed views remain deferred')
    if parent_path is None:
        result = deepcopy(paint)
        result.update(references=references, review_status='source_browser_semantics_reviewed',
            semantic_review=review, semantic_review_sha256=sha256_file(review_path),
            raw_measurements_sha256=sha256_file(paint_path),
            annotation_frames_manifest_sha256=paint['frames_manifest_sha256'],
            frames_manifest_sha256=sha256_file(frames_path),
            fit_frame_indices=split['fit_frame_indices'], check_frame_indices=split['check_frame_indices'])
        if deferred:
            result['additional_reviewed_reference_observations'] = deferred
    else:
        parent_path = Path(parent_path)
        result = json.loads(parent_path.read_text())
        if result['source_sha256'] != source or set(result['fit_frame_indices']) != fit or set(result['check_frame_indices']) != check:
            raise ValueError('Parent source/split mismatch')
        result.update(parent_measurements_path=str(parent_path.resolve()),
            parent_measurements_sha256=sha256_file(parent_path))
        if boundary_label == 'touch_near':
            from calibration.field_recovery import RecoveryAtlas
            parent = RecoveryAtlas.load(parent_atlas_path)
            if (parent.payload['source']['sha256'] != source or
                    parent.payload['provenance']['measurements_sha256'] != sha256_file(parent_path)):
                raise ValueError('Parent atlas/source/measurement mismatch')
            result.update(parent_atlas_sha256=sha256_file(parent_atlas_path),
                near_boundary_observations=references, near_boundary_source_review=review,
                near_boundary_source_review_sha256=sha256_file(review_path),
                near_boundary_raw_measurements_sha256=sha256_file(paint_path))
        else:
            result.update(boundary_observations=references,
                boundary_source_review=review, boundary_source_review_sha256=sha256_file(review_path),
                boundary_raw_measurements_sha256=sha256_file(paint_path))
    result['assembly_provenance'] = dict(helper_sha256=sha256_file(__file__), split_sha256=sha256_file(split_path),
        semantic_features_without_fit_samples=missing_fit_features,
        semantic_features_without_fit_samples_reason='Fewer than three sampler-selected source points; not promoted')
    if reference_indices is not None:
        result['assembly_provenance']['selected_reference_frame_indices'] = sorted(set(reference_indices))
        result['assembly_provenance']['deferred_reference_frame_indices'] = [r['frame_index'] for r in deferred]
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return dict(output=str(output), output_sha256=sha256_file(output),
        reference_frames=[r['frame_index'] for r in references], accepted_review_features=len(accepted),
        fit_features=sum(len(r['features']) for r in references), missing_fit_features=missing_fit_features,
        deferred_reference_frames=[r['frame_index'] for r in deferred])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('frames', 'paint', 'review', 'split', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--parent-measurements', type=Path)
    parser.add_argument('--reference-frame', type=int, action='append', help='Explicit source-selected chart reference; retain other reviewed views as deferred observations')
    parser.add_argument('--boundary-label', choices=('touch_far', 'touch_near'), default='touch_far')
    parser.add_argument('--parent-atlas', type=Path, help='Exact saved parent required for a near-boundary revision')
    args = parser.parse_args()
    print(json.dumps(assemble(args.frames, args.paint, args.review, args.split, args.out,
                             args.parent_measurements, args.reference_frame, args.boundary_label, args.parent_atlas), indent=2))


if __name__ == '__main__':
    main()
