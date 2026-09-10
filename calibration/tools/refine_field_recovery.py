"""Refine a frozen fresh candidate's boundary, preserving its measured image motion.

This is a development revision, not an independently refitted chart or motion run.
Parent hashes, chart fits and warnings remain explicit in the resulting artifact.
"""
import argparse
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from calibration import field_recovery as recovery
from calibration.tools.fit_field_recovery import fitting_frame_bindings, load_inputs


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def refine(parent_path, frames_path, measurements_path, output, *, render=True, spatial_profile=None):
    started = time.monotonic()
    parent_path, frames_path, measurements_path, output = map(
        Path, (parent_path, frames_path, measurements_path, output))
    manifest, measurements, by_index = load_inputs(frames_path, measurements_path)
    parent = recovery.RecoveryAtlas.load(parent_path)
    payload = deepcopy(parent.payload)
    chosen_profile = spatial_profile or (payload.get('spatial_boundary') or {}).get(
        'weight_profile', recovery.SMOOTH_SPATIAL_WEIGHT)
    if chosen_profile not in (recovery.SMOOTH_SPATIAL_WEIGHT, recovery.LINEAR_SPATIAL_WEIGHT):
        raise ValueError('Unknown spatial boundary profile')
    if payload['source']['sha256'] != manifest['source']['sha256']:
        raise ValueError('Parent/source mismatch')
    if measurements.get('parent_measurements_sha256') != payload['provenance']['measurements_sha256']:
        raise ValueError('Revision must explicitly bind the frozen parent measurements')
    parent_inputs_path = measurements.get('parent_measurements_path')
    if not parent_inputs_path or sha(parent_inputs_path) != measurements['parent_measurements_sha256']:
        raise ValueError('Frozen parent input file missing or changed')
    parent_inputs = json.loads(Path(parent_inputs_path).read_text())
    if any(set(measurements[k]) != set(parent_inputs[k]) for k in ('fit_frame_indices', 'check_frame_indices')):
        raise ValueError('Boundary refinement must preserve the frozen parent fit/check split')
    if measurements['references'] != parent_inputs['references']:
        raise ValueError('Boundary-only refinement cannot silently change fitted chart inputs')
    if len(payload['frames']) != len(manifest['frames']):
        raise ValueError('Parent/source frame coverage mismatch')
    for saved, row in zip(payload['frames'], manifest['frames']):
        for key in ('index', 'source_pts', 'source_time_base', 'native_size', 'image_sha256'):
            if saved[key] != row[key]:
                raise ValueError(f'Parent exact frame mismatch: {key}')
    if output.exists():
        raise ValueError('Output must be a new directory')
    output.mkdir(parents=True)
    def write(name, data):
        (output/name).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    write('declaration.json', dict(parent_atlas_sha256=sha(parent_path),
          frames_sha256=sha(frames_path), measurements_sha256=sha(measurements_path),
          helper_sha256=sha(recovery.__file__), refinement_helper_sha256=sha(__file__),
          changed='v7 fixed diagnostic blending and declared fit-frame boundary corrections',
          spatial_boundary_weight_profile=chosen_profile,
          image_motion='reused exactly from source-bound parent', chart_fits='unchanged from parent'))
    payload['projection_mode'] = (recovery.SINGLE_REFERENCE_PROJECTION_MODE
        if parent.projection_mode == recovery.SINGLE_REFERENCE_PROJECTION_MODE
        else recovery.V7_PROJECTION_MODE)
    payload['spatial_boundary'] = None
    payload['temporal_boundary'] = None
    for row in payload['frames']:
        row['boundary_offset_native_y_px'] = 0.
        # Only geometry/boundary warnings are recomputed for the changed mapping.
        row['warnings'] = [w for w in row['warnings'] if w not in (
            'invalid_or_absent_visible_supported_geometry', 'outside_temporal_boundary_fit_support')]
    saved_by_index = {row['index']: row for row in payload['frames']}
    allowed = set(measurements['fit_frame_indices'])
    grouped = {}
    for ref in measurements['references'] + measurements.get('boundary_observations', []):
        index = ref['frame_index']
        if index not in allowed:
            raise ValueError('Boundary fitting attempted outside declared fit frames')
        frame = by_index[index]
        for key in ('source_pts', 'source_time_base', 'image_sha256'):
            if ref.get(key) != frame[key]:
                raise ValueError(f'Boundary exact frame binding absent or wrong: {key}')
        for feature in ref['features']:
            if feature['label'] == 'touch_far':
                p = recovery.points(feature['points_native'])
                if np.any(p < 0) or np.any(p >= frame['native_size']):
                    raise ValueError('Boundary samples outside native source image')
                grouped.setdefault(index, []).extend(p.tolist())
    boundary = [dict(frame_index=i, points_native=p,
                     reference_to_native=saved_by_index[i]['reference_to_native'])
                for i, p in sorted(grouped.items())]
    payload['spatial_boundary'] = recovery.fit_spatial_boundary(boundary, allowed)
    if payload['spatial_boundary']:
        payload['spatial_boundary']['weight_profile'] = chosen_profile
    spatial_atlas = recovery.RecoveryAtlas.from_payload(payload)
    world = np.c_[np.linspace(-recovery.HALF_L, recovery.HALF_L, 8001),
                  np.full(8001, -recovery.HALF_W)]
    temporal_observations, unsupported = [], []
    for row in boundary:
        frame = by_index[row['frame_index']]
        mapping = spatial_atlas.frame(frame['source_pts'], frame['source_time_base'],
                    source_sha256=manifest['source']['sha256'], native_size=frame['native_size'])
        curve = mapping.project(world)
        visible = np.isfinite(curve).all(axis=1) & (curve >= 0).all(axis=1) & (curve < frame['native_size']).all(axis=1)
        curve = curve[visible]
        # Use the visible monotone far boundary; never sort reflected branches together.
        if len(curve) < 10 or np.any(np.diff(curve[:, 0]) <= 0) or np.max(np.diff(curve[:, 0])) > 100:
            unsupported.append(dict(frame_index=frame['index'], reason='no continuous monotone visible far curve'))
            continue
        native = recovery.points(row['points_native'])
        selected = (native[:, 0] >= curve[0, 0]) & (native[:, 0] <= curve[-1, 0])
        if selected.sum() < 10:
            unsupported.append(dict(frame_index=frame['index'], reason='fewer than 10 in-domain boundary samples'))
            continue
        errors = native[selected, 1] - np.interp(native[selected, 0], curve[:, 0], curve[:, 1])
        med = float(np.median(errors))
        temporal_observations.append(dict(frame_index=frame['index'],
            time_s=float(frame['source_pts']*Fraction(frame['source_time_base'])),
            samples=len(errors), median_offset_px=med, mad_px=float(np.median(abs(errors-med)))))
    payload['temporal_boundary'] = recovery.fit_temporal_boundary(temporal_observations, allowed)
    for row in payload['frames']:
        if payload['temporal_boundary']:
            try:
                row['boundary_offset_native_y_px'] = recovery.temporal_offset(payload['temporal_boundary'],
                    float(row['source_pts']*Fraction(row['source_time_base'])))
            except ValueError:
                row['warnings'].append('outside_temporal_boundary_fit_support')
    payload['provenance'].update(parent_atlas_sha256=sha(parent_path),
        parent_measurements_sha256=measurements['parent_measurements_sha256'],
        measurements_sha256=sha(measurements_path), image_motion_recomputed=False,
        chart_fits_recomputed=False, refinement_helper_sha256=sha(__file__))
    evidence = payload['provenance'].setdefault('fitting_frames', [])
    evidence.extend(row for row in fitting_frame_bindings(manifest, grouped) if row not in evidence)
    atlas = recovery.RecoveryAtlas.from_payload(payload)
    gx, gy = np.meshgrid(np.linspace(-recovery.HALF_L, recovery.HALF_L, 181),
                         np.linspace(-recovery.HALF_W, recovery.HALF_W, 107))
    grid = np.c_[gx.ravel(), gy.ravel()]
    geometry = []
    for row in manifest['frames']:
        mapping = atlas.frame(row['source_pts'], row['source_time_base'],
            source_sha256=manifest['source']['sha256'], native_size=row['native_size'])
        check = recovery.visible_geometry_check(mapping, grid)
        geometry.append(dict(frame_index=row['index'], **check))
        if not check['valid']:
            saved_by_index[row['index']]['warnings'].append('invalid_or_absent_visible_supported_geometry')
    atlas = recovery.RecoveryAtlas.from_payload(payload)
    atlas.save(output/'atlas.json')
    atlas = recovery.RecoveryAtlas.load(output/'atlas.json')
    if render:
        (output/'overlay').mkdir()
        for i, row in enumerate(manifest['frames']):
            path = (frames_path.parent/row['file']).resolve()
            if not path.is_relative_to(frames_path.parent.resolve()) or sha(path) != row['image_sha256']:
                raise ValueError('Native source image path/hash mismatch')
            im = cv2.imread(str(path))
            mapping = atlas.frame(row['source_pts'], row['source_time_base'],
                source_sha256=manifest['source']['sha256'], native_size=row['native_size'])
            cv2.imwrite(str(output/'overlay'/f'{i+1:05d}.jpg'), recovery.render_frame(im, mapping),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            if i % 50 == 0:
                print(f'render refined saved mapping {i}/{len(manifest["frames"])}', flush=True)
    report = dict(accepted=False, status='approximate development refinement; independent checks pending',
        parent_atlas_sha256=sha(parent_path), atlas_sha256=sha(output/'atlas.json'),
        total_frames=len(payload['frames']), frames_with_warnings=sum(bool(r['warnings']) for r in payload['frames']),
        temporal_observations=temporal_observations, unsupported_boundary_observations=unsupported,
        geometry=geometry, image_motion_recomputed=False, chart_fits_recomputed=False,
        elapsed_seconds=time.monotonic()-started)
    write('report.json', report)
    print(json.dumps({k: report[k] for k in ('total_frames', 'frames_with_warnings', 'elapsed_seconds')}), flush=True)
    print('FIELD_RECOVERY_REFINEMENT_COMPLETE', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--frames', type=Path, required=True)
    parser.add_argument('--measurements', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--skip-render', action='store_true')
    parser.add_argument('--spatial-profile', choices=(recovery.SMOOTH_SPATIAL_WEIGHT, recovery.LINEAR_SPATIAL_WEIGHT))
    args = parser.parse_args()
    refine(args.parent, args.frames, args.measurements, args.out, render=not args.skip_render,
           spatial_profile=args.spatial_profile)


if __name__ == '__main__':
    main()
