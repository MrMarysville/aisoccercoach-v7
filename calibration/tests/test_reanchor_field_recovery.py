"""Synthetic drift, frozen evidence and failure coverage; no private footage."""
from copy import deepcopy
import json

import numpy as np
import pytest

from calibration import field_recovery as r
from calibration.alignment_trial import sha256_file
from calibration.tools import reanchor_field_recovery as reanchor
from calibration.tools.score_field_recovery import score, qualification_receipt
from calibration.tests.test_propagation_resume import inputs, invoke
from calibration.tools import propagate_field_recovery as raw_runner


def translation(x, y=0.):
    return np.array([[1., 0., x], [0., 1., y], [0., 0., 1.]])


def atlas_and_grid(times, boundary):
    payload = dict(status="approximate", metric_certified=False, source=dict(game_id="synthetic", sha256="a" * 64),
        field=r.FIELD, projection_mode=r.POLYNOMIAL_REFERENCE_PROJECTION_MODE,
        charts=[dict(chart="mid", reference_frame_index=0, field_to_reference=[[2, 0, 160], [0, 2, 90], [0, 0, 1]],
                     support_world_polygon=[[-40, -20], [40, -20], [40, 20], [-40, 20]])],
        reference_polynomial=dict(degree=2, coefficients_native_px=[[.2, -.1]] + [[0., 0.]] * 5,
                                  origin_native_px=[160., 90.], scale_native_px=320.),
        frames=[dict(index=i, source_pts=round(t * 1000), source_time_base="1/1000", image_sha256=str(i),
                     native_size=[320, 180], reference_to_native=translation(2 * abs(t - times[boundary])).tolist(),
                     boundary_offset_native_y_px=.3, warnings=[]) for i, t in enumerate(times)])
    xx, yy = np.meshgrid(np.linspace(-50, 50, 31), np.linspace(-30, 30, 21))
    return r.RecoveryAtlas.from_payload(payload), np.c_[xx.ravel(), yy.ravel()]


@pytest.mark.parametrize("backward", [False, True])
def test_known_drift_recovers_in_both_directions_without_changing_model_or_boundary(backward):
    times = np.arange(61) / 10 + 20
    boundary = len(times) - 1 if backward else 0
    atlas, grid = atlas_and_grid(times, boundary)
    original = deepcopy(atlas.document)
    observations = [np.eye(3) if i % 10 == 0 else None for i in range(len(times))]
    corrected, statuses, _ = reanchor.bounded_correction(atlas, atlas.payload["frames"], observations, times, boundary, grid)
    assert atlas.document == original and corrected[boundary] == atlas.payload["frames"][boundary]
    assert all(s["supported"] for s in statuses)
    assert max(abs(np.array(f["reference_to_native"])[0, 2]) for f in corrected) < 1.5
    assert max(s["bounds"]["max_px"] for s in statuses) <= 12
    assert all(f["boundary_offset_native_y_px"] == .3 for f in corrected)
    # The first direction is reference-to-native, not its inverse.
    actual = r.project(translation(10) @ translation(4), [[1., 2.]])
    np.testing.assert_allclose(actual, [[15., 2.]])


def test_missing_observations_limits_and_inherited_failures_keep_raw_maps_unsupported(monkeypatch):
    times = np.arange(101) / 10 + 20
    atlas, grid = atlas_and_grid(times, 0)
    observations = [np.eye(3) if i % 10 == 0 and i not in (30, 40, 50) else None for i in range(len(times))]
    atlas.payload["frames"][10]["warnings"] = ["failed_adjacent_motion_step_1"]
    corrected, statuses, _ = reanchor.bounded_correction(atlas, atlas.payload["frames"], observations, times, 0, grid)
    for i in (10, 35, 45, 55, 100):
        assert not statuses[i]["supported"]
        assert corrected[i]["reference_to_native"] == atlas.payload["frames"][i]["reference_to_native"]
    assert "inherited_mapping_warning" in statuses[10]["reasons"]
    assert "missing_temporal_registration_support" in statuses[35]["reasons"]
    assert statuses[-1]["bounds"]["max_px"] > 12
    intervals = reanchor.unsupported_intervals(statuses)
    assert len(intervals) >= 3 and intervals[-1]["last_source_pts"] == 30000
    empty, unsupported, _ = reanchor.bounded_correction(atlas, atlas.payload["frames"], [None] * len(times), times, 0, grid)
    assert not any(s["supported"] for s in unsupported)
    assert [f["reference_to_native"] for f in empty] == [f["reference_to_native"] for f in atlas.payload["frames"]]
    def singular(*args, **kwargs):
        raise np.linalg.LinAlgError("singular averaged correction")
    monkeypatch.setattr(r, "smooth_reference_drift", singular)
    _, failed, evidence = reanchor.bounded_correction(atlas, atlas.payload["frames"], observations, times, 0, grid)
    assert not any(s["supported"] for s in failed) and "singular" in evidence["failure"]


def freeze(path, schema, payload):
    path.write_text(json.dumps(dict(schema=schema, payload=payload, payload_sha256=r.digest(payload))))
    return path


def prepare_run(tmp_path, monkeypatch, *, resumed=False, v7=False):
    args = inputs(tmp_path)
    document = json.loads((tmp_path / "atlas.json").read_text())
    document["payload"]["charts"][0]["reference_frame_index"] = 0
    document["payload"]["frames"][0]["warnings"] = []
    if v7:
        payload = document["payload"]
        payload["projection_mode"] = r.V7_PROJECTION_MODE
        chart = payload["charts"][0]
        payload["charts"] = [dict(deepcopy(chart), chart=name,
            support_world_polygon=[[-r.HALF_L, -r.HALF_W], [r.HALF_L, -r.HALF_W],
                                   [r.HALF_L, r.HALF_W], [-r.HALF_L, r.HALF_W]])
            for name in ("left", "mid", "right")]
        payload["spatial_boundary"] = dict(coefficients=[90 - 2*r.HALF_W + .2, 0., 0.],
                                           x_origin_px=160., x_scale_px=320.)
        payload["temporal_boundary"] = r.fit_temporal_boundary([
            dict(frame_index=i, time_s=4. + i*.5, median_offset_px=.4, samples=100, mad_px=0.)
            for i in range(5)], range(5))
        payload["frames"][0]["boundary_offset_native_y_px"] = .4
        payload["frames"][0]["reference_to_native"] = [[1.02, .02, 7], [.01, .98, 2], [.0001, .0002, 1]]
    document["payload_sha256"] = r.digest(document["payload"])
    (tmp_path / "atlas.json").write_text(json.dumps(document))
    monkeypatch.setattr(r, "adjacent_motion", lambda *a: (translation(1), dict(image_registration_pass=True)))
    monkeypatch.setattr(r, "reference_features", lambda image, **kw: int(image[0, 0, 0]))
    monkeypatch.setattr(reanchor.propagation, "reference_features", lambda image, **kw: int(image[0, 0, 0]))
    monkeypatch.setattr(r, "register_reference", lambda a, b, size: (translation(b - a), dict(image_registration_pass=True,
        inlier_hull_fraction=.5, check_consistent_fraction=1., check_median_px=.1)))
    monkeypatch.setattr(raw_runner, "gauge_frame", lambda *a, **kw: (dict(measured=True, samples=50, median_px=0., p95_px=0.), {}))
    raw = tmp_path / "raw"
    if resumed:
        invoke(monkeypatch, args, raw, "--max-frames", "1")
        invoke(monkeypatch, args, raw, "--resume")
    else:
        invoke(monkeypatch, args, raw)
    return raw


def run_reanchor(tmp_path, raw, direction, output, references=None):
    frames = tmp_path / f"frames-{direction}/frames.json"
    manifest = json.loads(frames.read_text())
    split = tmp_path / f"split-{direction}.json"
    if not split.exists():
        freeze(split, "field-recovery-reanchor-split-v1", dict(source_sha256="a" * 64,
            frames_manifest_sha256=sha256_file(frames), frames=[{k: row[k] for k in (*reanchor.IDENTITY_KEYS, "index")}
                | dict(role="fit" if row["index"] in (0, 2) else "check") for row in manifest["frames"]]))
    return reanchor.run(tmp_path / "atlas.json", tmp_path / "clip/frames.json", raw, frames, split,
                        tmp_path / "plan.json", tmp_path / "catalog.json", direction, output, references_path=references)


@pytest.mark.parametrize("direction", ["forward", "backward"])
@pytest.mark.parametrize("v7", [False, True])
def test_resumed_and_uninterrupted_reanchor_equivalent_with_immutable_parent(tmp_path, monkeypatch, direction, v7):
    roots = [tmp_path / name for name in ("full", "resumed")]
    outputs = []
    for root, resume in zip(roots, (False, True)):
        root.mkdir()
        raw = prepare_run(root, monkeypatch, resumed=resume, v7=v7)
        parents = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in [root / "atlas.json", *raw.glob("progress-*/*.json")]}
        report = run_reanchor(root, raw, direction, root / "reanchored")
        assert report["boundary_mapping_unchanged"] and report["frame_count"] == 3
        for p, (data, mtime) in parents.items():
            assert p.read_bytes() == data and p.stat().st_mtime_ns == mtime
        atlas = r.RecoveryAtlas.load(root / "reanchored/atlas.json")
        assert not atlas.payload["metric_certified"]
        assert atlas.payload["provenance"]["parent_provenance"]["fit_frame_indices"] == [0]
        assert len(atlas.payload["provenance"]["fitting_frames"]) == 3
        connections = json.loads((root / "reanchored/reference-connections.json").read_text())
        parent = r.RecoveryAtlas.load(root / "atlas.json")
        assert {k: v for k, v in atlas.payload.items() if k not in ("frames", "provenance")} == {
            k: v for k, v in parent.payload.items() if k not in ("frames", "provenance")}
        if v7:
            assert connections["gauge"]["chart"] == "mid" and len(atlas.charts) == 3
            assert all(row["boundary_offset_native_y_px"] == .4 for row in atlas.payload["frames"])
            assert report["supported_frames"] == 3
        gauge = next(v for v in connections["views"] if v["manifest"] == "parent")
        for view in connections["views"]:
            assert view["connection_id"] == round((reanchor.identity(view) - reanchor.identity(gauge)) * 1_000_000_000)
            if direction == "backward" and view["manifest"] == "extension":
                assert view["connection_id"] < 0
        outputs.append(atlas.payload["frames"])
        with pytest.raises(FileExistsError):
            run_reanchor(root, raw, direction, root / "reanchored")
        changed = json.loads((raw / f"result-{direction}.json").read_text())
        changed["records"][0]["reference_to_native"][0][2] += 1
        (raw / f"result-{direction}.json").write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="checkpoint"):
            run_reanchor(root, raw, direction, root / "changed")
    assert outputs[0] == outputs[1]


def test_v7_boundary_time_support_and_missing_charts_cannot_claim_support(tmp_path, monkeypatch):
    prepare_run(tmp_path, monkeypatch, v7=True)
    parent = r.RecoveryAtlas.load(tmp_path / "atlas.json")
    grid = np.array([[x, y] for x in (-30., 0., 30.) for y in (-25., 0., 25.)])
    records = [dict(deepcopy(parent.payload["frames"][0]), index=i, source_pts=5000 + 1000*i)
               for i in range(3)]
    transforms = [r.matrix(row["reference_to_native"]) for row in records]
    corrected, statuses, _ = reanchor.bounded_correction(parent, records, transforms, [5., 6., 7.], 0, grid)
    assert corrected[0] == records[0]
    assert "outside_temporal_boundary_fit_support" in statuses[2]["reasons"]
    assert not statuses[2]["supported"] and corrected[2]["reference_to_native"] == records[2]["reference_to_native"]
    partial = deepcopy(parent.payload)
    partial["charts"].pop()
    partial["temporal_boundary"] = None
    _, statuses, _ = reanchor.bounded_correction(r.RecoveryAtlas.from_payload(partial), records, transforms,
                                                [5., 6., 7.], 0, grid)
    assert all(not row["supported"] for row in statuses)
    assert {"incomplete_v7_spatial_charts", "missing_v7_boundary_fit"} <= set(statuses[0]["reasons"])


def test_check_plan_keeps_missing_portions_ambiguous_groups_and_missing_maps(tmp_path):
    atlas, _ = atlas_and_grid([20., 20.1, 20.2, 20.3], 0)
    rows = [dict(row, source_seconds=float(reanchor.identity(row))) for row in atlas.payload["frames"]]
    manifest = dict(source=atlas.payload["source"], frames=rows)
    frames_path = tmp_path / "frames.json"
    frames_path.write_text(json.dumps(manifest))
    measurements, groups = [], []
    for i, label, status, ids in [(1, "halfway", "visible", ["a", "missing"]),
                                  (1, "touch_far", "ambiguous", []), (2, "halfway", "visible", ["b"])]:
        row = rows[i]
        groups.append(dict(frame_index=i, label=label, **{k: row[k] for k in reanchor.IDENTITY_KEYS},
                           observation_status=status, feature_ids=ids))
        if ids:
            pixels = r.RecoveryFrame(atlas, row).project([[0, -2], [0, 0], [0, 2]])
            measurements.append(dict(frame_index=i, label=label, feature_id=ids[0], selected_count=3,
                                     searched_count=3, points_native=pixels.tolist(), review_status="source_reviewed"))
    checks_path = tmp_path / "checks.json"
    checks_path.write_text(json.dumps(dict(source_sha256="a" * 64, frames_manifest_sha256=sha256_file(frames_path), measurements=measurements)))
    plan = dict(source_sha256="a" * 64, frames_manifest_sha256=sha256_file(frames_path),
                fit_frame_indices=[0, 3], check_frame_indices=[1, 2], groups=groups)
    plan_path = freeze(tmp_path / "plan.json", "field-recovery-check-plan-v1", plan)
    atlas.save(tmp_path / "atlas.json")
    old = score(tmp_path / "atlas.json", frames_path, checks_path, tmp_path / "legacy.json")
    assert old["marking_frame_count"] == 2 and old["all_marking_frames_pass"]
    frozen = score(tmp_path / "atlas.json", frames_path, checks_path, tmp_path / "frozen.json", check_plan=plan_path)
    assert frozen["marking_frame_count"] == 3 and frozen["passing_marking_frames"] == 1
    assert not frozen["all_marking_frames_pass"]
    data = json.loads((tmp_path / "frozen.json").read_text())
    assert data["missing_or_unmeasured_marking_frames"] == 2
    missing = deepcopy(atlas.payload)
    missing["frames"] = [row for row in missing["frames"] if row["index"] != 2]
    r.RecoveryAtlas.from_payload(missing).save(tmp_path / "missing-map.json")
    result = score(tmp_path / "missing-map.json", frames_path, checks_path, tmp_path / "missing-score.json", check_plan=plan_path)
    assert result["passing_marking_frames"] == 0
    reused = deepcopy(atlas.payload)
    reused["provenance"] = dict(fitting_frames=[rows[1]])
    r.RecoveryAtlas.from_payload(reused).save(tmp_path / "fitted.json")
    with pytest.raises(ValueError, match="fitting evidence"):
        score(tmp_path / "fitted.json", frames_path, checks_path, tmp_path / "invalid.json", check_plan=plan_path)


def test_reviewed_reference_switch_and_failed_bridge_remain_explicit(tmp_path, monkeypatch):
    raw = prepare_run(tmp_path, monkeypatch)
    run_reanchor(tmp_path, raw, "forward", tmp_path / "direct")
    path = tmp_path / "frames-forward/frames.json"
    row = json.loads(path.read_text())["frames"][-1]
    pixels = r.project(translation(2) @ np.array([[2, 0, 160], [0, 2, 90], [0, 0, 1]]), [[0, -10], [0, 0], [0, 10]])
    ref = dict(frame_index=2, **{k: row[k] for k in reanchor.IDENTITY_KEYS}, review_status="source_reviewed",
               required_labels=["halfway"], features=[dict(feature_id="fit", label="halfway", points_native=pixels.tolist())])
    references = freeze(tmp_path / "references.json", "field-recovery-reanchor-references-v1", dict(
        source_sha256="a" * 64, frames_manifest_sha256=sha256_file(path),
        split_sha256=sha256_file(tmp_path / "split-forward.json"), references=[ref]))
    report = run_reanchor(tmp_path, raw, "forward", tmp_path / "switch", references)
    evidence = json.loads((tmp_path / "switch/reference-connections.json").read_text())
    assert evidence["reference_paint_checks"][0]["approved"]
    assert report["reference_changes"] and not report["reference_changes"][0]["independent_neighbor_checks_pass"]
    empty = json.loads(references.read_text())["payload"]
    empty["references"][0]["features"][0]["points_native"] = []
    empty_path = freeze(tmp_path / "empty-reference.json", "field-recovery-reanchor-references-v1", empty)
    run_reanchor(tmp_path, raw, "forward", tmp_path / "empty-paint", empty_path)
    assert not json.loads((tmp_path / "empty-paint/reference-connections.json").read_text())["reference_paint_checks"][0]["approved"]
    ids = iter([10, 20, 30])
    monkeypatch.setattr(reanchor.propagation, "reference_features", lambda image, **kw: next(ids))
    def bridge(a, b, size):
        if (a, b) == (10, 30):
            return None, dict(image_registration_pass=False, reason="no direct overlap")
        return translation(2 if b == 30 else 0), dict(image_registration_pass=True,
            inlier_hull_fraction=.5, check_consistent_fraction=1., check_median_px=.1)
    monkeypatch.setattr(r, "register_reference", bridge)
    run_reanchor(tmp_path, raw, "forward", tmp_path / "bridge", references)
    evidence = json.loads((tmp_path / "bridge/reference-connections.json").read_text())
    assert evidence["reference_paint_checks"][0]["approved"]
    ids_by_frame = {v["index"]: v["connection_id"] for v in evidence["views"] if v["manifest"] == "extension"}
    assert evidence["paths"][str(ids_by_frame[2])]["path"] == [0, ids_by_frame[0], ids_by_frame[2]]
    monkeypatch.setattr(reanchor.propagation, "reference_features", lambda image, **kw: int(image[0, 0, 0]))
    monkeypatch.setattr(r, "register_reference", lambda a, b, size: (None, dict(image_registration_pass=False, reason="failed bridge")))
    failed = run_reanchor(tmp_path, raw, "forward", tmp_path / "failed", references)
    evidence = json.loads((tmp_path / "failed/reference-connections.json").read_text())
    assert not evidence["reference_paint_checks"][0]["approved"]
    assert failed["supported_frames"] == 0 and failed["unsupported_intervals"]


def test_qualification_requires_all_groups_geometry_transition_review_and_resume(tmp_path):
    binding = dict(atlas_sha256="atlas", frames_sha256="frames", source_sha256="source")
    score_data = dict(binding, check_plan_sha256="plan", checks_sha256="paint", required_marking_frames=2,
        passing_marking_frames=2, missing_or_unmeasured_marking_frames=0, pooled_diagnostic={}, elapsed_seconds=.1,
        rows=[dict(frame_index=i, experiment_pixel_target_pass=True, evidence_complete=True,
                   samples=3, finite_samples=3, median_px=1., p95_px=2.) for i in (1, 3)])
    report = dict(binding, direction="forward", frame_count=3, parent_inputs_unchanged=True, boundary_mapping_unchanged=True,
        elapsed_seconds=1., unsupported_intervals=[], reference_changes=[dict(required_neighbor_check_indices=[1, 3])],
        frames=[dict(source_pts=i, source_time_base="1/10", supported=True,
                     geometry=dict(valid=True, nonfinite_supported_projections=0), bounds=dict(valid=True, max_px=2.)) for i in range(3)])
    for name, value in (("score", score_data), ("report", report),
        ("review", dict(binding, full_sequence_reviewed=True, passed=True, reviewed_frame_count=3)),
        ("equivalence", dict(binding, raw_maps_equal=True, corrected_maps_equal=True, parent_artifacts_unchanged=True))):
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    def qualify(name, **kwargs):
        return qualification_receipt(tmp_path / "score.json", tmp_path / "report.json", tmp_path / f"{name}.json", **kwargs)
    missing = qualify("missing")
    assert not missing["pixel_qualified"]
    assert not missing["gates"]["full_sequence_visual_review_pass"] and not missing["gates"]["resume_equivalence_pass"]
    args = dict(visual_review_path=tmp_path / "review.json", equivalence_path=tmp_path / "equivalence.json")
    passed = qualify("passed", **args)
    assert passed["pixel_qualified"] and not passed["metric_certified"] and not passed["full_game_accepted"]
    report["reference_changes"][0]["required_neighbor_check_indices"] = [1, 2]
    (tmp_path / "report.json").write_text(json.dumps(report))
    assert not qualify("missing-neighbor", **args)["pixel_qualified"]
    report["frames"][1]["geometry"]["valid"] = False
    (tmp_path / "report.json").write_text(json.dumps(report))
    assert not qualify("geometry", **args)["gates"]["supported_geometry_pass"]
    score_data["frames_sha256"] = "changed"
    (tmp_path / "score.json").write_text(json.dumps(score_data))
    with pytest.raises(ValueError, match="matching"):
        qualify("different-source", **args)
