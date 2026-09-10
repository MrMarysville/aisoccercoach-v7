from types import SimpleNamespace

import pytest

from calibration.tools import check_environment


def test_pinned_distribution_names_versions_and_opencv_collision(monkeypatch):
    actual = check_environment.pinned_packages()
    assert actual["opencv-python-headless"] == "4.10.0.84"
    assert "opencv-python" not in actual
    monkeypatch.setattr(check_environment.metadata, "version", lambda name: "wrong")
    with pytest.raises(ValueError, match="requirements.txt"):
        check_environment.pinned_packages()
    monkeypatch.setattr(check_environment.metadata, "version", actual.__getitem__)
    monkeypatch.setattr(check_environment.metadata, "distributions", lambda: [
        SimpleNamespace(metadata={"Name": name})
        for name in ("opencv_python_headless", "opencv-python")])
    with pytest.raises(ValueError, match="only the pinned"):
        check_environment.pinned_packages()
