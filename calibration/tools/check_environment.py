"""Validate the pinned Python distributions before creating experiment outputs."""
from importlib import metadata
import json
from pathlib import Path
import re


def pinned_packages():
    requirements = Path(__file__).resolve().parents[2] / "requirements.txt"
    pins = dict(line.split("==") for raw in requirements.read_text().splitlines()
                if (line := raw.split("#", 1)[0].strip()))
    installed = {name: metadata.version(name) for name in pins}
    if installed != pins:
        raise ValueError(f"Installed versions differ from requirements.txt: {installed}")
    # All four wheel variants install cv2; co-installing them overwrites files.
    variants = {"opencv-python", "opencv-python-headless", "opencv-contrib-python",
                "opencv-contrib-python-headless"}
    present = {re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower()
               for dist in metadata.distributions()}
    if present & variants != {"opencv-python-headless"}:
        raise ValueError("Install only the pinned opencv-python-headless distribution")
    return installed


if __name__ == "__main__":
    print(json.dumps(pinned_packages(), indent=2))
