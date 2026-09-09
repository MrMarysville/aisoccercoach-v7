"""Local JSON interface for authoritative manual field-adjustment sessions."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from calibration.field_adjustment import AdjustmentError, AdjustmentStore, prepare


class JsonParser(argparse.ArgumentParser):
    def error(self, message):
        raise AdjustmentError(message, "invalid_arguments")


def main():
    parser = JsonParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("prepare")
    for name in ("atlas", "frames", "out", "id", "title"):
        command.add_argument("--" + name, required=True)
    command.add_argument("--video-name")
    for name in ("state", "save", "locate"):
        commands.add_parser(name).add_argument("--root", required=True)
    try:
        args = parser.parse_args()
        if args.command == "prepare":
            result = prepare(args.atlas, args.frames, args.out, args.id, args.title, args.video_name)
        else:
            store = AdjustmentStore(args.root)
            if args.command == "state":
                result = store.state()
            else:
                request = json.load(sys.stdin)
                if not isinstance(request, dict):
                    raise AdjustmentError("Expected a JSON object")
                result = getattr(store, args.command)(request)
        print(json.dumps(result, allow_nan=False))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(json.dumps(dict(error=str(exc), code=getattr(exc, "code", "invalid_request"))))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
