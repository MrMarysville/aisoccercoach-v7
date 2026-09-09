"""Query approximate field coordinates from an exact, source-bound native frame."""
import argparse
import json

from calibration.field_atlas import FieldAtlas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("atlas")
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--pts", type=int, required=True)
    parser.add_argument("--time-base", required=True)
    parser.add_argument("--native-size", type=int, nargs=2, required=True, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--pixel", type=float, nargs=2, action="append", required=True, metavar=("X", "Y"))
    args = parser.parse_args()
    atlas = FieldAtlas.load(args.atlas)
    frame = atlas.frame(args.pts, args.time_base, source_sha256=args.source_sha256,
                        native_size=args.native_size)
    print(json.dumps(dict(source=atlas.source, source_pts=args.pts, time_base=args.time_base,
                          metric_certified=False, results=frame.locate(args.pixel)), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
