"""CLI wrapper for preprocessing Magnus PGN files into an NPZ dataset."""

from __future__ import annotations

import argparse

from aftermarketfish.magnus_preprocess import process_magnus_games


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Magnus games to NPZ dataset")
    parser.add_argument(
        "--magnus_dir",
        type=str,
        default="data/Magnus",
        help="Directory containing Raw/ and Processed/ subdirectories",
    )
    args = parser.parse_args()
    process_magnus_games(args.magnus_dir)


if __name__ == "__main__":
    main()
