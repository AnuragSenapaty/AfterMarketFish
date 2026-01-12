"""CLI wrapper to append Nakamura data onto the Magnus dataset NPZ."""

from __future__ import annotations

import argparse

from aftermarketfish.nakamura_preprocess import process_nakamura_games_append


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--magnus_dir", type=str, default="data/Magnus", help="Magnus base dir (has Processed/)")
    ap.add_argument("--nakamura_dir", type=str, default="data/Nakamura", help="Nakamura base dir (has Raw/)")
    ap.add_argument("--player_name", type=str, default="Nakamura,Hikaru", help="Player name as in PGN headers")
    ap.add_argument("--out_name", type=str, default="magnus_dataset.npz", help="Output NPZ filename in Magnus/Processed/")
    args = ap.parse_args()

    process_nakamura_games_append(
        magnus_dir=args.magnus_dir,
        nakamura_dir=args.nakamura_dir,
        target_player_name=args.player_name,
        out_name=args.out_name,
        require_existing=True,
    )


if __name__ == "__main__":
    main()
