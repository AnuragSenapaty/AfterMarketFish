"""Pre-processing for Hikaru Nakamura PGN datasets (append to Magnus dataset)."""

from __future__ import annotations

import os
import glob
from typing import Optional, Tuple

import numpy as np
import chess.pgn

from .paths import resolve_repo_path
from .pgn_parser import game_to_training_pairs


def _load_existing_npz(path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not os.path.exists(path):
        return None, None
    d = np.load(path, allow_pickle=True)
    if "X" not in d or "y" not in d:
        raise ValueError(f"Existing dataset at {path!r} missing X/y arrays.")
    return d["X"], d["y"]


def process_nakamura_games_append(
    magnus_dir: str,
    nakamura_dir: str,
    target_player_name: str = "Nakamura,Hikaru",
    out_name: str = "magnus_dataset.npz",
    require_existing: bool = True,
) -> str:
    """
    Process Nakamura PGNs under *nakamura_dir* and append pairs to the existing dataset
    stored under *magnus_dir*/Processed/<out_name>.

    By default, requires that the existing dataset already exists (require_existing=True).
    """
    magnus_dir_abs = resolve_repo_path(magnus_dir)
    nakamura_dir_abs = resolve_repo_path(nakamura_dir)

    proc_dir = os.path.join(magnus_dir_abs, "Processed")
    os.makedirs(proc_dir, exist_ok=True)

    out_path = os.path.join(proc_dir, out_name)

    X0, y0 = _load_existing_npz(out_path)
    if require_existing and X0 is None:
        raise FileNotFoundError(
            f"Expected existing dataset at {out_path!r} but it was not found. "
            f"Run process_magnus_games() first."
        )

    raw_dir = os.path.join(nakamura_dir_abs, "Raw")
    pgn_paths = sorted(glob.glob(os.path.join(raw_dir, "*.pgn")))
    if not pgn_paths:
        pgn_paths = sorted(glob.glob(os.path.join(raw_dir, "**", "*.pgn"), recursive=True))

    if not pgn_paths:
        raise FileNotFoundError(f"No PGN files found under {raw_dir!r}")

    all_inputs = []
    all_moves: list[str] = []
    total_games = 0
    total_pairs = 0

    for pgn_file in pgn_paths:
        with open(pgn_file, "r", encoding="utf-8", errors="ignore") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break

                total_games += 1

                pairs = game_to_training_pairs(
                    game,
                    target_player_name=target_player_name,
                    target_color="black",
                )

                if not pairs:
                    continue

                for planes, move_uci in pairs:
                    all_inputs.append(planes.astype(np.uint8))
                    all_moves.append(move_uci)

                total_pairs += len(pairs)

    if not all_inputs:
        raise RuntimeError(f"No Nakamura training pairs generated from PGNs in {raw_dir!r}")

    X1 = np.stack(all_inputs, axis=0)
    y1 = np.array(all_moves, dtype=object)

    if X0 is None:
        X = X1
        y = y1
    else:
        X = np.concatenate([X0, X1], axis=0)
        y = np.concatenate([y0, y1], axis=0)

    np.savez_compressed(out_path, X=X, y=y)

    base_count = 0 if X0 is None else int(X0.shape[0])
    print(f"Loaded existing positions: {base_count}")
    print(f"Added Nakamura positions:  {int(X1.shape[0])}")
    print(f"Saved combined positions:  {int(X.shape[0])} -> {out_path}")
    print(f"Nakamura games parsed: {total_games}, pairs generated: {total_pairs}")

    return out_path
