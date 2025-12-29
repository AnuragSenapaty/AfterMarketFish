"""Pre-processing for Magnus Carlsen PGN datasets."""

from __future__ import annotations

import os
import glob
from typing import Optional

import numpy as np
import chess.pgn

from .paths import resolve_repo_path
from .pgn_parser import game_to_training_pairs


def process_magnus_games(magnus_dir: str, target_player_name: str = "Carlsen,Magnus") -> str:
    """Process Magnus PGNs under *magnus_dir* and create magnus_dataset.npz."""
    magnus_dir_abs = resolve_repo_path(magnus_dir)

    raw_dir = os.path.join(magnus_dir_abs, "Raw")
    proc_dir = os.path.join(magnus_dir_abs, "Processed")
    os.makedirs(proc_dir, exist_ok=True)

    pgn_paths = sorted(glob.glob(os.path.join(raw_dir, "*.pgn")))
    if not pgn_paths:
        pgn_paths = sorted(glob.glob(os.path.join(raw_dir, "**", "*.pgn"), recursive=True))

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
        raise RuntimeError(f"No training pairs generated from PGNs in {raw_dir!r}")

    X = np.stack(all_inputs, axis=0)
    y = np.array(all_moves, dtype=object)

    out_path = os.path.join(proc_dir, "magnus_dataset.npz")
    np.savez_compressed(out_path, X=X, y=y)

    print(f"Saved {X.shape[0]} positions to {out_path}")
    print(f"Games parsed: {total_games}, pairs generated: {total_pairs}")

    return out_path
