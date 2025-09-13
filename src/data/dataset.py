# src/data/preprocess.py
import os
import glob
import numpy as np
import chess.pgn

# local import (same folder)
import loadData  # has game_to_training_pairs

def process_magnus_games(magnus_dir: str, target_player_name="Carlsen,Magnus"):
    """
    Read all PGNs under <project_root>/<magnus_dir>/Raw and save processed data under .../Processed.
    Example call: process_magnus_games("Data/Magnus")
    """
    # Resolve repo root from this file's location: .../AfterMarketFish
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    magnus_dir_abs = os.path.join(project_root, magnus_dir)
    raw_dir  = os.path.join(magnus_dir_abs, "Raw")
    proc_dir = os.path.join(magnus_dir_abs, "Processed")
    os.makedirs(proc_dir, exist_ok=True)

    print(f"[INFO] cwd           : {os.getcwd()}")
    print(f"[INFO] project_root  : {project_root}")
    print(f"[INFO] magnus_dir_abs: {magnus_dir_abs}")
    print(f"[INFO] raw_dir       : {raw_dir}")
    print(f"[INFO] proc_dir      : {proc_dir}")
    print(f"[INFO] raw_dir exists? {os.path.isdir(raw_dir)}")

    # Find PGNs (non-recursive first; then recursive as fallback)
    pgn_paths = sorted(glob.glob(os.path.join(raw_dir, "*.pgn")))
    if not pgn_paths:
        pgn_paths = sorted(glob.glob(os.path.join(raw_dir, "**", "*.pgn"), recursive=True))
    print(f"[INFO] Found {len(pgn_paths)} PGN file(s)")

    all_inputs = []
    all_moves  = []
    total_games = 0
    total_pairs = 0

    for pgn_file in pgn_paths:
        print(f"[PROCESS] {pgn_file}")
        with open(pgn_file, "r", encoding="utf-8", errors="ignore") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                total_games += 1
                exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
                pgn_text = game.accept(exporter)

                pairs = loadData.game_to_training_pairs(
                    pgn_text,
                    target_player_name=target_player_name,
                    target_color="black"  # fallback if name doesn't match
                )
                if not pairs:
                    continue

                for planes, move_uci in pairs:
                    all_inputs.append(planes.astype(np.uint8))
                    all_moves.append(move_uci)
                total_pairs += len(pairs)

    if not all_inputs:
        print("[WARN] No training pairs produced.")
        print("       Check paths and capitalization, and ensure PGNs are in <magnus_dir>/Raw.")
        print(f"       Looked in: {raw_dir}")
        return ""

    X = np.stack(all_inputs)   # (N, 12, 8, 8)
    y = np.array(all_moves)    # (N,)
    out_path = os.path.join(proc_dir, "magnus_dataset.npz")
    np.savez_compressed(out_path, X=X, y=y)
    print(f"[DONE] Saved {X.shape[0]} positions to {out_path} (X: {X.shape}, y: {y.shape})")
    print(f"[STATS] games parsed: {total_games}, pairs generated: {total_pairs}")
    return out_path

if __name__ == "__main__":
    # Example manual run: python -m src.data.preprocess Data/Magnus
    import sys
    magnus_dir = sys.argv[1] if len(sys.argv) > 1 else "data/Magnus"
    process_magnus_games(magnus_dir)
