# tests/test_magnus_preprocess.py
from pathlib import Path
import numpy as np

from aftermarketfish.magnus_preprocess import process_magnus_games


def test_process_magnus_games_creates_npz(tmp_path: Path):
    magnus_dir = tmp_path / "Magnus"
    raw_dir = magnus_dir / "Raw"
    proc_dir = magnus_dir / "Processed"
    raw_dir.mkdir(parents=True)
    proc_dir.mkdir(parents=True)

    # Minimal PGN with a few moves. We don't need Magnus specifically because
    # process_magnus_games passes target_color="black" anyway.
    pgn_text = """[Event "Test"]
[Site "Local"]
[Date "2025.08.19"]
[Round "1"]
[White "Someone"]
[Black "Carlsen,Magnus"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 *
"""
    (raw_dir / "game1.pgn").write_text(pgn_text, encoding="utf-8")

    out_path = process_magnus_games(str(magnus_dir))
    out = Path(out_path)
    assert out.exists()
    assert out.name == "magnus_dataset.npz"

    data = np.load(out, allow_pickle=True)
    assert "X" in data and "y" in data
    X = data["X"]
    y = data["y"]
    assert X.ndim == 4 and X.shape[1:] == (12, 8, 8)
    assert len(X) == len(y)
    assert len(X) > 0
