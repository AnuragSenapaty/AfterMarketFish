import sys, os, io, glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data.loadData import game_to_training_pairs, board_to_bitmap, mirror_move
import chess
import chess.pgn
import numpy as np

# ---------- helpers ----------

def _magnus_raw_dir():
    # You said you put the file in Data/magnus/raw (capital D).
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'Magnus', 'Raw'))

def _first_pgn_path():
    base = _magnus_raw_dir()
    matches = sorted(glob.glob(os.path.join(base, "*.pgn")))
    assert matches, f"No PGN found under {base}"
    return matches[0]

def _read_first_game_text(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        game = chess.pgn.read_game(f)
    assert game is not None, f"Could not read a game from {path}"
    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
    return game.accept(exporter)

def _first_move_for_color(pgn_text: str, color: chess.Color) -> chess.Move:
    """Return the first actual move (as a chess.Move) played by `color` in the main line."""
    g = chess.pgn.read_game(io.StringIO(pgn_text))
    b = g.board()
    node = g
    while node.variations:
        mv = node.variation(0).move
        if b.turn == color:
            return mv
        b.push(mv)
        node = node.variation(0)
    raise AssertionError(f"No move found for color={color}")

def _uci_of_mirrored_move(mv: chess.Move) -> str:
    return mirror_move(mv).uci()

def _assert_all_pairs_legal(pgn_text: str, pairs, target_color: chess.Color):
    """Every (planes, move_uci) must be legal in the corresponding un-normalized board position."""
    g = chess.pgn.read_game(io.StringIO(pgn_text))
    b = g.board()
    idx = 0
    node = g
    while node.variations:
        mv = node.variation(0).move
        if b.turn == target_color:
            planes, uci = pairs[idx]
            assert isinstance(planes, np.ndarray) and planes.shape == (12, 8, 8)
            if target_color == chess.BLACK:
                # un-mirror to compare with the true board
                from_sq = chess.square_mirror(chess.parse_square(uci[:2]))
                to_sq   = chess.square_mirror(chess.parse_square(uci[2:4]))
                promo = None
                if len(uci) > 4:
                    promo_map = {'q': chess.QUEEN, 'r': chess.ROOK, 'b': chess.BISHOP, 'n': chess.KNIGHT}
                    promo = promo_map.get(uci[4].lower(), None)
                chk = chess.Move(from_sq, to_sq, promotion=promo)
            else:
                chk = chess.Move.from_uci(uci)
            assert chk in b.legal_moves, f"Illegal pair {idx}: {uci}"
            idx += 1
        b.push(mv)
        node = node.variation(0)
    assert idx == len(pairs), "Pair count mismatch vs traversed moves"

# ---------- tests ----------

def test_magnus_first_game_black_mirroring_dynamic():
    """
    Load the first PGN under Data/magnus/raw, target Magnus (Black), and verify:
      - pair count equals number of Black moves,
      - first pair's move equals the MIRROR of the game's first Black move,
      - mirrored input shows the correct piece on the from-square,
      - every paired move is legal.
    """
    pgn_path = _first_pgn_path()
    pgn_text = _read_first_game_text(pgn_path)

    # Compute the expected first mirrored UCI directly from the PGN
    first_black_move = _first_move_for_color(pgn_text, chess.BLACK)     # e.g., g8f6
    expected_first_uci = _uci_of_mirrored_move(first_black_move)        # e.g., b1c3

    # Build pairs with Magnus as target by name, fallback color black
    pairs = game_to_training_pairs(
        pgn_text,
        target_player_name="Carlsen,Magnus",
        target_color="black"
    )

    # Count expected black moves in the PGN
    # (we can re-walk the game quickly)
    g = chess.pgn.read_game(io.StringIO(pgn_text))
    b = g.board()
    expected_black = 0
    node = g
    while node.variations:
        mv = node.variation(0).move
        if b.turn == chess.BLACK: expected_black += 1
        b.push(mv); node = node.variation(0)

    assert len(pairs) == expected_black, f"Expected {expected_black} black moves, got {len(pairs)}"

    # First pair assertions
    planes0, move0 = pairs[0]
    assert move0 == expected_first_uci, f"Expected mirrored first black move {expected_first_uci!r}, got {move0!r}"
    assert planes0.shape == (12, 8, 8)
    # Moving piece should be present on the mirrored from-square
    # Decode expected_from in mirrored coords to rank/file indices
    from_sq = chess.parse_square(move0[:2])       # e.g., 'b1'
    file_idx = chess.square_file(from_sq)         # a=0..h=7
    rank_idx = 7 - chess.square_rank(from_sq)     # our planes index 0=rank8
    # We don't know which channel (knight/pawn/etc) a priori, but at least one "white" channel (0..5) must have a 1 there
    assert int(planes0[0:6, rank_idx, file_idx].sum()) == 1, "Mirrored input should show exactly one white piece on the from-square"
    assert planes0.sum() >= 2, "Sanity: at least 2 pieces on the board"

    # All pairs legal
    _assert_all_pairs_legal(pgn_text, pairs, chess.BLACK)

if __name__ == "__main__":
    test_magnus_first_game_black_mirroring_dynamic()
    print("✅ Magnus PGN test passed (manual run)")
