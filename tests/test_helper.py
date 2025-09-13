import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data.loadData import board_to_bitmap, game_to_training_pairs
import chess
import chess.pgn
import numpy as np
import io

def test_starting_position_bitmap():
    """Test that the bitmap of the starting position is correct in shape and piece count."""
    planes = board_to_bitmap(chess.STARTING_FEN)

    assert planes.shape == (12, 8, 8), "Bitmap should be 12x8x8"
    assert planes.sum() == 32, "Total pieces should be 32"

    # White pawns on rank 2 (row index 6)
    assert (planes[0][6] == 1).sum() == 8, "White pawns should be on rank 2"
    # Black pawns on rank 7 (row index 1)
    assert (planes[6][1] == 1).sum() == 8, "Black pawns should be on rank 7"

    # White back rank
    assert planes[1][7][1] == 1  # Knight b1
    assert planes[1][7][6] == 1  # Knight g1
    assert planes[2][7][2] == 1  # Bishop c1
    assert planes[2][7][5] == 1  # Bishop f1
    assert planes[3][7][0] == 1  # Rook a1
    assert planes[3][7][7] == 1  # Rook h1
    assert planes[4][7][3] == 1  # Queen d1
    assert planes[5][7][4] == 1  # King e1

    # Black back rank
    assert planes[7][0][1] == 1  # Knight b8
    assert planes[7][0][6] == 1  # Knight g8
    assert planes[8][0][2] == 1  # Bishop c8
    assert planes[8][0][5] == 1  # Bishop f8
    assert planes[9][0][0] == 1  # Rook a8
    assert planes[9][0][7] == 1  # Rook h8
    assert planes[10][0][3] == 1 # Queen d8
    assert planes[11][0][4] == 1 # King e8

    print("✅ test_starting_position_bitmap passed")

def test_custom_position_one():
    """Test known custom position (just kings)."""
    fen = "8/8/8/4k3/8/4K3/8/8 w - - 0 1"
    planes = board_to_bitmap(fen)

    assert planes.sum() == 2, "Only 2 kings should be present"
    assert planes[5][5][4] == 1  # White King e3
    assert planes[11][3][4] == 1 # Black King e5

    print("✅ test_custom_position_one passed")

def test_custom_position_two():
    """Test position after 1.d4 (white pawn moved from d2 to d4)."""
    fen = "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1"
    planes = board_to_bitmap(fen)

    assert planes[0].sum() == 8  # total white pawns unchanged
    assert planes[0][4][3] == 1  # pawn on d4
    assert planes[0][6][3] == 0  # pawn no longer on d2
    assert planes[6].sum() == 8  # black pawns unchanged
    assert planes.sum() == 32, "Still 32 pieces"

    print("✅ test_custom_position_two passed")

# ---------- Game → training pairs tests ----------

PGN_TEXT = """[Event "Test"]
[Site "Local"]
[Date "2025.08.19"]
[Round "1"]
[White "Anurag Senapaty"]
[Black "Opponent"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 *
"""

def _count_moves_for_color(pgn_text: str, color: chess.Color) -> int:
    """Utility: count how many plies the given color played in the main line."""
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    board = game.board()
    count = 0
    node = game
    while node.variations:
        move = node.variation(0).move
        if board.turn == color:
            count += 1
        board.push(move)
        node = node.variation(0)
    return count

def _assert_moves_are_legal(pairs, target_color: chess.Color):
    """Sanity: each move should be legal in the provided input position."""
    # Rebuild a board that always puts target as the mover in the same normalization
    game = chess.pgn.read_game(io.StringIO(PGN_TEXT))
    board = game.board()
    idx = 0
    node = game
    while node.variations:
        move = node.variation(0).move
        if board.turn == target_color:
            planes, move_uci = pairs[idx]
            # quick shape check
            assert planes.shape == (12, 8, 8)
            # verify legality in the un-normalized board, accounting for mirroring for black:
            if target_color == chess.BLACK:
                # pairs are mirrored; un-mirror to compare with actual board
                from_sq = chess.square_mirror(chess.parse_square(move_uci[:2]))
                to_sq   = chess.square_mirror(chess.parse_square(move_uci[2:]))
                check_move = chess.Move(from_sq, to_sq, promotion=None)
            else:
                check_move = chess.Move.from_uci(move_uci)
            assert check_move in board.legal_moves, f"Illegal move in pair {idx}: {move_uci}"
            idx += 1
        board.push(move)
        node = node.variation(0)
    # length should match
    assert idx == len(pairs)

def test_game_pairs_white_target_color():
    """White is target by color. Expect 3 pairs (white moves) and first move e2e4."""
    expected = _count_moves_for_color(PGN_TEXT, chess.WHITE)  # should be 3
    pairs = game_to_training_pairs(PGN_TEXT, target_color="white")
    assert isinstance(pairs, list)
    assert len(pairs) == expected, f"White made {expected} moves in this game fragment"

    planes0, move0 = pairs[0]
    assert planes0.shape == (12, 8, 8)
    assert planes0[0][6][4] == 1, "White pawn should be at e2 before e4"
    assert move0 == "e2e4"

    _assert_moves_are_legal(pairs, chess.WHITE)

def test_game_pairs_black_target_color_mirrored():
    """Black is target by color. Expect 3 pairs (black moves) and first mirrored move e2e4 (from e7e5)."""
    expected = _count_moves_for_color(PGN_TEXT, chess.BLACK)  # should be 3
    pairs = game_to_training_pairs(PGN_TEXT, target_color="black")
    assert isinstance(pairs, list)
    assert len(pairs) == expected, f"Black made {expected} moves in this game fragment"

    planes0, move0 = pairs[0]
    assert planes0.shape == (12, 8, 8)
    assert move0 == "e2e4", "Black's first move e7e5 should mirror to e2e4"
    assert planes0[0][6][4] == 1, "Mirrored view should show target pawn at e2 before moving"
    assert planes0.sum() == 32

    _assert_moves_are_legal(pairs, chess.BLACK)

def test_game_pairs_target_by_player_name():
    """Target using the PGN header name; should resolve to White in this PGN."""
    pairs = game_to_training_pairs(PGN_TEXT, target_player_name="Anurag Senapaty")
    assert len(pairs) == 3
    _, first_move = pairs[0]
    assert first_move == "e2e4"

if __name__ == "__main__":
    # Manual runs without pytest
    test_starting_position_bitmap()
    test_custom_position_one()
    test_custom_position_two()
    test_game_pairs_white_target_color()
    test_game_pairs_black_target_color_mirrored()
    test_game_pairs_target_by_player_name()
    print("✅ All tests passed (manual run)")
