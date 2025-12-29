import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import io
import chess
import chess.pgn
import numpy as np

from data.loadData import board_to_bitmap, game_to_training_pairs


def test_starting_position_bitmap():
    planes = board_to_bitmap(chess.STARTING_FEN)

    assert planes.shape == (12, 8, 8)
    assert planes.sum() == 32

    assert (planes[0][6] == 1).sum() == 8
    assert (planes[6][1] == 1).sum() == 8

    assert planes[1][7][1] == 1
    assert planes[1][7][6] == 1
    assert planes[2][7][2] == 1
    assert planes[2][7][5] == 1
    assert planes[3][7][0] == 1
    assert planes[3][7][7] == 1
    assert planes[4][7][3] == 1
    assert planes[5][7][4] == 1

    assert planes[7][0][1] == 1
    assert planes[7][0][6] == 1
    assert planes[8][0][2] == 1
    assert planes[8][0][5] == 1
    assert planes[9][0][0] == 1
    assert planes[9][0][7] == 1
    assert planes[10][0][3] == 1
    assert planes[11][0][4] == 1


def test_custom_position_one():
    fen = "8/8/8/4k3/8/4K3/8/8 w - - 0 1"
    planes = board_to_bitmap(fen)

    assert planes.sum() == 2
    assert planes[5][5][4] == 1
    assert planes[11][3][4] == 1


def test_custom_position_two():
    fen = "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1"
    planes = board_to_bitmap(fen)

    assert planes[0].sum() == 8
    assert planes[0][4][3] == 1
    assert planes[0][6][3] == 0
    assert planes[6].sum() == 8
    assert planes.sum() == 32


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
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    board = game.board()
    count = 0
    for move in game.mainline_moves():
        if board.turn == color:
            count += 1
        board.push(move)
    return count


def _assert_moves_are_legal(pairs, target_color: chess.Color, pgn_text: str):
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    board = game.board()
    i = 0
    for move in game.mainline_moves():
        if board.turn == target_color:
            planes, move_uci = pairs[i]
            assert planes.shape == (12, 8, 8)
            norm_board = board if target_color == chess.WHITE else board.mirror()
            assert chess.Move.from_uci(move_uci) in norm_board.legal_moves
            i += 1
        board.push(move)
    assert i == len(pairs)


def test_game_pairs_white_target_color():
    expected = _count_moves_for_color(PGN_TEXT, chess.WHITE)
    pairs = game_to_training_pairs(PGN_TEXT, target_color="white")

    assert isinstance(pairs, list)
    assert len(pairs) == expected

    planes0, move0 = pairs[0]
    assert planes0.shape == (12, 8, 8)
    assert planes0[0][6][4] == 1
    assert move0 == "e2e4"

    _assert_moves_are_legal(pairs, chess.WHITE, PGN_TEXT)


def test_game_pairs_black_target_color_mirrored():
    expected = _count_moves_for_color(PGN_TEXT, chess.BLACK)
    pairs = game_to_training_pairs(PGN_TEXT, target_color="black")

    assert isinstance(pairs, list)
    assert len(pairs) == expected

    planes0, move0 = pairs[0]
    assert planes0.shape == (12, 8, 8)
    assert move0 == "e2e4"
    assert planes0[0][6][4] == 1
    assert planes0.sum() == 32

    _assert_moves_are_legal(pairs, chess.BLACK, PGN_TEXT)


def test_game_pairs_target_by_player_name():
    pairs = game_to_training_pairs(PGN_TEXT, target_player_name="Anurag Senapaty")
    assert len(pairs) == 3
    _, first_move = pairs[0]
    assert first_move == "e2e4"


if __name__ == "__main__":
    test_starting_position_bitmap()
    test_custom_position_one()
    test_custom_position_two()
    test_game_pairs_white_target_color()
    test_game_pairs_black_target_color_mirrored()
    test_game_pairs_target_by_player_name()
    print("All tests passed")
