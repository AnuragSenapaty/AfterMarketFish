# tests/test_bitboards.py
import chess
import numpy as np

from aftermarketfish.bitboards import board_to_bitmap


def test_starting_position_bitmap():
    planes = board_to_bitmap(chess.STARTING_FEN)

    assert isinstance(planes, np.ndarray)
    assert planes.shape == (12, 8, 8)
    assert planes.sum() == 32

    # White pawns on rank 2 -> rank index 1
    assert planes[0, 1].sum() == 8
    # Black pawns on rank 7 -> rank index 6
    assert planes[6, 6].sum() == 8

    # White major pieces on rank 1 -> rank index 0
    assert planes[1, 0, 1] == 1  # white knight b1
    assert planes[1, 0, 6] == 1  # white knight g1
    assert planes[2, 0, 2] == 1  # white bishop c1
    assert planes[2, 0, 5] == 1  # white bishop f1
    assert planes[3, 0, 0] == 1  # white rook a1
    assert planes[3, 0, 7] == 1  # white rook h1
    assert planes[4, 0, 3] == 1  # white queen d1
    assert planes[5, 0, 4] == 1  # white king e1

    # Black major pieces on rank 8 -> rank index 7
    assert planes[7, 7, 1] == 1   # black knight b8
    assert planes[7, 7, 6] == 1   # black knight g8
    assert planes[8, 7, 2] == 1   # black bishop c8
    assert planes[8, 7, 5] == 1   # black bishop f8
    assert planes[9, 7, 0] == 1   # black rook a8
    assert planes[9, 7, 7] == 1   # black rook h8
    assert planes[10, 7, 3] == 1  # black queen d8
    assert planes[11, 7, 4] == 1  # black king e8


def test_custom_position_one_two_kings():
    fen = "8/8/8/4k3/8/4K3/8/8 w - - 0 1"
    planes = board_to_bitmap(fen)

    assert planes.sum() == 2

    # White king at e3: file=4 rank=3 -> rank index 2
    assert planes[5, 2, 4] == 1
    # Black king at e5: file=4 rank=5 -> rank index 4
    assert planes[11, 4, 4] == 1


def test_custom_position_two_pawn_advanced():
    fen = "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1"
    planes = board_to_bitmap(fen)

    # White pawn count still 8
    assert planes[0].sum() == 8
    # Pawn moved to d4: file=3, rank=4 -> rank index 3
    assert planes[0, 3, 3] == 1
    # Pawn is no longer on d2: file=3, rank=2 -> rank index 1
    assert planes[0, 1, 3] == 0

    assert planes[6].sum() == 8  # black pawns
    assert planes.sum() == 32
