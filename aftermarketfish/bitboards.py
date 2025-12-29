"""Bitboard / bitmap utilities.

The network takes a 12×8×8 bitmap representation of the board:
- 6 piece types × 2 colours = 12 planes
- Each plane is an 8×8 board with 1s where that piece is present.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import chess

# Mapping of (piece_type, is_white) → channel index
PIECE_PLANES: Dict[Tuple[int, bool], int] = {
    (chess.PAWN,   True):  0,
    (chess.KNIGHT, True):  1,
    (chess.BISHOP, True):  2,
    (chess.ROOK,   True):  3,
    (chess.QUEEN,  True):  4,
    (chess.KING,   True):  5,
    (chess.PAWN,   False): 6,
    (chess.KNIGHT, False): 7,
    (chess.BISHOP, False): 8,
    (chess.ROOK,   False): 9,
    (chess.QUEEN,  False): 10,
    (chess.KING,   False): 11,
}


def board_to_bitmap(board_or_fen: chess.Board | str) -> np.ndarray:
    """Convert a board (or FEN string) to a 12×8×8 numpy array of 0/1."""
    if isinstance(board_or_fen, str):
        board = chess.Board(board_or_fen)
    else:
        board = board_or_fen

    planes = np.zeros((12, 8, 8), dtype=np.uint8)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        key = (piece.piece_type, piece.color)
        plane_idx = PIECE_PLANES.get(key)
        if plane_idx is None:
            continue
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        planes[plane_idx, rank, file] = 1

    return planes


def _mirror_square(sq: chess.Square) -> chess.Square:
    """Mirror a square vertically (along the horizontal axis)."""
    rank = chess.square_rank(sq)
    file = chess.square_file(sq)
    mirrored_rank = 7 - rank
    return chess.square(file, mirrored_rank)


def mirror_move(move: chess.Move) -> chess.Move:
    """Mirror a move vertically (for black perspective training)."""
    new_from = _mirror_square(move.from_square)
    new_to = _mirror_square(move.to_square)
    return chess.Move(new_from, new_to, promotion=move.promotion)
