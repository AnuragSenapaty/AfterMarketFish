"""PGN → training pair conversion.

A training pair is:
- X: a 12×8×8 bitboard tensor of the position
- y: the UCI move string played from that position
"""

from __future__ import annotations

import io
from typing import List, Tuple, Optional, Literal

import numpy as np
import chess
import chess.pgn

from .bitboards import board_to_bitmap, mirror_move

ColorLiteral = Literal["white", "black"]


def resolve_target_color(
    game: chess.pgn.Game,
    target_player_name: Optional[str],
    target_color: Optional[ColorLiteral],
) -> chess.Color:
    """Decide which colour we are training for in this game."""
    if target_color is not None:
        if target_color.lower() == "white":
            return chess.WHITE
        if target_color.lower() == "black":
            return chess.BLACK
        raise ValueError(f"Unknown target_color: {target_color!r}")

    if target_player_name:
        white_name = (game.headers.get("White") or "").strip()
        black_name = (game.headers.get("Black") or "").strip()
        if white_name == target_player_name:
            return chess.WHITE
        if black_name == target_player_name:
            return chess.BLACK

    # Fallback if we can't match anything
    return chess.WHITE


def game_to_training_pairs(
    pgn_or_game: str | chess.pgn.Game,
    target_player_name: Optional[str] = None,
    target_color: Optional[ColorLiteral] = None,
) -> List[Tuple[np.ndarray, str]]:
    """Convert a PGN string or Game to (planes, move_uci) pairs."""
    if isinstance(pgn_or_game, chess.pgn.Game):
        game = pgn_or_game
    else:
        game = chess.pgn.read_game(io.StringIO(pgn_or_game))

    if game is None:
        return []

    tgt_color = resolve_target_color(game, target_player_name, target_color)
    board = game.board()
    pairs: List[Tuple[np.ndarray, str]] = []

    for move in game.mainline_moves():
        if board.turn == tgt_color:
            if tgt_color == chess.WHITE:
                planes = board_to_bitmap(board)
                move_uci = move.uci()
            else:
                planes = board_to_bitmap(board.mirror())
                move_uci = mirror_move(move).uci()
            pairs.append((planes, move_uci))
        board.push(move)

    return pairs
