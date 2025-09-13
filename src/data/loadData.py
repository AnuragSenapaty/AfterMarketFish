import numpy as np
import chess
import io
from typing import List, Tuple, Optional, Literal
import chess.pgn

PIECE_PLANES = {
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
    (chess.QUEEN,  False):10,
    (chess.KING,   False):11,
}

def board_to_bitmap(x):
    board = chess.Board(x) if isinstance(x, str) else x
    planes = np.zeros((12, 8, 8), dtype=np.uint8)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        ch = PIECE_PLANES[(piece.piece_type, piece.color)]
        rank = 7 - chess.square_rank(square)
        file = chess.square_file(square)
        planes[ch, rank, file] = 1

    return planes

def mirror_move(move: chess.Move) -> chess.Move:
    """Return the move mirrored (rotate board 180° and swap colors)."""
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion  # piece type id is color-agnostic
    )

def resolve_target_color(game: chess.pgn.Game,
                         target_player_name: Optional[str],
                         fallback_color: Optional[Literal["white", "black"]]
                         ) -> chess.Color:
    """Figure out whether the target is white or black in this game."""
    if target_player_name:
        white_name = (game.headers.get("White") or "").strip().lower()
        black_name = (game.headers.get("Black") or "").strip().lower()
        name = target_player_name.strip().lower()
        if name == white_name:
            return chess.WHITE
        if name == black_name:
            return chess.BLACK
        # If the name doesn't match, fall back (or default to white)
    if fallback_color:
        return chess.WHITE if fallback_color == "white" else chess.BLACK
    return chess.WHITE  # sensible default

def game_to_training_pairs(
    pgn: str,
    target_player_name: Optional[str] = None,
    target_color: Optional[Literal["white", "black"]] = None,
) -> List[Tuple[np.ndarray, str]]:
    """
    Convert a PGN game into (input_planes, output_move_uci) pairs such that
    the target player is always the mover. If the target is Black, the board
    and move are mirrored so the model always sees the target's pieces as 'white'.

    Args:
        pgn: Full PGN text for a single game.
        target_player_name: If provided, matches against PGN headers 'White'/'Black'.
        target_color: Explicit fallback ('white' or 'black') if name is absent or unmatched.

    Returns:
        List of tuples (planes, move_uci), where:
          - planes is (12, 8, 8) uint8 from board_to_bitmap()
          - move_uci is a UCI string of the (possibly mirrored) move
    """
    game = chess.pgn.read_game(io.StringIO(pgn))
    if game is None:
        return []

    tgt_color = resolve_target_color(game, target_player_name, target_color)

    pairs: List[Tuple[np.ndarray, str]] = []
    board = game.board()

    node = game
    while node.variations:
        move = node.variation(0).move

        # The mover is board.turn BEFORE pushing the move.
        mover_is_target = (board.turn == tgt_color)

        if mover_is_target:
            if tgt_color == chess.WHITE:
                # Use position as-is
                planes = board_to_bitmap(board)
                move_uci = move.uci()
            else:
                # Mirror board and move so target (black) looks like "white"
                mirrored_board = board.mirror()
                mirrored_move = mirror_move(move)
                planes = board_to_bitmap(mirrored_board)
                move_uci = mirrored_move.uci()

            pairs.append((planes, move_uci))

        # Advance to next node / position
        board.push(move)
        node = node.variation(0)

    return pairs