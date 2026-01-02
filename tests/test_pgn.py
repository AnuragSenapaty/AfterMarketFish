# tests/test_pgn_parser.py
import io
import chess
import chess.pgn

from aftermarketfish.pgn_parser import game_to_training_pairs


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

            # For black training we mirror the board so legality is checked on mirrored board.
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
    assert move0 == "e2e4"

    _assert_moves_are_legal(pairs, chess.WHITE, PGN_TEXT)


def test_game_pairs_black_target_color_mirrored():
    expected = _count_moves_for_color(PGN_TEXT, chess.BLACK)
    pairs = game_to_training_pairs(PGN_TEXT, target_color="black")

    assert isinstance(pairs, list)
    assert len(pairs) == expected

    planes0, move0 = pairs[0]
    assert planes0.shape == (12, 8, 8)

    # First black move in the sample PGN is e7e5, mirrored becomes e2e4
    assert move0 == "e2e4"
    assert planes0.sum() == 32

    _assert_moves_are_legal(pairs, chess.BLACK, PGN_TEXT)


def test_game_pairs_target_by_player_name_defaults_to_white():
    # In this PGN, the named player is White, so we should get White’s moves.
    pairs = game_to_training_pairs(PGN_TEXT, target_player_name="Anurag Senapaty")
    assert len(pairs) == 3
    _, first_move = pairs[0]
    assert first_move == "e2e4"
