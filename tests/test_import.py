import os
import sys
import io
import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import chess
import chess.pgn
import numpy as np

from data.loadData import game_to_training_pairs, board_to_bitmap, mirror_move


def _magnus_raw_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'Magnus', 'Raw'))


def _first_pgn_path() -> str:
    base = _magnus_raw_dir()
    matches = sorted(glob.glob(os.path.join(base, '*.pgn')))
    assert matches, f'No PGN found under {base}'
    return matches[0]


def _read_first_game_text(path: str) -> str:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        game = chess.pgn.read_game(f)
    assert game is not None, f'Could not read a game from {path}'
    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
    return game.accept(exporter)


def _first_move_for_color(pgn_text: str, color: chess.Color) -> chess.Move:
    g = chess.pgn.read_game(io.StringIO(pgn_text))
    b = g.board()
    for mv in g.mainline_moves():
        if b.turn == color:
            return mv
        b.push(mv)
    raise AssertionError(f'No move found for color={color}')


def _uci_of_mirrored_move(mv: chess.Move) -> str:
    return mirror_move(mv).uci()


def _assert_all_pairs_legal(pgn_text: str, pairs, target_color: chess.Color) -> None:
    g = chess.pgn.read_game(io.StringIO(pgn_text))
    b = g.board()
    i = 0
    for mv in g.mainline_moves():
        if b.turn == target_color:
            planes, uci = pairs[i]
            assert isinstance(planes, np.ndarray) and planes.shape == (12, 8, 8)
            norm_board = b if target_color == chess.WHITE else b.mirror()
            assert chess.Move.from_uci(uci) in norm_board.legal_moves, f'Illegal pair {i}: {uci}'
            i += 1
        b.push(mv)
    assert i == len(pairs), 'Pair count mismatch vs traversed moves'


def test_magnus_first_game_black_mirroring_dynamic():
    pgn_path = _first_pgn_path()
    pgn_text = _read_first_game_text(pgn_path)

    first_black_move = _first_move_for_color(pgn_text, chess.BLACK)
    expected_first_uci = _uci_of_mirrored_move(first_black_move)

    pairs = game_to_training_pairs(
        pgn_text,
        target_player_name='Carlsen,Magnus',
        target_color='black',
    )

    g = chess.pgn.read_game(io.StringIO(pgn_text))
    b = g.board()
    expected_black = 0
    for mv in g.mainline_moves():
        if b.turn == chess.BLACK:
            expected_black += 1
        b.push(mv)

    assert len(pairs) == expected_black

    planes0, move0 = pairs[0]
    assert move0 == expected_first_uci
    assert planes0.shape == (12, 8, 8)

    from_sq = chess.parse_square(move0[:2])
    file_idx = chess.square_file(from_sq)
    rank_idx = 7 - chess.square_rank(from_sq)
    assert int(planes0[0:6, rank_idx, file_idx].sum()) == 1
    assert planes0.sum() >= 2

    _assert_all_pairs_legal(pgn_text, pairs, chess.BLACK)


if __name__ == '__main__':
    test_magnus_first_game_black_mirroring_dynamic()
    print('All tests passed')
