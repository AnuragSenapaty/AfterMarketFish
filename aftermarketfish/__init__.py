"""Core package for the AfterMarketFish chess engine.

This package contains:
- Bitboard utilities
- PGN → training pair conversion
- Magnus-games preprocessing
- Dataset & vocabulary helpers
- Feed-forward network model
- Training loop utilities
- Inference helpers
"""

from .bitboards import board_to_bitmap
from .pgn_parser import game_to_training_pairs
from .magnus_preprocess import process_magnus_games
from .dataset import MoveDataset
from .model_ffnn import FFNN, load_checkpoint
from .training_loop import train_ffnn
from .inference import (
    load_vocab,
    invert_vocab,
    predict_move_from_board,
    predict_move_from_fen,
)

__all__ = [
    "board_to_bitmap",
    "game_to_training_pairs",
    "process_magnus_games",
    "MoveDataset",
    "FFNN",
    "load_checkpoint",
    "train_ffnn",
    "load_vocab",
    "invert_vocab",
    "predict_move_from_board",
    "predict_move_from_fen",
]
