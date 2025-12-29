"""Inference / prediction helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import chess

from .bitboards import board_to_bitmap
from .model_ffnn import load_checkpoint


def load_vocab(vocab_path: str | Path) -> Dict[str, int]:
    vocab_path = Path(vocab_path)
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab: Dict[str, int] = json.load(f)
    return vocab


def invert_vocab(vocab: Dict[str, int]) -> Dict[int, str]:
    return {idx: move for move, idx in vocab.items()}


def _prepare_input(board: chess.Board) -> torch.Tensor:
    planes = board_to_bitmap(board).astype("float32")
    x = torch.from_numpy(planes).view(1, -1)  # (1, 12*8*8)
    return x


@torch.no_grad()
def predict_move_from_board(
    board: chess.Board,
    ckpt_path: str | Path,
    vocab_path: str | Path,
    device: str | torch.device = "cpu",
) -> Tuple[str, float]:
    """Predict a move (UCI) from a chess.Board.

    Returns (best_move_uci, confidence_softmax).
    """
    device = torch.device(device)

    model, _ = load_checkpoint(ckpt_path, map_location=device)
    vocab = load_vocab(vocab_path)
    inv_vocab = invert_vocab(vocab)

    model.to(device)
    model.eval()

    x = _prepare_input(board).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    top_prob, top_idx = probs.max(dim=1)

    move_idx = top_idx.item()
    move_uci = inv_vocab.get(move_idx, "")

    return move_uci, float(top_prob.item())


@torch.no_grad()
def predict_move_from_fen(
    fen: str,
    ckpt_path: str | Path,
    vocab_path: str | Path,
    device: str | torch.device = "cpu",
) -> Tuple[str, float]:
    board = chess.Board(fen)
    return predict_move_from_board(board, ckpt_path, vocab_path, device=device)
