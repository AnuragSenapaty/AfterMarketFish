#!/usr/bin/env python3
"""
Lichess bot runner for AfterMarketFish.

Usage:
  export LICHESS_TOKEN="your_token_here"
  python run_lichess_bot.py --ckpt models/ffnn.pt --vocab models/vocab.json

Notes:
- Your Lichess account must be a BOT account.
- Token needs scope: "Play games with the bot API" (bot:play).
- This script:
  * listens to incoming events
  * accepts challenges
  * streams game state for each started game
  * plays model moves (top-k legality filtering), random fallback if needed
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import berserk
import chess
import numpy as np
import torch


# -----------------------------
# AfterMarketFish imports
# -----------------------------
try:
    from aftermarketfish.bitboards import board_to_bitmap
    from aftermarketfish.model_ffnn import load_checkpoint
except Exception as e:
    raise SystemExit(
        "Could not import AfterMarketFish package.\n"
        "Run this from repo root and ensure:\n"
        "  - you have AfterMarketFish/aftermarketfish/__init__.py\n"
        "  - you are in venv (source scripts/activate.sh)\n"
        f"Import error: {e}"
    )


# -----------------------------
# Helpers: mirroring for black
# -----------------------------
def mirror_square_name(sq: str) -> str:
    """Mirror a square like 'e2' -> 'e7' (vertical flip)."""
    idx = chess.parse_square(sq)
    midx = chess.square_mirror(idx)
    return chess.square_name(midx)

def mirror_uci(uci: str) -> str:
    """Mirror a UCI move like e2e4 -> e7e5 (and keep promotion)."""
    uci = uci.strip()
    if len(uci) not in (4, 5):
        return uci
    a = mirror_square_name(uci[0:2])
    b = mirror_square_name(uci[2:4])
    promo = uci[4:5] if len(uci) == 5 else ""
    return f"{a}{b}{promo}"


# -----------------------------
# Model wrapper
# -----------------------------
@dataclass
class ModelBundle:
    model: torch.nn.Module
    inv_vocab: List[str]  # index -> uci
    device: torch.device

def load_vocab(vocab_path: str) -> List[str]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab: Dict[str, int] = json.load(f)
    # invert
    inv = [""] * (max(vocab.values()) + 1)
    for uci, idx in vocab.items():
        inv[idx] = uci
    if any(x == "" for x in inv):
        raise ValueError("Vocab indices are not contiguous / inversion failed.")
    return inv

def load_model_bundle(ckpt_path: str, vocab_path: str, device_str: str = "cpu") -> ModelBundle:
    device = torch.device(device_str)
    model, meta = load_checkpoint(ckpt_path)
    model.to(device)
    model.eval()
    inv_vocab = load_vocab(vocab_path)
    # sanity: meta num_classes should match vocab size (if present)
    if "num_classes" in meta and meta["num_classes"] != len(inv_vocab):
        raise ValueError(f"Checkpoint num_classes={meta['num_classes']} but vocab size={len(inv_vocab)}")
    return ModelBundle(model=model, inv_vocab=inv_vocab, device=device)

@torch.no_grad()
def predict_topk_uci(
    bundle: ModelBundle,
    board: chess.Board,
    my_color: chess.Color,
    k: int = 10,
) -> List[Tuple[str, float]]:
    """
    Returns list of (uci, prob) in the ORIGINAL board coordinate system.
    If playing black, we mirror board to match training normalization, then mirror predicted moves back.
    """
    # normalize board for model
    norm_board = board if my_color == chess.WHITE else board.mirror()

    # encode
    planes = board_to_bitmap(norm_board.fen())  # (12,8,8)
    x = torch.tensor(planes.reshape(1, -1), dtype=torch.float32, device=bundle.device)

    logits = bundle.model(x)[0]  # (C,)
    probs = torch.softmax(logits, dim=0)

    k = min(k, probs.numel())
    top_probs, top_idx = torch.topk(probs, k=k)

    out: List[Tuple[str, float]] = []
    for p, idx in zip(top_probs.tolist(), top_idx.tolist()):
        uci_norm = bundle.inv_vocab[idx]
        uci_real = uci_norm if my_color == chess.WHITE else mirror_uci(uci_norm)
        out.append((uci_real, float(p)))
    return out


# -----------------------------
# Lichess bot logic
# -----------------------------
def should_accept_challenge(challenge_event: dict) -> bool:
    # Keep it permissive to start.
    # You can add filters: variant, time control, rated, etc.
    return True

def build_board_from_moves(moves_str: str) -> chess.Board:
    board = chess.Board()
    moves_str = (moves_str or "").strip()
    if moves_str:
        for uci in moves_str.split():
            try:
                board.push_uci(uci)
            except Exception:
                # If something is malformed, bail to avoid desync
                break
    return board

def play_game_thread(
    client: berserk.Client,
    bundle: ModelBundle,
    game_id: str,
    my_user_id: str,
    topk: int,
    think_ms: int,
) -> None:
    """
    Handles one game to completion. Runs in a thread.
    """
    try:
        stream = client.bots.stream_game_state(game_id)
        game_full = next(stream)  # first event should be gameFull
    except StopIteration:
        return
    except Exception as e:
        print(f"[{game_id}] Failed to open game stream: {e}")
        return

    # Determine our color
    try:
        white_id = game_full["white"]["id"]
        black_id = game_full["black"]["id"]
    except Exception:
        print(f"[{game_id}] Unexpected gameFull format.")
        return

    if my_user_id == white_id:
        my_color = chess.WHITE
    elif my_user_id == black_id:
        my_color = chess.BLACK
    else:
        print(f"[{game_id}] Our user id not in game? my={my_user_id} white={white_id} black={black_id}")
        return

    print(f"[{game_id}] Started. I am {'White' if my_color==chess.WHITE else 'Black'}.")

    last_moves_seen = None

    for event in stream:
        et = event.get("type")
        if et == "gameState":
            moves_str = event.get("moves", "")
            status = event.get("status", "")
            if status and status not in ("started", "created"):
                print(f"[{game_id}] Game ended with status={status}")
                return

            # avoid re-processing same state
            if moves_str == last_moves_seen:
                continue
            last_moves_seen = moves_str

            board = build_board_from_moves(moves_str)

            # our turn?
            if board.turn != my_color:
                continue

            # small think delay (optional)
            if think_ms > 0:
                time.sleep(think_ms / 1000.0)

            # choose a move (legal-filtering)
            candidates = predict_topk_uci(bundle, board, my_color=my_color, k=topk)

            chosen = None
            for uci, prob in candidates:
                try:
                    mv = chess.Move.from_uci(uci)
                    if mv in board.legal_moves:
                        chosen = uci
                        break
                except Exception:
                    continue

            if chosen is None:
                # fallback: random legal
                chosen = random.choice(list(board.legal_moves)).uci()

            ok = False
            try:
                ok = client.bots.make_move(game_id, chosen)
            except Exception as e:
                print(f"[{game_id}] make_move failed: {e}")

            print(f"[{game_id}] Played {chosen} (ok={ok})")

        elif et == "chatLine":
            # optional: ignore or respond
            pass
        elif et == "gameFull":
            # already handled
            pass
        else:
            # other event types can exist; ignore
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to model checkpoint, e.g. models/ffnn.pt")
    ap.add_argument("--vocab", required=True, help="Path to vocab.json, e.g. models/vocab.json")
    ap.add_argument("--token", default=os.environ.get("LICHESS_TOKEN", ""), help="Lichess API token (or set LICHESS_TOKEN)")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--topk", type=int, default=10, help="Try top-k predicted moves and pick first legal")
    ap.add_argument("--think-ms", type=int, default=50, help="Artificial delay before moving")
    args = ap.parse_args()

    if not args.token:
        raise SystemExit(
            "Missing Lichess token. Set env var LICHESS_TOKEN or pass --token.\n"
            "Token needs scope: bot:play."
        )

    # load model
    bundle = load_model_bundle(args.ckpt, args.vocab, device_str=args.device)

    # connect
    session = berserk.TokenSession(args.token)
    client = berserk.Client(session=session)

    # determine our user id
    me = client.account.get()
    my_user_id = me.get("id")
    if not my_user_id:
        raise SystemExit("Could not determine account id from Lichess. Is token valid?")

    print(f"Connected as: {my_user_id}")
    print("Listening for challenges / games...")

    # main event loop
    for event in client.bots.stream_incoming_events():
        et = event.get("type")
        if et == "challenge":
            ch = event.get("challenge", {})
            ch_id = ch.get("id")
            if not ch_id:
                continue

            if should_accept_challenge(event):
                print(f"[challenge] accepting {ch_id}")
                try:
                    client.bots.accept_challenge(ch_id)
                except Exception as e:
                    print(f"[challenge] accept failed: {e}")
            else:
                print(f"[challenge] declining {ch_id}")
                try:
                    client.bots.decline_challenge(ch_id)
                except Exception as e:
                    print(f"[challenge] decline failed: {e}")

        elif et == "gameStart":
            game = event.get("game", {})
            game_id = game.get("id")
            if not game_id:
                continue
            t = threading.Thread(
                target=play_game_thread,
                args=(client, bundle, game_id, my_user_id, args.topk, args.think_ms),
                daemon=True,
            )
            t.start()
            print(f"[gameStart] spawned handler for game {game_id}")

        else:
            # other event types: gameFinish, etc. Ignore for now.
            pass


if __name__ == "__main__":
    main()
