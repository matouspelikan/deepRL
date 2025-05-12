#!/usr/bin/env python3
"""
PyTorch implementation of the Pisqorky (Gomoku) AlphaZero agent.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import math
import os
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Silence TF‑specific env vars left in the original project – harmless here.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from pisqorky import Pisqorky  # project‑local import
import pisqorky_cpp          # C++ MCTS extension module
import pisqorky_evaluator
import pisqorky_player_heuristic
import wrappers

# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
# ReCodEx‑controlled switches
parser.add_argument("--recodex", default=False, action="store_true")
parser.add_argument("--render_each", default=0, type=int)
parser.add_argument("--seed", default=None, type=int)
parser.add_argument("--threads", default=1, type=int)
# Configurable hyper‑parameters
parser.add_argument("--alpha", default=0.1, type=float)
parser.add_argument("--batch_size", default=512, type=int)
parser.add_argument("--epsilon", default=0.25, type=float)
parser.add_argument("--evaluate_each", default=200, type=int)
parser.add_argument("--learning_rate", default=0.001, type=float)
parser.add_argument("--model_path", default="pisqorky_cpp.pt", type=str)
parser.add_argument("--num_simulations", default=100, type=int)
parser.add_argument("--sampling_moves", default=8, type=int)
parser.add_argument("--show_sim_games", default=False, action="store_true")
parser.add_argument("--sim_games", default=16, type=int)
parser.add_argument("--train_for", default=1, type=int)
parser.add_argument("--window_length", default=100_000, type=int)
parser.add_argument("--workers", default=512, type=int)

# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _to_tensor(x: np.ndarray | torch.Tensor, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert a NumPy array (or Tensor) to a torch Tensor on DEVICE.
    * NHWC images (≥4‑D) → NCHW tensors
    * 1‑D/2‑D/3‑D inputs are passed through unchanged (no axis juggling).
    """
    if isinstance(x, torch.Tensor):
        return x.to(device=DEVICE, dtype=dtype)

    x = np.asarray(x)

    # Add batch dim for a single board (H, W, C)
    if x.ndim == 3:
        x = x[None]

    # Only tensors with (N, H, W, C) layout need axis swap
    if x.ndim >= 4:
        x = np.moveaxis(x, -1, 1)  # NHWC → NCHW

    return torch.as_tensor(x, device=DEVICE, dtype=dtype)

def _to_numpy(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().numpy()
    # Inference outputs of the network (policy) are (N, C) so we leave them.
    if x.ndim == 4:  # NCHW → NHWC for images
        x = np.moveaxis(x, 1, -1)
    return x

# ─────────────────────────────────────────────────────────────────────────────
# Neural network definition
# ─────────────────────────────────────────────────────────────────────────────
class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        layers: List[nn.Module] = []
        in_ch = Pisqorky.C
        for i in range(5):
            layers.append(nn.Conv2d(in_ch, 20 if i == 4 else 15, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_ch = 20 if i == 4 else 15
        self.stem = nn.Sequential(*layers)

        # Policy head
        self.policy_conv = nn.Conv2d(in_ch, 2, kernel_size=3, padding=1)
        self.policy_fc   = nn.Linear(2 * Pisqorky.N * Pisqorky.N, Pisqorky.ACTIONS)

        # Value head
        self.value_conv = nn.Conv2d(in_ch, 2, kernel_size=3, padding=1)
        self.value_fc   = nn.Linear(2 * Pisqorky.N * Pisqorky.N, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        # Policy
        p = F.relu(self.policy_conv(x))
        p = self.policy_fc(p.flatten(1))
        p = F.softmax(p, dim=-1)
        # Value
        v = F.relu(self.value_conv(x))
        v = torch.tanh(self.value_fc(v.flatten(1))).squeeze(-1)  # (N,)
        return p, v

# ─────────────────────────────────────────────────────────────────────────────
# Agent wrapper
# ─────────────────────────────────────────────────────────────────────────────
class Agent:
    def __init__(self, args: argparse.Namespace):
        self._model = _Net().to(DEVICE)
        self._opt   = torch.optim.Adam(self._model.parameters(), lr=args.learning_rate)

    # —— (de)serialization ————————————————————————————
    def save(self, path: str) -> None:
        torch.save(self._model.state_dict(), path)

    @classmethod
    def load(cls, path: str, args: argparse.Namespace) -> "Agent":
        agent = cls.__new__(cls)
        agent._model = _Net().to(DEVICE)
        agent._model.load_state_dict(torch.load(path, map_location=DEVICE))
        agent._opt = torch.optim.Adam(agent._model.parameters(), lr=args.learning_rate)
        return agent

    # —— training ————————————————————————————————
    def train(self, boards: np.ndarray, target_policies: np.ndarray, target_values: np.ndarray) -> tuple[float, float, float]:
        boards = np.asarray(boards, np.float32)
        target_policies = np.asarray(target_policies, np.float32)
        target_values = np.asarray(target_values, np.float32)

        boards_t = _to_tensor(boards)
        target_policies_t = _to_tensor(target_policies)
        target_values_t = torch.as_tensor(target_values, dtype=torch.float32, device=DEVICE)  # (N,)

        self._opt.zero_grad(set_to_none=True)
        pred_p, pred_v = self._model(boards_t)

        # Categorical cross‑entropy with soft labels
        loss_p = (-target_policies_t * torch.log(torch.clamp(pred_p, 1e-9, 1.0))).sum(dim=1).mean()
        loss_v = F.mse_loss(pred_v, target_values_t)
        loss = loss_p + loss_v
        loss.backward()
        self._opt.step()
        
        # Return losses as float values for logging
        return loss.item(), loss_p.item(), loss_v.item()

    # —— inference ————————————————————————————————
    def predict(self, boards: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        boards = np.asarray(boards, np.float32)
        with torch.no_grad():
            pol, val = self._model(_to_tensor(boards))
        pol_np, val_np = _to_numpy(pol), _to_numpy(val)
        return pol_np, val_np

# ─────────────────────────────────────────────────────────────────────────────
# Training loop and evaluation
# ─────────────────────────────────────────────────────────────────────────────
ReplayBufferEntry = collections.namedtuple("ReplayBufferEntry", ["board", "policy", "outcome"])

def train(args: argparse.Namespace) -> Agent:
    agent = Agent(args)
    replay_buffer = wrappers.ReplayBuffer(max_length=args.window_length)

    iteration = 0
    training = True
    best_evaluation = -1.0

    pisqorky_cpp.simulated_games_start(
        args.workers,
        args.num_simulations,
        args.sampling_moves,
        args.epsilon,
        args.alpha,
    )

    while training:
        iteration += 1

        # Generate simulated games
        for _ in range(args.sim_games):
            game = pisqorky_cpp.simulated_game(agent.predict)
            replay_buffer.extend(game)

        # Optional pretty print of one generated game
        if iteration % args.evaluate_each == 0 and args.show_sim_games:
            log: List[List[str]] = [[] for _ in range(Pisqorky.N + 1)]
            for i, (board, policy, outcome) in enumerate(game):
                log[0].append(f"Move {i}, result {outcome}".center(28))
                action = 0
                for row in range(Pisqorky.N):
                    for col in range(Pisqorky.N):
                        cell = (
                            " XX " if board[row, col, 1] else
                            " .. " if board[row, col, 2] else
                            f"{policy[action] * 100:>3.0f} "
                        )
                        log[1 + row].append(cell)
                        action += 1
                print(*["".join(line) for line in log], sep="\n")

        # Parameter updates
        total_loss_sum = 0
        policy_loss_sum = 0
        value_loss_sum = 0
        for _ in range(args.train_for):
            batch = replay_buffer.sample(min(len(replay_buffer), args.batch_size), np.random)
            boards, policies, outcomes = zip(*batch)
            total_loss, policy_loss, value_loss = agent.train(boards, policies, outcomes)
            total_loss_sum += total_loss
            policy_loss_sum += policy_loss
            value_loss_sum += value_loss
        
        # Log losses every 10 iterations
        if iteration % args.evaluate_each == 0:
            print(f"Iteration {iteration}: Loss = {total_loss_sum/args.train_for:.4f}, " +
                  f"Policy Loss = {policy_loss_sum/args.train_for:.4f}, " +
                  f"Value Loss = {value_loss_sum/args.train_for:.4f}")

        # Periodic evaluation
        if iteration % args.evaluate_each == 0:
            print(f"Evaluation after iteration {iteration}, {datetime.datetime.now()}")
            score = pisqorky_evaluator.evaluate(
                [Player(agent, argparse.Namespace(num_simulations=0)), pisqorky_player_heuristic.Player()],
                games=100, render=False, verbose=False, first_chosen=False)
            print(f"Evaluation: {100 * score:.1f}%")
            if score > best_evaluation or math.isclose(score, 1.0):
                agent.save(f"{args.model_path}-{iteration/1000:06.1f}k-{100*score:.0f}.pt")
                best_evaluation = score
            print(flush=True)

    pisqorky_cpp.simulated_games_stop()
    return agent

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation‑time wrapper
# ─────────────────────────────────────────────────────────────────────────────
class Player:
    def __init__(self, agent: Agent, args: argparse.Namespace):
        self.agent = agent
        self.args = args

    def play(self, game: Pisqorky) -> int:
        policy = pisqorky_cpp.mcts(
            game._board,
            game._to_play,
            self.agent.predict,
            self.args.num_simulations,
            0.0,
            0.0,
        )
        return max(game.valid_actions(), key=lambda a: policy[a])

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> Player:
    # Reproducibility – note that PyTorch randomness is separate from numpy
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Threading consistency: limit intra/inter‑op threads to respect `--threads`
    torch.set_num_threads(args.threads)

    if args.recodex:
        agent = Agent.load(args.model_path, args)
    else:
        agent = train(args)

    return Player(agent, args)

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    player = main(args)

    pisqorky_evaluator.evaluate(
        [player, pisqorky_player_heuristic.Player(seed=args.seed)],
        games=100,
        render=False,
        verbose=True,
    )
