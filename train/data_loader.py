import os
import json
import torch
import numpy as np
from envs.chess_env import ChessEnv


def load_training_data(data_dir="data/games"):
    X, y = [], []
    files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
    for fname in files:
        with open(os.path.join(data_dir, fname), 'r') as f:
            game = json.load(f)
            board = ChessEnv()
            board.reset()
            for move in game['moves']:
                X.append(board.get_state())
                y.append(game['reward'])
                board.step(move)
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)
    return X, y
