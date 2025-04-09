import chess
import numpy as np


class ChessEnv:
    def __init__(self):
        self.board = chess.Board()

    def reset(self):
        self.board.reset()
        return self.get_state()

    def get_state(self):
        # Encode the board as a simple 8x8x12 binary matrix (one-hot for piece type & color)
        state = np.zeros((12, 8, 8), dtype=np.float32)
        piece_map = self.board.piece_map()

        for square, piece in piece_map.items():
            piece_type = piece.piece_type - 1  # 0 to 5
            color_offset = 0 if piece.color == chess.WHITE else 6
            row, col = divmod(square, 8)
            state[color_offset + piece_type][row][col] = 1

        return state

    def step(self, move_uci):
        move = chess.Move.from_uci(move_uci)
        if move in self.board.legal_moves:
            self.board.push(move)
            done = self.board.is_game_over()
            outcome = self.board.outcome() if done else None
            reward = self.get_reward(outcome) if outcome else 0
            return self.get_state(), reward, done, {}
        else:
            return self.get_state(), -1, True, {"illegal": True}

    def legal_moves(self):
        return [m.uci() for m in self.board.legal_moves]

    def get_reward(self, outcome):
        if outcome.winner is None:
            return 0  # Draw
        return 1 if outcome.winner == chess.WHITE else -1

    def render(self):
        print(self.board)