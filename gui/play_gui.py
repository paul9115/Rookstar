import pygame
import chess
import sys
import torch
import numpy as np
import os
import json
import datetime
from envs.chess_env import ChessEnv
from models.net import RookstarNet

# Constants
WIDTH, HEIGHT = 640, 640
SQUARE_SIZE = WIDTH // 8
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREY = (150, 150, 150)
LIGHT_BROWN = (240, 217, 181)
DARK_BROWN = (181, 136, 99)
GREEN = (0, 255, 0)

PIECE_IMAGES = {}
ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "games")
os.makedirs(DATA_DIR, exist_ok=True)

PROMOTION_OPTIONS = {
    pygame.K_q: chess.QUEEN,
    pygame.K_r: chess.ROOK,
    pygame.K_b: chess.BISHOP,
    pygame.K_n: chess.KNIGHT
}

def load_images():
    pieces = ['r', 'n', 'b', 'q', 'k', 'p']
    for color in ['w', 'b']:
        for piece in pieces:
            name = color + piece
            path = os.path.join(ASSET_DIR, f"{name}.png")
            img = pygame.image.load(path)
            PIECE_IMAGES[name] = pygame.transform.scale(img, (SQUARE_SIZE, SQUARE_SIZE))

def draw_board(screen, board, selected_square=None, legal_moves=None):
    for row in range(8):
        for col in range(8):
            color = LIGHT_BROWN if (row + col) % 2 == 0 else DARK_BROWN
            pygame.draw.rect(screen, color, pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

    if legal_moves:
        for move in legal_moves:
            row = 7 - chess.square_rank(move.to_square)
            col = chess.square_file(move.to_square)
            center = (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2)
            pygame.draw.circle(screen, GREY, center, 8)

    if selected_square is not None:
        row = 7 - chess.square_rank(selected_square)
        col = chess.square_file(selected_square)
        pygame.draw.rect(screen, GREEN, pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), width=3)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)
            color = 'w' if piece.color == chess.WHITE else 'b'
            img = PIECE_IMAGES[color + piece.symbol().lower()]
            screen.blit(img, pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

def square_at_pixel(x, y):
    col = x // SQUARE_SIZE
    row = 7 - (y // SQUARE_SIZE)
    return chess.square(col, row)

def model_move(env, model):
    best_move = None
    best_value = -float('inf') if env.board.turn == chess.WHITE else float('inf')

    for move in env.board.legal_moves:
        env.board.push(move)
        state = torch.tensor(env.get_state()).unsqueeze(0)
        _, value = model(state)
        value = value.item()
        env.board.pop()

        if env.board.turn == chess.WHITE:
            if value > best_value:
                best_value = value
                best_move = move
        else:
            if value < best_value:
                best_value = value
                best_move = move

    return best_move

def save_game_json(board, result):
    temp_board = chess.Board()
    move_stack = list(board.move_stack)
    san_moves = []
    for move in move_stack: 
        try:
            san_moves.append(temp_board.san(move))
            temp_board.push(move)
        except Exception:
            san_moves.append("?")
            break
    data = {
        "moves": [move.uci() for move in move_stack],
        "san": san_moves,
        "reward": result,
        "timestamp": datetime.datetime.now().isoformat()
    }
    filename = f"game_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Game saved to {filepath}")
    

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Play Rookstar")
    load_images()

    env = ChessEnv()
    env.reset()

    model = RookstarNet()
    model.load_state_dict(torch.load("models/weights/latest.pt"))
    model.eval()

    selected_square = None
    legal_targets = []
    pending_promotion = None
    clock = pygame.time.Clock()

    while True:
        draw_board(screen, env.board, selected_square, legal_targets)
        pygame.display.flip()
        clock.tick(30)

        if env.board.is_game_over():
            result = env.board.result()
            reward = 1 if result == "0-1" else -1 if result == "1-0" else 0
            save_game_json(env.board, reward)
            print(f"Game Over! Result: {result}")
            pygame.time.wait(3000)
            pygame.quit()
            sys.exit()

        if env.board.turn == chess.BLACK:
            move = model_move(env, model)
            if move:
                env.step(move.uci())
            continue

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if pending_promotion and event.type == pygame.KEYDOWN:
                if event.key in PROMOTION_OPTIONS:
                    promo_type = PROMOTION_OPTIONS[event.key]
                    move = chess.Move(from_square=pending_promotion[0], to_square=pending_promotion[1], promotion=promo_type)
                    if move in env.board.legal_moves:
                        env.step(move.uci())
                    pending_promotion = None
                    selected_square = None
                    legal_targets = []

            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                square = square_at_pixel(*pos)

                if selected_square is None:
                    piece = env.board.piece_at(square)
                    if piece and piece.color == chess.WHITE:
                        selected_square = square
                        legal_targets = [m for m in env.board.legal_moves if m.from_square == selected_square]
                else:
                    if square == selected_square or not any(m.to_square == square for m in legal_targets):
                        selected_square = None
                        legal_targets = []
                    else:
                        possible_moves = [m for m in legal_targets if m.to_square == square]
                        if possible_moves:
                            move = possible_moves[0]
                            if env.board.piece_at(selected_square).piece_type == chess.PAWN and chess.square_rank(square) in [0, 7]:
                                pending_promotion = (selected_square, square)
                                print("Select promotion piece: Q (queen), R (rook), B (bishop), N (knight)")
                            else:
                                env.step(move.uci())
                            selected_square = None
                            legal_targets = []

if __name__ == '__main__':
    main()