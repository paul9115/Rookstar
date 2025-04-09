# gui/play_gui.py
import pygame
import chess
import sys
import torch
import numpy as np
import os
from envs.chess_env import ChessEnv
from models.net import RookstarNet

# Constants
WIDTH, HEIGHT = 640, 640
SQUARE_SIZE = WIDTH // 8
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
LIGHT_BROWN = (240, 217, 181)
DARK_BROWN = (181, 136, 99)
SELECTED_HIGHLIGHT = (0, 255, 0, 100)
LEGAL_HIGHLIGHT = (0, 0, 255, 100)

PIECE_IMAGES = {}
ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")

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

    if selected_square is not None:
        row = 7 - chess.square_rank(selected_square)
        col = chess.square_file(selected_square)
        pygame.draw.rect(screen, SELECTED_HIGHLIGHT[:3], pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

    if legal_moves:
        for move in legal_moves:
            row = 7 - chess.square_rank(move.to_square)
            col = chess.square_file(move.to_square)
            pygame.draw.rect(screen, LEGAL_HIGHLIGHT[:3], pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

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
    clock = pygame.time.Clock()

    while True:
        draw_board(screen, env.board, selected_square, legal_targets)
        pygame.display.flip()
        clock.tick(30)

        if env.board.is_game_over():
            print("Game over:", env.board.outcome())
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

            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                square = square_at_pixel(*pos)

                if selected_square is None:
                    piece = env.board.piece_at(square)
                    if piece and piece.color == chess.WHITE:
                        selected_square = square
                        legal_targets = [m for m in env.board.legal_moves if m.from_square == selected_square]
                else:
                    move = chess.Move(from_square=selected_square, to_square=square)
                    if move in env.board.legal_moves:
                        if env.board.piece_at(selected_square).piece_type == chess.PAWN and chess.square_rank(square) in [0, 7]:
                            move = chess.Move(from_square=selected_square, to_square=square, promotion=chess.QUEEN)
                        if move in env.board.legal_moves:
                            env.step(move.uci())
                    selected_square = None
                    legal_targets = []

if __name__ == '__main__':
    main()
