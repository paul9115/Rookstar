import os
import random
import torch
import json
import chess
import chess.pgn
from datetime import datetime
from envs.chess_env import ChessEnv


def save_game_pgn(moves, result, game_id):
    game = chess.pgn.Game()
    game.headers["Event"] = "Rookstar Self-Play"
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game.headers["Result"] = result

    node = game
    for move in moves:
        node = node.add_variation(move)

    pgn_path = f"data/games/game_{game_id:06}.pgn"
    with open(pgn_path, "w") as f:
        print(game, file=f)


def save_game_json(moves, reward, game_id):
    board = chess.Board()
    san_moves = []
    for move in moves:
        san_moves.append(board.san(move))
        board.push(move)

    json_path = f"data/games/game_{game_id:06}.json"
    with open(json_path, "w") as f:
        json.dump({
            "moves": [m.uci() for m in moves],
            "san": san_moves,
            "reward": reward,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)


def get_next_game_id():
    os.makedirs("data/games", exist_ok=True)
    existing = [f for f in os.listdir("data/games") if f.endswith(".json")]
    ids = [int(f.split("_")[1].split(".")[0]) for f in existing]
    return max(ids) + 1 if ids else 1


def play_random_game():
    env = ChessEnv()
    state = env.reset()
    done = False
    moves = []

    while not done:
        legal_moves = env.board.legal_moves
        move = random.choice(list(legal_moves))
        moves.append(move)
        state, reward, done, info = env.step(move.uci())

    game_id = get_next_game_id()
    outcome = env.board.outcome()
    result = outcome.result() if outcome else "*"

    save_game_pgn(moves, result, game_id)
    save_game_json(moves, reward, game_id)
    print(f"Game {game_id:06} saved. Result: {result}, Reward: {reward}")


def play_human_vs_random():
    env = ChessEnv()
    state = env.reset()
    done = False

    print("Welcome to Rookstar! You are playing as White.")
    env.render()

    while not done:
        if env.board.turn == chess.WHITE:
            move = input("Your move (in UCI, e.g., e2e4): ")
            if move not in env.legal_moves():
                print("Illegal move. Try again.")
                continue
        else:
            move = random.choice(env.legal_moves())
            print(f"Rookstar plays: {move}")

        state, reward, done, info = env.step(move)
        env.render()

    outcome = env.board.outcome()
    result = outcome.result() if outcome else "*"
    print(f"Game Over. Result: {result}, Reward: {reward}")

def generate_batch(n):
    for i in range(n):
        print(f"Generating game {i + 1}/{n}")
        play_random_game()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Rookstar Game Runner")

    parser.add_argument('--batch', type=int, default=1,
                        help="Number of random games to generate in batch mode.")
    args = parser.parse_args()
    if args.batch > 1:
        generate_batch(args.batch)
    else:
        play_human_vs_random()    
