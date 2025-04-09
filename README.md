# Rookstar: Reinforcement Learning Chess engine

A neural network-based chess engine trained entirely through self-play inspired by AlphaZero. The goal is to start from scratch (around 100 ELO) and eventually reach grand-master level strength (~2000+ ELO) through continuous self play and reinforcement learning.

## Features
- Reinforcement learning (no supervised pretraining)
- Monte Carlo Tree Search (future)
- Self play for training and data generation 
- Neural network model with policy and value heads

## Setup
```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Project structure 
- `envs/`: Chess environment wrapped for RL 
- `models/`: neural networks for move prediction and evaluation
- `train/`: Self-play generation and model training
- `data/`: Generated self-play games for training logs

## Goals 
- ✅ Self-play vs random agent
- ⌛ Basic NN + value head 
- ⌛ Self-play dataset -> training loop 
- ⌛ Policy + value head with MCTS
- ⌛ Lichess.org bot integration

## License
MIT

