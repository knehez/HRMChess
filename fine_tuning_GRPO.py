import pandas as pd
import numpy as np
import torch
import os
import chess
from Chess import fen_to_bitboard_tensor
from tqdm import tqdm  # a fájl tetejére!

def load_puzzles(csv_path, max_positions=10000, skip_positions=0, min_rating=800, max_rating=2200):
    """
    Betölti a puzzle-öket a lichess_db_puzzle.csv-ből, skip_positions-től indulva, max_positions darabot, rating szűréssel.
    A Chess.py-beli load_puzzle_data logikáját követi, pandas-szal.
    """
    print(f"\n🧩 LOADING TACTICAL PUZZLES from {csv_path} (skip: {skip_positions}, max: {max_positions})")
    df = pd.read_csv(csv_path, skiprows=range(1, skip_positions+1), nrows=max_positions)
    print(f"Loaded {len(df)} puzzles.")

    puzzle_states = []
    puzzle_policies = []
    puzzle_fens = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing puzzles"):
        try:
            fen = row['FEN']
            moves = row['Moves'].split()
            rating = int(row['Rating']) if 'Rating' in row else 1500
            if not (min_rating <= rating <= max_rating):
                continue
            if len(moves) < 1:
                continue
            first_move = moves[0]
            board = chess.Board(fen)
            move = chess.Move.from_uci(first_move)
            if move not in board.legal_moves:
                continue
            state = fen_to_bitboard_tensor(fen)
            policy = (move.from_square, move.to_square)
            puzzle_states.append(state)
            puzzle_policies.append(policy)
            puzzle_fens.append(fen)
        except Exception as e:
            continue

    print(f"✅ TACTICAL PUZZLES: {len(puzzle_states):,} pozíció betöltve")
    return np.array(puzzle_states), np.array(puzzle_policies), puzzle_fens


def save_dataset(states, policies, out_path):
    """
    Elmenti a datasetet torch tensor formátumban.
    """
    print(f"Saving dataset to {out_path} ...")
    torch.save({'states': states, 'policies': policies}, out_path)
    print("Done.")

# === GRPO Fine-tuning script ===
if __name__ == "__main__":
    csv_path = 'lichess_db_puzzle.csv'
    out_path = 'puzzle_dataset.pt'
    max_positions = 2000000
    skip_positions = 3000000
    
    if not os.path.exists(out_path):
        states, policies, fens = load_puzzles(csv_path, max_positions, skip_positions)
        save_dataset(states, policies, out_path)
    else:
        print(f"{out_path} already exists, skipping generation.")
    
    import sys
    from hrm_model import PolicyDataset, HRMChess, train_GRPO_loop

    # --- Load dataset ---
    dataset_path = "puzzle_dataset.pt"
    if not os.path.exists(dataset_path):
        print(f"Dataset file not found: {dataset_path}")
        sys.exit(1)
    data = torch.load(dataset_path, weights_only=False)  # Fix for PyTorch 2.6+ numpy arrays
    states = data['states']
    policies = data['policies']
    dataset = PolicyDataset(states, policies)

    # --- Load best model checkpoint ---
    checkpoint_path = "best_hrm_chess_model.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Model checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    hparams = checkpoint.get('hyperparams', {})
    input_dim = 20 # hparams.get('input_dim', 20)
    hidden_dim = hparams.get('hidden_dim', 192)
    N = hparams.get('N', 8)
    T = hparams.get('T', 8)

    # --- Recreate model and load weights ---
    model = HRMChess(input_dim=input_dim, hidden_dim=hidden_dim, N=N, T=T)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {checkpoint_path} (input_dim={input_dim}, hidden_dim={hidden_dim}, N={N}, T={T})")

    # --- Fine-tune with GRPO ---
    train_GRPO_loop(model, dataset, epochs=5, batch_size=64, lr=2e-5, beta=0.01)
