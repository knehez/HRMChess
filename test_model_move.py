import chess
import torch
from elo_measurement import ELORatingSystem

def test_model_move(fen):
    board = chess.Board(fen)
    move, confidence = elo_system.model_move(board, temperature=1.0, debug=True)
    print(f"model_move() választott lépés: {move}, confidence: {confidence:.3f}")
    # Top k moves kiírása
    k = 10
    # A model_move debug=True esetén kiírja a top 3-at, de itt explicit újra lekérjük
    # Újra meghívjuk, hogy move_info_sorted-ot is elérjük
    import torch
    from hrm_model import fen_to_bitplanes, bin_to_score, inference_with_amp
    legal_moves = list(board.legal_moves)
    next_fens = []
    for move in legal_moves:
        board_copy = board.copy()
        board_copy.push(move)
        next_fens.append(board_copy.fen())
    import numpy as np
    bitplane_np = np.array([fen_to_bitplanes(fen) for fen in next_fens], dtype=np.float32)
    bitplane_batch = torch.from_numpy(bitplane_np).to(elo_system.device)
    move_scores = [-float('inf')] * len(legal_moves)
    
    # Use optimized AMP inference
    out = inference_with_amp(elo_system.model, bitplane_batch, use_amp=True)
    
    for i, logits in enumerate(out):
        value_probs = torch.softmax(logits, dim=0)
        expected_bin = (value_probs * torch.arange(len(value_probs), device=value_probs.device)).sum().item()
        win_percent = bin_to_score(expected_bin, num_bins=len(value_probs))
        move_scores[i] = win_percent
    move_info = list(zip(legal_moves, move_scores))
    move_info_sorted = sorted(move_info, key=lambda x: x[1], reverse=True)
    print(f"Top {k} lépés:")
    for m, s in move_info_sorted[:k]:
        print(f"  {m}: {s:.3f}")
    assert move in board.legal_moves, "A választott lépés nem legális!"
    assert 0.0 <= confidence <= 1.0, "A confidence érték nincs 0 és 1 között!"
    print("Teszt sikeres: model_move() működik.")

def test_sequential_moves(fen, num_moves=20):
    board = chess.Board(fen)
    
    print(f"\nKezdőállás: {fen}")
    moves_played = []
    for i in range(num_moves):
        if board.is_game_over():
            print(f"Játék vége: {board.result()} ({'matt' if board.is_checkmate() else 'patt'})")
            break
        move, confidence = elo_system.model_move(board, temperature=1.0, debug=False)
        moves_played.append(board.san(move))
        board.push(move)
    print("Lépéssorozat:", ", ".join(moves_played))

if __name__ == "__main__":
    fen = "6k1/5ppp/4r3/8/8/8/5PPP/3R2K1 w - -"
    
    # Enable float16 optimization for faster inference (requires CUDA)
    use_float16 = torch.cuda.is_available()
    if use_float16:
        print("🚀 Float16 optimization enabled for testing")
    
    elo_system = ELORatingSystem(use_half=use_float16)
    test_model_move(fen)
    test_sequential_moves(fen)
