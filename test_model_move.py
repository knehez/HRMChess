import chess
from elo_measurement import ELORatingSystem

def test_model_move():
    # Példa FEN: kezdőállás
    fen = "6k1/5ppp/4r3/8/8/8/5PPP/3R2K1 w - -"
    board = chess.Board(fen)
    elo_system = ELORatingSystem()
    move, confidence = elo_system.model_move(board, temperature=1.0, debug=True)
    print(f"model_move() választott lépés: {move}, confidence: {confidence:.3f}")
    # Top k moves kiírása
    k = 5
    # A model_move debug=True esetén kiírja a top 3-at, de itt explicit újra lekérjük
    # Újra meghívjuk, hogy move_info_sorted-ot is elérjük
    import torch
    from hrm_model import fen_to_tokens, bin_to_score
    legal_moves = list(board.legal_moves)
    fen_tokens = torch.tensor([fen_to_tokens(board.fen())], dtype=torch.long).repeat(len(legal_moves), 1).to(elo_system.device)
    uci_indices = [elo_system.uci_move_to_idx.get(m.uci(), -1) for m in legal_moves]
    valid_mask = torch.tensor([i != -1 for i in uci_indices], dtype=torch.bool)
    uci_tensor = torch.tensor([i if i != -1 else 0 for i in uci_indices], dtype=torch.long).to(elo_system.device)
    move_scores = [-float('inf')] * len(legal_moves)
    with torch.no_grad():
        if valid_mask.any():
            fen_tokens_valid = fen_tokens[valid_mask]
            uci_tensor_valid = uci_tensor[valid_mask]
            out = elo_system.model(fen_tokens_valid, uci_tensor_valid)
            valid_indices = [i for i, idx in enumerate(uci_indices) if idx != -1]
            for j, logits in enumerate(out):
                value_probs = torch.softmax(logits, dim=0)
                expected_bin = (value_probs * torch.arange(len(value_probs), device=value_probs.device)).sum().item()
                win_percent = bin_to_score(expected_bin, num_bins=len(value_probs))
                move_scores[valid_indices[j]] = win_percent
    move_info = list(zip(legal_moves, move_scores))
    move_info_sorted = sorted(move_info, key=lambda x: x[1], reverse=True)
    print(f"Top {k} lépés:")
    for m, s in move_info_sorted[:k]:
        print(f"  {m}: {s:.3f}")
    assert move in board.legal_moves, "A választott lépés nem legális!"
    assert 0.0 <= confidence <= 1.0, "A confidence érték nincs 0 és 1 között!"
    print("Teszt sikeres: model_move() működik.")

if __name__ == "__main__":
    test_model_move()
