import torch

# Feltételezzük, hogy a dataset elemei (fen_tokens, uci_token, target_bin) tuple-k

def print_first_10_from_dataset(dataset_path="fen_move_score_dataset.pt"):
    data = torch.load(dataset_path, weights_only=False)
    # Ha dict, próbáljuk meg a legnagyobb listát vagy torch Dataset-et keresni benne
    if isinstance(data, dict):
        # Próbáljuk meg a legnagyobb listát vagy torch Dataset-et keresni
        candidates = [v for v in data.values() if isinstance(v, (list, tuple)) or hasattr(v, '__getitem__')]
        if candidates:
            data_seq = max(candidates, key=lambda x: len(x))
        else:
            raise ValueError("Nem található lista vagy indexelhető adat a dict-ben!")
    else:
        data_seq = data
    print(f"Dataset elemszám: {len(data_seq)}")

    # --- Uniqueness diagnostics for first 100 elements ---
    from hrm_model import score_to_bin
    uci_indices = []
    bin_indices = []
    score_values = []
    num_bins = 128  # update if needed
    for item in data_seq[:100]:
        if len(item) == 3:
            _, uci_token, score = item
        elif len(item) == 2:
            _, uci_token = item
            score = None
        else:
            continue
        uci_indices.append(uci_token)
        if score is not None:
            try:
                bin_idx = score_to_bin(float(score), num_bins=num_bins)
            except Exception:
                bin_idx = score
            bin_indices.append(bin_idx)
            score_values.append(score)

    print("Első 100 elemben:")
    print(f"  Egyedi uci_token-ek száma: {len(set(uci_indices))}")
    print(f"  Egyedi bin indexek száma: {len(set(bin_indices)) if bin_indices else 'N/A'}")
    print(f"  Score tartomány: {min(score_values) if score_values else 'N/A'} .. {max(score_values) if score_values else 'N/A'}")
    print()

    # --- Print first 10 as before ---
    for i, item in enumerate(data_seq[:10]):

        fen_str, uci_str, score = item
        
        print(f"#{i+1}")
        print(f"  fen: {fen_str}")
        print(f"  uci: {uci_str}")
        print(f"  score: {score}")
        print()

if __name__ == "__main__":
    print_first_10_from_dataset()
