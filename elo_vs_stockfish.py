"""
HRM Chess Model ELO Rating vs Stockfish only
"""

import torch
import chess
import chess.engine
import numpy as np
import os
from hrm_model import HRMChess, load_model_with_amp, inference_with_amp
from elo_measurement import ELORatingSystem

def find_stockfish_path():
    """Auto-detect Stockfish path for different operating systems"""
    if os.name == 'nt':  # Windows
        candidates = [
            "stockfish.exe",
            "./stockfish.exe",
            "C:/stockfish/stockfish.exe",
            "C:/Program Files/stockfish/stockfish.exe"
        ]
    else:
        candidates = [
            "/usr/bin/stockfish",
            "/usr/local/bin/stockfish", 
            "./stockfish",
            "stockfish",
            "/opt/stockfish/stockfish",
            "/usr/games/stockfish"
        ]
    for path in candidates:
        if os.path.exists(path):
            print(f"✅ Found Stockfish at: {path}")
            return path
    return None

def main():
    print("🏆 HRM Chess Model ELO Rating vs Stockfish")
    print("=" * 60)
    # Modell betöltése (automatikus választás)
    try:
        elo_system = ELORatingSystem()
    except FileNotFoundError:
        print("❌ No model found!")
        return
    stockfish_path = find_stockfish_path()
    if not stockfish_path:
        print("❌ Stockfish not found. Download from: https://stockfishchess.org/download/")
        return
    # Csak Stockfish elleni mérés, depth=1-3
    results = []
    for depth in [1, 2, 3]:
        print(f"\n🤖 Playing vs Stockfish (depth={depth})...")
        game_result = elo_system.play_vs_stockfish(stockfish_path, depth=depth, time_limit=0.05, pgn=True, model_is_white=False)
        if game_result:
            results.append((depth, game_result))
    # ELO becslés depth alapján
    stockfish_elos = []
    for depth, result in results:
        if depth == 1:
            base_elo = 1200
        elif depth == 2:
            base_elo = 1400
        else:
            base_elo = 1600
        if result["winner"] == "Model":
            estimated = base_elo + 50
        elif result["winner"] == "Draw":
            estimated = base_elo - 50
        else:
            estimated = base_elo - 200
        estimated = max(400, estimated)
        stockfish_elos.append(estimated)
        print(f"Depth {depth}: {result['result']} Winner: {result['winner']} → {estimated} ELO")
    if stockfish_elos:
        avg_elo = int(np.mean(stockfish_elos))
        print("\n📈 FINAL ELO ESTIMATION vs Stockfish only:")
        print(f"Average ELO: {avg_elo}")
    else:
        print("No valid games played.")

if __name__ == "__main__":
    main()
