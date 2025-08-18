"""
Pure Vision Transformer Chess Model ELO Rating vs Stockfish
Ez a script a Python stockfish modult használja ELO-alapú erősségbeállítással.
"""

import torch
import chess
import numpy as np
import os
from stockfish import Stockfish
from hrm_model import PureViTChess, load_model_with_amp, inference_with_amp
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
    
    # Test if we can initialize Stockfish with the path
    for path in candidates:
        if os.path.exists(path):
            try:
                # Test with Python stockfish module
                test_stockfish = Stockfish(path=path)
                # Try to get a simple move to validate the engine
                test_stockfish.set_position(["e2e4"])
                if test_stockfish.get_best_move() is not None:
                    print(f"✅ Found Stockfish at: {path}")
                    return path
            except Exception:
                continue
    return None

def main():
    print("🏆 Pure ViT Chess Model ELO Rating vs Stockfish")
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
    
    # ELO-alapú Stockfish erősségek tesztelése
    # UCI_LimitStrength = true és UCI_Elo paraméterekkel
    elo_levels = [600, 600, 600, 600, 600, 600, 600, 600, 600, 600]
    results = []
    
    for target_elo in elo_levels:
        print(f"\n🤖 Playing vs Stockfish (ELO={target_elo})...")
        game_result = play_vs_stockfish_elo(elo_system, stockfish_path, target_elo, model_is_white=True)
        if game_result:
            results.append((target_elo, game_result))
    
    # ELO becslés eredmények alapján
    calculate_model_elo(results)

def play_vs_stockfish_elo(elo_system, stockfish_path, target_elo, model_is_white=True):
    """Játék Stockfish ellen megadott ELO erősséggel"""
    try:
        # Initialize Stockfish with Python module
        stockfish = Stockfish(path=stockfish_path)
        
        # Set ELO-based strength limiting
        stockfish_settings = {
            "Threads": 1,
            "Hash": 16,
            "UCI_Chess960": "false",
            "UCI_LimitStrength": "true",  # Enable strength limiting
            "UCI_Elo": target_elo         # Set target ELO
        }
        stockfish.update_engine_parameters(stockfish_settings)
        
        board = chess.Board()
        moves_played = []
        
        # Track game history for the model
        game_history = {
            'starting_fen': board.fen(),
            'moves': []
        }
        
        print(f"🎮 Starting game: Model is {'White' if model_is_white else 'Black'} vs Stockfish ELO {target_elo}")
        
        while not board.is_game_over() and len(moves_played) < 150:
            if (board.turn == chess.WHITE) == model_is_white:
                # Model lépése
                try:
                    move, _ = elo_system.model_move(board, temperature=0.3, debug=False, game_history=game_history)
                    board.push(move)
                    game_history['moves'].append(move.uci())
                    moves_played.append(f"Model: {move}")
                except Exception as e:
                    print(f"❌ Model error: {e}")
                    break
            else:
                # Stockfish lépése
                try:
                    stockfish.set_position(game_history['moves'])
                    stockfish_move = stockfish.get_best_move()
                    if stockfish_move is None:
                        print("❌ Stockfish couldn't find a move")
                        break
                        
                    move = chess.Move.from_uci(stockfish_move)
                    if move not in board.legal_moves:
                        print(f"❌ Stockfish suggested illegal move: {stockfish_move}")
                        break
                        
                    board.push(move)
                    game_history['moves'].append(move.uci())
                    moves_played.append(f"SF: {move}")
                except Exception as e:
                    print(f"❌ Stockfish error: {e}")
                    break
        
        # Game ended - evaluate result
        result = board.result()
        winner = None
        termination_reason = "Normal"
        
        if board.is_checkmate():
            termination_reason = "Checkmate"
        elif board.is_stalemate():
            termination_reason = "Stalemate"
        elif board.is_insufficient_material():
            termination_reason = "Insufficient material"
        elif len(moves_played) >= 150:
            termination_reason = "Move limit"
        
        if result == "1-0":
            winner = "Model" if model_is_white else "Stockfish"
        elif result == "0-1":
            winner = "Stockfish" if model_is_white else "Model"
        else:
            winner = "Draw"
        
        game_data = {
            "opponent": f"Stockfish_ELO{target_elo}",
            "result": result,
            "winner": winner,
            "moves": len(moves_played),
            "termination": termination_reason,
            "model_color": "White" if model_is_white else "Black",
            "target_elo": target_elo
        }
        
        print(f"Game ended: {result} - Winner: {winner} ({len(moves_played)} moves)")
        print(f"Termination: {termination_reason}")
        
        return game_data
            
    except Exception as e:
        print(f"❌ Hiba: {e}")
        return None

def calculate_model_elo(results):
    """Model ELO kiszámítása a Stockfish eredmények alapján"""
    if not results:
        print("No valid games played.")
        return
    
    print("\n📊 GAME RESULTS:")
    print("-" * 50)
    
    estimated_elos = []
    for target_elo, result in results:
        if result["winner"] == "Model":
            # Model won: estimate model is ~100 ELO stronger than opponent
            estimated = target_elo + 100
        elif result["winner"] == "Draw":
            # Draw: estimate model is roughly equal to opponent
            estimated = target_elo
        else:
            # Model lost: estimate model is ~100 ELO weaker than opponent
            estimated = target_elo - 100
        
        estimated = max(400, estimated)  # Minimum ELO floor
        estimated_elos.append(estimated)
        
        print(f"vs ELO {target_elo:4}: {result['result']:7} Winner: {result['winner']:9} → Est. {estimated:4} ELO")
    
    if estimated_elos:
        avg_elo = int(np.mean(estimated_elos))
        median_elo = int(np.median(estimated_elos))
        
        print("-" * 50)
        print("📈 FINAL ELO ESTIMATION:")
        print(f"Average ELO: {avg_elo}")
        print(f"Median ELO:  {median_elo}")
        print(f"Range: {min(estimated_elos)} - {max(estimated_elos)}")
        
        # ELO kategorizálás
        final_elo = median_elo  # Use median as more robust estimate
        if final_elo < 1000:
            category = "Beginner"
        elif final_elo < 1200:
            category = "Novice" 
        elif final_elo < 1400:
            category = "Casual Player"
        elif final_elo < 1600:
            category = "Club Player"
        elif final_elo < 1800:
            category = "Strong Club Player"
        elif final_elo < 2000:
            category = "Expert"
        else:
            category = "Master Level"
        
        print(f"Category: {category}")
        print("=" * 50)

if __name__ == "__main__":
    main()
