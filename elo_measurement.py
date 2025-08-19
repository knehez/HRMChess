"""
Convolutional HRM Chess Model ELO Rating

Ez a script a Python stockfish modult használja a chess.engine helyett.
Telepítés: pip install stockfish
"""

import torch
import chess
import chess.pgn
import numpy as np
import time
from hrm_model import PureViTChess, load_model_with_amp, inference_with_amp
from collections import defaultdict
import json
import os
import glob
from stockfish import Stockfish

class ELORatingSystem:
    def __init__(self, model_path=None, use_half=False):
        """ELO mérési rendszer inicializálása - transformer-alapú HRM modellhez float16 optimalizációval"""
        import sys
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_half = use_half and self.device.type == 'cuda'
        
        if model_path is None:
            model_path = self._select_model()
        self.model_path = model_path
        print(f"🎯 Loading transformer HRM model from: {model_path}")
        
        if self.use_half:
            print("🚀 Float16 optimization enabled for faster inference")
        
        # Load model with AMP optimization
        try:
            self.model, self.model_info = load_model_with_amp(
                self.model_path, 
                device=self.device, 
                use_half=self.use_half
            )
            
            # Build model type string (compatible with Vision Transformer models)
            hidden_dim = self.model_info.get('hidden_dim', 256)
            nhead = self.model_info.get('nhead', 8)
            dim_feedforward = self.model_info.get('dim_feedforward', hidden_dim * 4)
            
            # Check if it's the new Vision Transformer model (no N/T parameters)
            if 'N' in self.model_info and 'T' in self.model_info:
                # Legacy HRM model
                N = self.model_info['N']
                T = self.model_info['T']
                self.model_type = f"HRM-{hidden_dim}-N{N}-T{T}-nhead{nhead}-dff{dim_feedforward}"
            else:
                # New Vision Transformer model
                self.model_type = f"ViT-ResNet-{hidden_dim}-nhead{nhead}-dff{dim_feedforward}"
            
            if self.use_half:
                self.model_type += "-FP16"
                
            print(f"✅ Model loaded successfully: {self.model_type}")
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"📊 Model parameters: {total_params:,}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            exit(0) # no fallback

        self.model.eval()
        self.games_played = []
        # Always initialize move index attributes to avoid AttributeError
        self.uci_move_list = []
        self.uci_move_to_idx = {}
    
    def _select_model(self):
        """Modell kiválasztása a felhasználó által"""
        print("🔍 Searching for available HRM chess models...")
        
        # Keresés .pt fájlokra
        model_files = glob.glob("*.pt")
        
        # Szűrés model fájlokra
        available_models = []
        for file in model_files:
            # Kihagyjuk a data/dataset fájlokat, de megtartjuk a *_model.pt fájlokat
            if any(skip in file.lower() for skip in ['data', 'dataset']) and not file.lower().endswith('_model.pt'):
                continue
            # Extra szűrés: kihagyjuk a train/test fájlokat kivéve ha model fájlok
            if any(skip in file.lower() for skip in ['train', 'test']) and not any(keep in file.lower() for keep in ['model', 'checkpoint']):
                continue
            
            try:
                # Load and check if it's a valid model file
                checkpoint = torch.load(file, map_location='cpu', weights_only=False)
                hidden_dim, model_type = self._detect_model_parameters(checkpoint)
                file_size = os.path.getsize(file) / (1024 * 1024)
                available_models.append({
                    'file': file,
                    'hidden_dim': hidden_dim,
                    'model_type': model_type,
                    'size_mb': file_size
                })
            except Exception as e:
                print(f"⚠️ Could not load {file}: {e}")
                continue
        
        if not available_models:
            print("❌ No HRM models found!")
            print("🔧 Please train a model first using Chess.py")
            raise FileNotFoundError("No models available")
        
        print(f"\n📋 Available models ({len(available_models)} found):")
        print("-" * 80)
        for i, model in enumerate(available_models):
            print(f"  {i+1}. {model['file']:<25} | Hidden: {model['hidden_dim']:<3} | Type: {model['model_type']:<12} | {model['size_mb']:.1f} MB")
        print("-" * 80)
        
        # Felhasználói választás
        while True:
            try:
                choice = input(f"\n🎯 Choose model (1-{len(available_models)}) or press Enter for first: ").strip()
                
                if choice == "":
                    choice_idx = 0
                else:
                    choice_idx = int(choice) - 1
                
                if 0 <= choice_idx < len(available_models):
                    selected = available_models[choice_idx]
                    print(f"✅ Selected: {selected['file']} (Hidden: {selected['hidden_dim']}, Type: {selected['model_type']})")
                    return selected['file']
                else:
                    print(f"❌ Invalid choice. Please enter 1-{len(available_models)}")
            except (ValueError, KeyboardInterrupt):
                print(f"❌ Invalid input. Please enter 1-{len(available_models)}")
    
    def _detect_model_parameters(self, checkpoint):
        """Detect model parameters from checkpoint (supports HRM, ViT-ResNet and Pure ViT models)"""
        hidden_dim = None
        
        # First check if hyperparams are saved in checkpoint
        if 'hyperparams' in checkpoint:
            hyperparams = checkpoint['hyperparams']
            hidden_dim = hyperparams.get('hidden_dim', None)
            model_type = hyperparams.get('model_type', None)
            
            # Check model type from hyperparams
            if model_type == 'pure_vit':
                model_type = "Pure ViT"
                print(f"🔍 Found Pure Vision Transformer model: hidden_dim={hidden_dim}")
            elif 'N' in hyperparams and 'T' in hyperparams:
                model_type = "HRM"
                print(f"🔍 Found HRM model: hidden_dim={hidden_dim}")
            else:
                model_type = "ViT-ResNet"
                print(f"🔍 Found Vision Transformer model: hidden_dim={hidden_dim}")
            return hidden_dim, model_type
        
        # Get the actual model state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Detect model type from layers
        if 'patch_embedding.weight' in state_dict:
            # Pure Vision Transformer model (has patch embedding)
            model_type = "Pure ViT"
            # Detect hidden_dim from patch embedding output dimension
            hidden_dim = state_dict['patch_embedding.weight'].shape[0]
            print(f"🔍 Auto-detected Pure Vision Transformer: hidden_dim={hidden_dim}")
        elif 'value_head.pos' in state_dict:
            # Vision Transformer model (has positional embeddings)
            model_type = "ViT-ResNet"
            # Detect hidden_dim from positional embedding
            if 'value_head.pos' in state_dict:
                hidden_dim = state_dict['value_head.pos'].shape[2]
                print(f"🔍 Auto-detected Vision Transformer: hidden_dim={hidden_dim}")
        else:
            # Legacy HRM model
            model_type = "HRM"
            # Fallback detection methods for hidden_dim
            if 'conv.0.weight' in state_dict:
                hidden_dim = state_dict['conv.0.weight'].shape[0]
                print(f"🔍 Auto-detected HRM model: hidden_dim={hidden_dim}")
        
        if hidden_dim is None:
            print("⚠️ Could not detect hidden_dim, using default: 256")
            hidden_dim = 256
            
        return hidden_dim, model_type
    
    def model_move(self, board, temperature=1.0, debug=False, game_history=None):
        """
        Value-based move selection: 
        1. Try each legal move
        2. Evaluate the resulting position with the model
        3. Choose the move that leads to the best position evaluation
        """
        from hrm_model import game_to_bitplanes, bin_to_score
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None, 0.0
        
        # Build compact history for current position
        if game_history is None or len(game_history) == 0:
            # No game history available - use current position with empty history
            current_compact_history = {
                'starting_fen': board.fen(),
                'moves': [],
                'score': 0.5
            }
        else:
            # Use the actual game history
            current_compact_history = {
                'starting_fen': game_history['starting_fen'],
                'moves': game_history['moves'],
                'score': 0.5
            }
        
        # Evaluate each possible move by looking at the resulting position
        move_evaluations = []
        
        for move in legal_moves:
            # Make the move temporarily
            board_copy = board.copy()
            board_copy.push(move)
            
            # Create history for the position AFTER the move
            new_history = current_compact_history.copy()
            new_history['moves'] = current_compact_history['moves'] + [move.uci()]
            
            # Convert the resulting position to bitplanes (reduced history_length)
            bitplanes = game_to_bitplanes(new_history)
            move_evaluations.append((move, bitplanes))
        
        # Batch evaluation of all resulting positions
        import numpy as np
        if len(move_evaluations) > 0:
            bitplane_batch = torch.from_numpy(np.array([eval[1] for eval in move_evaluations], dtype=np.float32)).to(self.device)
            
            # Use optimized AMP inference
            with torch.no_grad():
                logits_batch = inference_with_amp(self.model, bitplane_batch, use_amp=True)  # [num_moves, num_bins]
            
            move_scores = []
            move_info = []
            
            for i, (move, _) in enumerate(move_evaluations):
                logits = logits_batch[i]
                value_probs = torch.softmax(logits / temperature, dim=0)
                expected_bin = (value_probs * torch.arange(len(value_probs), device=value_probs.device)).sum().item()
                win_percent = bin_to_score(expected_bin, num_bins=len(value_probs))

                move_scores.append(win_percent)
                move_info.append((move, win_percent))
            
            # Choose the best move
            best_idx = int(np.argmax(move_scores))
            selected_move = move_evaluations[best_idx][0]
            move_confidence = float(move_scores[best_idx])
            
            if debug:
                move_info_sorted = sorted(move_info, key=lambda x: x[1], reverse=True)
                print(f"  Top moves: {[(str(m), round(s,3)) for m,s in move_info_sorted[:3]]}")
                
            return selected_move, move_confidence
        
        # Fallback: return first legal move
        return legal_moves[0], 0.5
    
    def play_vs_stockfish(self, stockfish_path, depth=1, time_limit=0.05, pgn=False, model_is_white=True):
        """Továbbfejlesztett játék Stockfish ellen - Python stockfish modult használva"""
        print(f"\n🤖 Playing against Stockfish (depth={depth}, time={time_limit}s)")
        
        try:
            # Initialize Stockfish with Python module
            stockfish = Stockfish(path=stockfish_path)
            
            # Set Stockfish parameters
            stockfish.set_depth(depth)
            stockfish_settings = {
                "Threads": 1,
                "Hash": 16,
                "UCI_Chess960": "false",
                "UCI_LimitStrength": "false"
            }
            stockfish.update_engine_parameters(stockfish_settings)
            
            board = chess.Board()
            moves_played = []
            game_log = []
            
            # Track game history for the model
            game_history = {
                'starting_fen': board.fen(),
                'moves': []
            }
            
            print(f"🎮 Starting game: Model is {'White' if model_is_white else 'Black'}")
            
            while not board.is_game_over() and len(moves_played) < 150:  # Longer games
                if (board.turn == chess.WHITE) == model_is_white:
                    # Model lépése - agresszívebb beállítás
                    try:
                        move, confidence = self.model_move(board, temperature=0.3, debug=False, game_history=game_history)
                        board.push(move)
                        game_history['moves'].append(move.uci())  # Track move in history
                        moves_played.append(f"Model: {move}")
                        game_log.append(f"Move {len(moves_played)}: Model plays {move} (conf: {confidence:.3f})")
                    except Exception as e:
                        print(f"❌ Model error: {e}")
                        break
                else:
                    # Stockfish lépése
                    try:
                        # Update Stockfish with current position before getting move
                        stockfish.set_position(game_history['moves'])
                        
                        # Get best move from Stockfish
                        stockfish_move = stockfish.get_best_move()
                        if stockfish_move is None:
                            print("❌ Stockfish couldn't find a move")
                            break
                            
                        move = chess.Move.from_uci(stockfish_move)
                        if move not in board.legal_moves:
                            print(f"❌ Stockfish suggested illegal move: {stockfish_move}")
                            break
                            
                        board.push(move)
                        game_history['moves'].append(move.uci())  # Track move in history
                        moves_played.append(f"SF: {move}")
                        game_log.append(f"Move {len(moves_played)}: Stockfish plays {move}")
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
                "opponent": f"Stockfish_D{depth}_T{time_limit}",
                "result": result,
                "winner": winner,
                "moves": len(moves_played),
                "termination": termination_reason,
                "model_color": "White" if model_is_white else "Black",
                "game_log": game_log[-10:]  # Last 10 moves for analysis
            }
            
            print(f"Game ended: {result} - Winner: {winner} ({len(moves_played)} moves)")
            print(f"Termination: {termination_reason}")
            
            # Show some key moves
            if len(game_log) > 5:
                print(f"Sample moves: {game_log[0]}, ..., {game_log[-1]}")
            # PGN output if requested
            if pgn:
                game_pgn = chess.pgn.Game.from_board(board)
                game_pgn.headers["White"] = "Model" if model_is_white else "Stockfish"
                game_pgn.headers["Black"] = "Stockfish" if model_is_white else "Model"
                print("\nPGN:\n", game_pgn)
            
            return game_data
                
        except FileNotFoundError:
            print("❌ Stockfish nem található! Telepítsd a stockfish Python modult: pip install stockfish")
            return None
        except Exception as e:
            print(f"❌ Hiba: {e}")
            return None
    
    def play_vs_random(self, games=10):
        """Továbbfejlesztett játék random ellenféllel - kevesebb döntetlen"""
        print(f"\n🎲 Playing {games} games vs Random")
        results = {"wins": 0, "draws": 0, "losses": 0}
        game_details = []
        
        for i in range(games):
            board = chess.Board()
            model_is_white = (i % 2 == 0)  # Váltogatjuk a színeket
            moves_played = 0
            max_moves = 80  # Rövidebb játszmák a döntetlen elkerülésére
            
            # Track game history for the model (same as in play_vs_stockfish)
            game_history = {
                'starting_fen': board.fen(),
                'moves': []
            }
            
            while not board.is_game_over() and moves_played < max_moves:
                if (board.turn == chess.WHITE) == model_is_white:
                    # Model lépése - agresszívebb beállítással
                    move, confidence = self.model_move(board, temperature=0.3, game_history=game_history)
                    board.push(move)
                    game_history['moves'].append(move.uci())  # Track move in history
                else:
                    # Random lépés - de nem teljesen véletlen
                    legal_moves = list(board.legal_moves)
                    
                    # 70% teljesen random, 30% "okosabb" random (captures/checks előnyben)
                    if np.random.random() < 0.7:
                        random_move = np.random.choice(legal_moves)
                    else:
                        # Prioritize captures and checks
                        priority_moves = []
                        for move in legal_moves:
                            if board.is_capture(move) or board.gives_check(move):
                                priority_moves.append(move)
                        
                        if priority_moves:
                            random_move = np.random.choice(priority_moves)
                        else:
                            random_move = np.random.choice(legal_moves)
                    
                    board.push(random_move)
                    game_history['moves'].append(random_move.uci())  # Track move in history
                
                moves_played += 1
            
            # Eredmény kiértékelése
            if board.is_game_over():
                result = board.result()
                print(f"Game {i+1}: {result} - {model_is_white}")
            else:
                # Ha elérte a max_moves limitet, értékeljük pozíció alapján
                # Egyszerű material count
                white_material = len([p for p in board.piece_map().values() if p.color == chess.WHITE])
                black_material = len([p for p in board.piece_map().values() if p.color == chess.BLACK])
                
                if white_material > black_material:
                    result = "1-0"
                elif black_material > white_material:
                    result = "0-1"
                else:
                    result = "1/2-1/2"
            
            # Eredmény rögzítése
            if result == "1-0":
                if model_is_white:
                    results["wins"] += 1
                    game_result = "win"
                else:
                    results["losses"] += 1
                    game_result = "loss"
            elif result == "0-1":
                if model_is_white:
                    results["losses"] += 1
                    game_result = "loss"
                else:
                    results["wins"] += 1
                    game_result = "win"
            else:
                results["draws"] += 1
                game_result = "draw"
            
            game_details.append({
                'game': i+1,
                'model_color': 'White' if model_is_white else 'Black',
                'result': result,
                'game_result': game_result,
                'moves': moves_played
            })
        
        win_rate = results["wins"] / games
        score = (results["wins"] + 0.5 * results["draws"]) / games
        
        print(f"Results: {results['wins']}W-{results['draws']}D-{results['losses']}L")
        print(f"Win rate: {win_rate:.1%}")
        print(f"Score vs Random: {score:.3f} ({results['wins']}W + {0.5 * results['draws']:.1f}D = {score*games:.1f}/{games})")
        
        # Improved ELO calculation
        if score >= 0.95:
            estimated_elo = 1200  # Dominant vs random
        elif score <= 0.05:
            estimated_elo = 400   # Very weak
        else:
            # More granular ELO estimation
            if score >= 0.8:
                estimated_elo = 1000 + (score - 0.8) * 1000  # 1000-1200 range
            elif score >= 0.6:
                estimated_elo = 800 + (score - 0.6) * 1000   # 800-1000 range
            elif score >= 0.4:
                estimated_elo = 600 + (score - 0.4) * 1000   # 600-800 range
            else:
                estimated_elo = 400 + score * 500            # 400-600 range
        
        return results, estimated_elo
        
    def puzzle_rating(self, puzzle_file=None):
        """Továbbfejlesztett taktikai puzzle elemzés debug információkkal"""
        print("\n🧩 TACTICAL PUZZLE RATING:")
        # Mindig puzzles.json-ből töltjük be
        try:
            with open("puzzles.json", "r", encoding="utf-8") as f:
                puzzles = json.load(f)
            print(f"✅ Loaded {len(puzzles)} puzzles from puzzles.json")
        except Exception as e:
            print(f"❌ Error loading puzzles.json: {e}")
            puzzles = []
        
        solved = 0
        total_rating = 0
        total_confidence = 0.0
        detailed_results = []
        
        for i, puzzle in enumerate(puzzles):
            print(f"\n🧩 Puzzle {i+1}: ({puzzle['rating']} ELO)")
            board = chess.Board(puzzle["fen"])
            print(f"Position: {puzzle['fen']}")
            
            # Get model move with debug info
            best_move, confidence = self.model_move(board, temperature=0.3, debug=False)
            
            # Check if move is correct (more flexible matching)
            correct = str(best_move) in puzzle["solution"]
            
            if correct:
                solved += 1
                total_rating += puzzle["rating"]
                total_confidence += confidence
                status = "✅ SOLVED"
            else:
                status = "❌ FAILED"
            
            detailed_results.append({
                'rating': puzzle['rating'],
                'model_move': str(best_move),
                'confidence': confidence,
                'correct_moves': puzzle['solution'],
                'solved': correct
            })
            
            print(f"  Result: {status}")
            print(f"  Model chose: {best_move} (Confidence: {confidence:.3f})")
            print(f"  Expected: {puzzle['solution']}")
        
        print(f"\n🏆 Puzzle Performance Summary:")
        print(f"Solved: {solved}/{len(puzzles)} ({100*solved//len(puzzles)}%)")
        
        if solved > 0:
            avg_puzzle_rating = total_rating // solved
            avg_puzzle_confidence = total_confidence / solved
            print(f"Average solved puzzle rating: {avg_puzzle_rating} ELO | Average solved puzzle confidence: {avg_puzzle_confidence:.3f}")
            # Bonus for multiple solutions
            bonus = min(50 * solved, 200)  # Max 200 ELO bonus
            final_puzzle_rating = avg_puzzle_rating + bonus
        else:
            print(f"No puzzles solved - analyzing move patterns...")
            
            # Analyze move patterns even when failing
            move_types = defaultdict(int)
            for result in detailed_results:
                move = result['model_move']
                if 'O-O' in move or move.startswith('e1g1') or move.startswith('e8g8'):
                    move_types['castling'] += 1
                elif move[0].isupper():  # Piece move
                    move_types['piece_move'] += 1
                else:  # Pawn move
                    move_types['pawn_move'] += 1
            
            print(f"Move pattern analysis: {dict(move_types)}")
            
            # Give some credit for reasonable moves
            final_puzzle_rating = 700 + min(solved * 100, 300)  # 700-1000 range
        
        return final_puzzle_rating
    
    def comprehensive_rating(self, stockfish_path=None):
        """Komplett ELO mérés több módszerrel"""
        print("=" * 60)
        print("🏆 COMPREHENSIVE ELO RATING MEASUREMENT")
        print("=" * 60)
        
        ratings = {}
        
        # 2. Puzzle rating
        puzzle_elo = self.puzzle_rating()
        ratings["puzzle_rating"] = puzzle_elo
        print(f"\n🧩 Puzzle Rating: {puzzle_elo:.0f} ELO")
        
        # 1. Random games alapmérés
        random_results, random_elo = self.play_vs_random(games=20)
        ratings["vs_random"] = random_elo
        print(f"\n📊 vs Random: {random_elo:.0f} ELO")
        
        # 3. Stockfish games (ha elérhető)
        if stockfish_path and os.path.exists(stockfish_path):
            print(f"\n🤖 Testing vs Stockfish...")
            
            # Different Stockfish levels
            stockfish_results = []
            for depth in [1, 2, 3]:
                game_result = self.play_vs_stockfish(stockfish_path, depth=depth, time_limit=0.1)
                if game_result:
                    stockfish_results.append((depth, game_result))
            
            if stockfish_results:
                # CORRECTED ELO calculation based on Stockfish performance
                stockfish_elos = []
                for depth, result in stockfish_results:
                    # More realistic Stockfish strength estimates
                    if depth == 1:
                        base_elo = 1200  # Stockfish depth 1 ≈ 1200 ELO
                    elif depth == 2:
                        base_elo = 1400  # Stockfish depth 2 ≈ 1400 ELO  
                    else:
                        base_elo = 1600  # Stockfish depth 3 ≈ 1600 ELO
                    
                    if result["winner"] == "Model":
                        estimated = base_elo + 50   # Model won: +50 ELO
                    elif result["winner"] == "Draw":
                        estimated = base_elo - 50   # Draw: -50 ELO (slightly weaker)
                    else:
                        estimated = base_elo - 200  # Model lost: -200 ELO
                    
                    # Ensure minimum rating
                    estimated = max(400, estimated)
                    stockfish_elos.append(estimated)
                
                avg_stockfish_elo = np.mean(stockfish_elos)
                ratings["vs_stockfish"] = avg_stockfish_elo
                print(f"\n🤖 vs Stockfish: {avg_stockfish_elo:.0f} ELO")
        
        # Final rating calculation
        weights = {"vs_random": 0.3, "puzzle_rating": 0.4, "vs_stockfish": 0.3}
        
        weighted_rating = 0
        total_weight = 0
        
        for method, rating in ratings.items():
            if method in weights:
                weight = weights[method]
                weighted_rating += rating * weight
                total_weight += weight
        
        final_elo = weighted_rating / total_weight if total_weight > 0 else 1200
        
        print("\n" + "=" * 60)
        print("📈 FINAL ELO ESTIMATION")
        print("=" * 60)
        for method, rating in ratings.items():
            print(f"{method:15}: {rating:4.0f} ELO")
        print("-" * 25)
        print(f"{'FINAL RATING':15}: {final_elo:4.0f} ELO")
        print("=" * 60)
        
        # Save results
        result_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "individual_ratings": ratings,
            "final_elo": final_elo,
            "random_games": random_results
        }
        
        return final_elo, ratings

def main():
    """Fő ELO mérési folyamat konvolúciós HRM modellhez"""
    print("🏆 Convolutional HRM Chess Model ELO Rating Measurement")
    print("=" * 60)
    
    # ELO mérő rendszer inicializálása (automatikus modell választással)
    import sys
    if len(sys.argv) > 1:
        # Explicit modell megadás
        model_path = sys.argv[1]
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return
        elo_system = ELORatingSystem(model_path=model_path)
    else:
        # Interaktív modell választás
        try:
            elo_system = ELORatingSystem()
        except FileNotFoundError:
            return
    
    # Cross-platform Stockfish path detection
    def find_stockfish_path():
        """Auto-detect Stockfish path for different operating systems"""
        if os.name == 'nt':  # Windows
            candidates = [
                "stockfish.exe",
                "./stockfish.exe",
                "C:/stockfish/stockfish.exe",
                "C:/Program Files/stockfish/stockfish.exe"
            ]
        else:  # Linux/Unix
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
    
    stockfish_path = find_stockfish_path()
    
    if not stockfish_path:
        print("⚠️ Stockfish not found or stockfish Python module not installed.")
        print("📍 Install with: pip install stockfish")
        print("📍 Download Stockfish binary from: https://stockfishchess.org/download/")
        print()
    
    # Teljes ELO mérés
    final_elo, _ = elo_system.comprehensive_rating(stockfish_path)
    
    # ELO kategorizálás
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
    
    print("\n🎯 CONVOLUTIONAL HRM MODEL STRENGTH ASSESSMENT:")
    print(f"Model Type: {elo_system.model_type}")
    print(f"ELO Rating: {final_elo:.0f}")
    print(f"Category: {category}")
    print(f"Percentile: Top {100-final_elo//20}% of chess players")

if __name__ == "__main__":
    main()
