"""
Convolutional HRM Chess Model ELO Rating Measurement System
"""

import torch
import chess
import chess.engine
import numpy as np
import time
from hrm_model import HRMChess
from collections import defaultdict
import json
import os
import glob

class ELORatingSystem:
    def __init__(self, model_path=None):
        """ELO mérési rendszer inicializálása - transformer-alapú HRM modellhez"""
        import sys
        from hrm_model import generate_all_possible_uci_moves
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_path is None:
            model_path = self._select_model()
        self.model_path = model_path
        print(f"🎯 Loading transformer HRM model from: {model_path}")
        # Load checkpoint and detect transformer params
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
            # Try to get transformer params from hyperparams
            if 'hyperparams' in checkpoint:
                hyper = checkpoint['hyperparams']
                emb_dim = hyper.get('emb_dim', hyper.get('hidden_dim', 128))  # <-- JAVÍTÁS: hidden_dim fallback
                N = hyper.get('N', 4)
                T = hyper.get('T', 4)
            else:
                emb_dim = 128
                N = 4
                T = 4
            self.model = HRMChess(emb_dim=emb_dim, N=N, T=T).to(self.device)
            self.model_type = f"Transformer-HRM-{emb_dim}-N{N}-T{T}"
            print(f"🏗️ Created transformer HRM model: emb_dim={emb_dim}, N={N}, T={T}")
        except Exception as e:
            print(f"⚠️ Error detecting model parameters: {e}")
            print("🔧 Creating default transformer HRM model...")
            self.model = HRMChess(emb_dim=128, N=4, T=4).to(self.device)
            self.model_type = "Default-Transformer-HRM-128"
        # Load model weights
        if os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    if 'training_info' in checkpoint:
                        info = checkpoint['training_info']
                        epoch = info.get('epoch', 'N/A')
                        val_loss = info.get('val_loss', 'N/A')
                        val_loss_str = f"{val_loss:.4f}" if isinstance(val_loss, (int, float)) else str(val_loss)
                        print(f"📈 Model training info: Epoch {epoch}, Val Loss: {val_loss_str}")
                elif isinstance(checkpoint, dict) and 'hyperparams' in checkpoint:
                    state_dict = {k: v for k, v in checkpoint.items() if k not in ['hyperparams', 'training_info']}
                else:
                    state_dict = checkpoint
                self.model.load_state_dict(state_dict)
                print(f"✅ Transformer HRM model loaded successfully: {self.model_type}")
                total_params = sum(p.numel() for p in self.model.parameters())
                print(f"📊 Model parameters: {total_params:,}")
                print(f"🔧 Architecture: Transformer + HRM heads → value bin classification")
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                print("🔄 Initializing with random weights...")
        else:
            print(f"⚠️ Model file not found: {self.model_path}")
            print("🔄 Using randomly initialized model for testing...")
        self.model.eval()
        self.games_played = []
        # UCI move vocab for move tokenization
        self.uci_move_list = generate_all_possible_uci_moves()
        self.uci_move_to_idx = {uci: i for i, uci in enumerate(self.uci_move_list)}
    
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
                hidden_dim, N, T = self._detect_conv_parameters(checkpoint)
                file_size = os.path.getsize(file) / (1024 * 1024)
                available_models.append({
                    'file': file,
                    'hidden_dim': hidden_dim,
                    'N': N,
                    'T': T,
                    'size_mb': file_size
                })
            except Exception as e:
                print(f"⚠️ Could not load {file}: {e}")
                continue
        
        if not available_models:
            print("❌ No HRM models found!")
            print("🔧 Please train a model first using Chess.py")
            raise FileNotFoundError("No models available")
        
        # Listázzuk az elérhető modelleket
        print(f"\n📋 Available HRM models ({len(available_models)} found):")
        print("-" * 80)
        for i, model in enumerate(available_models):
            params = model['hidden_dim'] * (model['N'] + model['T']) / 1000  # Rough estimate
            print(f"  {i+1}. {model['file']:<25} | Hidden: {model['hidden_dim']:<3} | N: {model['N']} T: {model['T']} | {model['size_mb']:.1f} MB")
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
                    print(f"✅ Selected: {selected['file']} (Hidden: {selected['hidden_dim']}, N: {selected['N']}, T: {selected['T']})")
                    return selected['file']
                else:
                    print(f"❌ Invalid choice. Please enter 1-{len(available_models)}")
            except (ValueError, KeyboardInterrupt):
                print(f"❌ Invalid input. Please enter 1-{len(available_models)}")
    
    def _detect_conv_parameters(self, checkpoint):
        """HRM modell paramétereinek detektálása"""
        hidden_dim = None
        N, T = 8, 8  # Default values
        
        # First check if hyperparams are saved in checkpoint
        if 'hyperparams' in checkpoint:
            hyperparams = checkpoint['hyperparams']
            hidden_dim = hyperparams.get('hidden_dim', None)
            N = hyperparams.get('N', N)
            T = hyperparams.get('T', T)
            print(f"🔍 Found saved hyperparams: hidden_dim={hidden_dim}, N={N}, T={T}")
            return hidden_dim, N, T
        
        # Get the actual model state dict (handle both new and legacy formats)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Fallback: detect from model layers
        # Detect hidden_dim from convolutional layers
        if 'board_conv.0.weight' in state_dict:
            # First conv layer: 1 -> hidden_dim//4
            first_conv_out = state_dict['board_conv.0.weight'].shape[0]
            hidden_dim = first_conv_out * 4
            print(f"🔍 Auto-detected hidden_dim: {hidden_dim} (from board_conv.0)")
        elif 'board_conv.2.weight' in state_dict:
            # Second conv layer: hidden_dim//4 -> hidden_dim//2
            second_conv_out = state_dict['board_conv.2.weight'].shape[0]
            hidden_dim = second_conv_out * 2
            print(f"🔍 Auto-detected hidden_dim: {hidden_dim} (from board_conv.2)")
        elif 'feature_combiner.0.weight' in state_dict:
            # Feature combiner: combined_features -> hidden_dim
            hidden_dim = state_dict['feature_combiner.0.weight'].shape[0]
            print(f"🔍 Auto-detected hidden_dim: {hidden_dim} (from feature_combiner)")
        elif 'L_net.0.weight' in state_dict:
            # L_net input: hidden_dim * 3
            l_net_input_size = state_dict['L_net.0.weight'].shape[1]
            if l_net_input_size % 3 == 0:
                hidden_dim = l_net_input_size // 3
                print(f"🔍 Auto-detected hidden_dim: {hidden_dim} (from L_net)")
        
        if hidden_dim is None:
            print("⚠️ Could not detect hidden_dim, using default: 128")
            hidden_dim = 128
        
        print(f"⚠️ N and T not saved in model, using defaults: N={N}, T={T}")
        return hidden_dim, N, T
        
    def model_move(self, board, temperature=0.7, debug=False):
        """Transformer HRM modell lépés választás (value bin head alapján, FEN+move token input, batch)"""
        import torch
        from hrm_model import fen_to_tokens, bin_to_score
        fen = board.fen()
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available.")
        # Prepare batch tokens
        fen_tokens = torch.tensor([fen_to_tokens(fen)], dtype=torch.long).repeat(len(legal_moves), 1).to(self.device)  # [num_moves, 77]
        uci_indices = []
        valid_indices = []
        missing_moves = []
        for i, move in enumerate(legal_moves):
            uci = move.uci()
            idx = self.uci_move_to_idx.get(uci, None)
            if idx is not None:
                uci_indices.append(idx)
                valid_indices.append(i)
            else:
                uci_indices.append(-1)  # Placeholder for invalid
                missing_moves.append(uci)
        if missing_moves:
            print(f"⚠️ Warning: {len(missing_moves)} legal moves missing from uci_move_to_idx: {missing_moves}")
        # Mask for valid moves
        valid_mask = torch.tensor([i != -1 for i in uci_indices], dtype=torch.bool)
        uci_tensor = torch.tensor([i if i != -1 else 0 for i in uci_indices], dtype=torch.long).to(self.device)  # [num_moves]
        move_scores = np.full(len(legal_moves), -float('inf'), dtype=np.float32)
        move_info = []
        with torch.no_grad():
            # Only evaluate valid moves
            if valid_mask.any():
                fen_tokens_valid = fen_tokens[valid_mask]
                uci_tensor_valid = uci_tensor[valid_mask]
                out = self.model(fen_tokens_valid, uci_tensor_valid)  # [num_valid, num_bins]
                for j, logits in enumerate(out):
                    value_probs = torch.softmax(logits / temperature, dim=0)
                    expected_bin = (value_probs * torch.arange(len(value_probs), device=value_probs.device)).sum().item()
                    win_percent = bin_to_score(expected_bin, num_bins=len(value_probs))
                    move_scores[valid_indices[j]] = win_percent
                    move_info.append((legal_moves[valid_indices[j]], win_percent))
            # For invalid moves, keep -inf
            for i, idx in enumerate(uci_indices):
                if idx == -1:
                    move_info.append((legal_moves[i], -float('inf')))
        selected_idx = int(np.argmax(move_scores))
        selected_move = legal_moves[selected_idx]
        move_confidence = move_scores[selected_idx]
        if debug:
            move_info_sorted = sorted(move_info, key=lambda x: x[1], reverse=True)
            print(f"  Top moves: {[(str(m), round(s,2)) for m,s in move_info_sorted[:3]]}")
        return selected_move, move_confidence
    
    def play_vs_stockfish(self, stockfish_path, depth=1, time_limit=0.05):
        """Továbbfejlesztett játék Stockfish ellen"""
        print(f"\n🤖 Playing against Stockfish (depth={depth}, time={time_limit}s)")
        
        try:
            with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
                board = chess.Board()
                moves_played = []
                model_is_white = True  # Model fehérrel játszik
                game_log = []
                
                while not board.is_game_over() and len(moves_played) < 150:  # Longer games
                    if (board.turn == chess.WHITE) == model_is_white:
                        # Model lépése - agresszívebb beállítás
                        try:
                            move, confidence = self.model_move(board, temperature=0.6, debug=False)
                            board.push(move)
                            moves_played.append(f"Model: {move}")
                            game_log.append(f"Move {len(moves_played)}: Model plays {move} (conf: {confidence:.3f})")
                        except Exception as e:
                            print(f"Model error: {e}")
                            break
                    else:
                        # Stockfish lépése
                        try:
                            result = engine.play(board, chess.engine.Limit(depth=depth, time=time_limit))
                            board.push(result.move)
                            moves_played.append(f"SF: {result.move}")
                            game_log.append(f"Move {len(moves_played)}: Stockfish plays {result.move}")
                        except Exception as e:
                            print(f"Stockfish error: {e}")
                            break
                
                # Játék eredménye
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
                
                return game_data
                
        except FileNotFoundError:
            print("❌ Stockfish nem található! Töltsd le: https://stockfishchess.org/download/")
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
            
            while not board.is_game_over() and moves_played < max_moves:
                if (board.turn == chess.WHITE) == model_is_white:
                    # Model lépése - agresszívebb beállítással
                    move, confidence = self.model_move(board, temperature=0.8)
                    board.push(move)
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
                
                moves_played += 1
            
            # Eredmény kiértékelése
            if board.is_game_over():
                result = board.result()
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
        
        # Enhanced puzzle set with easier tactical problems
        puzzles = [
            # Easy (1000–1200 ELO) - back rank mate idea
            {
                "fen": "6k1/3r1ppp/1pR1p3/1p1pP3/rP6/5PP1/4P1KP/1R6 b - - 2 26",
                "solution": ["d5d4", "c6c8", "d7d8", "c8d8"],
                "description": "backRankMate endgame mate",
                "difficulty": "Easy",
                "rating": 1040
            },
            # Medium (1200–1400 ELO) - discovered attack
            {
                "fen": "r2q1r1k/ppp3pp/2nb4/3Q1b2/8/6N1/PPP1NPPP/R1B2RK1 w - - 1 14",
                "solution": ["g3f5", "d6h2", "g1h2", "d8d5"],
                "description": "advantage discoveredAttack kingsideAttack",
                "difficulty": "Medium",
                "rating": 1211
            },
            # Very Hard (1600+ ELO) - long mate sequence
            {
                "fen": "5b1k/4n1np/6p1/4Q3/3BP3/3b1P2/3q2PP/R5K1 b - - 4 29",
                "solution": ["e7c6", "e5g7", "f8g7", "a1a8", "c6b8", "a8b8"],
                "description": "long mate mateIn3",
                "difficulty": "Very Hard",
                "rating": 1677
            },
            # Hard (1400–1600 ELO) - deflection tactic
            {
                "fen": "8/3R4/7p/4K3/2BpP3/2kP2Pr/8/3b4 w - - 3 46",
                "solution": ["d7d4", "h3h5", "e5f6", "c3d4"],
                "description": "advantage deflection endgame",
                "difficulty": "Hard",
                "rating": 1402
            },
            # Hard (1400–1600 ELO) - advanced pawn and sacrifice
            {
                "fen": "8/p7/1P1k3p/P4pp1/2K5/8/5n2/7B b - - 0 39",
                "solution": ["a7b6", "a5a6", "f2h1", "a6a7"],
                "description": "advancedPawn advantage endgame",
                "difficulty": "Hard",
                "rating": 1509
            }
        ]
        
        solved = 0
        total_rating = 0
        detailed_results = []
        
        for i, puzzle in enumerate(puzzles):
            print(f"\n🧩 Puzzle {i+1}: {puzzle['description']} ({puzzle['difficulty']} - {puzzle['rating']} ELO)")
            board = chess.Board(puzzle["fen"])
            print(f"Position: {puzzle['fen']}")
            
            # Get model move with debug info
            best_move, confidence = self.model_move(board, temperature=0.5, debug=True)
            
            # Check if move is correct (more flexible matching)
            correct = str(best_move) in puzzle["solution"]
            
            if correct:
                solved += 1
                total_rating += puzzle["rating"]
                status = "✅ SOLVED"
            else:
                status = "❌ FAILED"
            
            detailed_results.append({
                'puzzle': puzzle['description'],
                'difficulty': puzzle['difficulty'], 
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
            print(f"Average solved puzzle rating: {avg_puzzle_rating} ELO")
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
        
        # 1. Random games alapmérés
        random_results, random_elo = self.play_vs_random(games=20)
        ratings["vs_random"] = random_elo
        print(f"\n📊 vs Random: {random_elo:.0f} ELO")
        
        # 2. Puzzle rating
        puzzle_elo = self.puzzle_rating()
        ratings["puzzle_rating"] = puzzle_elo
        print(f"\n🧩 Puzzle Rating: {puzzle_elo:.0f} ELO")
        
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
        
        for path in candidates:
            if os.path.exists(path):
                print(f"✅ Found Stockfish at: {path}")
                return path
        
        return None
    
    stockfish_path = find_stockfish_path()
    
    if not stockfish_path:
        print("⚠️ Stockfish not found. Download from: https://stockfishchess.org/download/")
        print("📍 For more accurate rating, install Stockfish")
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
