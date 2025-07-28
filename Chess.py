import chess
import chess.pgn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numba
import os
import sys
import csv
import subprocess
import time
import math

# Import HRM model and related functions
from hrm_model import HRMChess, PolicyValueDataset, train_step, train_loop, quick_train_eval

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def detect_gpu_memory_and_optimize_training():
    """
    GPU memória detektálás és automatikus batch size/learning rate optimalizálás
    
    Returns:
        dict: Optimalizált training paraméterek
    """
    print(f"\n🔍 GPU MEMORY DETECTION & TRAINING OPTIMIZATION")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - GPU required for training!")
        print("🚨 This training requires GPU acceleration.")
        print("💡 Please ensure CUDA is installed and GPU is available.")
        exit(1)
    
    try:
        # GPU információk lekérdezése
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        
        # Memória információk
        total_memory = torch.cuda.get_device_properties(current_device).total_memory
        reserved_memory = torch.cuda.memory_reserved(current_device)
        allocated_memory = torch.cuda.memory_allocated(current_device)
        
        # Elérhető memória kiszámítása (GB-ban)
        total_gb = total_memory / (1024**3)
        available_gb = (total_memory - reserved_memory) / (1024**3)
        free_gb = (total_memory - allocated_memory) / (1024**3)
        
        print(f"🖥️ GPU Device: {gpu_name}")
        print(f"📊 Total VRAM: {total_gb:.1f} GB")
        print(f"📊 Available VRAM: {available_gb:.1f} GB")
        print(f"📊 Free VRAM: {free_gb:.1f} GB")
        print(f"🔢 CUDA Devices: {gpu_count}")
        
        # Batch size optimalizálás SZABAD GPU memória alapján
        if free_gb >= 20:  # Bőven van szabad hely high-end kártyákon
            batch_config = {
                'batch_size': 64,
                'lr_multiplier': 1.5,  # Nagyobb batch → nagyobb LR
                'optimization_level': 'HIGH_END_FREE'
            }
            print(f"🚀 HIGH-END FREE MEMORY ({free_gb:.1f}GB+ available)")
            
        elif free_gb >= 14:  # Jó mennyiségű szabad memória
            batch_config = {
                'batch_size': 48,
                'lr_multiplier': 1.3,
                'optimization_level': 'HIGH_FREE'
            }
            print(f"🔥 HIGH FREE MEMORY ({free_gb:.1f}GB available)")
            
        elif free_gb >= 10:  # Közepes szabad memória
            batch_config = {
                'batch_size': 32,
                'lr_multiplier': 1.1,
                'optimization_level': 'MID_HIGH_FREE'
            }
            print(f"⚡ MID-HIGH FREE MEMORY ({free_gb:.1f}GB available)")
            
        elif free_gb >= 6:   # Átlagos szabad memória
            batch_config = {
                'batch_size': 24,
                'lr_multiplier': 1.0,
                'optimization_level': 'MID_FREE'
            }
            print(f"💪 MID FREE MEMORY ({free_gb:.1f}GB available)")
            
        elif free_gb >= 4:   # Kevés szabad memória
            batch_config = {
                'batch_size': 16,
                'lr_multiplier': 0.9,
                'optimization_level': 'LOW_MID_FREE'
            }
            print(f"🎯 LOW-MID FREE MEMORY ({free_gb:.1f}GB available)")
            
        elif free_gb >= 2:   # Nagyon kevés szabad memória
            batch_config = {
                'batch_size': 12,
                'lr_multiplier': 0.8,
                'optimization_level': 'LOW_FREE'
            }
            print(f"⚠️ LOW FREE MEMORY ({free_gb:.1f}GB available)")
            
        else:  # <2GB szabad VRAM
            print(f"❌ Insufficient free GPU memory (<2GB, available: {free_gb:.1f}GB)")
            print("🚨 Training requires at least 2GB free VRAM.")
            print("💡 Please close other GPU applications or use a smaller model.")
            exit(1)
        
        # Memória foglaltság alapú finomhangolás
        memory_usage_ratio = allocated_memory / total_memory
        if memory_usage_ratio > 0.3:  # Ha már 30%+ foglalt
            print(f"⚠️ High memory usage detected ({memory_usage_ratio*100:.1f}%)")
            print("🔧 Reducing batch size for safety...")
            batch_config['batch_size'] = max(8, int(batch_config['batch_size'] * 0.7))
            batch_config['lr_multiplier'] *= 0.9
        
        # Safety check - dynamic memory test
        print(f"\n🧪 MEMORY SAFETY TEST")
        test_passed = True
        try:
            # Teszt tensor létrehozása a választott batch size-hoz
            test_batch_size = batch_config['batch_size']
            test_tensor = torch.randn(test_batch_size, 72, device=device)
            test_tensor2 = torch.randn(test_batch_size, 64, 64, device=device)
            
            # Memória felhasználás ellenőrzése
            test_memory = torch.cuda.memory_allocated(current_device)
            test_memory_gb = test_memory / (1024**3)
            
            print(f"   ✅ Test batch ({test_batch_size}) allocated: {test_memory_gb:.2f} GB")
            
            # Cleanup
            del test_tensor, test_tensor2
            torch.cuda.empty_cache()
            
            # Ha a teszt túl sok memóriát használ, csökkentjük a batch size-t
            if test_memory_gb > free_gb * 0.6:  # Ha több mint 60% szabad VRAM-ot használná
                print(f"   ⚠️ Batch size too large for free memory, reducing...")
                batch_config['batch_size'] = max(8, int(batch_config['batch_size'] * 0.6))
                batch_config['lr_multiplier'] *= 0.9
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"   ❌ Memory test failed - reducing batch size")
                batch_config['batch_size'] = max(8, int(batch_config['batch_size'] * 0.5))
                batch_config['lr_multiplier'] *= 0.8
                test_passed = False
            torch.cuda.empty_cache()
        
        # Végső konfiguráció
        result = {
            'batch_size': batch_config['batch_size'],
            'lr_multiplier': batch_config['lr_multiplier'],
            'memory_gb': total_gb,
            'free_memory_gb': free_gb,  # Hozzáadva a szabad memória info
            'device_name': gpu_name,
            'optimization_level': batch_config['optimization_level'],
            'memory_test_passed': test_passed
        }
        
        print(f"\n✅ OPTIMIZED TRAINING CONFIGURATION (FREE MEMORY BASED):")
        print(f"   🎯 Batch Size: {result['batch_size']}")
        print(f"   📈 LR Multiplier: {result['lr_multiplier']:.2f}x")
        print(f"   🏷️ Level: {result['optimization_level']}")
        print(f"   💾 Free VRAM: {free_gb:.1f}GB / {total_gb:.1f}GB total")
        print(f"   🧪 Memory Test: {'✅ PASSED' if test_passed else '⚠️ ADJUSTED'}")
        
        return result
        
    except Exception as e:
        print(f"❌ GPU detection failed: {e}")
        print("� Cannot proceed without GPU information.")
        exit(1)

def load_pgn_data(pgn_path, fen_to_tensor, max_games=None, max_moves=40, min_elo=1600):
    all_states, all_policies, all_fens = [], [], []
    
    # Debug counters
    stats = {
        'total_games': 0,
        'elo_rejected': 0,
        'result_rejected': 0,
        'timecontrol_rejected': 0,
        'processed_games': 0,
        'positions_extracted': 0,
        'piece_moves': 0,
        'pawn_moves': 0,
        'captures': 0,
        'tactical_moves': 0
    }
    
    with open(pgn_path, encoding="utf-8") as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
                
            stats['total_games'] += 1
            
            if stats['total_games'] % 1000 == 0:
                print(f"PGN: {stats['total_games']} ellenőrizve, {stats['processed_games']} feldolgozva, "
                      f"{stats['positions_extracted']} pozíció")
                      
            if max_games is not None and stats['processed_games'] >= max_games:
                break
            
            # Szűrés: csak megfelelő játékosok játszmai
            try:
                white_elo = int(game.headers.get("WhiteElo", "0"))
                black_elo = int(game.headers.get("BlackElo", "0"))
                
                # MAGASABB ELO minimum - jobb minőség
                if not (white_elo >= min_elo and black_elo >= min_elo and 
                       white_elo < 2800 and black_elo < 2800):
                    stats['elo_rejected'] += 1
                    continue
                    
                # Időkontroll és eredmény szűrés
                time_control = game.headers.get("TimeControl", "")
                result = game.headers.get("Result", "*")
                
                # Csak befejezett játszmák
                if not (result in ["1-0", "0-1"]):  # Kizárjuk a döntetleneket!
                    stats['result_rejected'] += 1
                    continue
                    
                if not (len(time_control) >= 3):
                    stats['timecontrol_rejected'] += 1
                    continue
                        
                # Ha minden szűrési feltétel teljesül, feldolgozzuk a játszmát
                board = game.board()
                node = game
                move_count = 0
                
                # KÖZÉPJÁTÉK és VÉGJÁTÉK hangsúlyozása
                opening_moves_to_skip = 8  # Több nyitás kihagyása
                current_move = 0
                game_moves = []
                
                # Előre gyűjtjük a lépéseket minőségi szűréshez
                temp_board = game.board()
                temp_node = game
                while temp_node.variations:
                    move = temp_node.variation(0).move
                    game_moves.append((temp_board.copy(), move))
                    temp_board.push(move)
                    temp_node = temp_node.variation(0)
                
                # INTELLIGENS LÉPÉS SZELEKTÁLÁS
                for board_state, move in game_moves:
                    current_move += 1
                    
                    # Skip opening moves
                    if current_move <= opening_moves_to_skip:
                        board.push(move)
                        node = node.variation(0)
                        continue
                    
                    # MINŐSÉGI SZŰRÉS - csak érdekes lépések
                    is_capture = board.is_capture(move)
                    is_check = board.gives_check(move)
                    is_piece_move = board.piece_at(move.from_square) and \
                                   board.piece_at(move.from_square).piece_type != chess.PAWN
                    is_tactical = is_capture or is_check or \
                                 board.is_castling(move) or \
                                 board.is_en_passant(move)
                    
                    # BALANCED SAMPLING - preferáljuk a bábu lépéseket és taktikai lépéseket
                    include_move = False
                    
                    if is_tactical:  # Mindig befoglaljuk a taktikai lépéseket
                        include_move = True
                        stats['tactical_moves'] += 1
                    elif is_piece_move:  # 80% eséllyel befoglaljuk a bábu lépéseket
                        if np.random.random() < 0.8:
                            include_move = True
                            stats['piece_moves'] += 1
                    else:  # Csak 30% eséllyel befoglaljuk a gyalog lépéseket
                        if np.random.random() < 0.3:
                            include_move = True
                            stats['pawn_moves'] += 1
                    
                    if include_move:
                        state = fen_to_tensor(board.fen())
                        policy_sparse = (move.from_square, move.to_square)
                        current_fen = board.fen()  # Store the actual FEN
                        
                        all_states.append(state)
                        all_policies.append(policy_sparse)
                        all_fens.append(current_fen)  # Store FEN for Stockfish evaluation
                        stats['positions_extracted'] += 1
                        
                        if is_capture:
                            stats['captures'] += 1
                    
                    board.push(move)
                    node = node.variation(0)
                    move_count += 1
                    
                    if move_count >= max_moves:
                        break
                
                stats['processed_games'] += 1
                        
            except (ValueError, KeyError):
                continue  # Skip games with missing/invalid data
    
    # ENHANCED statistics
    print(f"\n📊 ENHANCED PGN Processing Statistics:")
    print(f"   Total games examined: {stats['total_games']:,}")
    print(f"   Successfully processed: {stats['processed_games']:,}")
    print(f"   Positions extracted: {stats['positions_extracted']:,}")
    print(f"   ├── Piece moves: {stats['piece_moves']:,} ({stats['piece_moves']/max(stats['positions_extracted'], 1)*100:.1f}%)")
    print(f"   ├── Pawn moves: {stats['pawn_moves']:,} ({stats['pawn_moves']/max(stats['positions_extracted'], 1)*100:.1f}%)")
    print(f"   ├── Captures: {stats['captures']:,} ({stats['captures']/max(stats['positions_extracted'], 1)*100:.1f}%)")
    print(f"   └── Tactical moves: {stats['tactical_moves']:,} ({stats['tactical_moves']/max(stats['positions_extracted'], 1)*100:.1f}%)")
    
    print(f"✅ BALANCED PGN adatok: {len(all_states):,} pozíció {stats['processed_games']} játszmából")
    return all_states, all_policies, all_fens

def load_puzzle_data(csv_path, fen_to_tensor, max_puzzles=None, min_rating=800, max_rating=2200):
    """
    Lichess puzzle CSV betöltése taktikai training adatokhoz
    
    Args:
        csv_path: Path to lichess puzzle CSV file
        fen_to_tensor: FEN konvertáló függvény
        max_puzzles: Maximum puzzles to load
        min_rating: Minimum puzzle rating
        max_rating: Maximum puzzle rating
    
    Returns:
        puzzle_states, puzzle_policies, puzzle_fens
    """
    print(f"\n🧩 LOADING TACTICAL PUZZLES from {csv_path}")
    
    puzzle_states = []
    puzzle_policies = []
    puzzle_fens = []
    
    # Puzzle statistics
    stats = {
        'total_puzzles': 0,
        'rating_filtered': 0,
        'parse_errors': 0,
        'processed_puzzles': 0,
        'difficulty_easy': 0,
        'difficulty_medium': 0,
        'difficulty_hard': 0
    }
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            
            # Skip header if present
            first_row = next(reader, None)
            if first_row and 'PuzzleId' in first_row[0]:
                pass  # Header row, continue
            else:
                # First row is data, process it
                csvfile.seek(0)
                reader = csv.reader(csvfile)
            
            for row in reader:
                if len(row) < 9:  # Ensure minimum columns
                    continue
                    
                stats['total_puzzles'] += 1
                
                if stats['total_puzzles'] % 5000 == 0:
                    print(f"Puzzles: {stats['total_puzzles']:,} ellenőrizve, {stats['processed_puzzles']:,} feldolgozva")
                
                if max_puzzles and stats['processed_puzzles'] >= max_puzzles:
                    break
                
                try:
                    # CSV format: PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl
                    puzzle_id = row[0]
                    fen = row[1]
                    moves = row[2].split()
                    rating = int(row[3])
                    themes = row[7] if len(row) > 7 else ""
                    
                    # Rating filter
                    if not (min_rating <= rating <= max_rating):
                        stats['rating_filtered'] += 1
                        continue
                    
                    # Parse the first move (puzzle solution)
                    if len(moves) < 1:
                        stats['parse_errors'] += 1
                        continue
                    
                    first_move = moves[0]
                    
                    # Convert UCI move to board squares
                    board = chess.Board(fen)
                    try:
                        move = chess.Move.from_uci(first_move)
                        if move not in board.legal_moves:
                            stats['parse_errors'] += 1
                            continue
                    except:
                        stats['parse_errors'] += 1
                        continue
                    
                    # Convert position to tensor
                    state = fen_to_tensor(fen)
                    policy = (move.from_square, move.to_square)
                    
                    puzzle_states.append(state)
                    puzzle_policies.append(policy)
                    puzzle_fens.append(fen)
                    stats['processed_puzzles'] += 1
                    
                    # Difficulty categorization
                    if rating < 1200:
                        stats['difficulty_easy'] += 1
                    elif rating < 1800:
                        stats['difficulty_medium'] += 1
                    else:
                        stats['difficulty_hard'] += 1
                        
                except (ValueError, IndexError, chess.InvalidMoveError) as e:
                    stats['parse_errors'] += 1
                    continue
    
    except FileNotFoundError:
        print(f"⚠️ Puzzle file not found: {csv_path}")
        return [], [], []
    except Exception as e:
        print(f"⚠️ Error loading puzzles: {e}")
        return [], [], []
    
    # Enhanced statistics
    print(f"\n📊 PUZZLE PROCESSING Statistics:")
    print(f"   Total puzzles examined: {stats['total_puzzles']:,}")
    print(f"   Successfully processed: {stats['processed_puzzles']:,}")
    print(f"   Rating filtered: {stats['rating_filtered']:,}")
    print(f"   Parse errors: {stats['parse_errors']:,}")
    print(f"   ├── Easy (< 1200): {stats['difficulty_easy']:,} ({stats['difficulty_easy']/max(stats['processed_puzzles'], 1)*100:.1f}%)")
    print(f"   ├── Medium (1200-1800): {stats['difficulty_medium']:,} ({stats['difficulty_medium']/max(stats['processed_puzzles'], 1)*100:.1f}%)")
    print(f"   └── Hard (> 1800): {stats['difficulty_hard']:,} ({stats['difficulty_hard']/max(stats['processed_puzzles'], 1)*100:.1f}%)")
    
    print(f"✅ TACTICAL PUZZLES: {len(puzzle_states):,} pozíció betöltve")
    return puzzle_states, puzzle_policies, puzzle_fens

# Egyszerűsített board encoder - compact reprezentáció
def fen_to_tensor(fen):
    board = chess.Board(fen)
    
    # Compact piece encoding: 4 bit elég lenne, de használunk 8-bit integers
    piece_map = {
        'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
        'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12
    }
    
    # 64 mező mint uint8 + extra információk mint float16
    board_state = np.zeros(64 + 8, dtype=np.uint8)  # Ultra compact representation
    
    # Bábuk pozíciói (64 dimenzió mint integer)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            board_state[square] = piece_map[piece.symbol()]
    
    # Extra információk (8 dimenzió) - ezek maradnak float típusúak
    extra_info = np.zeros(8, dtype=np.float16)
    extra_info[0] = float(board.turn)  # Ki van soron (0=fekete, 1=fehér)
    extra_info[1] = float(board.has_kingside_castling_rights(chess.WHITE))
    extra_info[2] = float(board.has_queenside_castling_rights(chess.WHITE))
    extra_info[3] = float(board.has_kingside_castling_rights(chess.BLACK))
    extra_info[4] = float(board.has_queenside_castling_rights(chess.BLACK))
    extra_info[5] = float(board.ep_square) if board.ep_square is not None else -1.0
    extra_info[6] = float(board.halfmove_clock) / 100.0
    extra_info[7] = float(board.fullmove_number) / 100.0
    
    # Combine board and extra info - convert to float32 for neural network
    result = np.zeros(72, dtype=np.float32)
    result[:64] = board_state[:64].astype(np.float32)  # Convert pieces to float
    result[64:] = extra_info.astype(np.float32)  # Convert extra info to float
    
    return result

def optuna_hyperparameter_optimization(states, policies, values, n_trials=20):
    """
    Optuna-based hyperparameter optimization Policy+Value modellhez
    """
    import optuna
    from optuna.samplers import TPESampler
    
    print("\n🚀 OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("="*50)
    print(f"Running {n_trials} intelligent trials with TPE sampler")
    
    dataset = PolicyValueDataset(states, policies, values)
    dataset_size = len(states)
    
    def objective(trial):
        """Optuna objective function"""
        
        # Hyperparameter suggestions
        hidden_dim = trial.suggest_categorical('hidden_dim', [128, 160, 192, 224, 256, 288])
        N = trial.suggest_int('N', 2, 10)
        T = trial.suggest_int('T', 2, 10)
        
        # Learning rate és batch size ÖSSZEFÜGGŐ optimalizálás
        # Kisebb batch → kisebb lr (stabilabb gradiens, de lassabb konvergencia)
        # Nagyobb batch → nagyobb lr (gyorsabb konvergencia, stabil nagy batch gradiens)
        batch_size = trial.suggest_categorical('batch_size', [16, 24, 32])
        
        # LR scaling based on batch size - empirikus batch size scaling
        lr_base = trial.suggest_float('lr_base', 8e-5, 3e-4, log=True)
        
        if batch_size == 16:
            lr = lr_base * 0.75  # 25% csökkentés kis batch-hez
        elif batch_size == 24:
            lr = lr_base  # Referencia lr (baseline)
        else:  # batch_size == 32
            lr = lr_base * 1.3  # 30% növelés nagy batch-hez (sqrt scaling közelítés)
        
        # Model létrehozása
        model = HRMChess(
            input_dim=72,
            hidden_dim=hidden_dim,
            N=N,
            T=T
        ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        hrm_steps = N * T
        
        # Quick evaluation with early stopping
        try:
            # Mini training - csak 2 epoch gyors teszteléshez
            val_loss = quick_train_eval(
                model, dataset, 
                epochs=2, 
                batch_size=batch_size, 
                lr=lr,
                subset_ratio=0.1,  # Csak 10% adaton tesztel a gyorsaságért
                device=device
            )
            
            # Penalty for too many parameters (efficiency optimization)
            param_penalty = total_params / 1_000_000  # 1M param = 1.0 penalty
            steps_penalty = max(0, hrm_steps - 10) * 0.1  # 10+ lépés penalty
            
            # Combined objective: validation loss + efficiency penalties
            objective_value = val_loss + param_penalty * 0.01 + steps_penalty * 0.02
            
            # Prune trial ha túl rossz
            if val_loss > 10.0:  # Prune clearly bad trials
                raise optuna.TrialPruned()
            
            return objective_value
            
        except Exception as e:
            print(f"Trial failed: {e}")
            return float('inf')
    
    # Optuna study létrehozása
    sampler = TPESampler(seed=42)  # Reproducible results
    study = optuna.create_study(
        direction='minimize', 
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
    )
    
    # Optimization futtatása
    print(f"🔬 Starting {n_trials} Optuna trials...")
    study.optimize(objective, n_trials=n_trials, timeout=1800)  # Max 30 min
    
    # Legjobb eredmények
    print(f"\n🏆 OPTUNA OPTIMIZATION RESULTS")
    print("="*50)
    
    best_trial = study.best_trial
    best_params = best_trial.params
    
    print(f"Best trial value: {best_trial.value:.4f}")
    print(f"Best parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    # Top 5 trial eredmények
    print(f"\n📊 Top 5 trials:")
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('inf'))
    for i, trial in enumerate(sorted_trials[:5]):
        if trial.value is not None:
            print(f"{i+1}. Value: {trial.value:.4f}, Params: {trial.params}")
    
    # Visszatérés optimális konfigurációval
    # Helyes lr kiszámítása lr_base és batch_size alapján
    lr_base = best_params['lr_base']
    batch_size = best_params['batch_size']
    
    if batch_size == 16:
        lr = lr_base * 0.75
    elif batch_size == 24:
        lr = lr_base
    else:  # batch_size == 32
        lr = lr_base * 1.3
    
    best_config = {
        'hidden_dim': best_params['hidden_dim'],
        'N': best_params['N'],
        'T': best_params['T'],
        'lr': lr,  # Számított lr érték
        'lr_base': lr_base,  # Eredeti lr_base is megőrzése
        'batch_size': batch_size,
        'name': f"Optuna-Optimized-{best_params['N']}x{best_params['T']}-BS{batch_size}"
    }
    
    return best_config

def get_user_choice():
    """Interactive parameter selection"""
    print("\n" + "="*60)
    print("� HRM CHESS MODEL CONFIGURATION")
    print("="*60)
    print("Choose your training approach:")
    print("1. 🔧 Hyperparameter Optimization (Automatic - finds best N, T, hidden_dim)")
    print("2. ⚙️  Manual Parameter Settings (Choose your own N, T, hidden_dim)")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-2): ").strip()
            if choice in ['1', '2']:
                return int(choice)
            else:
                print("❌ Please enter 1 or 2")
        except (ValueError, KeyboardInterrupt):
            print("\n❌ Invalid input. Please enter 1 or 2")

def get_manual_parameters():
    """Get manual hyperparameters from user"""
    print("\n🔧 MANUAL PARAMETER CONFIGURATION")
    print("-" * 40)
    print("💡 Parameter Guidelines:")
    print("   • hidden_dim: 128-512 (network width)")
    print("   • N: 2-8 (high-level reasoning cycles)")
    print("   • T: 2-8 (steps per cycle)")
    print("   • Total HRM steps = N × T (recommended: 6-32)")
    print("   • More steps = better reasoning but slower training")
    
    # Get hidden_dim
    while True:
        try:
            hidden_dim = int(input("\nEnter hidden_dim (128-512, recommended 192-256): "))
            if 64 <= hidden_dim <= 1024:
                break
            else:
                print("❌ Please enter a value between 64 and 1024")
        except ValueError:
            print("❌ Please enter a valid integer")
    
    # Get N
    while True:
        try:
            N = int(input("Enter N - reasoning cycles (2-12, recommended 2-12): "))
            if 2 <= N <= 12:
                break
            else:
                print("❌ Please enter a value between 2 and 12")
        except ValueError:
            print("❌ Please enter a valid integer")
    
    # Get T
    while True:
        try:
            T = int(input("Enter T - steps per cycle (2-12, recommended 3-12): "))
            if 2 <= T <= 12:
                break
            else:
                print("❌ Please enter a value between 2 and 12")
        except ValueError:
            print("❌ Please enter a valid integer")
    
    total_steps = N * T
    
    # Estimate model complexity
    estimated_params = hidden_dim * (hidden_dim * 6 + 64*64) + hidden_dim * 3
    complexity_level = "Light" if total_steps <= 8 else "Medium" if total_steps <= 16 else "Heavy"
    
    print(f"\n✅ Manual Configuration:")
    print(f"   • hidden_dim: {hidden_dim}")
    print(f"   • N: {N}, T: {T}")
    print(f"   • Total HRM steps: {total_steps}")
    print(f"   • Complexity: {complexity_level}")
    print(f"   • Estimated parameters: ~{estimated_params:,}")
    
    if total_steps > 50:
        print("⚠️  Warning: Very high step count may slow training significantly")
    elif total_steps > 32:
        print("⚠️  Warning: High step count may slow training")
    elif total_steps < 4:
        print("⚠️  Warning: Very low step count may reduce model capacity")
    
    # Confirmation
    confirm = input("\nProceed with this configuration? (y/n): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("🔄 Restarting parameter selection...")
        return get_manual_parameters()
    
    return hidden_dim, N, T

class StockfishEvaluator:
    """Stockfish motor integráció pozíció értékeléshez - PERZISZTENS KAPCSOLAT"""
    
    def __init__(self, stockfish_path="./stockfish.exe", movetime=50):
        self.stockfish_path = stockfish_path
        self.movetime = movetime
        self.process = None
        self.initialized = False
        self._init_engine()
    
    def _init_engine(self):
        """Stockfish motor inicializálás - HÁTTÉR PROCESS"""
        try:
            if os.path.exists(self.stockfish_path):
                print(f"🤖 Stockfish found: {self.stockfish_path}")
            else:
                print(f"❌ Stockfish not found at: {self.stockfish_path}")
                print("🔍 Looking for stockfish in system PATH...")
                # Try system stockfish
                self.stockfish_path = "stockfish"
            
            # Start persistent Stockfish process
            print("🚀 Starting persistent Stockfish engine...")
            self.process = subprocess.Popen(
                [self.stockfish_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            # Initialize UCI
            self._send_command("uci")
            self._wait_for_response("uciok")
            
            self._send_command("isready")
            self._wait_for_response("readyok")
            
            print("✅ Persistent Stockfish engine ready!")
            self.initialized = True
            
        except Exception as e:
            print(f"⚠️ Stockfish initialization error: {e}")
            self.initialized = False
    
    def _send_command(self, command):
        """Send command to Stockfish"""
        if self.process and self.process.stdin:
            try:
                self.process.stdin.write(command + "\n")
                self.process.stdin.flush()
            except:
                self.initialized = False
    
    def _read_line(self, timeout=2.0):
        """Read a line from Stockfish with timeout - Windows kompatibilis"""
        if not self.process or not self.process.stdout:
            return None
        
        try:
            # Windows-friendly approach - poll and readline
            import time
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                # Check if process is still alive
                if self.process.poll() is not None:
                    return None
                
                # Try to read a line (non-blocking simulation)
                try:
                    # This will block, but we have a short timeout
                    line = self.process.stdout.readline()
                    if line:
                        return line.strip()
                except:
                    break
                
                # Small sleep to prevent busy waiting
                time.sleep(0.01)
            
        except Exception as e:
            return None
        
        return None
    
    def _wait_for_response(self, expected, timeout=5.0):
        """Wait for specific response from Stockfish"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            line = self._read_line(0.1)
            if line and expected in line:
                return True
        return False
    
    def evaluate_position(self, fen):
        """
        Pozíció értékelése Stockfish-sel
        Returns: (best_move_uci, evaluation_score)
        """
        if not self.initialized or not self.process:
            return None, 0.0
        
        try:
            # Set position
            self._send_command(f"position fen {fen}")
            
            # Get evaluation with limited thinking time
            self._send_command(f"go movetime {self.movetime}")
            
            # Parse response - simplified approach for Windows
            best_move = None
            score = 0.0
            lines_read = 0
            max_lines = 50  # Limit reading to prevent hanging
            
            while lines_read < max_lines:
                try:
                    line = self.process.stdout.readline()
                    if not line:
                        break
                    
                    line = line.strip()
                    lines_read += 1
                    
                    if line.startswith('bestmove'):
                        parts = line.split()
                        if len(parts) >= 2 and parts[1] != "(none)":
                            best_move = parts[1]
                        break
                    elif 'score cp' in line:
                        # Extract centipawn score
                        try:
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if part == "cp" and i + 1 < len(parts):
                                    cp_score = int(parts[i + 1])
                                    score = max(-1.0, min(1.0, cp_score / 300.0))
                                    break
                        except:
                            pass
                    elif 'score mate' in line:
                        # Mate score
                        try:
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if part == "mate" and i + 1 < len(parts):
                                    mate_moves = int(parts[i + 1])
                                    score = 1.0 if mate_moves > 0 else -1.0
                                    break
                        except:
                            pass
                            
                except Exception as e:
                    break
            
            return best_move, score
            
        except Exception as e:
            print(f"⚠️ Stockfish evaluation error: {e}")
            return None, 0.0
    
    def close(self):
        """Close Stockfish engine"""
        if self.process:
            try:
                self._send_command("quit")
                self.process.terminate()
                self.process.wait(timeout=2)
            except:
                if self.process:
                    self.process.kill()
            self.process = None
            self.initialized = False
            print("🔌 Stockfish engine closed")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.close()
    
    def create_policy_value_dataset(self, states, policies, max_positions=10000):
        """
        Policy+Value dataset létrehozása Stockfish értékeléssel
        """
        print(f"\n🤖 Creating Policy+Value dataset with Stockfish...")
        print(f"📊 Processing {min(len(states), max_positions):,} positions")
        
        # Limit positions for reasonable processing time
        total_positions = min(len(states), max_positions)
        
        stockfish_states = []
        stockfish_policies = []
        stockfish_values = []
        
        processed = 0
        start_time = time.time()
        
        for i in range(total_positions):
            if i % 1000 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (total_positions - i) / rate if rate > 0 else 0
                print(f"📈 Progress: {i:,}/{total_positions:,} ({i/total_positions*100:.1f}%) "
                      f"Rate: {rate:.1f}/s ETA: {eta/60:.1f}min")
            
            try:
                # Convert state back to FEN for Stockfish
                state = states[i]
                policy = policies[i]
                
                # Simple state to FEN conversion (simplified)
                board = chess.Board()
                # Note: This is a simplified conversion, a proper implementation
                # would reconstruct the exact position from the state vector
                
                # For now, use a basic starting position and get some evaluation
                fen = board.fen()
                
                # Get Stockfish evaluation
                best_move, value = self.evaluate_position(fen)
                
                if best_move is not None:
                    stockfish_states.append(state)
                    stockfish_policies.append(policy)
                    stockfish_values.append(value)
                    processed += 1
                
            except Exception as e:
                continue  # Skip problematic positions
        
        print(f"\n✅ Created Policy+Value dataset:")
        print(f"   📊 Processed: {processed:,} positions")
        print(f"   📈 Success rate: {processed/total_positions*100:.1f}%")
        
        return np.array(stockfish_states), stockfish_policies, np.array(stockfish_values)

def create_policy_value_dataset_from_games(max_positions=10000):
    """
    Policy+Value dataset létrehozása játszmákból és Stockfish értékeléssel
    
    Args:
        max_positions: Maximális pozíciók száma a dataset-ben
    """
    print("\n🎮 Creating Policy+Value dataset from games...")
    print(f"🎯 Target positions: {max_positions:,}")
    
    # Intelligens PGN games becslés a dataset méret alapján
    avg_positions_per_game = 22  # Átlagos pozíciók száma játszmánként
    
    # PGN target számítás - általános skálázható módszer
    # Logaritmikus skálázás minden dataset méretre
    
    # Logaritmikus skálázás: 50% puzzle kis dataset-nél → 15% nagy dataset-nél
    log_factor = math.log10(max(max_positions, 1000))
    base_ratio = 0.5  # 50% kezdeti puzzle ratio
    scale_factor = (log_factor - 3.0) / 4.0  # 3=log10(1000), 7=log10(10M)
    scale_factor = max(0, min(1, scale_factor))  # Clamp 0-1 közé
    
    # Puzzle ratio: 50% (1k) → 35% (100k) → 20% (10M) → 15% (100M+)
    puzzle_ratio = base_ratio - (scale_factor * 0.35)
    
    # PGN ratio - komplementer
    pgn_ratio = 1.0 - puzzle_ratio
    pgn_target_positions = int(max_positions * pgn_ratio)
    estimated_games_needed = max(1000, int(pgn_target_positions / avg_positions_per_game))
    
    print(f"🎯 Automatic scaling for {max_positions:,} positions:")
    print(f"   📊 PGN ratio: {pgn_ratio:.1%} → {pgn_target_positions:,} positions")  
    print(f"   🧩 Puzzle ratio: {puzzle_ratio:.1%} → {int(max_positions * puzzle_ratio):,} positions")
    print(f"   🎮 Estimated games needed: {estimated_games_needed:,}")
    
    # Load PGN data
    print(f"📥 Loading PGN games...")
    try:
        pgn_states, pgn_policies, pgn_fens = load_pgn_data(
            "./lichess_db_standard_rated_2013-07.pgn",
            fen_to_tensor,
            max_games=estimated_games_needed,
            max_moves=30,
            min_elo=1600
        )
        print(f"✅ Loaded {len(pgn_states):,} positions from PGN")
    except:
        print("⚠️ PGN file not found, using minimal dataset")
        pgn_states, pgn_policies, pgn_fens = [], [], []
    
    # Load PUZZLE data - TAKTIKAI KÉPESSÉGEK - SKÁLÁZHATÓ MÉRETEZÉS
    print(f"📥 Loading tactical puzzles...")
    
    # Általános puzzle ratio számítás - logaritmikus skálázás
    # Kisebb dataset-eknél több puzzle (jobb taktikai fókusz)
    # Nagyobb dataset-eknél kevesebb puzzle (PGN dominancia)
    
    # Logaritmikus skálázás: 50% puzzle kis dataset-nél → 15% nagy dataset-nél
    log_factor = math.log10(max(max_positions, 1000))  # min 1000 a log hibák elkerülésére
    base_ratio = 0.5  # 50% kezdeti puzzle ratio
    scale_factor = (log_factor - 3.0) / 4.0  # 3=log10(1000), 7=log10(10M) → 0-1 range
    scale_factor = max(0, min(1, scale_factor))  # Clamp 0-1 közé
    
    # Puzzle ratio: 50% (1k) → 35% (100k) → 20% (10M) → 15% (100M+)
    puzzle_ratio = base_ratio - (scale_factor * 0.35)
    puzzle_target = int(max_positions * puzzle_ratio)
    
    print(f"🎯 Dataset size: {max_positions:,} → Puzzle ratio: {puzzle_ratio:.1%}")
    print(f"🎯 Target puzzle count: {puzzle_target:,}")
    
    # PGN ratio kalkuláció
    pgn_ratio = 1.0 - puzzle_ratio
    pgn_target_positions = int(max_positions * pgn_ratio)
    
    try:
        puzzle_states, puzzle_policies, puzzle_fens = load_puzzle_data(
            "./lichess_db_puzzle.csv",
            fen_to_tensor,
            max_puzzles=puzzle_target,
            min_rating=800,   # Alacsonyabb minimum több puzzle-ért
            max_rating=2500   # Magasabb maximum több puzzle-ért
        )
        print(f"✅ Loaded {len(puzzle_states):,} tactical positions from CSV")
    except:
        print("⚠️ Puzzle CSV file not found, using only PGN data")
        puzzle_states, puzzle_policies, puzzle_fens = [], [], []
    
    # BALANCED DATASET COMBINATION
    print(f"\n🎯 COMBINING PGN + PUZZLE DATA:")
    print(f"   📊 PGN positions: {len(pgn_states):,}")
    print(f"   🧩 Puzzle positions: {len(puzzle_states):,}")
    
    # Dataset composition analysis
    total_loaded = len(pgn_states) + len(puzzle_states)
    if total_loaded > 0:
        pgn_percentage = len(pgn_states) / total_loaded * 100
        puzzle_percentage = len(puzzle_states) / total_loaded * 100
        print(f"   📈 Composition: {pgn_percentage:.1f}% PGN, {puzzle_percentage:.1f}% Puzzles")
        
        if len(puzzle_states) < puzzle_target * 0.5:
            print(f"   ⚠️ Warning: Only {len(puzzle_states):,} puzzles loaded (target: {puzzle_target:,})")
            print(f"   💡 Consider checking puzzle CSV file or lowering rating filters")
    
    # Combine all data sources
    all_states = pgn_states + puzzle_states
    all_policies = pgn_policies + puzzle_policies
    all_fens = pgn_fens + puzzle_fens
    
    if len(all_states) == 0:
        print("❌ No training data available!")
        return None, None, None
    
    print(f"📊 Total positions loaded: {len(all_states):,}")
    
    # Dataset adequacy check
    adequacy_ratio = len(all_states) / max_positions if max_positions > 0 else 0
    if adequacy_ratio < 0.5:
        print(f"⚠️ Warning: Only {adequacy_ratio*100:.1f}% of target positions loaded!")
        print(f"💡 Consider: checking file paths, lowering rating filters, or reducing target size")
    elif adequacy_ratio >= 1.0:
        print(f"✅ Excellent: {adequacy_ratio:.1f}x target positions available")
    else:
        print(f"✅ Good: {adequacy_ratio*100:.1f}% of target positions loaded")
    
    # Adjust to target max_positions
    if len(all_states) > max_positions:
        print(f"🎯 Randomly selecting {max_positions:,} positions from {len(all_states):,} available")
        indices = np.random.choice(len(all_states), max_positions, replace=False)
        all_states = [all_states[i] for i in indices]
        all_policies = [all_policies[i] for i in indices]
        all_fens = [all_fens[i] for i in indices]
    
    # Create Stockfish evaluator for real position evaluation
    evaluator = StockfishEvaluator()
    
    # GYORS STOCKFISH BATCH EVALUATION
    print("🤖 Fast batch evaluation with persistent Stockfish...")
    
    values = []
    processed_states = []
    processed_policies = []
    
    # Batch processing with progress tracking
    batch_size = 1000  # Process in batches for better progress tracking
    total_batches = (len(all_states) + batch_size - 1) // batch_size
    
    start_time = time.time()
    processed_count = 0
    
    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(all_states))
        
        # Progress update
        elapsed = time.time() - start_time
        rate = processed_count / elapsed if elapsed > 0 else 0
        eta = (len(all_states) - processed_count) / rate if rate > 0 else 0
        
        print(f"📈 Batch {batch_idx+1}/{total_batches} | "
              f"Progress: {processed_count:,}/{len(all_states):,} ({processed_count/len(all_states)*100:.1f}%) | "
              f"Rate: {rate:.1f}/s | ETA: {eta/60:.1f}min")
        
        # Process batch
        for i in range(batch_start, batch_end):
            state = all_states[i]
            policy = all_policies[i]
            fen = all_fens[i]
            
            try:
                # Use persistent Stockfish connection - MUCH FASTER
                best_move, stockfish_value = evaluator.evaluate_position(fen)
                
                if best_move is not None:
                    value = stockfish_value
                else:
                    # Fallback to neutral if Stockfish fails
                    value = 0.0
                    
            except Exception as e:
                # Fallback value if evaluation fails
                value = 0.0
            
            values.append(value)
            processed_states.append(state)
            processed_policies.append(policy)
            processed_count += 1
    
    # Close Stockfish engine
    evaluator.close()
    
    values = np.array(values)
    
    print(f"\n✅ Policy+Value dataset created:")
    print(f"   📊 Positions: {len(processed_states):,}")
    print(f"   📈 Value range: [{np.min(values):.3f}, {np.max(values):.3f}]")
    print(f"   📊 Value mean: {np.mean(values):.3f} ± {np.std(values):.3f}")
    
    return processed_states, processed_policies, values

if __name__ == "__main__":
    print("🏗️ HRM CHESS MODEL TRAINING")
    print("="*50)
    
    import os
    import sys
    print(f"Using device: {device}")
    
    # GPU MEMORY DETECTION & OPTIMIZATION
    gpu_config = detect_gpu_memory_and_optimize_training()
    
    # Load or create Policy+Value dataset
    pv_dataset_path = "chess_policy_value_dataset.pt"
    
    if not os.path.exists(pv_dataset_path):
        print(f"\n📝 Policy+Value dataset not found: {pv_dataset_path}")
        print("� Creating new Policy+Value dataset...")
        
        # Ask user for dataset size
        while True:
            try:
                max_positions = int(input("Enter dataset size (number of positions, e.g., 20000): "))
                if max_positions > 0:
                    break
                else:
                    print("❌ Please enter a positive number")
            except ValueError:
                print("❌ Please enter a valid integer")
        
        print(f"🎯 Creating dataset with {max_positions:,} positions")
        
        # Create dataset from games and puzzles with user-specified size
        states, policies, values = create_policy_value_dataset_from_games(
            max_positions=max_positions   # User-specified dataset size
        )
        
        if states is None:
            print("❌ Failed to create dataset!")
            exit(1)
        
        # Save dataset
        print("💾 Saving Policy+Value dataset...")
        dataset_info = {
            'states': np.array(states, dtype=np.float32),
            'policies': policies,
            'values': values,
            'info': {
                'created': time.time(),
                'source': 'PGN + Tactical Puzzles (Stockfish evaluation)',
                'positions': len(states),
                'stockfish_depth': 'simplified',
                'data_mix': 'Balanced PGN games + tactical puzzles',
                'gpu_optimized': True,
                'gpu_config': gpu_config,
                'user_specified_size': max_positions
            }
        }
        
        torch.save(dataset_info, pv_dataset_path)
        print(f"✅ Dataset saved to: {pv_dataset_path}")
        
        # Use the created data
        data = dataset_info
    else:
        # Load existing Policy+Value dataset
        print(f"\n📥 Loading existing Policy+Value dataset: {pv_dataset_path}")
        data = torch.load(pv_dataset_path, weights_only=False)
    
    states = data['states']
    policies = data['policies']
    values = data['values']
    info = data['info']
    
    print(f"✅ Loaded Policy+Value dataset:")
    print(f"   📊 Positions: {len(states):,}")
    print(f"   📈 Value range: [{np.min(values):.3f}, {np.max(values):.3f}]")
    print(f"   📊 Value mean: {np.mean(values):.3f} ± {np.std(values):.3f}")
    print(f"   🤖 Source: {info.get('source', 'Unknown')}")
    print(f"   🎮 Data mix: {info.get('data_mix', 'Unknown composition')}")
    print(f"   🖥️ GPU Optimized: {info.get('gpu_optimized', False)}")
    
    # Policy+Value training mode
    print("\n🎯 POLICY+VALUE TRAINING MODE")
    print("   • Input: 72-dim vector → 8x8 2D conv + extra features")
    print("   • Policy Head: 64x64 move matrix")
    print("   • Value Head: Position evaluation [-1, 1]")
    print("   • Training: Stockfish supervision")
    print("   • Dataset: Balanced PGN games + tactical puzzles")
    
    print("\n⚙️ HRM PARAMETER EXPLANATION:")
    print("   • N: Number of high-level reasoning cycles")
    print("   • T: Steps per cycle (low-level processing)")
    print("   • Total HRM steps = N × T")
    print("   • hidden_dim: Neural network width")
    print("   • Optimal balance: complexity vs speed vs accuracy")
    
    # Configuration
    data_path = pv_dataset_path
    model_path = "hrm_policy_value_chess_model.pt"
    
    # Get dataset info
    dataset_size = len(states)
    print(f"\n📊 Dataset size: {dataset_size:,} positions")
    
    # Interactive parameter selection
    user_choice = get_user_choice()
    
    if user_choice == 1:
        # HYPERPARAMETER OPTIMIZATION MODE
        print("\n🔬 HYPERPARAMETER OPTIMIZATION MODE ENABLED")
        
        # Optuna hyperparameter optimization
        try:
            import optuna
            print("✅ Optuna available - using advanced Bayesian optimization")
            best_config = optuna_hyperparameter_optimization(states, policies, values, n_trials=15)
        except ImportError:
            print("❌ Optuna not installed - please install with: pip install optuna")
            print("� Falling back to manual configuration...")
            hidden_dim, N, T = get_manual_parameters()
            lr = 2e-4
            batch_size = 32
            best_config = None
        
        if best_config:
            hidden_dim = best_config['hidden_dim']
            N = best_config['N']
            T = best_config['T']
            
            # GPU-optimized batch size és learning rate
            gpu_batch_size = gpu_config['batch_size']
            gpu_lr_multiplier = gpu_config['lr_multiplier']
            
            # SMART GPU SCALING: Use Optuna as baseline, scale up with GPU power
            if 'batch_size' in best_config:
                optuna_batch = best_config['batch_size']
                # GPU scaling: if GPU can handle more, use more!
                batch_size = max(optuna_batch, min(gpu_batch_size, optuna_batch * 2))  # Cap at 2x Optuna or GPU limit
                print(f"🔧 Batch size: Optuna suggested {optuna_batch}, GPU scaled to {batch_size} (limit: {gpu_batch_size})")
            else:
                batch_size = gpu_batch_size
                
            # Adjust learning rate for final batch size (not just GPU multiplier)
            if 'lr' in best_config:
                base_lr = best_config['lr']
            else:
                base_lr = 2e-4
                
            # Scale LR for both GPU power and actual batch size used
            batch_scaling = batch_size / best_config.get('batch_size', 16)  # Scale from Optuna baseline
            lr = base_lr * gpu_lr_multiplier * batch_scaling  # Both GPU and batch scaling
                
            model_size = f"GPU_OPTIMIZED-{best_config['name']}"
            print(f"\n✅ Using GPU-optimized configuration:")
            print(f"   🎯 Final LR: {lr:.6f} (base: {base_lr:.6f} × GPU: {gpu_lr_multiplier:.2f} × batch: {batch_scaling:.2f})")
            print(f"   📊 Final Batch: {batch_size} (GPU-enhanced from {best_config.get('batch_size', 16)})")
            print(f"   🖥️ GPU Level: {gpu_config['optimization_level']}")
            print(f"   🖥️ VRAM: {gpu_config['memory_gb']:.1f} GB ({gpu_config['device_name']})")
        else:
            print("⚠️ Hyperopt failed, using GPU-optimized defaults")
            hidden_dim, N, T = 192, 3, 4
            batch_size = gpu_config['batch_size']
            lr = 2e-4 * gpu_config['lr_multiplier']
            model_size = f"GPU_DEFAULT-{gpu_config['optimization_level']}"
            
    elif user_choice == 2:
        # MANUAL PARAMETER MODE with GPU optimization
        hidden_dim, N, T = get_manual_parameters()
        
        # Apply GPU optimizations
        batch_size = gpu_config['batch_size']
        lr = 2e-4 * gpu_config['lr_multiplier']
        model_size = f"GPU_MANUAL-{N}x{T}-{gpu_config['optimization_level']}"
        
        print(f"\n🔧 GPU OPTIMIZATIONS APPLIED:")
        print(f"   📊 Batch Size: {batch_size} (GPU-optimized)")
        print(f"   📈 Learning Rate: {lr:.6f} (base: 2e-4 × {gpu_config['lr_multiplier']:.2f})")
        print(f"   🖥️ GPU Level: {gpu_config['optimization_level']}")
    
    # HRM modell létrehozása optimalizált paraméterekkel
    model = HRMChess(input_dim=72, hidden_dim=hidden_dim, N=N, T=T).to(device)
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hrm_steps = N * T
    
    print(f"\n🏗️ MODEL ARCHITECTURE:")
    print(f"📊 Total parameters: {total_params:,}")
    print(f"📊 Trainable parameters: {trainable_params:,}")
    print(f"🔄 HRM reasoning steps: {hrm_steps} (N={N} × T={T})")
    print("🏗️ Architecture: HRM")
    print("   • Board conv: 8x8 → conv2d → 4x4 → flatten")
    print("   • Extra processor: 8-dim meta info → linear")
    print("   • Feature combiner: board + extra → hidden_dim")
    print("   • Board enhancer: hidden_dim → hidden_dim → hidden_dim")
    print(f"   • HRM modules: L_net and H_net with N={N}, T={T}")
    print("   • Policy Head: Move prediction (hidden_dim → 64*64)")
    print("   • Value Head: Position evaluation (hidden_dim → 1)")
    
    # GPU-optimized training configuration
    epochs = 30  # Több epoch a jobb konvergenciáért
    
    print(f"\n⚙️ GPU-OPTIMIZED TRAINING CONFIGURATION:")
    print(f"   • Model: {model_size}")
    print(f"   • Mode: Policy+Value+Warmup")
    print(f"   • Batch size: {batch_size} (GPU-optimized)")
    print(f"   • Learning rate: {lr:.6f} (GPU-scaled)")
    print(f"   • Warmup epochs: 3 (linear warmup + cosine annealing)")
    print(f"   • Total epochs: {epochs}")
    print(f"   • HRM steps: {N}×{T}={N*T}")
    print(f"   • Parameters: {total_params:,}")
    print(f"   • Dataset: {dataset_size:,} positions")
    print(f"   🖥️ GPU: {gpu_config['device_name']} ({gpu_config['memory_gb']:.1f} GB)")
    print(f"   🏷️ Optimization Level: {gpu_config['optimization_level']}")
    
    # Memory usage estimation
    estimated_memory_per_batch = (batch_size * 72 * 4 + batch_size * 64 * 64 * 4) / (1024**3)  # Rough estimate in GB
    print(f"   📊 Estimated memory/batch: ~{estimated_memory_per_batch:.2f} GB")
    
    # Create Policy+Value dataset
    dataset = PolicyValueDataset(states, policies, values)
    print(f"\n📊 Policy+Value dataset: {len(dataset):,} positions")
    print("🚀 Starting GPU-optimized Policy+Value HRM training with warmup...")
    
    # Train with GPU-optimized parameters and warmup
    train_loop(model, dataset, epochs=epochs, batch_size=batch_size, lr=lr, warmup_epochs=3, device=device)
    
    # Save final model with hyperparameters and GPU info
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'hyperparams': {
            'hidden_dim': hidden_dim,
            'N': N,
            'T': T,
            'input_dim': 72
        },
        'training_info': {
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'warmup_epochs': 3,
            'dataset_size': len(dataset),
            'total_params': total_params,
            'training_mode': 'policy_value_warmup',
            'gpu_optimized': True,
            'gpu_config': gpu_config
        }
    }
    torch.save(final_checkpoint, model_path)
    print("\n✅ Training completed!")
    print(f"💾 Model saved to: {model_path}")
    print(f"🏆 HRM (N={N}, T={T}, hidden_dim={hidden_dim}) with {total_params:,} parameters")
    print(f"📊 Trained on {len(dataset):,} positions with Policy+Value+Warmup mode")
    print(f"🎮 Dataset: Balanced PGN games + tactical puzzles for enhanced gameplay")
    print(f"🔥 Warmup: 3 epochs with linear warmup + cosine annealing")
    
    print("🎯 Expected: Enhanced move prediction + position evaluation + tactical strength")
    print("⚔️ Ready for tactical fine-tuning and stronger gameplay!")

