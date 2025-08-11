
import chess
import numpy as np
from hrm_model import fen_to_bitplanes

# --- FEN to bitplane conversion ---
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
import random

# Import HRM model and related functions
from hrm_model import HRMChess, ValueBinDataset, train_loop

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42  # vagy bármilyen választott seed

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def detect_gpu_memory_and_optimize_training():
    """
    GPU memória detektálás és automatikus batch size/learning rate optimalizálás
    
    Returns:
        dict: Optimalizált training paraméterek
    """
    print("\nGPU Memory Detection & Training Optimization")
    print("-" * 50)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available - GPU required for training!")
        exit(1)
    
    try:
        # GPU információk lekérdezése
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        
        # Memória információk
        total_memory = torch.cuda.get_device_properties(current_device).total_memory
        allocated_memory = torch.cuda.memory_allocated(current_device)
        
        # Elérhető memória kiszámítása (GB-ban)
        total_gb = total_memory / (1024**3)
        free_gb = (total_memory - allocated_memory) / (1024**3)
        
        print(f"GPU: {gpu_name} ({total_gb:.1f}GB total, {free_gb:.1f}GB free)")
        
        # Batch size optimalizálás SZABAD GPU memória alapján
        if free_gb >= 20:  # RTX 4090 territory
            batch_config = {
                'batch_size': 128,
                'lr_multiplier': 2.0,
                'optimization_level': 'RTX_4090_ULTRA'
            }
        elif free_gb >= 16:  # RTX 4080/3090 territory  
            batch_config = {
                'batch_size': 96,
                'lr_multiplier': 1.8,
                'optimization_level': 'HIGH_END_PLUS'
            }
        elif free_gb >= 12:  # High-end territory
            batch_config = {
                'batch_size': 80,
                'lr_multiplier': 1.6,
                'optimization_level': 'HIGH_END_FREE'
            }
        elif free_gb >= 10:  # Good amount of memory
            batch_config = {
                'batch_size': 64,
                'lr_multiplier': 1.4,
                'optimization_level': 'HIGH_FREE'
            }
        elif free_gb >= 8:   # Mid-high memory
            batch_config = {
                'batch_size': 48,
                'lr_multiplier': 1.2,
                'optimization_level': 'MID_HIGH_FREE'
            }
        elif free_gb >= 6:   # Average memory
            batch_config = {
                'batch_size': 32,
                'lr_multiplier': 1.0,
                'optimization_level': 'MID_FREE'
            }
        elif free_gb >= 4:   # Low-mid memory
            batch_config = {
                'batch_size': 24,
                'lr_multiplier': 0.9,
                'optimization_level': 'LOW_MID_FREE'
            }
        elif free_gb >= 2:   # Low memory
            batch_config = {
                'batch_size': 16,
                'lr_multiplier': 0.8,
                'optimization_level': 'LOW_FREE'
            }
        else:  # <2GB szabad VRAM
            print(f"ERROR: Insufficient GPU memory ({free_gb:.1f}GB < 2GB required)")
            exit(1)

        # Ha több GPU van, szorozzuk fel a batch_size-t és lr_multiplier-t
        if gpu_count > 1:
            batch_config['batch_size'] *= gpu_count
            batch_config['lr_multiplier'] *= gpu_count * 0.9
            batch_config['optimization_level'] += f"_MULTIGPUx{gpu_count}"
        
        # Memória foglaltság alapú finomhangolás
        memory_usage_ratio = allocated_memory / total_memory
        if memory_usage_ratio > 0.3:  # Ha már 30%+ foglalt
            batch_config['batch_size'] = max(8, int(batch_config['batch_size'] * 0.7))
            batch_config['lr_multiplier'] *= 0.9
        
        # Memory safety test
        test_passed = True
        try:
            test_batch_size = batch_config['batch_size']
            test_input = torch.randn(test_batch_size, 20, 8, 8, device=device)
            test_hidden = torch.randn(test_batch_size, 256, device=device)
            test_conv_features = torch.randn(test_batch_size, 128, 8, 8, device=device)
            
            test_memory = torch.cuda.memory_allocated(current_device)
            test_memory_gb = test_memory / (1024**3)
            
            # Cleanup
            del test_input, test_hidden, test_conv_features
            torch.cuda.empty_cache()
            
            memory_threshold = 0.75 if free_gb >= 20 else 0.6
            if test_memory_gb > free_gb * memory_threshold:
                batch_config['batch_size'] = max(16, int(batch_config['batch_size'] * 0.7))
                batch_config['lr_multiplier'] *= 0.9
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                batch_config['batch_size'] = max(16, int(batch_config['batch_size'] * 0.6))
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
        
        print(f"Optimized config: batch_size={result['batch_size']}, "
              f"lr_multiplier={result['lr_multiplier']:.2f}x, "
              f"level={result['optimization_level']}")
        
        return result
        
    except Exception as e:
        print(f"❌ GPU detection failed: {e}")
        print(" Cannot proceed without GPU information.")
        exit(1)

def load_pgn_data(pgn_path, max_positions=None, max_moves=40, min_elo=1600):
    all_fens = []
    
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
            
            if stats['total_games'] % 5000 == 0:
                print(f"PGN processing: {stats['total_games']} games checked, "
                      f"{stats['processed_games']} processed, {stats['positions_extracted']} positions")
            
            if max_positions < stats['positions_extracted']:
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
                
                if len(time_control) <= 3:
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
                        current_fen = board.fen()  # Store the actual FEN
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
    
    # Statistics
    print(f"\nPGN Statistics: {stats['processed_games']:,} games processed, "
          f"{stats['positions_extracted']:,} positions extracted")
    
    # Duplikált FEN-ek szűrése
    unique_fen_idx = {}
    for idx, fen in enumerate(all_fens):
        if fen not in unique_fen_idx:
            unique_fen_idx[fen] = idx
    num_duplicates = len(all_fens) - len(unique_fen_idx)
    
    unique_indices = list(unique_fen_idx.values())
    all_fens = [all_fens[i] for i in unique_indices]

    print(f"Final dataset: {len(all_fens):,} unique positions "
          f"({num_duplicates} duplicates removed)")
    return all_fens

def get_manual_parameters():
    """Get manual hyperparameters from user"""
    print("\nManual Parameter Configuration")
    print("Guidelines: hidden_dim(128-512), N(2-8), T(2-8), total_steps=N×T(6-32)")
    
    # Get hidden_dim
    while True:
        try:
            hidden_dim = int(input("\nEnter hidden_dim (64-1024): "))
            if 64 <= hidden_dim <= 1024:
                break
            else:
                print("Please enter a value between 64 and 1024")
        except ValueError:
            print("Please enter a valid integer")

    # Get N
    while True:
        try:
            N = int(input("Enter N - reasoning cycles (2-20): "))
            if 2 <= N <= 20:
                break
            else:
                print("Please enter a value between 2 and 20")
        except ValueError:
            print("Please enter a valid integer")

    # Get T
    while True:
        try:
            T = int(input("Enter T - steps per cycle (2-20): "))
            if 2 <= T <= 20:
                break
            else:
                print("Please enter a value between 2 and 20")
        except ValueError:
            print("Please enter a valid integer")

    # Get nhead
    while True:
        try:
            nhead_in = input("Enter nhead (default 4): ").strip()
            if nhead_in == '':
                nhead = 4
                break
            nhead = int(nhead_in)
            if nhead >= 1 and nhead <= 32:
                break
            else:
                print("Please enter a value between 1 and 32")
        except ValueError:
            print("Please enter a valid integer")

    # Get dim_feedforward
    while True:
        try:
            dff_in = input(f"Enter dim_feedforward (default {hidden_dim*2}): ").strip()
            if dff_in == '':
                dim_feedforward = hidden_dim * 2
                break
            dim_feedforward = int(dff_in)
            if dim_feedforward >= hidden_dim:
                break
            else:
                print(f"Please enter a value >= {hidden_dim}")
        except ValueError:
            print("Please enter a valid integer")

    total_steps = N * T
    if total_steps <= 8:
        complexity_level = "Light"
    elif total_steps <= 16:
        complexity_level = "Medium"
    else:
        complexity_level = "Heavy"

    print(f"\nConfiguration: hidden_dim={hidden_dim}, N={N}, T={T}, "
          f"total_steps={total_steps}, complexity={complexity_level}")

    confirm = input("Proceed? (y/n): ").strip().lower()
    if confirm not in ['y', 'yes']:
        return get_manual_parameters()

    return hidden_dim, N, T, nhead, dim_feedforward

# Import StockfishEvaluator and ParallelStockfishEvaluator from the new module
from stockfish_eval import StockfishEvaluator, ParallelStockfishEvaluator

def create_dataset_from_games(max_positions=10000):
    """
    Dataset létrehozása játszmákból - csak PGN alapú, puzzle nélkül
    
    Args:
        max_positions: Maximális pozíciók száma a dataset-ben
    """
    import math
    import time

    print("\nCreating dataset from games...")

    print("Loading PGN games...")
    try:
        pgn_fens = load_pgn_data(
            "./lichess_db_standard_rated_2015-06.pgn",
            max_positions=max_positions,
            max_moves=100,
            min_elo=1000
        )
        print(f"Loaded {len(pgn_fens):,} positions from PGN")
    except Exception:
        print("PGN file not found")
        exit(0)

    all_fens = pgn_fens

    if len(all_fens) == 0:
        print("No training data available!")
        exit(0)

    print(f"Total positions loaded: {len(all_fens):,}")

    # Stockfish értékelés minden pozícióra
    print("\nEvaluating all legal moves for all positions...")
    stockfish = ParallelStockfishEvaluator(stockfish_path="stockfish.exe", movetime=10, num_evaluators=int(os.cpu_count() * 0.8) or 2)
    all_move_evals = stockfish.evaluate_positions_parallel(all_fens)
    stockfish.close()

    print(f"Stockfish-evaluated dataset: {len(all_fens):,} positions")
    return all_fens, all_move_evals

if __name__ == "__main__":
    try:
        print("HRM CHESS MODEL TRAINING")
        print("=" * 30)
        print(f"Using device: {device}")
        
        # GPU MEMORY DETECTION & OPTIMIZATION
        if torch.cuda.is_available():
            gpu_config = detect_gpu_memory_and_optimize_training()
        else:
            print("CUDA not available - using CPU training parameters")
            gpu_config = {
                'batch_size': 8,
                'lr_multiplier': 1.0,
                'memory_gb': 0,
                'free_memory_gb': 0,
                'device_name': 'cpu',
                'optimization_level': 'CPU_DEFAULT',
                'memory_test_passed': True
            }
        
        # Load or create dataset
        dataset_path = "fen_move_score_dataset.pt"
        if not os.path.exists(dataset_path):
            print(f"\nDataset not found: {dataset_path}")
            print("Creating new dataset...")
            # Ask user for dataset size
            while True:
                try:
                    max_positions = int(input("Enter number of positions for training dataset (e.g. 20000): "))
                    if max_positions > 0:
                        break
                    else:
                        print("Please enter a positive number!")
                except ValueError:
                    print("Please enter a valid integer!")
            print(f"Creating dataset with {max_positions:,} positions")
            
            # Create dataset
            fens, moves = create_dataset_from_games(max_positions)
            
            # Save as vector of (fen, score) tuples
            from tqdm import tqdm
            fen_move_score_vec = []
            for fen, move_list in tqdm(zip(fens, moves), total=len(fens), desc="Processing positions"):
                board = chess.Board(fen)
                for move_tuple in move_list:
                    move, score = move_tuple
                    try:
                        board.push(chess.Move.from_uci(move))
                        resulting_fen = board.fen()
                        fen_move_score_vec.append((resulting_fen, score))
                        board.pop()
                    except Exception as e:
                        continue  # Skip illegal moves
                        
            # Deduplicate by FEN
            print("\nDeduplicating by FEN...")
            unique_fen_score = {}
            for fen, score in fen_move_score_vec:
                if fen not in unique_fen_score:
                    unique_fen_score[fen] = score
            num_duplicates = len(fen_move_score_vec) - len(unique_fen_score)
            fen_move_score_vec = [(fen, score) for fen, score in unique_fen_score.items()]
            print(f"Deduplicated: {len(fen_move_score_vec):,} unique positions, removed {num_duplicates:,} duplicates")
            
            output_pt = "fen_move_score_dataset.pt"
            dataset_info = {
                'data': fen_move_score_vec,
                'info': {
                    'created': time.time(),
                    'source': 'PGN + Stockfish (all legal moves, deduped FENs)',
                    'base_positions': len(fens),
                    'total_positions': len(fen_move_score_vec),
                    'stockfish_evaluation': 'all_legal_moves',
                    'evaluation_method': 'all_moves_winpercent',
                    'data_format': '(fen, score)'
                }
            }
            torch.save(dataset_info, output_pt)
            print(f"Saved {len(fen_move_score_vec):,} (fen, score) pairs to {output_pt}")
            data = dataset_info
        else:
            # Load existing dataset
            print(f"\nLoading existing dataset: {dataset_path}")
            data = torch.load(dataset_path, weights_only=False)
            
        # Extract data
        dataset_info = data if 'data' in data else data.get('dataset_info', {})
        fen_move_score_vec = dataset_info['data']
        info = dataset_info['info']
        print(f"Dataset loaded: {len(fen_move_score_vec):,} positions")
        print(f"Source: {info.get('source', 'Unknown')}")
        
        # MANUAL PARAMETERS
        hidden_dim, N, T, nhead, dim_feedforward = get_manual_parameters()
        
        # Apply GPU optimizations
        batch_size = gpu_config['batch_size']
        lr = 1e-4 * gpu_config['lr_multiplier']
        
        print(f"\nTraining config: batch_size={batch_size}, lr={lr:.6f}")
        
        # Create model
        model = HRMChess(hidden_dim=hidden_dim, N=N, T=T, nhead=nhead, dim_feedforward=dim_feedforward).to(device)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs (DataParallel)")
            model = torch.nn.DataParallel(model)
            
        # Model info
        total_params = sum(p.numel() for p in model.parameters())
        hrm_steps = N * T
        print(f"Model: {total_params:,} parameters, {hrm_steps} HRM steps (N={N} × T={T})")
        
        # Create dataset
        fen_list = [fen for fen, score in fen_move_score_vec]
        score_list = [score for fen, score in fen_move_score_vec]
        dataset = ValueBinDataset(fen_list, score_list, num_bins=128)
        
        epochs = 30
        print(f"\nStarting training: {epochs} epochs, {len(dataset):,} positions")
        
        # Train
        train_loop(model, dataset, epochs=epochs, batch_size=batch_size, lr=lr, warmup_epochs=3, device=device, use_amp=True)
        
        # Save model
        if isinstance(model, torch.nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
            
        final_checkpoint = {
            'model_state_dict': state_dict,
            'hyperparams': {
                'hidden_dim': hidden_dim,
                'N': N,
                'T': T,
                'input_dim': 20,
                'nhead': nhead,
                'dim_feedforward': dim_feedforward
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
        
        model_path = "hrm_chess_model.pt"
        torch.save(final_checkpoint, model_path)
        print(f"\nTraining completed! Model saved to: {model_path}")
        print(f"HRM model (N={N}, T={T}, hidden_dim={hidden_dim}) with {total_params:,} parameters")
        print(f"Trained on {len(dataset):,} positions")
        
    except KeyboardInterrupt:
        import sys
        sys.exit(0)
