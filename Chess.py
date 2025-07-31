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
from hrm_model import HRMChess, ValueBinDataset, train_loop

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
        
        print("\n✅ OPTIMIZED TRAINING CONFIGURATION (FREE MEMORY BASED):")
        print(f"   🎯 Batch Size: {result['batch_size']}")
        print(f"   📈 LR Multiplier: {result['lr_multiplier']:.2f}x")
        print(f"   🏷️ Level: {result['optimization_level']}")
        print(f"   💾 Free VRAM: {free_gb:.1f}GB / {total_gb:.1f}GB total")
        print(f"   🧪 Memory Test: {'✅ PASSED' if test_passed else '⚠️ ADJUSTED'}")
        
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
            
            if stats['total_games'] % 1000 == 0:
                print(f"PGN: {stats['total_games']} ellenőrizve, {stats['processed_games']} feldolgozva, "
                      f"{stats['positions_extracted']} pozíció")
            
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
    
    # ENHANCED statistics
    print("\n📊 ENHANCED PGN Processing Statistics:")
    print(f"   Total games examined: {stats['total_games']:,}")
    print(f"   Successfully processed: {stats['processed_games']:,}")
    print(f"   Positions extracted: {stats['positions_extracted']:,}")
    print(f"   ├── Piece moves: {stats['piece_moves']:,} ({stats['piece_moves']/max(stats['positions_extracted'], 1)*100:.1f}%)")
    print(f"   ├── Pawn moves: {stats['pawn_moves']:,} ({stats['pawn_moves']/max(stats['positions_extracted'], 1)*100:.1f}%)")
    print(f"   ├── Captures: {stats['captures']:,} ({stats['captures']/max(stats['positions_extracted'], 1)*100:.1f}%)")
    print(f"   └── Tactical moves: {stats['tactical_moves']:,} ({stats['tactical_moves']/max(stats['positions_extracted'], 1)*100:.1f}%)")
    
    # Duplikált FEN-ek szűrése indexekkel (memóriahatékony)
    unique_fen_idx = {}
    for idx, fen in enumerate(all_fens):
        if fen not in unique_fen_idx:
            unique_fen_idx[fen] = idx
    num_duplicates = len(all_fens) - len(unique_fen_idx)
    if num_duplicates > 0:
        print(f"⚠️ Duplikált pozíciók száma: {num_duplicates}")
    else:
        print("✅ Nincsenek duplikált pozíciók a PGN adatok között.")
    # Csak az egyedi pozíciók és policy-k
    unique_indices = list(unique_fen_idx.values())
    all_fens = [all_fens[i] for i in unique_indices]

    print(f"✅ BALANCED PGN adatok: {len(all_fens):,} pozíció {stats['processed_games']} játszmából")
    return all_fens

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
            hidden_dim = int(input("\nEnter hidden_dim (64-512): "))
            if 64 <= hidden_dim <= 1024:
                break
            else:
                print("❌ Please enter a value between 64 and 1024")
        except ValueError:
            print("❌ Please enter a valid integer")
    
    # Get N
    while True:
        try:
            N = int(input("Enter N - reasoning cycles (2-12): "))
            if 2 <= N <= 12:
                break
            else:
                print("❌ Please enter a value between 2 and 12")
        except ValueError:
            print("❌ Please enter a valid integer")
    
    # Get T
    while True:
        try:
            T = int(input("Enter T - steps per cycle (2-12): "))
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
    
    print("\n✅ Manual Configuration:")
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

    print("\n🎮 Creating dataset from games...")

    print("📥 Loading PGN games...")
    try:
        pgn_fens = load_pgn_data(
            "./lichess_db_standard_rated_2015-05.pgn",
            max_positions=max_positions,
            max_moves=30,
            min_elo=1600
        )
        print(f"✅ Loaded {len(pgn_fens):,} positions from PGN")
    except:
        print("⚠️ PGN file not found")
        exit(0)

    all_fens = pgn_fens

    # Duplikált FEN-ek szűrése indexekkel (memóriahatékony)
    unique_fen_idx = {}
    for idx, fen in enumerate(all_fens):
        if fen not in unique_fen_idx:
            unique_fen_idx[fen] = idx
    num_duplicates = len(all_fens) - len(unique_fen_idx)
    if num_duplicates > 0:
        print(f"⚠️ Duplikált pozíciók száma: {num_duplicates}")
    else:
        print("✅ Nincsenek duplikált pozíciók a PGN adatok között.")
    
    # Csak az egyedi pozíciók és policy-k
    unique_indices = list(unique_fen_idx.values())
    all_fens = [all_fens[i] for i in unique_indices]

    if len(all_fens) == 0:
        print("❌ No training data available!")
        exit(0)

    print(f"📊 Total positions loaded: {len(all_fens):,}")

    # --- Stockfish értékelés minden pozícióra ---
    print("\n🤖 Evaluating all legal moves for all positions...")
    stockfish = ParallelStockfishEvaluator(stockfish_path="stockfish.exe", movetime=10, num_evaluators=int(os.cpu_count() * 0.8) or 2)
    all_move_evals = stockfish.evaluate_positions_parallel(all_fens)
    stockfish.close()

    print(f"✅ Stockfish-evaluated dataset: {len(all_fens):,} positions")
    return all_fens, all_move_evals

if __name__ == "__main__":
    print("🏗️ HRM CHESS MODEL TRAINING")
    print("="*50)
    
    import os
    import sys
    print(f"Using device: {device}")
    
    # GPU MEMORY DETECTION & OPTIMIZATION
    gpu_config = detect_gpu_memory_and_optimize_training()
    
    # Load or create dataset
    dataset_path = "fen_move_score_dataset.pt"
    
    if not os.path.exists(dataset_path):
        print(f"\n📝 Dataset not found: {dataset_path}")
        print("📊 Creating new dataset...")
        
        # Ask user for dataset size
        while True:
            try:
                max_games = int(input("Enter number of games (number of games, e.g., 20000): "))
                if max_games > 0:
                    break
                else:
                    print("❌ Please enter a positive number")
            except ValueError:
                print("❌ Please enter a valid integer")
        
        print(f"🎯 Creating dataset with {max_games:,} games")
        
        # Create dataset from games and puzzles with user-specified size
        fens, moves = create_dataset_from_games(max_games)

        # Save as vector of (fen, move, score) tuples in a .pt file, with metadata
        fen_move_score_vec = []
        for fen, move_list in zip(fens, moves):
            for move_tuple in move_list:
                move, score = move_tuple
                fen_move_score_vec.append((fen, move, score))
                
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
                'data_format': '(fen, move, score)',
                'number_of_games': max_games
            }
        }
        torch.save(dataset_info, output_pt)
        print(f"✅ Saved {len(fen_move_score_vec):,} (fen + move, score) pairs and metadata to {output_pt}")
        
        # Use the created data
        data = dataset_info
    else:
        # Load existing dataset
        print(f"\n📥 Loading existing dataset: {dataset_path}")
        data = torch.load(dataset_path, weights_only=False)
    
    # Extract (fen + move, score) tuples from dataset_info
    dataset_info = data if 'data' in data else data.get('dataset_info', {})
    fen_move_score_vec = dataset_info['data']
    info = dataset_info['info']

    print("✅ Loaded dataset:")
    print(f"   📊 Positions: {len(fen_move_score_vec):,}")
    print(f"   🤖 Source: {info.get('source', 'Unknown')}")
    print(f"   🎮 Data mix: {info.get('data_mix', 'Unknown composition')}")
    print(f"   🖥️ GPU Optimized: {info.get('gpu_optimized', False)}")

    # --- UCI move vocabulary and binning ---
    from hrm_model import generate_all_possible_uci_moves, score_to_bin
    uci_vocab = generate_all_possible_uci_moves()
    uci2idx = {uci: i for i, uci in enumerate(uci_vocab)}
    num_bins = 128  # should match model

    fen_tokens = []
    uci_tokens = []
    target_bins = []
    debug_prints = 0
    for fen, move, score in fen_move_score_vec:
        fen_ascii = [ord(c) for c in fen.ljust(77)[:77]]
        uci_idx = uci2idx.get(move)
        if uci_idx is None:
            if debug_prints < 10:
                print(f"[DEBUG] Skipped: move not in uci2idx: {move}")
                debug_prints += 1
            continue
        try:
            bin_idx = score_to_bin(float(score), num_bins=num_bins)
        except Exception as e:
            if debug_prints < 10:
                print(f"[DEBUG] Skipped: score conversion error: {score}, error: {e}")
                debug_prints += 1
            continue
        fen_tokens.append(fen_ascii)
        uci_tokens.append([uci_idx])
        target_bins.append(bin_idx)

    dataset_size = len(fen_tokens)
    print(f"\n📊 Dataset size: {dataset_size:,} positions")
    
    # MANUAL PARAMETERS
    hidden_dim, N, T = get_manual_parameters()
    
    # Apply GPU optimizations
    batch_size = gpu_config['batch_size']
    lr = 1e-4 * gpu_config['lr_multiplier']
    model_size = f"GPU_MANUAL-{N}x{T}-{gpu_config['optimization_level']}"
    
    print("\n🔧 GPU OPTIMIZATIONS APPLIED:")
    print(f"   📊 Batch Size: {batch_size} (GPU-optimized)")
    print(f"   📈 Learning Rate: {lr:.6f} (base: 2e-4 × {gpu_config['lr_multiplier']:.2f})")
    print(f"   🖥️ GPU Level: {gpu_config['optimization_level']}")
    
    # HRM modell létrehozása optimalizált paraméterekkel
    model = HRMChess(emb_dim=hidden_dim, N=N, T=T).to(device)
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hrm_steps = N * T
    
    print("\n🏗️ MODEL ARCHITECTURE:")
    print(f"📊 Total parameters: {total_params:,}")
    print(f"📊 Trainable parameters: {trainable_params:,}")
    print(f"🔄 HRM reasoning steps: {hrm_steps} (N={N} × T={T})")
    
    # GPU-optimized training configuration
    epochs = 30  # Több epoch a jobb konvergenciáért
    
    print("\n⚙️ GPU-OPTIMIZED TRAINING CONFIGURATION:")
    print(f"   • Model: {model_size}")
    print(f"   • Batch size: {batch_size} (GPU-optimized)")
    print(f"   • Learning rate: {lr:.6f} (GPU-scaled)")
    print("   • Warmup epochs: 3 (linear warmup + cosine annealing)")
    print(f"   • Total epochs: {epochs}")
    print(f"   • HRM steps: {N}×{T}={N*T}")
    print(f"   • Parameters: {total_params:,}")
    print(f"   • Dataset: {dataset_size:,} positions")
    print(f"   🖥️ GPU: {gpu_config['device_name']} ({gpu_config['memory_gb']:.1f} GB)")
    print(f"   🏷️ Optimization Level: {gpu_config['optimization_level']}")
    
    # Create value bin dataset
    dataset = ValueBinDataset(fen_tokens, uci_tokens, target_bins)
    print(f"\n📊 Dataset: {len(dataset):,} positions")
    print("🚀 Starting GPU-optimized HRM training with warmup...")
    
    # Train with GPU-optimized parameters and warmup
    train_loop(model, dataset, epochs=epochs, batch_size=batch_size, lr=lr, warmup_epochs=3, device=device)
    
    # Save final model with hyperparameters and GPU info
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'hyperparams': {
            'hidden_dim': hidden_dim,
            'N': N,
            'T': T,
            'input_dim': 20
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
    print("\n✅ Training completed!")
    print("💾 Model saved to: {model_path}")
    print("🏆 HRM (N={N}, T={T}, hidden_dim={hidden_dim}) with {total_params:,} parameters")
    print("📊 Trained on {len(dataset):,} positions with Warmup mode")
    print("🎮 Dataset: Balanced PGN games + tactical puzzles for enhanced gameplay")
    print("🔥 Warmup: 3 epochs with linear warmup + cosine annealing")
