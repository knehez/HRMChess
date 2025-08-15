import chess
import numpy as np

# --- FEN to bitplane conversion ---
import chess.pgn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import pickle
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from stockfish_eval import StockfishEvaluator, ParallelStockfishEvaluator
import time
import random

# Import HRM model and related functions
from hrm_model import HRMChess, ValueBinDataset, train_loop, game_to_bitplanes

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

# Old PGN-based functions removed - using Stockfish parallel generation instead

# ================================
# PARALLEL STOCKFISH DATASET GENERATOR
# ================================

class ParallelStockfishDatasetGenerator:
    """Parallel chess dataset generator using multiple Stockfish engines"""
    
    def __init__(self, num_workers=4, movetime=50):
        """Initialize the parallel dataset generator
        
        Args:
            num_workers: Number of parallel Stockfish instances
            movetime: Time in milliseconds for Stockfish analysis
        """
        self.num_workers = num_workers
        self.movetime = movetime
        self.dataset = []
        
    def generate_games_worker(self, worker_id: int, games_to_generate: List[int], 
                             max_moves: int = 60, randomness: float = 0.15) -> List[Dict[str, Any]]:
        """Generate multiple games in a single worker thread with one evaluator"""
        try:
            # Create one Stockfish evaluator for this worker for all games
            evaluator = StockfishEvaluator(movetime=self.movetime)
            
            all_positions = []
            total_games = len(games_to_generate)
            start_time = time.time()
            
            for i, game_id in enumerate(games_to_generate):
                try:
                    game_positions = self._generate_single_game(
                        evaluator, game_id, max_moves, randomness
                    )
                    all_positions.extend(game_positions)
                    
                    if worker_id == 0:
                        elapsed = time.time() - start_time
                        avg_time = elapsed / (i+1)
                        remaining = avg_time * (total_games - (i+1))
                        mins, secs = divmod(int(remaining), 60)
                        print(f"   🎮 Worker {worker_id}: Game {i+1}/{total_games} done | "
                                f"Positions: {len(game_positions)} | "
                                f"Total: {len(all_positions)} | "
                                f"ETA: {mins:02d}:{secs:02d}")
                    
                except Exception as e:
                    print(f"⚠️  Worker {worker_id} game {game_id} error: {e}")
                    continue
            
            # Clean up evaluator only once per worker
            evaluator.close()
            return all_positions
            
        except Exception as e:
            print(f"⚠️  Worker {worker_id} error: {e}")
            return []
    
    def _generate_single_game(self, evaluator: 'StockfishEvaluator', game_id: int, 
                             max_moves: int, randomness: float) -> List[Dict[str, Any]]:
        """Generate a single game using existing evaluator"""
        board = chess.Board()
        game_positions = []
        moves_played = []
        move_count = 0
        
        random_player_is_white = random.choice([True, False])
        
        while not board.is_game_over() and move_count < max_moves:
            try:
                # Get Stockfish best move and score
                best_move, score = evaluator.get_best_move_only(board.fen())
                
                if best_move == "CHECKMATE":
                    # Add final position with checkmate score
                    compact_history = self.create_compact_history(board, moves_played, score)
                    if compact_history:
                        game_positions.append(compact_history)
                    break
                elif best_move in ["STALEMATE", "DRAW"] or best_move is None:
                    break
                
                # Create compact history for current position
                compact_history = self.create_compact_history(board, moves_played, score)
                if compact_history:
                    game_positions.append(compact_history)
                
                # Determine whose turn it is
                current_player_is_white = board.turn
                
                # Only the selected player plays random moves
                if current_player_is_white == random_player_is_white:
                    if random.random() < randomness:
                        legal_moves = list(board.legal_moves)
                        if legal_moves:
                            move = random.choice(legal_moves)
                            move_to_play = move.uci()
                        else:
                            break
                    else:
                        move_to_play = best_move
                else:
                    move_to_play = best_move
                
                # Make the move
                try:
                    move = chess.Move.from_uci(move_to_play)
                    if move in board.legal_moves:
                        board.push(move)
                        moves_played.append(move_to_play)
                        move_count += 1
                    else:
                        break
                except (ValueError, chess.InvalidMoveError):
                    break
                    
            except Exception as e:
                print(f"⚠️  Game {game_id} move error: {e}")
                break
        
        return game_positions
    
    def create_compact_history(self, board: 'chess.Board', moves_played: List[str], score: float, 
                             history_length: int = 8) -> Dict[str, Any]:
        """Create compact history for current position"""
        max_moves_before = history_length - 1
        actual_moves_count = min(len(moves_played), max_moves_before)
        
        if actual_moves_count == 0:
            # No history needed, use current position
            position_starting_fen = board.fen()
            position_moves = []
        else:
            # Create a temporary board to find the correct starting position
            temp_board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            
            # Play all moves up to the history window start
            moves_before_history = moves_played[:-actual_moves_count] if actual_moves_count < len(moves_played) else []
            
            for move_uci in moves_before_history:
                try:
                    move = chess.Move.from_uci(move_uci)
                    if move in temp_board.legal_moves:
                        temp_board.push(move)
                    else:
                        break
                except (ValueError, chess.InvalidMoveError):
                    break
            
            position_starting_fen = temp_board.fen()
            position_moves = moves_played[-actual_moves_count:].copy()
        
        return {
            'starting_fen': position_starting_fen,
            'moves': position_moves,
            'score': score
        }
    
    def generate_dataset_parallel(self, num_games: int = 1000, max_moves: int = 60,
                                randomness: float = 0.15) -> List[Dict[str, Any]]:
        """Generate dataset using parallel workers with persistent evaluators"""
        print("🚀 PARALLEL Stockfish dataset generation")
        print(f"   Workers: {self.num_workers}")
        print(f"   Games: {num_games}")
        print(f"   Max moves per game: {max_moves}")
        print(f"   Randomness: {randomness:.1%}")
        print(f"   Stockfish time: {self.movetime}ms per position")
        
        start_time = time.time()
        completed_games = 0
        
        # Distribute games among workers
        games_per_worker = num_games // self.num_workers
        remaining_games = num_games % self.num_workers
        
        worker_game_assignments = []
        current_game_id = 0
        
        for worker_id in range(self.num_workers):
            # Each worker gets base amount + 1 extra if there are remaining games
            worker_games = games_per_worker + (1 if worker_id < remaining_games else 0)
            worker_game_list = list(range(current_game_id, current_game_id + worker_games))
            worker_game_assignments.append(worker_game_list)
            current_game_id += worker_games
        
        print(f"   Game distribution: {[len(games) for games in worker_game_assignments]}")
        
        # Process games in batches using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit worker tasks (each worker handles multiple games)
            future_to_worker = {
                executor.submit(
                    self.generate_games_worker, 
                    worker_id, 
                    worker_game_assignments[worker_id], 
                    max_moves, 
                    randomness
                ): worker_id
                for worker_id in range(self.num_workers)
            }
            
            # Process completed workers as they finish
            for future in as_completed(future_to_worker):
                worker_id = future_to_worker[future]
                try:
                    worker_positions = future.result()
                    self.dataset.extend(worker_positions)
                    worker_games = len(worker_game_assignments[worker_id])
                    completed_games += worker_games
                    
                    # Progress reporting
                    positions_per_game = len(self.dataset) / completed_games if completed_games > 0 else 0
                    
                    print(f"   Worker {worker_id} completed {worker_games} games | "
                          f"Total games: {completed_games}/{num_games} | "
                          f"Positions: {len(self.dataset):6d} | "
                          f"Avg/game: {positions_per_game:.1f}")
                        
                except Exception as e:
                    worker_games = len(worker_game_assignments[worker_id])
                    print(f"⚠️  Error processing worker {worker_id} ({worker_games} games): {e}")
                    completed_games += worker_games  # Still count as completed to avoid hanging
        
        total_time = time.time() - start_time
        print("✅ PARALLEL dataset generation complete!")
        print(f"   Total games: {num_games}")
        print(f"   Total positions: {len(self.dataset)}")
        print(f"   Average positions per game: {len(self.dataset)/num_games:.1f}")
        print(f"   Total time: {total_time/60:.1f} minutes")
        print(f"   Speed: {len(self.dataset)/(total_time/3600):.0f} positions/hour")
        print(f"   Speedup: ~{self.num_workers:.1f}x theoretical")
        
        return self.dataset
    
    def save_dataset(self, filename: str = "parallel_stockfish_dataset.pt"):
        """Save dataset to file"""
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.dataset, f)
        except Exception as e:
            print(f"❌ Error saving dataset: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        if not self.dataset:
            return {}
        
        scores = [pos['score'] for pos in self.dataset]
        move_counts = [len(pos['moves']) for pos in self.dataset]
        
        # Opening move analysis
        opening_moves = []
        for pos in self.dataset:
            if len(pos['moves']) == 1:  # First move positions
                opening_moves.append(pos['moves'][0])
        
        return {
            'total_positions': len(self.dataset),
            'score_range': (min(scores), max(scores)) if scores else (0, 0),
            'avg_score': sum(scores) / len(scores) if scores else 0,
            'move_count_range': (min(move_counts), max(move_counts)) if move_counts else (0, 0),
            'avg_moves_per_position': sum(move_counts) / len(move_counts) if move_counts else 0,
            'opening_moves_sample': opening_moves[:20]  # First 20 opening moves found
        }


def generate_stockfish_dataset_parallel(num_games=1000, num_workers=4, movetime=50, 
                                       max_moves=60, randomness=0.15):
    """Main function to generate parallel Stockfish dataset"""
    print("🤖 Parallel Stockfish Dataset Generator")
    print("=" * 50)
    
    generator = ParallelStockfishDatasetGenerator(num_workers, movetime)
    
    try:
        dataset = generator.generate_dataset_parallel(
            num_games=num_games,
            max_moves=max_moves,
            randomness=randomness
        )
        
        # Show statistics
        stats = generator.get_statistics()
        print("📊 Final Dataset Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        print("🎯 Ready for training!")
        
        return dataset
        
    except KeyboardInterrupt:
        print("⚠️ Generation interrupted by user")
        if len(generator.dataset) > 0:
            backup_filename = f"stockfish_dataset_backup_{len(generator.dataset)}.pt"
            generator.save_dataset(backup_filename)
            print(f"💾 Backup saved: {backup_filename}")
        return generator.dataset

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
        dataset_path = "game_history_dataset.pt"
        if not os.path.exists(dataset_path):
            print(f"\nDataset not found: {dataset_path}")
            print("Creating new dataset with game history...")
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
            
            # Ask for history length
            while True:
                try:
                    history_length = int(input("Enter history length in ply (default 8): ") or "8")
                    if 1 <= history_length <= 16:
                        break
                    else:
                        print("Please enter a value between 1 and 16!")
                except ValueError:
                    print("Please enter a valid integer!")
            
            print(f"Creating dataset with {max_positions:,} positions and {history_length}-ply history")
            
            # Use new Stockfish-based parallel dataset generation
            print("🚀 Using Stockfish parallel dataset generation...")
            
            # Calculate number of games needed (roughly 40 positions per game)
            estimated_games = max(100, max_positions // 40)
            print(f"Generating ~{estimated_games} games to reach {max_positions:,} positions")
            
            # Generate dataset using parallel Stockfish
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            num_workers = max(1, int(cpu_count * 0.8))
            compact_histories = generate_stockfish_dataset_parallel(
                            num_games=estimated_games,
                            num_workers=num_workers,
                            movetime=50,
                            max_moves=60,
                            randomness=0.5
                        )
            
            # Save as compact game history dataset
            from tqdm import tqdm
            dataset_info = {
                'data': compact_histories,  # Now each item contains its own score
                'info': {
                    'created': time.time(),
                    'source': 'PGN with compact game history',
                    'total_positions': len(compact_histories),
                    'history_length': history_length,
                    'data_format': 'compact_game_history (with embedded score)'
                }
            }
            torch.save(dataset_info, dataset_path)
            print(f"Saved {len(compact_histories):,} compact game histories to {dataset_path}")
            data = dataset_info
        else:
            # Load existing dataset
            print(f"\nLoading existing dataset: {dataset_path}")
            data = torch.load(dataset_path, weights_only=False)
        
        # Deduplicate loaded data (by starting_fen, moves)
        if 'data' in data:
            loaded_data = data['data']
            unique_positions = set()
            unique_entries = []
            for entry in loaded_data:
                key = (entry['starting_fen'], tuple(entry['moves']))
                if key in unique_positions:
                    continue
                unique_positions.add(key)
                unique_entries.append(entry)
            print(f"   Positions before deduplication: {len(loaded_data)}")
            print(f"   Positions after deduplication: {len(unique_entries)}")
            dataset_info = {
                'data': unique_entries,
                'info': data.get('info', {})
            }
            # Free memory: remove large objects
            del loaded_data
            del unique_positions
            del data
            import gc
            gc.collect()
        else:
            print("   Warning: No 'data' key found in loaded dataset for deduplication.")
            dataset_info = data.get('dataset_info', {})
            
        # Extract data
        game_history_data = dataset_info['data']
        info = dataset_info['info']
        print(f"Dataset loaded: {len(game_history_data):,} positions")
        print(f"Source: {info.get('source', 'Unknown')}")
        print(f"History length: {info.get('history_length', 'Unknown')} ply")
        
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
        
        # Create dataset with compact game histories (scores embedded)
        compact_histories = game_history_data  # Each item now contains its own score
        history_length = info.get('history_length', 8)
        dataset = ValueBinDataset(compact_histories, num_bins=128, history_length=history_length)
        
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
                'input_dim': 129,  # Updated for AlphaZero-style encoding with additional evaluation bitplanes
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
                'training_mode': 'game_history_training',
                'gpu_optimized': True,
                'gpu_config': gpu_config,
                'history_length': history_length
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
