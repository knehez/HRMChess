import chess
import numpy as np
import chess.pgn
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import pickle
import random
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
from stockfish_eval import StockfishEvaluator, ParallelStockfishEvaluator

# Import Pure Vision Transformer model and related functions
from hrm_model import PureViTChess, ValueBinDataset, train_loop, game_to_bitplanes

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


def get_hyperparameters():
    """Get hyperparameters from command line arguments or user input"""
    parser = argparse.ArgumentParser(description='Train Pure Vision Transformer Chess Model')
    parser.add_argument('--hidden_dim', type=int, help='Hidden dimension (64-1024)')
    parser.add_argument('--n_heads', type=int, help='Number of attention heads (4-16)')
    parser.add_argument('--n_layers', type=int, help='Number of transformer layers (3-8)')
    parser.add_argument('--ff_mult', type=int, help='Feedforward multiplier (2-6)')
    parser.add_argument('--dropout', type=float, help='Dropout rate (0.0-0.3)')
    parser.add_argument('--lr', type=float, help='Learning rate (1e-5 to 1e-3)')
    parser.add_argument('--batch_size', type=int, help='Batch size (16-128)')
    parser.add_argument('--weight_decay', type=float, help='Weight decay (1e-6 to 1e-3)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--dataset_positions', type=int, help='Number of training positions (or "all")')
    
    args = parser.parse_args()
    
    # If any required parameter is missing, get from user input
    if not all([args.hidden_dim, args.n_heads, args.n_layers, args.ff_mult, 
               args.dropout, args.lr, args.batch_size, args.weight_decay]):
        print("\n🎯 OPTIMIZED HYPERPARAMETER CONFIGURATION")
        print("=" * 50)
        print("Enter the optimized hyperparameters from hyperopt_search.py:")
        
        params = {}
        
        # Get parameters from user input if not provided via command line
        params['hidden_dim'] = args.hidden_dim or get_parameter_input(
            "Hidden dimension", int, 64, 1024, 
            "Controls model capacity (64-1024)"
        )
        
        params['n_heads'] = args.n_heads or get_parameter_input(
            "Number of attention heads", int, 4, 16,
            f"Must divide hidden_dim ({params['hidden_dim']})", 
            lambda x: params['hidden_dim'] % x == 0
        )
        
        params['n_layers'] = args.n_layers or get_parameter_input(
            "Number of transformer layers", int, 3, 8,
            "Controls model depth"
        )
        
        params['ff_mult'] = args.ff_mult or get_parameter_input(
            "Feedforward multiplier", int, 2, 6,
            "Feedforward dim = hidden_dim * ff_mult"
        )
        
        params['dropout'] = args.dropout or get_parameter_input(
            "Dropout rate", float, 0.0, 0.3,
            "Regularization strength"
        )
        
        params['lr'] = args.lr or get_parameter_input(
            "Learning rate", float, 1e-5, 1e-3,
            "Optimizer learning rate (scientific notation: 3.5e-4)"
        )
        
        params['batch_size'] = args.batch_size or get_parameter_input(
            "Batch size", int, 16, 128,
            "Training batch size"
        )
        
        params['weight_decay'] = args.weight_decay or get_parameter_input(
            "Weight decay", float, 1e-6, 1e-3,
            "L2 regularization (scientific notation: 3.7e-5)"
        )
        
        params['epochs'] = args.epochs
        params['dataset_positions'] = args.dataset_positions
        
        return params
    else:
        # All parameters provided via command line
        return {
            'hidden_dim': args.hidden_dim,
            'n_heads': args.n_heads, 
            'n_layers': args.n_layers,
            'ff_mult': args.ff_mult,
            'dropout': args.dropout,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'weight_decay': args.weight_decay,
            'epochs': args.epochs,
            'dataset_positions': args.dataset_positions
        }


def get_parameter_input(name, param_type, min_val, max_val, description, validator=None):
    """Helper function to get a single parameter from user input with validation"""
    while True:
        try:
            user_input = input(f"\n{name} ({min_val}-{max_val}): ")
            if param_type == float:
                # Handle scientific notation
                value = float(user_input)
            else:
                value = param_type(user_input)
            
            if not (min_val <= value <= max_val):
                print(f"❌ Value must be between {min_val} and {max_val}")
                continue
                
            if validator and not validator(value):
                print(f"❌ Invalid value. {description}")
                continue
                
            print(f"✅ {name}: {value} - {description}")
            return value
            
        except ValueError:
            print(f"❌ Please enter a valid {param_type.__name__}")


def print_configuration(params):
    """Print the final configuration for user confirmation"""
    print(f"\n🔧 FINAL CONFIGURATION")
    print("=" * 40)
    print(f"Model Architecture:")
    print(f"  Hidden Dimension: {params['hidden_dim']}")
    print(f"  Attention Heads: {params['n_heads']}")
    print(f"  Transformer Layers: {params['n_layers']}")
    print(f"  FF Multiplier: {params['ff_mult']}")
    print(f"  Dropout Rate: {params['dropout']:.4f}")
    
    print(f"\nTraining Parameters:")
    print(f"  Learning Rate: {params['lr']:.2e}")
    print(f"  Batch Size: {params['batch_size']}")
    print(f"  Weight Decay: {params['weight_decay']:.2e}")
    print(f"  Epochs: {params['epochs']}")
    
    # Calculate estimated parameters
    dim_feedforward = params['hidden_dim'] * params['ff_mult']
    # Rough estimate for Pure ViT parameters
    estimated_params = (
        params['hidden_dim'] * params['hidden_dim'] * 4 +  # Patch embedding + pos embedding  
        params['n_layers'] * (
            params['hidden_dim'] * params['hidden_dim'] * 4 +  # Self-attention (Q, K, V, O)
            params['hidden_dim'] * dim_feedforward * 2 +         # FFN
            params['hidden_dim'] * 4                             # Layer norms
        ) +
        params['hidden_dim'] * 128  # Classification head
    )
    
    print(f"\nModel Complexity:")
    print(f"  Feedforward Dim: {dim_feedforward}")
    print(f"  Estimated Parameters: {estimated_params:,}")
    
    if estimated_params < 1_000_000:
        complexity = "Light"
    elif estimated_params < 5_000_000:
        complexity = "Medium"
    else:
        complexity = "Heavy"
    print(f"  Complexity Level: {complexity}")
    
    while True:
        confirm = input(f"\n✅ Proceed with this configuration? (y/n): ").strip().lower()
        if confirm in ['y', 'yes']:
            return True
        elif confirm in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' or 'n'")

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
        self.pgn_games = []  # Store PGN games separately
        
    def generate_games_worker(self, worker_id: int, games_to_generate: List[int], 
                             max_moves: int = 60, randomness: float = 0.15) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Generate multiple games in a single worker thread with one evaluator"""
        try:
            # Create one Stockfish evaluator for this worker for all games
            evaluator = StockfishEvaluator(movetime=self.movetime)
            
            all_positions = []
            all_pgn_games = []
            total_games = len(games_to_generate)
            start_time = time.time()
            
            for i, game_id in enumerate(games_to_generate):
                try:
                    game_positions, pgn_game = self._generate_single_game(
                        evaluator, game_id, max_moves, randomness
                    )
                    all_positions.extend(game_positions)
                    if pgn_game:  # Only add non-empty PGN games
                        all_pgn_games.append(pgn_game)
                    
                    if worker_id == 0:
                        elapsed = time.time() - start_time
                        avg_time = elapsed / (i+1)
                        remaining = avg_time * (total_games - (i+1))
                        mins, secs = divmod(int(remaining), 60)
                        print(f"   🎮 Worker {worker_id}: Game {i+1}/{total_games} done | "
                                f"Positions: {len(game_positions)} | "
                                f"Total pos: {len(all_positions)} | PGN: {len(all_pgn_games)} | "
                                f"ETA: {mins:02d}:{secs:02d}")
                    
                except Exception as e:
                    print(f"⚠️  Worker {worker_id} game {game_id} error: {e}")
                    continue
            
            # Clean up evaluator only once per worker
            evaluator.close()
            return all_positions, all_pgn_games
            
        except Exception as e:
            print(f"⚠️  Worker {worker_id} error: {e}")
            return []
    
    def _generate_single_game(self, evaluator: 'StockfishEvaluator', game_id: int, 
                             max_moves: int, randomness: float) -> Tuple[List[Dict[str, Any]], str]:
        """Generate a single game using existing evaluator - returns positions and PGN string"""
        board = chess.Board()
        game_positions = []
        moves_played = []
        moves_with_scores = []  # For PGN generation
        move_count = 0
        
        random_player_is_white = random.choice([True, False])

        mating_approach = False

        while not board.is_game_over() and move_count < max_moves:
            try:
                # Determine whose turn it is and if they play randomly
                current_player_is_white = board.turn
                will_play_random = (current_player_is_white == random_player_is_white and 
                                   random.random() < randomness)
                
                # Decide which move to play
                if will_play_random:
                    # Random move - no need for Stockfish best move
                    legal_moves = list(board.legal_moves)
                    if not legal_moves:
                        break
                    move = random.choice(legal_moves)
                    move_to_play = move.uci()
                    
                    # Convert to SAN before making the move
                    san_move = board.san(move)
                    board.push(move)
                    moves_played.append(move_to_play)
                    
                    # Evaluate position after random move for both compact history and PGN
                    if not mating_approach:
                        move_score = evaluator.get_position_evaluation(board.fen(), not board.turn)
                    else:
                        move_score = 1.0
                else:
                    # Best move - get from Stockfish
                    best_move, original_score = evaluator.get_best_move_and_score(board.fen(), board.turn)
                    
                    if original_score == 1 or original_score == 0:
                        mating_approach = True
                    
                    try:
                        move = chess.Move.from_uci(best_move)
                        if move not in board.legal_moves:
                            break
                        
                        # Convert to SAN before making the move
                        san_move = board.san(move)
                        board.push(move)
                        moves_played.append(best_move)
                        
                        # Use the original position score for best moves
                        move_score = original_score
                        
                    except (ValueError, chess.InvalidMoveError):
                        break
                
                # Add move with score to PGN list
                moves_with_scores.append((san_move, move_score))
                move_count += 1
                
                # Create compact history AFTER making the move - this saves 1 evaluation per turn
                # because we don't create compact history for empty move history
                compact_history = self.create_compact_history(board, moves_played, move_score)
                if compact_history:
                    game_positions.append(compact_history)
                    
            except Exception as e:
                print(f"⚠️  Game {game_id} move error: {e}")
                break
        
        # Create PGN string with move numbers
        pgn_moves = []
        for i, (move_san, score) in enumerate(moves_with_scores):
            move_num = (i // 2) + 1
            if i % 2 == 0:  # White move
                pgn_moves.append(f"{move_num}. {move_san} {score:.2f}")
            else:  # Black move
                pgn_moves.append(f"{move_san} {score:.2f}")
        
        pgn_game = " ".join(pgn_moves)
        
        return game_positions, pgn_game
    
    def create_compact_history(self, board: 'chess.Board', moves_played: List[str], score: float, 
                             history_length: int = 8) -> Dict[str, Any]:
        """Create compact history for current position"""
        max_moves_before = history_length - 1
        actual_moves_count = min(len(moves_played), max_moves_before)
        
        if actual_moves_count == 0:
            return None
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
                                randomness: float = 0.15) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Generate dataset using parallel workers with persistent evaluators - returns positions and PGN games"""
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
                    worker_positions, worker_pgn_games = future.result()
                    self.dataset.extend(worker_positions)
                    self.pgn_games.extend(worker_pgn_games)
                    worker_games = len(worker_game_assignments[worker_id])
                    completed_games += worker_games
                    
                    # Progress reporting
                    positions_per_game = len(self.dataset) / completed_games if completed_games > 0 else 0
                    
                    print(f"   Worker {worker_id} completed {worker_games} games | "
                          f"Total games: {completed_games}/{num_games} | "
                          f"Positions: {len(self.dataset):6d} | PGN games: {len(self.pgn_games)} | "
                          f"Avg pos/game: {positions_per_game:.1f}")
                        
                except Exception as e:
                    worker_games = len(worker_game_assignments[worker_id])
                    print(f"⚠️  Error processing worker {worker_id} ({worker_games} games): {e}")
                    completed_games += worker_games  # Still count as completed to avoid hanging
        
        total_time = time.time() - start_time
        print("✅ PARALLEL dataset generation complete!")
        print(f"   Total games: {num_games}")
        print(f"   Total positions: {len(self.dataset)}")
        print(f"   Total PGN games: {len(self.pgn_games)}")
        print(f"   Average positions per game: {len(self.dataset)/num_games:.1f}")
        print(f"   Total time: {total_time/60:.1f} minutes")
        print(f"   Speed: {len(self.dataset)/(total_time/3600):.0f} positions/hour")
        print(f"   Speedup: ~{self.num_workers:.1f}x theoretical")
        
        return self.dataset, self.pgn_games
    
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
        dataset, pgn_games = generator.generate_dataset_parallel(
            num_games=num_games,
            max_moves=max_moves,
            randomness=randomness
        )
        
        # Save PGN games to text file
        pgn_filename = f"stockfish_games_{len(pgn_games)}.pgn"
        print(f"💾 Saving {len(pgn_games)} PGN games to {pgn_filename}...")
        with open(pgn_filename, 'w', encoding='utf-8') as f:
            # Write header with metadata
            f.write(f"# Stockfish Generated Chess Games with Scores\n")
            f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total games: {len(pgn_games)}\n")
            f.write(f"# Format: move_number. move score (e.g., 1. e4 0.54 e5 0.48)\n")
            f.write(f"# Workers: {num_workers}, Movetime: {movetime}ms\n")
            f.write(f"# Max moves: {max_moves}, Randomness: {randomness:.1%}\n")
            f.write(f"#\n")
            
            # Write PGN games (one per line)
            for i, pgn_game in enumerate(pgn_games):
                f.write(f"{pgn_game}\n\n")
        
        print(f"✅ PGN games saved to {pgn_filename}")
        
        # Show statistics
        stats = generator.get_statistics()
        print("📊 Final Dataset Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        print("🎯 Ready for training!")
        
        return dataset, pgn_games
        
    except KeyboardInterrupt:
        print("⚠️ Generation interrupted by user")
        if len(generator.dataset) > 0 or len(generator.pgn_games) > 0:
            backup_filename = f"stockfish_dataset_backup_{len(generator.dataset)}.pt"
            generator.save_dataset(backup_filename)
            
            # Also save PGN backup
            if len(generator.pgn_games) > 0:
                pgn_backup = f"stockfish_pgn_backup_{len(generator.pgn_games)}.pgn"
                with open(pgn_backup, 'w', encoding='utf-8') as f:
                    for i, pgn_game in enumerate(generator.pgn_games):
                        f.write(f"{pgn_game}\n\n")
                print(f"💾 PGN backup saved: {pgn_backup}")
            
            print(f"💾 Dataset backup saved: {backup_filename}")
        return generator.dataset, generator.pgn_games

# Import StockfishEvaluator and ParallelStockfishEvaluator from the new module
from stockfish_eval import StockfishEvaluator, ParallelStockfishEvaluator

if __name__ == "__main__":
    try:
        print("🏆 PURE VISION TRANSFORMER CHESS MODEL TRAINING")
        print("=" * 60)
        print(f"Using device: {device}")
        
        # Get optimized hyperparameters
        params = get_hyperparameters()
        
        # Show configuration and get user confirmation
        if not print_configuration(params):
            print("Training cancelled by user.")
            sys.exit(0)
        
        # Load or create dataset
        dataset_path = "game_history_dataset.pt"
        if not os.path.exists(dataset_path):
            print(f"\n📂 Dataset not found: {dataset_path}")
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
            compact_histories, pgn_games = generate_stockfish_dataset_parallel(
                            num_games=estimated_games,
                            num_workers=num_workers,
                            movetime=50,
                            max_moves=80,
                            randomness=0.3
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
            print(f"\n📂 Loading existing dataset: {dataset_path}")
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
        
        # Create Pure ViT model with optimized parameters
        model = PureViTChess(
            hidden_dim=params['hidden_dim'], 
            n_heads=params['n_heads'], 
            n_layers=params['n_layers'], 
            ff_mult=params['ff_mult'],
            dropout=params['dropout']
        ).to(device)
        
        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs (DataParallel)")
            model = torch.nn.DataParallel(model)
            
        # Model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n🤖 Model: {total_params:,} parameters")
        print(f"Architecture: Pure ViT (heads={params['n_heads']}, layers={params['n_layers']}, dim={params['hidden_dim']})")
        
        # Create dataset with compact game histories (scores embedded)
        compact_histories = game_history_data  # Each item now contains its own score
        history_length = info.get('history_length', 8)
        full_dataset = ValueBinDataset(compact_histories, num_bins=128, history_length=history_length)
        
        print(f"\n📊 Dataset loaded: {len(full_dataset):,} positions available")
        
        # Determine training positions
        if params['dataset_positions'] is None:
            # Ask user how many positions to train on
            while True:
                try:
                    user_input = input(f"How many positions do you want to train on? (1-{len(full_dataset):,}, or 'all'): ").strip().lower()
                    
                    if user_input == 'all':
                        dataset = full_dataset
                        training_positions = len(full_dataset)
                        break
                    else:
                        training_positions = int(user_input.replace(',', ''))
                        if 1 <= training_positions <= len(full_dataset):
                            # Create subset of dataset
                            if training_positions == len(full_dataset):
                                dataset = full_dataset
                            else:
                                # Randomly sample positions
                                indices = torch.randperm(len(full_dataset))[:training_positions]
                                subset_data = [full_dataset.compact_game_history_list[i] for i in indices]
                                dataset = ValueBinDataset(subset_data, num_bins=128, history_length=history_length)
                            break
                        else:
                            print(f"❌ Please enter a number between 1 and {len(full_dataset):,}")
                            
                except ValueError:
                    print("❌ Please enter a valid number or 'all'")
        else:
            # Use specified number of positions
            training_positions = params['dataset_positions']
            if training_positions >= len(full_dataset):
                dataset = full_dataset
                training_positions = len(full_dataset)
            else:
                indices = torch.randperm(len(full_dataset))[:training_positions]
                subset_data = [full_dataset.compact_game_history_list[i] for i in indices]
                dataset = ValueBinDataset(subset_data, num_bins=128, history_length=history_length)
        
        print(f"\n🚀 Starting training:")
        print(f"   Epochs: {params['epochs']}")
        print(f"   Positions: {len(dataset):,}")
        print(f"   Batch size: {params['batch_size']}")
        print(f"   Learning rate: {params['lr']:.2e}")
        print(f"   Weight decay: {params['weight_decay']:.2e}")
        print(f"📈 Training on {(len(dataset)/len(full_dataset)*100):.1f}% of available data")
        
        # Train with optimized parameters
        train_loop(
            model=model, 
            dataset=dataset, 
            epochs=params['epochs'], 
            batch_size=params['batch_size'], 
            lr=params['lr'], 
            device=device, 
            use_amp=True
        )
        
        # Save model with optimized hyperparameters
        if isinstance(model, torch.nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
            
        final_checkpoint = {
            'model_state_dict': state_dict,
            'hyperparams': {
                'hidden_dim': params['hidden_dim'],
                'n_heads': params['n_heads'],
                'n_layers': params['n_layers'], 
                'dim_feedforward': params['hidden_dim'] * params['ff_mult'],
                'dropout': params['dropout'],
                'model_type': 'pure_vit'
            },
            'training_info': {
                'epochs': params['epochs'],
                'batch_size': params['batch_size'],
                'lr': params['lr'],
                'weight_decay': params['weight_decay'],
                'dataset_size': len(dataset),
                'total_params': total_params,
                'training_mode': 'optimized_hyperparameters',
                'history_length': history_length,
                'optimized': True
            }
        }
        
        model_path = f"optimized_vit_chess_model_{params['hidden_dim']}d_{params['n_layers']}l.pt"
        torch.save(final_checkpoint, model_path)
        print(f"\n✅ Training completed! Model saved to: {model_path}")
        print(f"🏆 Optimized Pure ViT model:")
        print(f"   Parameters: {total_params:,}")
        print(f"   Architecture: {params['hidden_dim']}d-{params['n_heads']}h-{params['n_layers']}l-{params['ff_mult']}ff")
        print(f"   Trained on: {len(dataset):,} positions")
        print(f"   Hyperparameters: Optimized via hyperopt_search.py")
        
    except KeyboardInterrupt:
        print("\n❌ Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)
