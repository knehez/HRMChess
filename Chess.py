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
from hrm_model import HRMChess, PolicyDataset, train_step, train_loop, quick_train_eval

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
        print(" Cannot proceed without GPU information.")
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
    """Stockfish motor integráció pozíció értékeléshez - EGYSZERŰ READLINE MEGKÖZELÍTÉS"""
    
    def __init__(self, stockfish_path="./stockfish.exe", movetime=50):
        self.stockfish_path = stockfish_path
        self.movetime = movetime
        self.process = None
        self.initialized = False
        self._init_engine()
    
    def _init_engine(self):
        """Stockfish motor inicializálás - egyszerű megközelítés"""
        try:
            if os.path.exists(self.stockfish_path):
                print(f"🤖 Stockfish found: {self.stockfish_path}")
            else:
                print(f"❌ Stockfish not found at: {self.stockfish_path}")
                print("🔍 Looking for stockfish in system PATH...")
                # Try system stockfish
                self.stockfish_path = "stockfish"
            
            # Start Stockfish process with simple configuration
            print("🚀 Starting Stockfish engine...")
            self.process = subprocess.Popen(
                [self.stockfish_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                universal_newlines=True
            )
            
            # Simple UCI initialization
            self._send_command("uci")
            self._wait_for_response("uciok")
            
            self._send_command("isready")
            self._wait_for_response("readyok")
            
            print("✅ Stockfish engine ready!")
            self.initialized = True
            
        except Exception as e:
            print(f"⚠️ Stockfish initialization error: {e}")
            self.initialized = False
    
    def _send_command(self, command):
        """Send command to Stockfish - egyszerű megközelítés"""
        if self.process and self.process.stdin:
            try:
                self.process.stdin.write(command + "\n")
                self.process.stdin.flush()
            except Exception as e:
                print(f"⚠️ Error sending command: {e}")
                self.initialized = False
    
    def _read_line(self, timeout=3.0):
        """Read a line from Stockfish - EGYSZERŰ READLINE"""
        if not self.process or not self.process.stdout:
            return None
        
        try:
            # Simple readline with basic timeout using threading
            import threading
            import queue
            
            result_queue = queue.Queue()
            
            def read_line():
                try:
                    line = self.process.stdout.readline()
                    result_queue.put(line)
                except Exception as e:
                    result_queue.put(None)
            
            # Start reading thread
            thread = threading.Thread(target=read_line)
            thread.daemon = True
            thread.start()
            
            # Wait for result with timeout
            thread.join(timeout=timeout)
            
            if not result_queue.empty():
                line = result_queue.get_nowait()
                if line:
                    return line.strip()
            
            return None
            
        except Exception as e:
            return None
    
    def _wait_for_response(self, expected, timeout=5.0):
        """Wait for specific response from Stockfish - egyszerű megközelítés"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            line = self._read_line(1.0)  # 1 second timeout per line
            if line and expected in line:
                return True
            if not line:  # If no line received, continue waiting
                continue
        return False

    def evaluate_all_legal_moves(self, fen):
        """
        Összes legális lépés értékelése Stockfish-sel - EGYSZERŰSÍTETT VERZIÓ + UCINEWGAME
        Returns: List of (move_tuple, evaluation_score) for all legal moves
        """
        if not self.initialized or not self.process:
            print("⚠️ Stockfish engine not initialized")
            return []
        
        try:
            # Reset engine state before evaluation
            self._send_command("ucinewgame")
            self._send_command("isready")
            self._wait_for_response("readyok", timeout=2.0)
            
            # Create board from FEN to get legal moves
            board = chess.Board(fen)
            legal_moves = list(board.legal_moves)
            
            if not legal_moves:
                return []
            
            move_evaluations = []
            
            # Evaluate each legal move with simplified approach
            for move in legal_moves:
                move_tuple = (move.from_square, move.to_square)
                score = 0.0  # Default neutral score
                
                try:
                    # Check if process is still alive
                    if self.process.poll() is not None:
                        print("⚠️ Stockfish process died, restarting...")
                        self._init_engine()
                        if not self.initialized:
                            move_evaluations.append((move_tuple, score))
                            continue
                    
                    # Make the move on a copy of the board
                    temp_board = board.copy()
                    temp_board.push(move)
                    
                    # Get the resulting position FEN
                    new_fen = temp_board.fen()
                    
                    # Set position after the move
                    self._send_command(f"position fen {new_fen}")
                    
                    # Get evaluation with short thinking time
                    eval_time = max(10, self.movetime // 4)  # Minimum 10ms
                    self._send_command(f"go movetime {eval_time}")
                    
                    # Parse response - simplified approach
                    evaluation_successful = False
                    max_attempts = 10
                    attempts = 0
                    
                    while attempts < max_attempts and not evaluation_successful:
                        line = self._read_line(1.0)  # 1 second timeout per line
                        if not line:
                            attempts += 1
                            continue
                        
                        attempts += 1
                        
                        if line.startswith('bestmove'):
                            evaluation_successful = True
                            break
                        elif 'score cp' in line:
                            # Extract centipawn score
                            try:
                                parts = line.split()
                                for i, part in enumerate(parts):
                                    if part == "cp" and i + 1 < len(parts):
                                        cp_score = int(parts[i + 1])
                                        # Negate score because we're evaluating from opponent's perspective
                                        score = max(-1.0, min(1.0, -cp_score / 300.0))
                                        break
                            except (ValueError, IndexError):
                                pass
                        elif 'score mate' in line:
                            # Mate score
                            try:
                                parts = line.split()
                                for i, part in enumerate(parts):
                                    if part == "mate" and i + 1 < len(parts):
                                        mate_moves = int(parts[i + 1])
                                        # Negate because we're evaluating from opponent's perspective
                                        score = -1.0 if mate_moves > 0 else 1.0
                                        break
                            except (ValueError, IndexError):
                                pass
                    
                    move_evaluations.append((move_tuple, score))
                    
                except Exception as e:
                    # If evaluation fails for this move, assign neutral score
                    move_evaluations.append((move_tuple, 0.0))
                    continue
            
            return move_evaluations
            
        except Exception as e:
            print(f"⚠️ Stockfish evaluation error: {e}")
            # Fallback: return all legal moves with neutral scores
            try:
                board = chess.Board(fen)
                legal_moves = list(board.legal_moves)
                return [(move.from_square, move.to_square, 0.0) for move in legal_moves]
            except:
                return []
    
    def close(self):
        """Close Stockfish engine - egyszerű cleanup"""
        if self.process:
            try:
                self._send_command("quit")
                self.process.terminate()
                self.process.wait(timeout=3)
            except:
                if self.process:
                    self.process.kill()
            finally:
                self.process = None
                self.initialized = False
                print("🔌 Stockfish engine closed")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.close()

def create_dataset_from_games(max_positions=10000):
    """
    Dataset létrehozása játszmákból és Stockfish értékeléssel - egyszerűsített megközelítés
    
    Args:
        max_positions: Maximális pozíciók száma a dataset-ben
    """
    import math
    import time

    print("\n🎮 Creating dataset from games with Stockfish evaluation...")
    print(f"🎯 Target positions: {max_positions:,}")

    avg_positions_per_game = 22
    log_factor = math.log10(max(max_positions, 1000))
    base_ratio = 0.5
    scale_factor = (log_factor - 3.0) / 4.0
    scale_factor = max(0, min(1, scale_factor))
    puzzle_ratio = base_ratio - (scale_factor * 0.35)
    pgn_ratio = 1.0 - puzzle_ratio
    pgn_target_positions = int(max_positions * pgn_ratio)
    estimated_games_needed = max(1000, int(pgn_target_positions / avg_positions_per_game))

    print(f"🎯 Automatic scaling for {max_positions:,} positions:")
    print(f"   📊 PGN ratio: {pgn_ratio:.1%} → {pgn_target_positions:,} positions")
    print(f"   🧩 Puzzle ratio: {puzzle_ratio:.1%} → {int(max_positions * puzzle_ratio):,} positions")
    print(f"   🎮 Estimated games needed: {estimated_games_needed:,}")

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

    print(f"📥 Loading tactical puzzles...")
    puzzle_target = int(max_positions * puzzle_ratio)

    try:
        puzzle_states, puzzle_policies, puzzle_fens = load_puzzle_data(
            "./lichess_db_puzzle.csv",
            fen_to_tensor,
            max_puzzles=puzzle_target,
            min_rating=800,
            max_rating=2500
        )
        print(f"✅ Loaded {len(puzzle_states):,} tactical positions from CSV")
    except:
        print("⚠️ Puzzle CSV file not found, using only PGN data")
        puzzle_states, puzzle_policies, puzzle_fens = [], [], []

    print(f"\n🎯 COMBINING PGN + PUZZLE DATA:")
    print(f"   📊 PGN positions: {len(pgn_states):,}")
    print(f"   🧩 Puzzle positions: {len(puzzle_states):,}")

    all_states = pgn_states + puzzle_states
    all_fens = pgn_fens + puzzle_fens

    if len(all_states) == 0:
        print("❌ No training data available!")
        return None, None

    print(f"📊 Total positions loaded: {len(all_states):,}")

    adequacy_ratio = len(all_states) / max_positions if max_positions > 0 else 0
    if adequacy_ratio < 0.5:
        print(f"⚠️ Warning: Only {adequacy_ratio*100:.1f}% of target positions loaded!")
        print(f"💡 Consider: checking file paths, lowering rating filters, or reducing target size")
    elif adequacy_ratio >= 1.0:
        print(f"✅ Excellent: {adequacy_ratio:.1f}x target positions available")
    else:
        print(f"✅ Good: {adequacy_ratio*100:.1f}% of target positions loaded")

    if len(all_states) > max_positions:
        print(f"🎯 Randomly selecting {max_positions:,} positions from {len(all_states):,} available")
        indices = np.random.choice(len(all_states), max_positions, replace=False)
        all_states = [all_states[i] for i in indices]
        all_fens = [all_fens[i] for i in indices]

    print(f"🤖 Starting comprehensive Stockfish evaluation...")
    print("📊 Now evaluating ALL legal moves for each position (this will take longer but provide richer training data)")
    print("📊 Progress updates will appear as positions are evaluated...")

    processed_states = []
    processed_policies = []
    
    start_time = time.time()

    evaluator = StockfishEvaluator(movetime=50)
    
    for i, fen in enumerate(all_fens):
        state = all_states[i]
        
        # NEW: Evaluate ALL legal moves instead of just the best one
        move_evaluations = evaluator.evaluate_all_legal_moves(fen)
        
        processed_states.append(state)
        processed_policies.append(move_evaluations)
        
        # Progress update every 5 positions (more frequent due to longer processing)
        if (i + 1) % 5 == 0 or (i + 1) == len(all_fens):
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(all_fens) - (i + 1)) / rate if rate > 0 else 0
            
            # Calculate average moves per position for this batch
            total_moves_so_far = sum(len(moves) for moves in processed_policies)
            avg_moves_per_pos = total_moves_so_far / len(processed_policies) if processed_policies else 0
            
            print(f"📈 Progress: {i + 1}/{len(all_fens):,} positions | "
                  f"Rate: {rate:.2f} pos/s | ETA: {eta/60:.1f} min | "
                  f"Avg moves/pos: {avg_moves_per_pos:.1f}")
    
    evaluator.close()

    total_moves = sum(len(moves) for moves in processed_policies)
    elapsed_total = time.time() - start_time
    
    print(f"\n✅ COMPREHENSIVE policy dataset created in {elapsed_total/60:.1f} minutes!")
    print(f"   📊 Positions evaluated: {len(processed_states):,}")
    print(f"   📊 Total moves evaluated: {total_moves:,}")
    print(f"   📊 Average moves per position: {total_moves/len(processed_states):.1f}")
    print(f"   ⚡ Average rate: {len(processed_states)/elapsed_total:.2f} positions/second")
    print(f"   🎯 Training data richness: {total_moves/len(processed_states):.1f}x more comprehensive than single-move approach")

    return processed_states, processed_policies


if __name__ == "__main__":
    print("🏗️ HRM CHESS MODEL TRAINING")
    print("="*50)
    
    import os
    import sys
    print(f"Using device: {device}")
    
    # GPU MEMORY DETECTION & OPTIMIZATION
    gpu_config = detect_gpu_memory_and_optimize_training()
    
    # Load or create dataset
    dataset_path = "chess_positions_dataset.pt"
    
    if not os.path.exists(dataset_path):
        print(f"\n📝 Dataset not found: {dataset_path}")
        print("📊 Creating new dataset...")
        
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
        states, policies = create_dataset_from_games(max_positions)
        
        if states is None:
            print("❌ Failed to create dataset!")
            exit(1)
        
        # Save dataset
        print("💾 Saving dataset...")
        # Calculate average moves per position from the policies data
        total_moves_in_dataset = sum(len(moves) for moves in policies)
        avg_moves_per_position = total_moves_in_dataset / len(states) if states else 0
        
        dataset_info = {
            'states': np.array(states, dtype=np.float32),
            'policies': policies,
            'info': {
                'created': time.time(),
                'source': 'PGN + Tactical Puzzles (Comprehensive Stockfish evaluation)',
                'positions': len(states),
                'stockfish_evaluation': 'ALL_LEGAL_MOVES',
                'evaluation_method': 'comprehensive_move_evaluation',
                'data_mix': 'Balanced PGN games + tactical puzzles',
                'gpu_optimized': True,
                'gpu_config': gpu_config,
                'user_specified_size': max_positions,
                'avg_moves_per_position': avg_moves_per_position
            }
        }
        
        torch.save(dataset_info, dataset_path)
        print(f"✅ Dataset saved to: {dataset_path}")
        
        # Use the created data
        data = dataset_info
    else:
        # Load existing dataset
        print(f"\n📥 Loading existing dataset: {dataset_path}")
        data = torch.load(dataset_path, weights_only=False)
    
    states = data['states']
    policies = data['policies']
    info = data['info']
    
    print(f"✅ Loaded dataset:")
    print(f"   📊 Positions: {len(states):,}")
    print(f"   🤖 Source: {info.get('source', 'Unknown')}")
    print(f"   🎮 Data mix: {info.get('data_mix', 'Unknown composition')}")
    print(f"   🖥️ GPU Optimized: {info.get('gpu_optimized', False)}")
    
    # training mode
    print("\n🎯 TRAINING MODE")
    print("   • Input: 72-dim vector → 8x8 2D conv + extra features")
    print("   • Policy Head: 64x64 move matrix")
    print("   • Dataset: Balanced PGN games + tactical puzzles")
    
    print("\n⚙️ HRM PARAMETER EXPLANATION:")
    print("   • N: Number of high-level reasoning cycles")
    print("   • T: Steps per cycle (low-level processing)")
    print("   • Total HRM steps = N × T")
    print("   • hidden_dim: Neural network width")
    print("   • Optimal balance: complexity vs speed vs accuracy")
    
    # Configuration
    data_path = dataset_path
    model_path = "hrm_chess_model.pt"
    
    # Get dataset info
    dataset_size = len(states)
    print(f"\n📊 Dataset size: {dataset_size:,} positions")
    
    # MANUAL PARAMETERS
    hidden_dim, N, T = get_manual_parameters()
    
    # Apply GPU optimizations
    batch_size = gpu_config['batch_size']
    lr = 2e-4 * gpu_config['lr_multiplier']
    model_size = f"GPU_MANUAL-{N}x{T}-{gpu_config['optimization_level']}"
    
    print("\n🔧 GPU OPTIMIZATIONS APPLIED:")
    print(f"   📊 Batch Size: {batch_size} (GPU-optimized)")
    print(f"   📈 Learning Rate: {lr:.6f} (base: 2e-4 × {gpu_config['lr_multiplier']:.2f})")
    print(f"   🖥️ GPU Level: {gpu_config['optimization_level']}")
    
    # HRM modell létrehozása optimalizált paraméterekkel
    model = HRMChess(input_dim=72, hidden_dim=hidden_dim, N=N, T=T).to(device)
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hrm_steps = N * T
    
    print("\n🏗️ MODEL ARCHITECTURE:")
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
    
    # Memory usage estimation
    estimated_memory_per_batch = (batch_size * 72 * 4 + batch_size * 64 * 64 * 4) / (1024**3)  # Rough estimate in GB
    print(f"   📊 Estimated memory/batch: ~{estimated_memory_per_batch:.2f} GB")
    
    # Create Policy dataset
    dataset = PolicyDataset(states, policies)
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
    print(f"📊 Trained on {len(dataset):,} positions with Warmup mode")
    print(f"🎮 Dataset: Balanced PGN games + tactical puzzles for enhanced gameplay")
    print(f"🔥 Warmup: 3 epochs with linear warmup + cosine annealing")
    
    print("🎯 Expected: Enhanced move prediction + position evaluation + tactical strength")
    print("⚔️ Ready for tactical fine-tuning and stronger gameplay!")
