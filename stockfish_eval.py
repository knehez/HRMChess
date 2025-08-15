import math
import chess
from stockfish import Stockfish
import os

class StockfishEvaluator:
    """Stockfish motor integráció pozíció értékeléshez - PYTHON STOCKFISH MODUL"""
    def __init__(self, stockfish_path="./stockfish.exe", movetime=50):
        self.stockfish_path = stockfish_path
        self.movetime = movetime
        self.stockfish = None
        self.initialized = False
        self._init_engine()

    def _init_engine(self):
        try:
            # Try to find stockfish executable
            tried_paths = [self.stockfish_path]
            if self.stockfish_path != "./stockfish.exe":
                tried_paths.append("./stockfish.exe")
            if self.stockfish_path != "stockfish":
                tried_paths.append("stockfish")
            
            stockfish_exe_path = None
            for path in tried_paths:
                if os.path.exists(path):
                    stockfish_exe_path = path
                    print(f"🤖 Stockfish found: {stockfish_exe_path}")
                    break
                elif path == "stockfish":
                    try:
                        # Try system PATH
                        self.stockfish = Stockfish(path="stockfish", parameters={"Threads": 1, "Hash": 32})
                        self.stockfish.set_depth(10)
                        stockfish_exe_path = "stockfish"
                        print("🔍 Using stockfish from system PATH")
                        break
                    except Exception:
                        continue
            
            if stockfish_exe_path is None:
                print(f"❌ Stockfish not found at any of: {tried_paths}")
                self.initialized = False
                return
            
            if self.stockfish is None:
                print("🚀 Starting Stockfish engine...")
                self.stockfish = Stockfish(
                    path=stockfish_exe_path,
                    parameters={
                        "Threads": 1,
                        "Hash": 32,
                        "UCI_Elo": 2000,
                        "Skill Level": 20
                    }
                )
                self.stockfish.set_depth(10)
            
            # Test if engine is working
            self.stockfish.set_fen_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            if self.stockfish.is_fen_valid("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"):
                print("✅ Stockfish engine ready!")
                self.initialized = True
            else:
                print("❌ Stockfish test failed")
                self.initialized = False
                
        except Exception as e:
            print(f"⚠️ Stockfish initialization error: {e}")
            self.initialized = False

    def stockfish_cp_to_winpercent(self, cp: int) -> float:
        return 0.5 * (2.0 / (1.0 + math.exp(-0.00368208 * cp)))

    def get_all_moves(self, fen):
        """
        Visszaadja az összes lehetséges lépést és értékét az adott FEN állásban.
        Minden lépést végrehajt, értékel, majd visszaállítja a táblát.
        Visszatérési érték: List[(move_uci, score)]
        """
        if not self.initialized or not self.stockfish:
            print("⚠️ Stockfish engine not initialized")
            return []
        try:
            board = chess.Board(fen)
            all_moves = list(board.legal_moves)
            move_scores = []
            
            for move in all_moves:
                move_uci = move.uci()
                # Make move on a copy of the board
                board_after = board.copy()
                board_after.push(move)
                
                # Set position and evaluate
                self.stockfish.set_fen_position(board_after.fen())
                evaluation = self.stockfish.get_evaluation()
                
                if evaluation:
                    if evaluation['type'] == 'cp':
                        # Convert centipawn to win percentage (invert for opponent's perspective)
                        score = self.stockfish_cp_to_winpercent(-evaluation['value'])
                    elif evaluation['type'] == 'mate':
                        # Mate score: positive if we mate opponent, negative if opponent mates us
                        mate_in_moves = evaluation['value']
                        score = 1.0 if mate_in_moves > 0 else 0.0
                    else:
                        score = 0.5
                else:
                    score = 0.5
                
                move_scores.append((move_uci, score))
            
            return move_scores
        except Exception as e:
            print(f"⚠️ Error getting all moves: {e}")
            return []

    def get_best_move_only(self, fen):
        """
        Get best move and evaluation for position (compatible with Chess.py)
        Returns: (best_move_uci, score)
        """
        if not self.initialized or not self.stockfish:
            print("⚠️ Stockfish engine not initialized")
            return None, 0.5
        
        try:
            # Check for game-ending conditions
            board = chess.Board(fen)
            
            if board.is_checkmate():
                return "CHECKMATE", 0.0 if board.turn else 1.0
            
            if board.is_stalemate() or board.is_insufficient_material():
                return "STALEMATE", 0.5
            
            if board.can_claim_draw():
                return "DRAW", 0.5
            
            # Set position in stockfish
            self.stockfish.set_fen_position(fen)
            
            # Get best move
            best_move = self.stockfish.get_best_move()
            if not best_move:
                return None, 0.5
            
            # Get evaluation
            evaluation = self.stockfish.get_evaluation()
            if evaluation:
                if evaluation['type'] == 'cp':
                    score = self.stockfish_cp_to_winpercent(-evaluation['value'])
                elif evaluation['type'] == 'mate':
                    mate_in_moves = evaluation['value']
                    score = 1.0 if mate_in_moves > 0 else 0.0
                else:
                    score = 0.5
            else:
                score = 0.5
            
            return best_move, score
            
        except Exception as e:
            print(f"⚠️ Error in get_best_move_only: {e}")
            return None, 0.5

    def close(self):
        """Clean up stockfish engine"""
        if self.stockfish:
            try:
                # The stockfish module handles cleanup automatically
                self.stockfish = None
                self.initialized = False
            except Exception as e:
                print(f"⚠️ Error closing Stockfish: {e}")


class ParallelStockfishEvaluator:
    """Parallel Stockfish evaluator using multiple engines"""
    def __init__(self, num_workers=4, movetime=50, stockfish_path="./stockfish"):
        self.num_workers = num_workers
        self.movetime = movetime
        self.stockfish_path = stockfish_path
        self.evaluators = []
        self._init_evaluators()

    def _init_evaluators(self):
        """Initialize multiple Stockfish evaluators"""
        print(f"🚀 Initializing {self.num_workers} Stockfish evaluators...")
        for i in range(self.num_workers):
            evaluator = StockfishEvaluator(self.stockfish_path, self.movetime)
            if evaluator.initialized:
                self.evaluators.append(evaluator)
                print(f"   ✅ Evaluator {i+1} ready")
            else:
                print(f"   ❌ Evaluator {i+1} failed to initialize")
        
        if not self.evaluators:
            print("❌ No Stockfish evaluators could be initialized!")
        else:
            print(f"✅ {len(self.evaluators)} evaluators ready for parallel processing")

    def evaluate_positions_parallel(self, fens):
        """
        Evaluate multiple positions in parallel
        Returns: List[Tuple[best_move, score]]
        """
        if not self.evaluators:
            print("⚠️ No evaluators available")
            return [(None, 0.5)] * len(fens)

        results = []
        evaluator_index = 0
        
        for fen in fens:
            # Round-robin assignment to evaluators
            evaluator = self.evaluators[evaluator_index % len(self.evaluators)]
            evaluator_index += 1
            
            try:
                best_move, score = evaluator.get_best_move_only(fen)
                results.append((best_move, score))
            except Exception as e:
                print(f"⚠️ Error evaluating position: {e}")
                results.append((None, 0.5))
        
        return results

    def close_all(self):
        """Close all evaluators"""
        for evaluator in self.evaluators:
            evaluator.close()
        self.evaluators.clear()
        print("🔄 All Stockfish evaluators closed")
