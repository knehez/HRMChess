import unittest
from stockfish_eval import StockfishEvaluator
import chess

class TestG4Position(unittest.TestCase):
    def test_g4_evaluation(self):
        evaluator = StockfishEvaluator(movetime=50)
        board = chess.Board()
        board.push_san("g3")
        board.push_san("e5")
        board.push_san("f4")
        board.push_san("exf4")
        board.push_san("g4")
        score = evaluator.get_position_evaluation(board.fen(), not board.turn)
        self.assertTrue(score < 0.5, f"g4 után a score túl magas: {score}")

if __name__ == "__main__":
    unittest.main()
