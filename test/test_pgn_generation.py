import unittest
import os
from Chess import ParallelStockfishDatasetGenerator

class TestPGNGeneration(unittest.TestCase):
    def test_short_pgn_generation(self):
        generator = ParallelStockfishDatasetGenerator(
            num_workers=1,
            movetime=30
        )
        dataset, pgn_games = generator.generate_dataset_parallel(
            num_games=2,
            max_moves=6,
            randomness=0.2
        )
        self.assertTrue(len(pgn_games) > 0, "No PGN games generated!")
        self.assertTrue(len(dataset) > 0, "No positions generated!")
        # Save PGN for manual inspection
        with open("test/test_short.pgn", "w", encoding="utf-8") as f:
            for pgn in pgn_games:
                f.write(pgn + "\n")
        print(f"PGN games saved: {len(pgn_games)}")

if __name__ == "__main__":
    unittest.main()
