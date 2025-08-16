import unittest
from Chess import ParallelStockfishDatasetGenerator

class TestConstructorParams(unittest.TestCase):
    def test_params(self):
        gen = ParallelStockfishDatasetGenerator(num_workers=2, movetime=42)
        self.assertEqual(gen.num_workers, 2)
        self.assertEqual(gen.movetime, 42)

if __name__ == "__main__":
    unittest.main()
