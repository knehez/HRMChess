import unittest
import torch
import os
from Chess import ParallelStockfishDatasetGenerator

class TestBinarySaveLoad(unittest.TestCase):
    def test_binary_save_and_load(self):
        generator = ParallelStockfishDatasetGenerator(num_workers=1, movetime=30)
        dataset, _ = generator.generate_dataset_parallel(num_games=1, max_moves=4, randomness=0.1)
        # Save as list of dicts
        torch.save(dataset, "test/test_binary.pt")
        loaded = torch.load("test/test_binary.pt")
        self.assertEqual(dataset, loaded, "Loaded positions do not match saved positions!")
        os.remove("test/test_binary.pt")

if __name__ == "__main__":
    unittest.main()
