import chess
import chess.engine
import csv
import json

def analyze_puzzles_from_csv(csv_path, stockfish_path="stockfish.exe", max_puzzles=1000, output_json="puzzles.json", depth=15, movetime=0.1):
    results = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_puzzles:
                break
            fen = row["FEN"]
            rating = int(row["Rating"])
            # Stockfish best move
            try:
                board = chess.Board(fen)
                with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
                    result = engine.play(board, chess.engine.Limit(depth=depth, time=movetime))
                    best_move = str(result.move)
            except Exception:
                best_move = None
            results.append({
                "fen": fen,
                "solution": [best_move] if best_move else [],
                "rating": rating
            })
            if (i+1) % 50 == 0:
                print(f"Processed {i+1} puzzles...")
    # Save to JSON
    with open(output_json, "w", encoding="utf-8") as out_f:
        json.dump(results, out_f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} puzzles to {output_json}")

if __name__ == "__main__":
    analyze_puzzles_from_csv("lichess_db_puzzle.csv")
