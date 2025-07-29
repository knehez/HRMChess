import chess
import subprocess
import os
import time

class StockfishEvaluator:
    """Stockfish motor integráció pozíció értékeléshez - EGYSZERŰ READLINE MEGKÖZELÍTÉS"""
    def __init__(self, stockfish_path="./stockfish.exe", movetime=50):
        self.stockfish_path = stockfish_path
        self.movetime = movetime
        self.process = None
        self.initialized = False
        self._init_engine()

    def _init_engine(self):
        try:
            if os.path.exists(self.stockfish_path):
                print(f"🤖 Stockfish found: {self.stockfish_path}")
            else:
                print(f"❌ Stockfish not found at: {self.stockfish_path}")
                print("🔍 Looking for stockfish in system PATH...")
                self.stockfish_path = "stockfish"
            print("🚀 Starting Stockfish engine...")
            self.process = subprocess.Popen(
                [self.stockfish_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                universal_newlines=True
            )
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
        if self.process and self.process.stdin:
            try:
                self.process.stdin.write(command + "\n")
                self.process.stdin.flush()
            except Exception as e:
                print(f"⚠️ Error sending command: {e}")
                self.initialized = False

    def _read_line(self, timeout=3.0):
        if not self.process or not self.process.stdout:
            return None
        try:
            import threading
            import queue
            result_queue = queue.Queue()
            def read_line():
                try:
                    line = self.process.stdout.readline()
                    result_queue.put(line)
                except Exception as e:
                    result_queue.put(None)
            thread = threading.Thread(target=read_line)
            thread.daemon = True
            thread.start()
            thread.join(timeout=timeout)
            if not result_queue.empty():
                line = result_queue.get_nowait()
                if line:
                    return line.strip()
            return None
        except Exception as e:
            return None

    def _wait_for_response(self, expected, timeout=5.0):
        start_time = time.time()
        while time.time() - start_time < timeout:
            line = self._read_line(1.0)
            if line and expected in line:
                return True
            if not line:
                continue
        return False

    def get_top_k_moves(self, fen, k=3):
        if not self.initialized or not self.process:
            print("⚠️ Stockfish engine not initialized")
            return []
        try:
            self._send_command("ucinewgame")
            self._send_command(f"setoption name MultiPV value {k}")
            self._send_command(f"position fen {fen}")
            self._send_command(f"go movetime {self.movetime}")
            top_moves = {}
            start_time = time.time()
            while time.time() - start_time < (self.movetime / 1000) + 2:
                line = self._read_line(timeout=1.0)
                if not line:
                    continue
                if line.startswith('bestmove'):
                    break
                if 'multipv' in line and ('score cp' in line or 'score mate' in line):
                    parts = line.split()
                    try:
                        pv_index = parts.index('multipv') + 1
                        move_index = parts.index('pv') + 1
                        rank = int(parts[pv_index])
                        move_uci = parts[move_index]
                        score = 0.0
                        if 'score cp' in line:
                            score_index = parts.index('cp') + 1
                            cp_score = int(parts[score_index])
                            score = max(-1.0, min(1.0, cp_score / 300.0))
                        elif 'score mate' in line:
                            mate_index = parts.index('mate') + 1
                            mate_in_moves = int(parts[mate_index])
                            score = 1.0 if mate_in_moves > 0 else -1.0
                        if rank <= k:
                            top_moves[rank] = (move_uci, score)
                    except (ValueError, IndexError):
                        continue
            self._send_command("setoption name MultiPV value 1")
            return [top_moves[i] for i in sorted(top_moves.keys())]
        except Exception as e:
            print(f"⚠️ Error getting top k moves: {e}")
            self._send_command("setoption name MultiPV value 1")
            return []

    def close(self):
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
        self.close()

class ParallelStockfishEvaluator:
    """Parallel Stockfish evaluator using multiple processes for faster evaluation"""
    def __init__(self, stockfish_path="./stockfish.exe", movetime=50, num_evaluators=2):
        self.stockfish_path = stockfish_path
        self.movetime = movetime
        self.num_evaluators = num_evaluators
        self.evaluators = []
        self.initialized = False
        print(f"🚀 Initializing {num_evaluators} parallel Stockfish evaluators...")
        self._init_parallel_engines()

    def _init_parallel_engines(self):
        import threading
        self.evaluators = []
        init_threads = []
        init_results = []
        def init_single_evaluator(evaluator_id):
            try:
                evaluator = StockfishEvaluator(
                    stockfish_path=self.stockfish_path,
                    movetime=self.movetime
                )
                evaluator.evaluator_id = evaluator_id
                init_results.append((evaluator_id, evaluator, True))
                print(f"✅ Evaluator {evaluator_id} initialized successfully")
            except Exception as e:
                print(f"❌ Failed to initialize evaluator {evaluator_id}: {e}")
                init_results.append((evaluator_id, None, False))
        for i in range(self.num_evaluators):
            thread = threading.Thread(target=init_single_evaluator, args=(i,))
            init_threads.append(thread)
            thread.start()
        for thread in init_threads:
            thread.join()
        init_results.sort(key=lambda x: x[0])
        for evaluator_id, evaluator, success in init_results:
            if success and evaluator:
                self.evaluators.append(evaluator)
        if len(self.evaluators) > 0:
            self.initialized = True
            print(f"🎯 Successfully initialized {len(self.evaluators)}/{self.num_evaluators} parallel evaluators")
        else:
            print("❌ Failed to initialize any parallel evaluators")
            self.initialized = False

    def evaluate_positions_parallel(self, fens):
        if not self.initialized or len(self.evaluators) == 0:
            print("⚠️ Parallel evaluators not initialized, exit")
            exit(0)
        import threading
        import queue
        import time
        work_queue = queue.Queue()
        results = {}
        results_lock = threading.Lock()
        start_time_global = time.time()
        for i, fen in enumerate(fens):
            work_queue.put((i, fen))
        def worker_thread(evaluator: StockfishEvaluator, thread_id):
            processed_count = 0
            while True:
                try:
                    position_idx, fen = work_queue.get(timeout=1.0)
                    start_time = time.time()
                    top_moves = evaluator.get_top_k_moves(fen, k=1)
                    move_evaluations = []
                    board = chess.Board(fen)
                    for move_uci, score_cp in top_moves:
                        try:
                            move = chess.Move.from_uci(move_uci)
                            move_evaluations.append(((move.from_square, move.to_square), score_cp))
                        except:
                            continue
                    eval_time = time.time() - start_time
                    with results_lock:
                        results[position_idx] = move_evaluations
                    processed_count += 1
                    if thread_id == 0 and processed_count % 5 == 0:
                        with results_lock:
                            total_completed = len(results)
                        elapsed = time.time() - start_time_global
                        rate = total_completed / elapsed if elapsed > 0 else 0
                        remaining = len(fens) - total_completed
                        eta = remaining / rate / 60 if rate > 0 else float('inf')
                        percent = (total_completed / len(fens)) * 100
                        print(f"🔄 Thread {thread_id}: {total_completed}/{len(fens)} positions "
                              f"({percent:.1f}%) processed | Rate: {rate:.2f} pos/s | ETA: {eta:.1f} min")
                    work_queue.task_done()
                except queue.Empty:
                    break
                except Exception as e:
                    print(f"⚠️ Worker thread {thread_id} error: {e}")
                    work_queue.task_done()
                    continue
        threads = []
        for i, evaluator in enumerate(self.evaluators):
            thread = threading.Thread(target=worker_thread, args=(evaluator, i))
            thread.daemon = True
            threads.append(thread)
            thread.start()
        print(f"⚡ Processing {len(fens)} positions with {len(self.evaluators)} parallel evaluators...")
        start_time = time.time()
        work_queue.join()
        for thread in threads:
            thread.join(timeout=5.0)
        elapsed_time = time.time() - start_time
        ordered_results = []
        for i in range(len(fens)):
            if i in results:
                ordered_results.append(results[i])
            else:
                print(f"⚠️ Missing result for position {i}, using empty evaluation")
                ordered_results.append([])
        total_moves = sum(len(moves) for moves in ordered_results)
        avg_moves_per_pos = total_moves / len(ordered_results) if ordered_results else 0
        positions_per_sec = len(fens) / elapsed_time if elapsed_time > 0 else 0
        print(f"✅ Parallel evaluation completed in {elapsed_time:.1f}s")
        print(f"   📊 Positions: {len(ordered_results):,}")
        print(f"   📊 Total moves: {total_moves:,}")
        print(f"   📊 Avg moves/position: {avg_moves_per_pos:.1f}")
        print(f"   ⚡ Rate: {positions_per_sec:.2f} positions/second")
        print(f"   🚀 Speedup: ~{len(self.evaluators):.1f}x theoretical")
        return ordered_results

    def close(self):
        print(f"🔌 Closing {len(self.evaluators)} parallel evaluators...")
        for i, evaluator in enumerate(self.evaluators):
            try:
                evaluator.close()
                print(f"✅ Evaluator {i} closed")
            except Exception as e:
                print(f"⚠️ Error closing evaluator {i}: {e}")
        self.evaluators = []
        self.initialized = False
        print("🔌 All parallel evaluators closed")

    def __del__(self):
        self.close()
