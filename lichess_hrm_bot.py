import berserk
import chess
import numpy as np
import onnxruntime as ort
import requests
import sys
import time
from hrm_model import fen_to_bitplanes, bin_to_score

# --- Lichess API token ---
# IMPORTANT: Replace this with your Bot account token!
LICHESS_TOKEN = "lip_RHHI4w9SK63nzvyoxpdS"

# --- Session alapú Lichess client ---
session = requests.Session()
session.headers.update({'Authorization': f'Bearer {LICHESS_TOKEN}'})
client = berserk.Client(session=session)

# --- ONNX modell betöltése ---
onnx_path = "hrm_model.onnx"
ort_session = ort.InferenceSession(onnx_path)

# --- Bot kihívás funkciók ---
def challenge_bot(username, time_control="3+2", color="random"):
    """Kihív egy másik botot játékra."""
    try:
        challenge = client.challenges.create(
            username=username,
            rated=True,  # RATED játék!
            clock_limit=int(time_control.split('+')[0]) * 60,
            clock_increment=int(time_control.split('+')[1]),
            color=color,
            variant="standard"
        )
        print(f"✅ Kihívás elküldve: {username} ({time_control}, {color}) - RATED")
        return challenge
    except Exception as e:
        print(f"❌ Hiba a kihívás során: {e}")
        return None

def challenge_multiple_bots(bot_list, time_control="3+2"):
    """Több botot hív ki egymás után."""
    print(f"🎯 {len(bot_list)} bot kihívása ({time_control})...")
    successful_challenges = 0
    
    for i, bot_name in enumerate(bot_list, 1):
        print(f"\n📞 [{i}/{len(bot_list)}] Kihívás: {bot_name}")
        result = challenge_bot(bot_name, time_control, "random")
        if result:
            successful_challenges += 1
        
        if i < len(bot_list):
            print("⏳ Várakozás 2 másodperc...")
            time.sleep(2)
    
    print(f"\n📊 Összesítés: {successful_challenges}/{len(bot_list)} sikeres kihívás")
    return successful_challenges

# --- Lépésválasztó függvény ---
def choose_move(board):
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
    
    next_fens = []
    for move in legal_moves:
        board_copy = board.copy()
        board_copy.push(move)
        next_fens.append(board_copy.fen())
    
    bitplane_np = np.array([fen_to_bitplanes(fen, history_length=8) for fen in next_fens], dtype=np.float32)
    outputs = ort_session.run(None, {"input": bitplane_np})
    
    move_scores = []
    for logits in outputs[0]:
        value_probs = np.exp(logits) / np.sum(np.exp(logits))
        expected_bin = np.sum(value_probs * np.arange(len(value_probs)))
        win_percent = bin_to_score(expected_bin, num_bins=len(value_probs))
        move_scores.append(win_percent)
    
    best_idx = int(np.argmax(move_scores))
    best_move = legal_moves[best_idx]
    best_score = move_scores[best_idx]
    
    print(f"🤔 Választott lépés: {best_move} (win%: {best_score:.2f})")
    return best_move

def handle_game(game_id):
    """Egy játék teljes kezelése."""
    board = chess.Board()
    
    account_info = client.account.get()
    our_username = account_info['id']
    
    game_active = True
    white_player = ""
    black_player = ""
    
    print(f"👤 Botunk neve: {our_username}")
    
    for state in client.bots.stream_game_state(game_id):
        if not game_active:
            break
            
        state_type = state.get('type', 'unknown')
        
        if state_type in ['gameFinish', 'gameEnd']:
            winner = state.get('winner')
            status = state.get('status', 'finished')
            print(f"🏁 Játék vége! Status: {status}")
            if winner:
                print(f"🏆 Győztes: {winner}")
            else:
                print("🤝 Döntetlen")
            game_active = False
            break
        
        if state_type == 'gameFull':
            moves_str = state.get('state', {}).get('moves', '')
            white_player = state.get('white', {}).get('id', '')
            black_player = state.get('black', {}).get('id', '')
            game_status = state.get('state', {}).get('status', 'started')
        elif state_type == 'gameState':
            moves_str = state.get('moves', '')
            game_status = state.get('status', 'started')
        else:
            continue
        
        if game_status in ['mate', 'resign', 'stalemate', 'timeout', 'draw']:
            print(f"🏁 Játék vége! Status: {game_status}")
            game_active = False
            break
        
        moves = moves_str.split() if moves_str else []
        board = chess.Board()
        
        try:
            for move in moves:
                board.push_uci(move)
        except ValueError as e:
            print(f"❌ Hibás lépés: {move}, hiba: {e}")
            continue
        
        if board.is_checkmate():
            winner = "fekete" if board.turn == chess.WHITE else "fehér"
            print(f"🏁 Sakkmatt! Győztes: {winner}")
            game_active = False
            break
        elif board.is_stalemate():
            print("🏁 Patt! Döntetlen")
            game_active = False
            break
        elif board.is_insufficient_material():
            print("🏁 Nem elegendő anyag! Döntetlen")
            game_active = False
            break
        
        our_turn = ((board.turn == chess.WHITE and our_username == white_player) or 
                    (board.turn == chess.BLACK and our_username == black_player))
        
        if our_turn:
            print(f"🤖 Mi lépünk! ({our_username})")
            move = choose_move(board)
            if move:
                print(f"✅ Lépés: {move.uci()}")
                try:
                    client.bots.make_move(game_id, move.uci())
                except Exception as move_error:
                    print(f"❌ Hiba a lépés küldésekor: {move_error}")
            else:
                print("❌ Nincs érvényes lépés!")
        else:
            print(f"⏳ Ellenfél lép... ({white_player} vs {black_player})")
    
    print(f"🔚 Játék befejezve: {game_id}")

def run_bot_mode():
    """Bot mód futtatása - eseményfigyelés és játék kezelés."""
    print("🤖 Bot mód aktív - várakozás eseményekre...")
    
    try:
        for event in client.bots.stream_incoming_events():
            event_type = event.get('type', 'unknown')
            print(f"📨 Esemény: {event_type}")
            
            if event_type == 'challenge':
                challenge_data = event.get('challenge', {})
                challenge_id = challenge_data.get('id')
                challenger = challenge_data.get('challenger', {}).get('id', 'unknown')
                time_control = challenge_data.get('timeControl', {})
                rated = challenge_data.get('rated', False)
                variant = challenge_data.get('variant', {}).get('key', 'standard')
                
                print(f"⚔️ Kihívás érkezett: {challenger} ({challenge_id})")
                print(f"   ⏰ Idő: {time_control}")
                print(f"   🏆 Rated: {rated}, Variant: {variant}")
                
                # Azonnal próbáljuk elfogadni a kihívást
                if variant == 'standard' and challenge_id:
                    try:
                        # Azonnali elfogadás
                        client.bots.accept_challenge(challenge_id)
                        print(f"✅ Kihívás elfogadva: {challenge_id}")
                    except Exception as e:
                        error_msg = str(e)
                        if '404' in error_msg or 'Not found' in error_msg:
                            print(f"⚠️ Kihívás már lejárt vagy visszavonva: {challenge_id}")
                        elif '400' in error_msg:
                            print(f"⚠️ Érvénytelen kihívás: {challenge_id}")
                        else:
                            print(f"❌ Ismeretlen hiba: {e}")
                else:
                    print(f"⏭️ Kihívás átugorgva: variant={variant}, id={challenge_id}")
                    
            elif event_type == 'gameStart':
                game_id = event['game']['id']
                print(f"🎮 Új játék indult: {game_id}")
                
                try:
                    handle_game(game_id)
                except Exception as game_error:
                    print(f"❌ Hiba a játék során ({game_id}): {game_error}")
                    continue
                    
            elif event_type == 'gameFinish':
                game_info = event.get('game', {})
                game_id = game_info.get('id', 'unknown')
                print(f"🔄 Játék {game_id} befejezve - kész új kihívásokra!")
                
            else:
                print(f"⏳ Várakozás további eseményekre...")

    except KeyboardInterrupt:
        print("\n🛑 Bot leállítása...")
    except Exception as e:
        print(f"❌ Váratlan hiba: {e}")
        import traceback
        traceback.print_exc()

# --- Fő program ---
print("🤖 HRM Chess Bot indítása...")
print(f"🔑 Token: {LICHESS_TOKEN[:8]}...")

# Gyors parancssor módok
if len(sys.argv) > 1:
    command = sys.argv[1].lower()
    
    if command == "challenge" and len(sys.argv) > 2:
        bot_name = sys.argv[2]
        time_control = sys.argv[3] if len(sys.argv) > 3 else "3+2"
        print(f"🎯 Gyors kihívás: {bot_name} ({time_control})")
        challenge_bot(bot_name, time_control, "random")
        print("✅ Kihívás elküldve! Most bot mód...")
        
    elif command == "weak":
        weak_bots = ["maia1", "RandomMover", "weak_bot"]
        print("🔥 Gyenge botok kihívása...")
        challenge_multiple_bots(weak_bots, "3+2")
        print("✅ Kihívások elküldve! Most bot mód...")
        
    elif command == "strong":
        strong_bots = ["maia9", "stockfish-bot", "komodo-bot", "leela-bot"]
        print("💪 Erős botok kihívása...")
        challenge_multiple_bots(strong_bots, "5+3")
        print("✅ Kihívások elküldve! Most bot mód...")
        
    elif command == "bullet":
        bullet_bots = ["maia5", "maia7", "RandomMover", "weak_bot", "fairy-stockfish"]
        print("⚡ BULLET CHESS (1+0) - gyors lépések előnye!")
        challenge_multiple_bots(bullet_bots, "1+0")
        print("✅ Bullet kihívások elküldve! Most bot mód...")
        
    elif command == "blitz":
        blitz_bots = ["maia7", "maia8", "fairy-stockfish", "komodo-bot"]
        print("🔥 BLITZ CHESS (3+0) - gyors döntések!")
        challenge_multiple_bots(blitz_bots, "3+0")
        print("✅ Blitz kihívások elküldve! Most bot mód...")
        
    elif command == "bot":
        print("🔄 Direkt bot mód...")
        
    else:
        print("❌ Ismeretlen parancs. Bot mód indítása...")

else:
    # Nincs parancssor argumentum - egyszerű választás
    print("\n" + "="*50)
    print("🤖 HRM Chess Bot")
    print("="*50)
    print("1. 🔥 Gyenge botok kihívása + bot mód")
    print("2. 💪 Erős botok kihívása + bot mód")
    print("3. ⚡ BULLET CHESS (1+0) - gyors lépések!")
    print("4. � BLITZ CHESS (3+0) - gyors döntések!")
    print("5. �🔄 Csak bot mód (várakozás kihívásokra)")
    print("="*50)
    
    choice = input("Válassz (1-5): ").strip()
    
    if choice == "1":
        weak_bots = ["maia1", "RandomMover", "weak_bot"]
        print("🔥 Gyenge botok kihívása...")
        challenge_multiple_bots(weak_bots, "3+2")
        print("✅ Kihívások elküldve! Most bot mód...")
    elif choice == "2":
        strong_bots = ["maia9", "stockfish-bot", "komodo-bot", "leela-bot"]
        print("💪 Erős botok kihívása...")
        challenge_multiple_bots(strong_bots, "5+3")
        print("✅ Kihívások elküldve! Most bot mód...")
    elif choice == "3":
        bullet_bots = ["maia5", "maia7", "RandomMover", "weak_bot", "fairy-stockfish"]
        print("⚡ BULLET CHESS (1+0) - gyors lépések előnye!")
        challenge_multiple_bots(bullet_bots, "1+0")
        print("✅ Bullet kihívások elküldve! Most bot mód...")
    elif choice == "4":
        blitz_bots = ["maia7", "maia8", "fairy-stockfish", "komodo-bot"]
        print("🔥 BLITZ CHESS (3+0) - gyors döntések!")
        challenge_multiple_bots(blitz_bots, "3+0")
        print("✅ Blitz kihívások elküldve! Most bot mód...")
    elif choice == "5":
        print("🔄 Csak bot mód...")
    else:
        print("🔄 Alapértelmezett: bot mód...")

# Most indítjuk el a bot módot
run_bot_mode()
