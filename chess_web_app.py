"""
Flask Chess Web Application
Interaktív sakk játék a betanított HRM modellel ChessBoard.js használatával
"""

from flask import Flask, render_template, request, jsonify, session
import torch
import chess
import chess.pgn
import numpy as np
import json
import os
from datetime import datetime
import uuid
import glob

# Import our trained model
from Chess import fen_to_tensor
from hrm_model import HRMChess

app = Flask(__name__)
app.secret_key = 'hrm_chess_secret_key_2025'  # Change in production

class ChessGameManager:
    def __init__(self, model_path=None):
        """Initialize the chess game manager with trained HRM model"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # If no model path provided, let user select
        if model_path is None:
            model_path = self._select_model()
        
        self.model_path = model_path
        
        # Auto-detect and load model
        self.load_model()
        
    def _select_model(self):
        """Modell kiválasztása a felhasználó által (web app verzió)"""
        print("🔍 Searching for available HRM chess models...")
        
        # Keresés .pt fájlokra
        model_files = glob.glob("*.pt")
        
        # Szűrés model fájlokra
        available_models = []
        for file in model_files:
            # Kihagyjuk a data/dataset fájlokat, de megtartjuk a *_model.pt fájlokat
            if any(skip in file.lower() for skip in ['data', 'dataset']) and not file.lower().endswith('_model.pt'):
                continue
            # Extra szűrés: kihagyjuk a train/test fájlokat kivéve ha model fájlok
            if any(skip in file.lower() for skip in ['train', 'test']) and not any(keep in file.lower() for keep in ['model', 'checkpoint']):
                continue
            
            try:
                # Load and check if it's a valid model file
                checkpoint = torch.load(file, map_location='cpu', weights_only=False)
                hidden_dim, N, T = self._detect_model_parameters(checkpoint)
                file_size = os.path.getsize(file) / (1024 * 1024)
                available_models.append({
                    'file': file,
                    'hidden_dim': hidden_dim,
                    'N': N,
                    'T': T,
                    'size_mb': file_size
                })
            except Exception as e:
                print(f"⚠️ Could not load {file}: {e}")
                continue
        
        if not available_models:
            print("❌ No HRM models found!")
            print("🔧 Please train a model first using Chess.py")
            print("🔄 Using default untrained model for testing...")
            return None
        
        # Listázzuk az elérhető modelleket
        print(f"\n📋 Available HRM models ({len(available_models)} found):")
        print("-" * 80)
        for i, model in enumerate(available_models):
            print(f"  {i+1}. {model['file']:<25} | Hidden: {model['hidden_dim']:<3} | N: {model['N']} T: {model['T']} | {model['size_mb']:.1f} MB")
        print("-" * 80)
        
        # Automatikus választás web app esetén - első elérhető modell
        if len(available_models) == 1:
            selected = available_models[0]
            print(f"✅ Auto-selected: {selected['file']} (only model available)")
            return selected['file']
        
        # Felhasználói választás
        while True:
            try:
                choice = input(f"\n🎯 Choose model (1-{len(available_models)}) or press Enter for first: ").strip()
                
                if choice == "":
                    choice_idx = 0
                else:
                    choice_idx = int(choice) - 1
                
                if 0 <= choice_idx < len(available_models):
                    selected = available_models[choice_idx]
                    print(f"✅ Selected: {selected['file']} (Hidden: {selected['hidden_dim']}, N: {selected['N']}, T: {selected['T']})")
                    return selected['file']
                else:
                    print(f"❌ Invalid choice. Please enter 1-{len(available_models)}")
            except (ValueError, KeyboardInterrupt):
                print(f"❌ Invalid input. Please enter 1-{len(available_models)}")
                # Auto-select first model on error for web app
                selected = available_models[0]
                print(f"🔄 Auto-selecting first model: {selected['file']}")
                return selected['file']
    
    def _detect_model_parameters(self, checkpoint):
        """HRM modell paramétereinek detektálása"""
        hidden_dim = None
        N, T = 8, 8  # Default values
        
        # First check if hyperparams are saved in checkpoint
        if 'hyperparams' in checkpoint:
            hyperparams = checkpoint['hyperparams']
            hidden_dim = hyperparams.get('hidden_dim', None)
            N = hyperparams.get('N', N)
            T = hyperparams.get('T', T)
            return hidden_dim, N, T
        
        # Get the actual model state dict (handle both new and legacy formats)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Fallback: detect from model layers
        # Detect hidden_dim from convolutional layers
        if 'board_conv.0.weight' in state_dict:
            # First conv layer: 1 -> hidden_dim//4
            first_conv_out = state_dict['board_conv.0.weight'].shape[0]
            hidden_dim = first_conv_out * 4
        elif 'board_conv.2.weight' in state_dict:
            # Second conv layer: hidden_dim//4 -> hidden_dim//2
            second_conv_out = state_dict['board_conv.2.weight'].shape[0]
            hidden_dim = second_conv_out * 2
        elif 'feature_combiner.0.weight' in state_dict:
            # Feature combiner: combined_features -> hidden_dim
            hidden_dim = state_dict['feature_combiner.0.weight'].shape[0]
        elif 'L_net.0.weight' in state_dict:
            # L_net input: hidden_dim * 3
            l_net_input_size = state_dict['L_net.0.weight'].shape[1]
            if l_net_input_size % 3 == 0:
                hidden_dim = l_net_input_size // 3
        
        if hidden_dim is None:
            hidden_dim = 192  # Default for web app
        
        return hidden_dim, N, T
        
    def load_model(self):
        """Load the trained HRM model with auto-detection"""
        print(f"🎯 Loading HRM Chess model...")
        
        if not self.model_path:
            print("❌ No trained HRM model found!")
            print("🔧 Please train the HRM model first using Chess.py")
            # Create a default HRM model for testing
            self.model = HRMChess(input_dim=72, hidden_dim=192, N=8, T=8).to(self.device)
            self.model_info = "HRM-Untrained-192"
        else:
            try:
                # Auto-detect HRM model parameters from enhanced checkpoint
                checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
                
                # Default values
                hidden_dim = 192
                N, T = 8, 8
                
                # First check if hyperparams are saved in checkpoint (new format)
                if isinstance(checkpoint, dict) and 'hyperparams' in checkpoint:
                    hyperparams = checkpoint['hyperparams']
                    hidden_dim = hyperparams.get('hidden_dim', hidden_dim)
                    N = hyperparams.get('N', N)
                    T = hyperparams.get('T', T)
                    print(f"🔍 Found saved hyperparams: hidden_dim={hidden_dim}, N={N}, T={T}")
                    
                    # Extract the actual model state dict
                    if 'model_state_dict' in checkpoint:
                        model_state_dict = checkpoint['model_state_dict']
                    else:
                        # Legacy format with hyperparams but direct state_dict
                        model_state_dict = {k: v for k, v in checkpoint.items() if k not in ['hyperparams', 'training_info']}
                else:
                    # Fallback: detect from model layers
                    model_state_dict = checkpoint
                    
                    # Try to detect from board_conv convolutional layers (new architecture)
                    if 'board_conv.0.weight' in model_state_dict:
                        # First conv layer: 1 -> hidden_dim//4
                        first_conv_out = model_state_dict['board_conv.0.weight'].shape[0]
                        hidden_dim = first_conv_out * 4
                        print(f"🔍 Auto-detected convolutional HRM hidden_dim: {hidden_dim} (from board_conv.0)")
                    elif 'board_conv.2.weight' in model_state_dict:
                        # Second conv layer: hidden_dim//4 -> hidden_dim//2
                        second_conv_out = model_state_dict['board_conv.2.weight'].shape[0]
                        hidden_dim = second_conv_out * 2
                        print(f"🔍 Auto-detected convolutional HRM hidden_dim: {hidden_dim} (from board_conv.2)")
                    elif 'feature_combiner.0.weight' in model_state_dict:
                        # Feature combiner output: -> hidden_dim
                        hidden_dim = model_state_dict['feature_combiner.0.weight'].shape[0]
                        print(f"🔍 Auto-detected convolutional HRM hidden_dim: {hidden_dim} (from feature_combiner)")
                    elif 'L_net.0.weight' in model_state_dict:
                        # L_net input: hidden_dim * 3
                        l_net_input_size = model_state_dict['L_net.0.weight'].shape[1]
                        if l_net_input_size % 3 == 0:
                            hidden_dim = l_net_input_size // 3
                            print(f"🔍 Auto-detected HRM hidden_dim: {hidden_dim} (from L_net)")
                
                # Create HRM model with detected parameters
                self.model = HRMChess(input_dim=72, hidden_dim=hidden_dim, N=N, T=T).to(self.device)
                self.model.load_state_dict(model_state_dict)
                
                # Determine architecture type
                if 'board_conv.0.weight' in model_state_dict:
                    self.model_info = f"Convolutional-HRM-Trained-{hidden_dim}-N{N}-T{T}"
                    print(f"✅ Convolutional HRM model loaded: {self.model_path}")
                else:
                    self.model_info = f"Linear-HRM-Trained-{hidden_dim}-N{N}-T{T}"
                    print(f"✅ Linear HRM model loaded: {self.model_path}")
                
                total_params = sum(p.numel() for p in self.model.parameters())
                print(f"📊 Model: {self.model_info} ({total_params:,} parameters)")
                
                # Show training info if available
                if isinstance(checkpoint, dict) and 'training_info' in checkpoint:
                    info = checkpoint['training_info']
                    val_loss = info.get('val_loss', 'N/A')
                    if isinstance(val_loss, (int, float)):
                        print(f"📈 Training info: Epoch {info.get('epoch', 'N/A')}, Val Loss: {val_loss:.4f}")
                    else:
                        print(f"📈 Training info: Epoch {info.get('epoch', 'N/A')}, Val Loss: {val_loss}")
                
            except Exception as e:
                print(f"❌ Error loading HRM model: {e}")
                print(f"📋 Checkpoint keys: {list(checkpoint.keys()) if 'checkpoint' in locals() else 'N/A'}")
                self.model = HRMChess(input_dim=72, hidden_dim=192, N=8, T=8).to(self.device)
                self.model_info = "HRM-Error-192"
        
        self.model.eval()
        
    def get_model_move(self, board):
        """Get the best move from the HRM model"""
        try:
            # Convert board to tensor using simplified representation
            state = fen_to_tensor(board.fen())
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # HRM model prediction
                model_output = self.model(state_tensor, return_value=True)
                
                if isinstance(model_output, tuple):
                    # model: move_logits
                    move_logits, values = model_output
                    position_value = values.item()
                else:
                    # Legacy Policy-only model
                    move_logits = model_output
                    position_value = None
                
                policy_probs = torch.softmax(move_logits.view(-1), dim=0)  # Flatten to (4096,)
            
            # Get legal moves and their probabilities
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return None, 0.0, 0.0
            
            legal_probs = []
            for move in legal_moves:
                idx = move.from_square * 64 + move.to_square
                legal_probs.append(policy_probs[idx].item())
            
            # Choose best move with temperature-based selection for stronger play
            legal_probs = np.array(legal_probs)
            
            # SIMPLIFIED: Just take the best move (highest probability)
            best_idx = np.argmax(legal_probs)  # Egyszerűen a legjobb lépés
            
            best_move = legal_moves[best_idx]
            confidence = legal_probs[best_idx]
            
            # Use actual position value if available from Policy+Value model
            if position_value is not None:
                # Policy+Value model provides actual position evaluation
                final_value = position_value
            else:
                # Legacy Policy-only model: calculate pseudo-value based on move confidence
                # Higher confidence moves get higher pseudo-values
                final_value = (confidence - 0.5) * 2.0  # Scale to [-1, 1] range
            
            return best_move, final_value, confidence
            
        except Exception as e:
            print(f"HRM model error: {e}")
            # Fallback to random move
            legal_moves = list(board.legal_moves)
            if legal_moves:
                return np.random.choice(legal_moves), 0.0, 0.0
            return None, 0.0, 0.0

# Global game manager instance
_game_manager = None

def get_game_manager():
    """Get or create game manager instance"""
    global _game_manager
    if _game_manager is None:
        print("🔧 Initializing game manager...")
        _game_manager = ChessGameManager()
    return _game_manager

@app.route('/')
def index():
    """Main chess game page"""
    manager = get_game_manager()
    return render_template('chess_game.html', model_info=manager.model_info)

@app.route('/new_game', methods=['POST'])
def new_game():
    """Start a new chess game"""
    manager = get_game_manager()
        
    # Clear previous sessions to prevent cookie overflow
    session.clear()
    
    # Use simple numeric game ID instead of UUID
    game_id = "game1"
    
    # Get player color preference
    data = request.get_json()
    player_color = data.get('color', 'white')  # white or black
    
    # Initialize game state
    board = chess.Board()
    
    session[game_id] = {
        'board': board.fen(),
        'color': player_color,
        'moves': 0,
        'status': 'active'
    }
    
    response = {
        'game_id': game_id,
        'board': board.fen(),
        'player_color': player_color,
        'status': 'active'
    }
    
    # If player chose black, make model's first move
    if player_color == 'black':
        model_move, value, confidence = manager.get_model_move(board)
        if model_move:
            board.push(model_move)
            session[game_id]['board'] = board.fen()
            session[game_id]['moves'] = 1
            
            response['model_move'] = str(model_move)
            response['board'] = board.fen()
            response['model_value'] = value
            response['model_confidence'] = confidence
    
    return jsonify(response)

@app.route('/make_move', methods=['POST'])
def make_move():
    """Handle player move and get model response"""
    data = request.get_json()
    game_id = data.get('game_id')
    player_move = data.get('move')
    
    if game_id not in session:
        return jsonify({'error': 'Game not found'}), 404
    
    game_state = session[game_id]
    board = chess.Board(game_state['board'])
    
    try:
        # Validate and make player move
        move = chess.Move.from_uci(player_move)
        if move not in board.legal_moves:
            return jsonify({'error': 'Illegal move'}), 400
        
        board.push(move)
        
        # Update move count
        game_state['moves'] += 1
        
        # Check if game is over
        if board.is_game_over():
            game_state['status'] = 'finished'
            session[game_id] = game_state
            
            return jsonify({
                'board': board.fen(),
                'game_over': True,
                'result': board.result(),
                'winner': get_winner(board.result(), game_state['color'])
            })
        
        # Get model's response move
        model_move, value, confidence = get_game_manager().get_model_move(board)
        
        if model_move is None:
            game_state['status'] = 'finished'
            session[game_id] = game_state
            return jsonify({
                'board': board.fen(),
                'game_over': True,
                'result': 'Model error',
                'winner': 'Player'
            })
        
        # Make model move
        board.push(model_move)
        
        # Update move count
        game_state['moves'] += 1
        
        # Update game state
        game_state['board'] = board.fen()
        
        # Check if game is over after model move
        if board.is_game_over():
            game_state['status'] = 'finished'
            session[game_id] = game_state
            
            return jsonify({
                'board': board.fen(),
                'model_move': str(model_move),
                'model_value': value,
                'model_confidence': confidence,
                'game_over': True,
                'result': board.result(),
                'winner': get_winner(board.result(), game_state['color'])
            })
        
        session[game_id] = game_state
        
        return jsonify({
            'board': board.fen(),
            'model_move': str(model_move),
            'model_value': value,
            'model_confidence': confidence,
            'game_over': False
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/game_status/<game_id>')
def game_status(game_id):
    """Get current game status"""
    if game_id not in session:
        return jsonify({'error': 'Game not found'}), 404
    
    game_state = session[game_id]
    board = chess.Board(game_state['board'])
    
    return jsonify({
        'game_id': game_id,
        'board': game_state['board'],
        'player_color': game_state['color'],
        'status': game_state['status'],
        'moves_count': game_state['moves'],
        'turn': 'white' if board.turn else 'black',
        'in_check': board.is_check(),
        'legal_moves': [str(move) for move in board.legal_moves]
    })

@app.route('/game_analysis/<game_id>')
def game_analysis(game_id):
    """Get simplified game analysis (limited data due to session size constraints)"""
    if game_id not in session:
        return jsonify({'error': 'Game not found'}), 404
    
    game_state = session[game_id]
    board = chess.Board(game_state['board'])
    
    # Simplified analysis since we don't store detailed move history
    analysis = {
        'total_moves': game_state['moves'],
        'current_position': game_state['board'],
        'game_status': game_state['status'],
        'player_color': game_state['color'],
        'current_turn': 'white' if board.turn else 'black',
        'material_balance': calculate_material_balance(board),
        'pieces_count': len(board.piece_map())
    }
    
    return jsonify(analysis)

def get_winner(result, player_color):
    """Determine the winner based on game result and player color"""
    if result == '1-0':  # White wins
        return 'Player' if player_color == 'white' else 'Model'
    elif result == '0-1':  # Black wins
        return 'Player' if player_color == 'black' else 'Model'
    else:  # Draw
        return 'Draw'

def calculate_material_balance(board):
    """Calculate material balance for simple analysis"""
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    
    white_material = 0
    black_material = 0
    
    for square, piece in board.piece_map().items():
        value = piece_values.get(piece.piece_type, 0)
        if piece.color == chess.WHITE:
            white_material += value
        else:
            black_material += value
    
    return white_material - black_material

@app.route('/model_info')
def model_info():
    """Get information about the loaded HRM model"""
    manager = get_game_manager()
    return jsonify({
        'model_type': manager.model_info,
        'architecture': 'HRM (72-dim input, Policy+Value compatible)',
        'input_dimension': 72,
        'hidden_dimension': manager.model.hidden_dim,
        'hrm_parameters': f"N={manager.model.N}, T={manager.model.T}",
        'hrm_steps': manager.model.N * manager.model.T,
        'device': str(manager.device),
        'parameters': sum(p.numel() for p in manager.model.parameters()),
        'model_file': manager.model_path,
        'features': [
            'Ultra-compact 72-dimensional input',
            f'Hierarchical reasoning (N={manager.model.N}, T={manager.model.T})',
            'Policy+Value prediction (if supported)',
            'Sparse policy representation',
            'Optimized for fast inference',
            '2D Convolutional layers for spatial pattern recognition' if 'Convolutional' in manager.model_info else 'Linear input processing'
        ]
    })

@app.route('/available_models')
def available_models():
    """Get list of available models"""
    manager = get_game_manager()
    model_files = glob.glob("*.pt")
    available_models = []
    
    for file in model_files:
        # Skip data/dataset files unless they are *_model.pt
        if any(skip in file.lower() for skip in ['data', 'dataset']) and not file.lower().endswith('_model.pt'):
            continue
        # Skip train/test files unless they contain 'model' or 'checkpoint'
        if any(skip in file.lower() for skip in ['train', 'test']) and not any(keep in file.lower() for keep in ['model', 'checkpoint']):
            continue
        
        try:
            checkpoint = torch.load(file, map_location='cpu', weights_only=False)
            hidden_dim, N, T = manager._detect_model_parameters(checkpoint)
            file_size = os.path.getsize(file) / (1024 * 1024)
            
            # Determine architecture type
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            arch_type = "Convolutional" if 'board_conv.0.weight' in state_dict else "Linear"
            
            available_models.append({
                'file': file,
                'hidden_dim': hidden_dim,
                'N': N,
                'T': T,
                'size_mb': file_size,
                'architecture': arch_type,
                'current': file == manager.model_path
            })
        except Exception as e:
            print(f"⚠️ Could not load {file}: {e}")
            continue
    
    return jsonify({
        'models': available_models,
        'current_model': manager.model_path
    })

@app.route('/solve_fen', methods=['POST'])
def solve_fen():
    """Solve a chess puzzle from FEN input using the loaded HRM model"""
    data = request.get_json()
    fen = data.get('fen', None)
    if not fen:
        return jsonify({'error': 'No FEN provided'}), 400
    try:
        board = chess.Board(fen)
    except Exception as e:
        return jsonify({'error': f'Invalid FEN: {e}'}), 400

    manager = get_game_manager()
    move, value, confidence = manager.get_model_move(board)
    if move is None:
        return jsonify({'error': 'No legal move found'}), 400
    return jsonify({
        'best_move': str(move),
        'value': value,
        'confidence': confidence
    })

@app.route('/switch_model', methods=['POST'])
def switch_model():
    """Switch to a different model"""
    data = request.get_json()
    new_model_path = data.get('model_path')
    
    if not new_model_path or not os.path.exists(new_model_path):
        return jsonify({'error': 'Model file not found'}), 400
    
    try:
        # Create new game manager with the selected model
        global _game_manager
        old_model_info = _game_manager.model_info if _game_manager else "None"
        _game_manager = ChessGameManager(model_path=new_model_path)
        
        # Clear all sessions since model changed
        session.clear()
        
        return jsonify({
            'success': True,
            'old_model': old_model_info,
            'new_model': _game_manager.model_info,
            'message': f'Successfully switched to {_game_manager.model_info}'
        })
    except Exception as e:
        return jsonify({'error': f'Failed to load model: {str(e)}'}), 500

if __name__ == '__main__':
    # Initialize game manager
    
    print("🚀 Starting HRM Chess Web Application...")
    manager = get_game_manager()  # This will initialize it once
    
    print(f"📱 Model loaded: {manager.model_info}")
    
    # Determine architecture type from model_info
    if "Convolutional" in manager.model_info:
        print(f"🧠 Architecture: 2D Convolutional HRM (8x8 conv + meta features)")
    else:
        print(f"🧠 Architecture: Linear HRM (72-dim input → move prediction)")
    
    print(f"⚡ Device: {manager.device}")
    print(f"🔄 HRM Parameters: N={manager.model.N}, T={manager.model.T} ({manager.model.N * manager.model.T} reasoning steps)")
    
    total_params = sum(p.numel() for p in manager.model.parameters())
    print(f"📊 Model parameters: {total_params:,}")
    
    print("🌐 Open browser: http://localhost:5000")
    print("🎮 Ready to play chess against HRM!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
