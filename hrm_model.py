"""
HRM Chess Model - Hierarchical Reasoning Machine for Chess
Konvolúciós HRM implementáció with Automatic Mixed Precision (float16)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
import chess
import numpy as np
import wandb

# --- FEN to bitplane conversion ---
def game_to_bitplanes(move_history_compact: dict, history_length: int = 8) -> np.ndarray:
    """
    AlphaZero-style game history to bitplane conversion: returns [129, 8, 8] numpy array.
    Memory-efficient version that reconstructs board states from starting FEN + UCI moves.
    
    0-55: White pieces (6 types x 8 ply history)
    56-111: Black pieces (6 types x 8 ply history)
    112: Side to move (1 if white, 0 if black)
    113-116: Castling rights (KQkq)
    117: En passant square
    118: Move count (normalized)
    119-122: King safety indicators (4 directions for each king)
    123-124: Material balance (normalized for white/black)
    125: Check indicator
    126: Pins and skewers
    127-128: Attack/defense maps
    
    Args:
        move_history_compact: Dict with keys:
            - 'starting_fen': FEN string of the starting position
            - 'moves': List of UCI move strings, last move is Stockfish's best move
            - 'score': Float score (0.0-1.0) for the last move from Stockfish
        history_length: Number of ply to include in history (default 8)
    """
    planes = np.zeros((129, 8, 8), dtype=np.float32)
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    
    # Reconstruct board positions from starting FEN + moves
    starting_fen = move_history_compact.get('starting_fen', chess.STARTING_FEN)
    moves = move_history_compact.get('moves', [])
    
    # Create board from starting position
    board = chess.Board(starting_fen)
    
    # Build history by applying moves step by step
    history_positions = [board.copy()]  # Include starting position
    
    for move_uci in moves:
        try:
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                board.push(move)
                history_positions.append(board.copy())
            else:
                print(f"⚠️ Illegal move {move_uci} in history, stopping reconstruction at ply {len(history_positions)}")
                break
        except (ValueError, chess.InvalidMoveError):
            print(f"⚠️ Invalid UCI move {move_uci}, stopping reconstruction at ply {len(history_positions)}")
            break
    
    # Get current board (final position)
    current_board = history_positions[-1] if history_positions else chess.Board()
    
    # Take the last history_length positions
    if len(history_positions) > history_length:
        history_positions = history_positions[-history_length:]
    
    # If we have fewer positions than history_length, pad with empty boards
    while len(history_positions) < history_length:
        history_positions.insert(0, chess.Board("8/8/8/8/8/8/8/8 w - - 0 1"))  # Empty board
    
    # Encode pieces for each ply in history
    for ply_idx, board_state in enumerate(history_positions):
        # White pieces (0-55: 6 types x 8 ply)
        for piece_idx, piece_type in enumerate(piece_types):
            for square in board_state.pieces(piece_type, chess.WHITE):
                row, col = divmod(square, 8)
                plane_idx = ply_idx * 6 + piece_idx
                planes[plane_idx, row, col] = 1
        
        # Black pieces (56-111: 6 types x 8 ply)
        for piece_idx, piece_type in enumerate(piece_types):
            for square in board_state.pieces(piece_type, chess.BLACK):
                row, col = divmod(square, 8)
                plane_idx = 56 + ply_idx * 6 + piece_idx
                planes[plane_idx, row, col] = 1
    
    # Current position metadata (112-118)
    # 112: side to move
    planes[112, :, :] = int(current_board.turn)
    
    # 113-116: castling rights
    planes[113, :, :] = int(current_board.has_kingside_castling_rights(chess.WHITE))
    planes[114, :, :] = int(current_board.has_queenside_castling_rights(chess.WHITE))
    planes[115, :, :] = int(current_board.has_kingside_castling_rights(chess.BLACK))
    planes[116, :, :] = int(current_board.has_queenside_castling_rights(chess.BLACK))
    
    # 117: en passant
    if current_board.ep_square is not None:
        row, col = divmod(current_board.ep_square, 8)
        planes[117, row, col] = 1
    
    # 118: move count (normalized)
    planes[118, :, :] = current_board.fullmove_number / 100.0
    
    # Additional evaluation bitplanes (119-128)
    
    # 119-122: King safety indicators (4 directions for each king)
    white_king_square = current_board.king(chess.WHITE)
    black_king_square = current_board.king(chess.BLACK)
    
    if white_king_square is not None:
        king_row, king_col = divmod(white_king_square, 8)
        # Check king safety in 4 directions (N, E, S, W)
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        for dir_idx, (dr, dc) in enumerate(directions):
            safety_score = 0
            for distance in range(1, 4):  # Check 3 squares in each direction
                r, c = king_row + dr * distance, king_col + dc * distance
                if 0 <= r < 8 and 0 <= c < 8:
                    square = chess.square(c, r)
                    piece = current_board.piece_at(square)
                    if piece is not None and piece.color == chess.WHITE:
                        safety_score += 1  # Own pieces provide safety
                    elif piece is not None and piece.color == chess.BLACK:
                        safety_score -= 1  # Enemy pieces reduce safety
                else:
                    break
            # Normalize safety score and set for white king
            normalized_safety = max(0, min(1, (safety_score + 3) / 6))
            planes[119 + dir_idx, king_row, king_col] = normalized_safety
    
    if black_king_square is not None:
        king_row, king_col = divmod(black_king_square, 8)
        # Same calculation for black king, but inverted
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        for dir_idx, (dr, dc) in enumerate(directions):
            safety_score = 0
            for distance in range(1, 4):
                r, c = king_row + dr * distance, king_col + dc * distance
                if 0 <= r < 8 and 0 <= c < 8:
                    square = chess.square(c, r)
                    piece = current_board.piece_at(square)
                    if piece is not None and piece.color == chess.BLACK:
                        safety_score += 1
                    elif piece is not None and piece.color == chess.WHITE:
                        safety_score -= 1
                else:
                    break
            normalized_safety = max(0, min(1, (safety_score + 3) / 6))
            planes[119 + dir_idx, king_row, king_col] = normalized_safety
    
    # 123-124: Material balance (normalized for white/black)
    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
                   chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
    white_material = sum(piece_values[piece.piece_type] 
                        for piece in current_board.piece_map().values() 
                        if piece.color == chess.WHITE)
    black_material = sum(piece_values[piece.piece_type] 
                        for piece in current_board.piece_map().values() 
                        if piece.color == chess.BLACK)
    
    total_material = white_material + black_material
    if total_material > 0:
        white_ratio = white_material / total_material
        black_ratio = black_material / total_material
    else:
        white_ratio = black_ratio = 0.5
    
    planes[123, :, :] = white_ratio
    planes[124, :, :] = black_ratio
    
    # 125: Check indicator
    planes[125, :, :] = float(current_board.is_check())
    
    # 126: Pins and skewers
    for square in chess.SQUARES:
        piece = current_board.piece_at(square)
        if piece is not None:
            # Check if this piece is pinned
            if current_board.is_pinned(piece.color, square):
                row, col = divmod(square, 8)
                planes[126, row, col] = 1
    
    # 127-128: Attack/defense maps
    # 127: White attack map
    for square in chess.SQUARES:
        if current_board.is_attacked_by(chess.WHITE, square):
            row, col = divmod(square, 8)
            planes[127, row, col] = 1
    
    # 128: Black attack map  
    for square in chess.SQUARES:
        if current_board.is_attacked_by(chess.BLACK, square):
            row, col = divmod(square, 8)
            planes[128, row, col] = 1
    
    return planes

def fen_to_bitplanes(fen: str, history_length: int = 8) -> np.ndarray:
    """
    Compatibility wrapper for single FEN conversion without history.
    Creates a minimal compact history with just the current position.
    
    Args:
        fen: FEN string of the current position
        history_length: Number of ply to pad the history to (default 8)
    
    Returns:
        np.ndarray: [129, 8, 8] bitplane array
    """
    # Create compact history format with just the current position
    move_history_compact = {
        'starting_fen': fen,
        'moves': [],  # No moves, just the current position
        'score': 0.5  # Neutral score for single FEN
    }
    return game_to_bitplanes(move_history_compact, history_length)



# Dataset for value bin classification with compact game history support
class ValueBinDataset(torch.utils.data.Dataset):
    """Dataset for compact_game_history tuples, generates bitplane tensor and bin index on-the-fly."""
    def __init__(self, compact_game_history_list, num_bins=128, history_length=8):
        self.compact_game_history_list = compact_game_history_list
        self.num_bins = num_bins
        self.history_length = history_length

    def __len__(self):
        return len(self.compact_game_history_list)

    def __getitem__(self, idx):
        compact_game_history = self.compact_game_history_list[idx]
        score = compact_game_history.get('score', 0.5)  # Extract score from compact history
        
        # Generate bitplanes from compact game history
        bitplanes = torch.tensor(game_to_bitplanes(compact_game_history, self.history_length), dtype=torch.float32)
        bin_idx = score_to_bin(float(score), num_bins=self.num_bins)
        bin_idx_tensor = torch.tensor(bin_idx, dtype=torch.long)
        return bitplanes, bin_idx_tensor


class HRMChess(nn.Module):
    def __init__(self, num_bins=128, hidden_dim=256, N=8, T=8, nhead=None, dim_feedforward=None):
        super().__init__()
        self.N = N
        self.T = T
        self.hidden_dim = hidden_dim
        self.num_bitplanes = 129
        self.board_size = 8
        self.seq_len = 8 * 8

        # Attention defaults
        if nhead is None:
            nhead = 4
        if dim_feedforward is None:
            dim_feedforward = hidden_dim * 2
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward

        # 2D convolution with BatchNorm2d
        self.conv = nn.Sequential(
            nn.Conv2d(self.num_bitplanes, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )

        # Attention modul (Transformer encoder block)
        self.attn = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=self.nhead, dim_feedforward=self.dim_feedforward,
            dropout=0.1, activation='gelu', batch_first=True
        )

        # Flatten + enhancement
        self.board_enhancer = nn.Sequential(
            nn.Linear(hidden_dim * self.seq_len, hidden_dim * (self.seq_len // 16)),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * (self.seq_len // 16), hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )

        # HRM modulok
        self.L_net = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2, batch_first=True
        )

        self.H_net = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2, batch_first=True
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_bins)
        )

    def forward(self, bitplanes, z_init=None):
        # Input: [B, 20, 8, 8]
        x = self.conv(bitplanes)  # -> [B, hidden_dim, 8, 8]
        x = x.flatten(2).transpose(1, 2)  # [B, 64, hidden_dim]

        # 🌟 Self-attention
        x = self.attn(x)  # [B, 64, hidden_dim]
        x = x.flatten(1)  # [B, 64 * hidden_dim]

        board_enhanced = self.board_enhancer(x)  # [B, hidden_dim]

        batch_size = bitplanes.size(0)
        if z_init is None:
            z_H = torch.zeros(batch_size, self.hidden_dim, device=x.device)
            z_L = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        else:
            z_H, z_L = z_init

        for i in range(self.N * self.T - 1):
            with torch.no_grad():
                l_input = torch.stack([z_L, z_H, board_enhanced], dim=1)  # [B, 3, hidden_dim]
                l_out = self.L_net(l_input)  # [B, 3, hidden_dim]
                z_L = l_out.mean(dim=1)      # [B, hidden_dim]
                if (i + 1) % self.T == 0:
                    h_input = torch.stack([z_H, z_L], dim=1)  # [B, 2, hidden_dim]
                    h_out = self.H_net(h_input)  # [B, 2, hidden_dim]
                    z_H = h_out.mean(dim=1)      # [B, hidden_dim]

        # Final step (with grad)
        l_input = torch.stack([z_L, z_H, board_enhanced], dim=1)  # [B, 3, hidden_dim]
        l_out = self.L_net(l_input)  # [B, 3, hidden_dim]
        z_L = l_out.mean(dim=1)      # [B, hidden_dim]
        h_input = torch.stack([z_H, z_L], dim=1)  # [B, 2, hidden_dim]
        h_out = self.H_net(h_input)  # [B, 2, hidden_dim]
        z_H = h_out.mean(dim=1)      # [B, hidden_dim]

        logits = self.value_head(z_H)  # [B, num_bins]
        return logits

def train_step(model, batch, optimizer, scaler=None, use_amp=True):
    """
    Training step for value bin classification with Automatic Mixed Precision.
    batch: (fen_tokens, target_bin)
    """
    fen_tokens, target_bin = batch[:2]
    optimizer.zero_grad()
    
    if use_amp and scaler is not None:
        with torch.amp.autocast('cuda'):
            logits = model(fen_tokens)
            loss = F.cross_entropy(logits, target_bin)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        logits = model(fen_tokens)
        loss = F.cross_entropy(logits, target_bin)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    return {'total_loss': loss.item()}

def train_loop(model, dataset, epochs=25, batch_size=16, lr=1e-4, warmup_epochs=3, device=None, use_amp=True):
    """
    Training loop for value bin classification (cross-entropy) with Automatic Mixed Precision
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize wandb
    wandb.init(project="hrmchess", name="training_run")

    print("\n🏗️ TRAINING LOOP")
    print("   ⚖️ Value bin classification (cross-entropy)")
    print(f"   🚀 AMP (float16): {'Enabled' if use_amp and device.type == 'cuda' else 'Disabled'}")
    print(f"   🔥 Warmup epochs: {warmup_epochs}")

    # Initialize AMP scaler if using CUDA and AMP
    scaler = torch.amp.GradScaler('cuda') if use_amp and device.type == 'cuda' else None

    # Train/validation split
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    import platform
    if platform.system() == "Windows":
        num_workers = 0
    else:
        num_workers = 8
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    total_steps = len(train_dataloader) * epochs
    warmup_steps = len(train_dataloader) * warmup_epochs

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    backup_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=7, factor=0.7, min_lr=1e-6
    )

    model.to(device)
    model.train()

    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 12

    print(f"📊 Training: {train_size:,} positions")
    print(f"📊 Validation: {val_size:,} positions")
    print(f"🔥 Total steps: {total_steps:,} (warmup: {warmup_steps:,})")

    global_step = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_batches = 0
        epoch_start_lr = optimizer.param_groups[0]['lr']

        batch_loss_accum = 0.0
        batch_count_accum = 0
        print_interval = 10000
        for i, batch in enumerate(train_dataloader):
            batch = tuple(b.to(device) for b in batch)
            loss_info = train_step(model, batch, optimizer, scaler, use_amp)
            train_loss += loss_info['total_loss']
            train_batches += 1
            batch_loss_accum += loss_info['total_loss']
            batch_count_accum += 1
            scheduler.step()
            global_step += 1
            if (i + 1) % print_interval == 0:
                avg_loss = batch_loss_accum / batch_count_accum
                current_lr = optimizer.param_groups[0]['lr']
                warmup_progress = min(1.0, global_step / warmup_steps) * 100
                print(f"Epoch {epoch}, Step {i+1}, Avg loss: {avg_loss:.4f}, "
                      f"LR: {current_lr:.2e}, Warmup: {warmup_progress:.1f}%")
                batch_loss_accum = 0.0
                batch_count_accum = 0

        # Validation phase
        model.eval()

        val_loss = 0.0
        val_batches = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_dataloader:
                batch = tuple(b.to(device) for b in batch)
                fen_tokens, target_bin = batch[:2]
                if use_amp and device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        logits = model(fen_tokens)
                        loss = F.cross_entropy(logits, target_bin)
                else:
                    logits = model(fen_tokens)
                    loss = F.cross_entropy(logits, target_bin)
                val_loss += loss.item()
                val_batches += 1
                # Accuracy számítás
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == target_bin).sum().item()
                val_total += target_bin.size(0)

        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        val_accuracy = val_correct / val_total if val_total > 0 else 0.0

        current_lr = optimizer.param_groups[0]['lr']
        epoch_end_lr = current_lr

        warmup_status = "🔥 WARMUP" if epoch < warmup_epochs else "📈 COSINE"

        print(f"📊 Epoch {epoch:2d}/{epochs} {warmup_status} | "
              f"Train loss: {avg_train_loss:.4f} | "
              f"Val loss: {avg_val_loss:.4f} | "
              f"Val accuracy: {val_accuracy:.3f} | "
              f"LR: {epoch_start_lr:.2e}→{epoch_end_lr:.2e}")

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": current_lr
        })

        if epoch >= warmup_epochs:
            backup_scheduler.step(avg_val_loss)

        # check loss and accuracy improvement
        if avg_val_loss < best_val_loss and val_accuracy > (checkpoint['training_info']['val_accuracy'] if 'checkpoint' in locals() else 0.0):
            best_val_loss = avg_val_loss
            patience_counter = 0
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'hyperparams': {
                    'hidden_dim': model.hidden_dim,
                    'N': model.N,
                    'T': model.T,
                    'nhead': model.nhead,
                    'dim_feedforward': model.dim_feedforward
                },
                'training_info': {
                    'epoch': epoch,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'val_accuracy': val_accuracy,
                    'lr': current_lr,
                    'model_type': 'HRM',
                    'warmup_epochs': warmup_epochs,
                    'global_step': global_step
                }
            }
            torch.save(checkpoint, "best_hrm_chess_model.pt")
            print(f"✅ Best model saved! (Val loss: {avg_val_loss:.4f}, Val acc: {val_accuracy:.3f})")
        else:
            patience_counter += 1
            #if patience_counter >= max_patience:
            #    print(f"⏹️ Early stopping at epoch {epoch} (patience exhausted)")
            #    break

    print("\n🎉 Training with warmup completed!")
    print("📁 Best model: best_hrm_chess_model.pt")
    print(f"📊 Best validation loss: {best_val_loss:.4f}")
    print(f"🔥 Total training steps: {global_step:,}")
    print(f"🔥 Warmup completed: {min(global_step, warmup_steps):,}/{warmup_steps:,} steps")
    
    # Finish wandb run
    wandb.finish()

def score_to_bin(score: float, num_bins: int = 128) -> int:
    """
    Egy 0.0–1.0 közötti score alapján visszaadja a bin indexet (0–num_bins-1).
    """
    score = max(0.0, min(score, 1.0))  # levágás
    bin_idx = int(score * num_bins)
    return min(bin_idx, num_bins - 1)  # felső határ korrekció

def bin_to_score(bin_idx: int, num_bins: int = 128) -> float:
    """
    Egy bin indexhez tartozó score-tartomány közepét adja vissza 0.0–1.0 között.
    """
    bin_width = 1.0 / num_bins
    return (bin_idx + 0.5) * bin_width

def load_model_with_amp(model_path, device=None, use_half=False):
    """
    Load HRMChess model with optional float16 optimization for inference.
    
    Args:
        model_path: Path to the model checkpoint
        device: Target device (cuda/cpu)
        use_half: Whether to convert to float16 (only for CUDA)
    
    Returns:
        model: Loaded HRMChess model
        model_info: Dictionary with model metadata
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Extract hyperparameters
    if 'hyperparams' in checkpoint:
        hyper = checkpoint['hyperparams']
        hidden_dim = hyper.get('hidden_dim', 256)
        N = hyper.get('N', 8)
        T = hyper.get('T', 8)
        nhead = hyper.get('nhead', 4)
        dim_feedforward = hyper.get('dim_feedforward', hidden_dim * 2)
    else:
        # Default parameters
        hidden_dim = 256
        N = 8
        T = 8
        nhead = 4
        dim_feedforward = hidden_dim * 2
    
    # Create model
    model = HRMChess(
        hidden_dim=hidden_dim, 
        N=N, 
        T=T, 
        nhead=nhead, 
        dim_feedforward=dim_feedforward
    )
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Move to device
    model.to(device)
    
    # Convert to half precision if requested and on CUDA
    if use_half and device.type == 'cuda':
        model.half()
        print("🚀 Model converted to float16 for faster inference")
    
    model.eval()
    
    model_info = {
        'hidden_dim': hidden_dim,
        'N': N,
        'T': T,
        'nhead': nhead,
        'dim_feedforward': dim_feedforward,
        'device': device,
        'use_half': use_half and device.type == 'cuda'
    }
    
    return model, model_info

def inference_with_amp(model, bitplanes, use_amp=True):
    """
    Perform inference with optional AMP acceleration.
    
    Args:
        model: The HRMChess model
        bitplanes: Input tensor [batch_size, 20, 8, 8]
        use_amp: Whether to use automatic mixed precision
    
    Returns:
        logits: Model output
    """
    device = next(model.parameters()).device
    bitplanes = bitplanes.to(device)
    
    with torch.no_grad():
        if use_amp and device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                logits = model(bitplanes)
        else:
            logits = model(bitplanes)
    
    return logits