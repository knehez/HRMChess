"""
Pure Vision Transformer Chess Model
Direct tokenization of bitplanes with full transformer architecture
Simplified from the original HRM (Hierarchical Reasoning Machine) approach
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from tqdm import tqdm
import chess
import numpy as np
import wandb

# --- FEN to bitplane conversion ---
def game_to_bitplanes(move_history_compact: dict) -> np.ndarray:
    """
    Two-position game history to bitplane conversion: returns [58, 8, 8] numpy array.
    Encodes starting position and current position separately for better tokenization.
    
    Position 1 (Starting): 0-28 channels (29 total)
    0-11: White/Black pieces (6 types each)
    12-19: Metadata (side to move, castling, en passant, move count)
    20-28: Evaluation features (king safety, material, check, pins, attacks)
    
    Position 2 (Current): 29-57 channels (29 total)
    Same structure as Position 1
    
    This allows tokenization as [B, 58, 64] → [B, 128, 29] where each position
    becomes 64 tokens with 29 channels each.
    
    Args:
        move_history_compact: Dict with keys:
            - 'starting_fen': FEN string of the starting position
            - 'moves': List of UCI move strings, last move is Stockfish's best move
            - 'score': Float score (0.0-1.0) for the last move from Stockfish
    """
    planes = np.zeros((58, 8, 8), dtype=np.float32)  # Two positions: 2 x 29 = 58 planes
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    
    # Reconstruct board positions from starting FEN + moves
    starting_fen = move_history_compact.get('starting_fen', chess.STARTING_FEN)
    moves = move_history_compact.get('moves', [])
    
    # Create current board and apply moves
    current_board = chess.Board(starting_fen)
    starting_board = current_board.copy()  # Will be updated to previous position
    
    # Apply moves one by one, keeping starting_board as previous position
    for move_uci in moves:
        try:
            move = chess.Move.from_uci(move_uci)
            if move in current_board.legal_moves:
                starting_board = current_board.copy()  # Save current as starting
                current_board.push(move)               # Apply move to current
            else:
                print(f"⚠️ Illegal move {move_uci}, stopping")
                break
        except (ValueError, chess.InvalidMoveError):
            print(f"⚠️ Invalid UCI move {move_uci}, stopping")
            break
    
    # Encode both positions (starting: 0-28, current: 29-57)
    for pos_idx, board_state in enumerate([starting_board, current_board]):
        base_idx = pos_idx * 29  # 29 channels per position
        
        # Encode pieces (0-11: 6 white + 6 black pieces) - NumPy optimized
        for piece_idx, piece_type in enumerate(piece_types):
            # White pieces - vectorized
            white_squares = np.array(list(board_state.pieces(piece_type, chess.WHITE)))
            if len(white_squares) > 0:
                rows, cols = np.divmod(white_squares, 8)
                planes[base_idx + piece_idx, rows, cols] = 1
            
            # Black pieces - vectorized
            black_squares = np.array(list(board_state.pieces(piece_type, chess.BLACK)))
            if len(black_squares) > 0:
                rows, cols = np.divmod(black_squares, 8)
                planes[base_idx + 6 + piece_idx, rows, cols] = 1
        
        # Metadata (12-19: 8 channels)
        # 12: side to move
        planes[base_idx + 12, :, :] = int(board_state.turn)
        
        # 13-16: castling rights
        planes[base_idx + 13, :, :] = int(board_state.has_kingside_castling_rights(chess.WHITE))
        planes[base_idx + 14, :, :] = int(board_state.has_queenside_castling_rights(chess.WHITE))
        planes[base_idx + 15, :, :] = int(board_state.has_kingside_castling_rights(chess.BLACK))
        planes[base_idx + 16, :, :] = int(board_state.has_queenside_castling_rights(chess.BLACK))
        
        # 17: en passant
        if board_state.ep_square is not None:
            row, col = divmod(board_state.ep_square, 8)
            planes[base_idx + 17, row, col] = 1
        
        # 18: move count (normalized)
        planes[base_idx + 18, :, :] = board_state.fullmove_number / 100.0
        
        # 19: reserved for future metadata
        # planes[base_idx + 19, :, :] = 0  # placeholder
        
        # Evaluation features (20-28: 9 channels)
        
        # 20-23: King safety indicators (4 directions for white king)
        white_king_square = board_state.king(chess.WHITE)
        if white_king_square is not None:
            king_row, king_col = divmod(white_king_square, 8)
            directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # N, E, S, W
            for dir_idx, (dr, dc) in enumerate(directions):
                safety_score = 0
                for distance in range(1, 4):
                    r, c = king_row + dr * distance, king_col + dc * distance
                    if 0 <= r < 8 and 0 <= c < 8:
                        square = chess.square(c, r)
                        piece = board_state.piece_at(square)
                        if piece is not None and piece.color == chess.WHITE:
                            safety_score += 1
                        elif piece is not None and piece.color == chess.BLACK:
                            safety_score -= 1
                    else:
                        break
                normalized_safety = max(0, min(1, (safety_score + 3) / 6))
                planes[base_idx + 20 + dir_idx, king_row, king_col] = normalized_safety
        
        # 24-25: Material balance - optimized
        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
                       chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
        
        # Vectorized material calculation
        pieces = board_state.piece_map()
        white_material = sum(piece_values[piece.piece_type] 
                            for piece in pieces.values() 
                            if piece.color == chess.WHITE)
        black_material = sum(piece_values[piece.piece_type] 
                            for piece in pieces.values() 
                            if piece.color == chess.BLACK)
        
        total_material = white_material + black_material
        if total_material > 0:
            white_ratio = white_material / total_material
            black_ratio = black_material / total_material
        else:
            white_ratio = black_ratio = 0.5
        
        planes[base_idx + 24, :, :] = white_ratio
        planes[base_idx + 25, :, :] = black_ratio
        
        # 26: Check indicator
        planes[base_idx + 26, :, :] = float(board_state.is_check())
        
        # 27-28: Attack maps - NumPy optimized
        white_attacked = []
        black_attacked = []
        pinned_squares = []
        
        for sq in chess.SQUARES:
            if board_state.is_attacked_by(chess.WHITE, sq):
                white_attacked.append(sq)
            if board_state.is_attacked_by(chess.BLACK, sq):
                black_attacked.append(sq)
            
            piece = board_state.piece_at(sq)
            if piece and board_state.is_pinned(piece.color, sq):
                pinned_squares.append(sq)
        
        # Vectorized assignment for attack maps
        if white_attacked:
            white_attacked = np.array(white_attacked)
            rows, cols = np.divmod(white_attacked, 8)
            planes[base_idx + 27, rows, cols] = 1.0
            
        if black_attacked:
            black_attacked = np.array(black_attacked)
            rows, cols = np.divmod(black_attacked, 8)
            planes[base_idx + 28, rows, cols] = 1.0

        # 19: Pins - vectorized
        if pinned_squares:
            pinned_squares = np.array(pinned_squares)
            rows, cols = np.divmod(pinned_squares, 8)
            planes[base_idx + 19, rows, cols] = 1.0
    
    return planes

def fen_to_bitplanes(fen: str, history_length: int = 2) -> np.ndarray:
    """
    Compatibility wrapper for single FEN conversion without history.
    Creates a minimal compact history with just the current position.
    
    Args:
        fen: FEN string of the current position
        history_length: Number of ply to pad the history to (default 2)
    
    Returns:
        np.ndarray: [58, 8, 8] bitplane array (2 positions x 29 channels each)
    """
    # Create compact history format with just the current position
    move_history_compact = {
        'starting_fen': fen,
        'moves': [],  # No moves, just the current position
        'score': 0.5  # Neutral score for single FEN
    }
    return game_to_bitplanes(move_history_compact)

def score_to_soft_bins(score: float, num_bins: int = 128, sigma: float = 1.5):
    """
    0..1 score -> lágy eloszlás a bin-tengelyen (Gaussian a közép körül).
    sigma: bin egységekben (1.0–3.0 tipikusan jó)
    """
    score = max(0.0, min(score, 1.0))
    centers = torch.linspace(0.0 + 0.5/num_bins, 1.0 - 0.5/num_bins, num_bins)  # bin-középpontok
    mu = torch.tensor(score)
    dist = torch.exp(-0.5 * ((centers - mu) * num_bins / sigma) ** 2)
    dist = dist / dist.sum()
    return dist

# Dataset for value bin classification with compact game history support
class ValueBinDataset(torch.utils.data.Dataset):
    """Dataset for compact_game_history tuples, generates bitplane tensor and bin index on-the-fly."""
    def __init__(self, compact_game_history_list, num_bins=128, history_length=2, soft_targets=False, sigma=1.5):
        self.compact_game_history_list = compact_game_history_list
        self.num_bins = num_bins
        self.history_length = history_length
        self.soft_targets = soft_targets
        self.sigma = sigma
        
    def __len__(self):
        return len(self.compact_game_history_list)

    def __getitem__(self, idx):
        compact_game_history = self.compact_game_history_list[idx]
        score = compact_game_history.get('score', 0.5)  # Extract score from compact history
        
        # Generate bitplanes from compact game history
        bitplanes = torch.tensor(game_to_bitplanes(compact_game_history), dtype=torch.float32)
        if self.soft_targets:
            target_dist = score_to_soft_bins(score, self.num_bins, self.sigma)  # [num_bins]
            return bitplanes, target_dist.float()
        else:
            bin_idx = score_to_bin(score, self.num_bins)
            return bitplanes, torch.tensor(bin_idx, dtype=torch.long)


class PureViTChess(nn.Module):
    """
    Pure Vision Transformer Chess Model
    Direct tokenization of bitplanes without CNN preprocessing
    """
    def __init__(self, num_bins=128, hidden_dim=256, n_layers=6, n_heads=8, ff_mult=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_positions = 2  # Starting and current positions
        self.channels_per_position = 29  # 29 channels per position
        self.total_bitplanes = 58  # 2 positions x 29 channels
        self.board_size = 8
        self.seq_len = 2 * 8 * 8  # 128 squares (2 positions)

        # Linear projection from channels per position to hidden_dim per square
        self.patch_embedding = nn.Linear(self.channels_per_position, hidden_dim)
        
        # Learnable positional embedding for 2 positions x 64 squares = 128 tokens
        self.pos_embedding = nn.Parameter(torch.zeros(1, 128, hidden_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # [CLS] token for global representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ÚJ: szegmens embedding a 2 pozícióhoz (starting/current)
        self.segment_embedding = nn.Parameter(torch.zeros(1, 2, hidden_dim))
        nn.init.trunc_normal_(self.segment_embedding, std=0.02)

        # ÚJ: token-szintű LayerNorm a patch-embedding után
        self.token_ln = nn.LayerNorm(hidden_dim)
        
        # Vision Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=ff_mult * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classification head
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_bins)
        )

    def forward(self, bitplanes, z_init=None):
        """
        Args:
            bitplanes: [B, 58, 8, 8] chess position bitplanes (2 positions x 29 channels)
        Returns:
            logits: [B, num_bins] position evaluation
        """
        B, C, H, W = bitplanes.shape
        assert C == self.total_bitplanes and H == 8 and W == 8, f"Expected [B, 58, 8, 8], got [B, {C}, {H}, {W}]"
        
        # [B,2,29,8,8] -> [B,2,64,29]
        x = bitplanes.view(B, self.num_positions, self.channels_per_position, H, W)
        x = x.view(B, self.num_positions, self.channels_per_position, H*W).transpose(2, 3)  # [B,2,64,29]

        # Patch-embedding per square
        x = self.patch_embedding(x)                              # [B,2,64,C]
        # Szegmens-embedding hozzáadása (pozíciótípus: 0=starting, 1=current)
        seg = self.segment_embedding[:, :, :].unsqueeze(2)       # [1,2,1,C]
        x = x + seg                                              # [B,2,64,C]

        # Összelapítás 128 tokenre és pos-embedding
        x = x.reshape(B, self.num_positions*H*W, self.hidden_dim)  # [B,128,C]
        x = self.token_ln(x + self.pos_embedding)                   # stabilizál

        # [CLS] előtag
        cls = self.cls_token.expand(B, 1, self.hidden_dim)
        x = torch.cat([cls, x], dim=1)                            # [B,129,C]

        # Transformer (már tartalmazza a n_layers darab réteget)
        x = self.transformer(x)                                   # [B,129,C]
        
        # [CLS] -> logits
        cls_out = self.norm(x[:, 0])
        logits = self.head(cls_out)                               # [B,num_bins]
        
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

def train_loop(model, dataset, epochs=25, batch_size=16, lr=1e-4, device=None, use_amp=True):
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

    # Initialize AMP scaler if using CUDA and AMP
    scaler = torch.amp.GradScaler('cuda') if use_amp and device.type == 'cuda' else None

    # Train/validation split
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    import platform
    if platform.system() == "Windows":
        num_workers = 4  # Windows can handle 4 workers efficiently
    else:
        num_workers = 8
    
    # Debug CUDA info
    print(f"🔍 CUDA available: {torch.cuda.is_available()}")
    print(f"🔍 Device: {device}")
    print(f"🔍 Device type: {device.type}")
    
    # Force pin_memory=True for CUDA (you have GPU)
    use_pin_memory = True  # Force True since you have CUDA GPU
    
    # Enable pin_memory for faster GPU transfer and persistent workers for efficiency
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    backup_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=7, factor=0.7, min_lr=1e-6
    )

    total_steps = len(train_dataloader) * epochs

    def lr_lambda(current_step):
        progress = float(current_step) / float(max(1, total_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.to(device)
    model.train()

    best_val_loss = float('inf')
    patience_counter = 0

    print(f"📊 Training: {train_size:,} positions")
    print(f"📊 Validation: {val_size:,} positions")
    # print(f"🔥 Total steps: {total_steps:,}")  # Removed: total_steps is not used
    print(f"🚀 DataLoader workers: {num_workers} (pin_memory=True)")
    
    # GPU utilization monitoring
    if device.type == 'cuda':
        print(f"🎯 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")

    global_step = 0

    for epoch in range(epochs):
        epoch_start_time = time.time()  # Track epoch start time
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
                
                # Time estimation
                elapsed_time = time.time() - epoch_start_time if 'epoch_start_time' in locals() else 0
                steps_done = i + 1
                total_steps_epoch = len(train_dataloader)
                
                if steps_done > 0:
                    avg_time_per_step = elapsed_time / steps_done
                    remaining_steps = total_steps_epoch - steps_done
                    eta_seconds = remaining_steps * avg_time_per_step
                    eta_minutes = int(eta_seconds // 60)
                    eta_seconds = int(eta_seconds % 60)
                    eta_str = f", ETA: {eta_minutes}m{eta_seconds}s"
                else:
                    eta_str = ""
                
                print(f"Epoch {epoch}, Step {i+1}/{total_steps_epoch}, Avg loss: {avg_loss:.4f}, "
                      f"LR: {current_lr:.2e}{eta_str}")
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

        print(f"📊 Epoch {epoch:2d}/{epochs} | "
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

        # Use ReduceLROnPlateau to adjust LR based on validation loss
        backup_scheduler.step(avg_val_loss)

        # check loss and accuracy improvement
        if avg_val_loss < best_val_loss and val_accuracy > (checkpoint['training_info']['val_accuracy'] if 'checkpoint' in locals() else 0.0):
            best_val_loss = avg_val_loss
            patience_counter = 0
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'hyperparams': {
                    'hidden_dim': model.hidden_dim,
                    'n_heads': 8,  # Default for PureViTChess
                    'n_layers': 6,  # Default for PureViTChess
                    'model_arch': 'pure_vit'
                },
                'training_info': {
                    'epoch': epoch,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'val_accuracy': val_accuracy,
                    'lr': current_lr,
                    'model_type': 'PureViT',
                    'global_step': global_step
                }
            }
            torch.save(checkpoint, "best_hrm_chess_model.pt")
            print(f"✅ Best model saved! (Val loss: {avg_val_loss:.4f}, Val acc: {val_accuracy:.3f})")
        else:
            patience_counter += 1

    print("\n🎉 Training completed!")
    print("📁 Best model: best_hrm_chess_model.pt")
    print(f"📊 Best validation loss: {best_val_loss:.4f}")
    print(f"🔥 Total training steps: {global_step:,}")
    
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
    Load PureViTChess model with optional float16 optimization for inference.
    
    Args:
        model_path: Path to the model checkpoint
        device: Target device (cuda/cpu)
        use_half: Whether to convert to float16 (only for CUDA)
    
    Returns:
        model: Loaded PureViTChess model
        model_info: Dictionary with model metadata
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Get state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Auto-detect number of transformer layers from state dict
    layer_indices = set()
    for key in state_dict.keys():
        if key.startswith('transformer.layers.'):
            # Extract layer index from key like 'transformer.layers.0.norm1.weight'
            try:
                parts = key.split('.')
                if len(parts) > 2 and parts[2].isdigit():
                    layer_indices.add(int(parts[2]))
            except (IndexError, ValueError):
                continue
    
    detected_n_layers = max(layer_indices) + 1 if layer_indices else 6
    print(f"🔍 Detected transformer layers: {sorted(layer_indices)} → {detected_n_layers} total layers")
    
    # Extract hyperparameters
    if 'hyperparams' in checkpoint:
        hyper = checkpoint['hyperparams']
        hidden_dim = hyper.get('hidden_dim', 256)
        nhead = hyper.get('nhead', 8)
        
        # Try to get dim_feedforward from hyperparams, if not found detect from state_dict
        if 'dim_feedforward' in hyper:
            dim_feedforward = hyper['dim_feedforward']
        elif 'transformer.layers.0.linear1.weight' in state_dict:
            dim_feedforward = state_dict['transformer.layers.0.linear1.weight'].shape[0]
            print(f"🔍 Auto-detected dim_feedforward from state_dict: {dim_feedforward}")
        else:
            dim_feedforward = hidden_dim * 4
            print(f"🔍 Using default dim_feedforward: {dim_feedforward}")
            
        # Use detected layers instead of hyperparams if they conflict
        n_layers_from_hyper = hyper.get('n_layers', detected_n_layers)
        n_layers = detected_n_layers if layer_indices else n_layers_from_hyper
    else:
        # Default parameters with auto-detected layers
        hidden_dim = 256
        nhead = 8
        n_layers = detected_n_layers
        
        # Try to detect hidden_dim from patch embedding if available
        if 'patch_embedding.weight' in state_dict:
            hidden_dim = state_dict['patch_embedding.weight'].shape[0]
        
        # Try to detect feedforward dimension from transformer layer
        if 'transformer.layers.0.linear1.weight' in state_dict:
            dim_feedforward = state_dict['transformer.layers.0.linear1.weight'].shape[0]
        else:
            dim_feedforward = hidden_dim * 4  # Default fallback
        
        print(f"🔍 Auto-detected parameters: hidden_dim={hidden_dim}, n_layers={n_layers}, dim_feedforward={dim_feedforward}")
    
    # Create Pure ViT model with detected parameters
    model = PureViTChess(
        hidden_dim=hidden_dim,
        n_heads=nhead,
        n_layers=n_layers,
        ff_mult=dim_feedforward // hidden_dim
    )
    print(f"🔍 Loading Pure Vision Transformer model ({n_layers} layers)")
    
    # Load state dict
    try:
        model.load_state_dict(state_dict)
        print(f"✅ Successfully loaded Pure ViT model with {n_layers} layers")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"🔧 Model architecture mismatch - parameters detected from state_dict:")
        print(f"    hidden_dim={hidden_dim}, dim_feedforward={dim_feedforward}, n_layers={n_layers}")
        raise RuntimeError(f"Model architecture mismatch: {e}")
    
    # Move to device
    model.to(device)
    
    # Convert to half precision if requested and on CUDA
    if use_half and device.type == 'cuda':
        model.half()
        print("🚀 Model converted to float16 for faster inference")
    
    model.eval()
    
    model_info = {
        'hidden_dim': hidden_dim,
        'nhead': nhead,
        'dim_feedforward': dim_feedforward,
        'n_layers': n_layers,
        'model_type': 'pure_vit',
        'device': device,
        'use_half': use_half and device.type == 'cuda'
    }
    
    return model, model_info

def inference_with_amp(model, bitplanes, use_amp=True):
    """
    Perform inference with optional AMP acceleration.
    
    Args:
        model: The HRMChess model
        bitplanes: Input tensor [batch_size, 58, 8, 8] - two positions with 29 channels each
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