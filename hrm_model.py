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
def fen_to_bitplanes(fen: str) -> np.ndarray:
    """
    Enhanced FEN to bitplane conversion with more chess-specific features.
    Returns [30, 8, 8] bitplane array with richer representation.
    
    Bitplanes:
    0-11: Piece types (6 white + 6 black)
    12-15: Castling rights (4 types)
    16: Side to move
    17: En passant square
    18: Halfmove clock (normalized)
    19: Fullmove number (normalized)
    20-23: King safety indicators (4 directions for each king)
    24-25: Material balance (normalized for white/black)
    26: Check indicator
    27: Pins and skewers
    28-29: Attack/defense maps
    """
    board = chess.Board(fen)
    bitplanes = np.zeros((30, 8, 8), dtype=np.float32)
    
    # Original piece bitplanes (0-11)
    piece_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            idx = piece_map[piece.symbol()]
            row, col = divmod(square, 8)
            bitplanes[idx, row, col] = 1.0
    
    # Castling rights (12-15)
    castling = [board.has_kingside_castling_rights(chess.WHITE),
                board.has_queenside_castling_rights(chess.WHITE),
                board.has_kingside_castling_rights(chess.BLACK),
                board.has_queenside_castling_rights(chess.BLACK)]
    for i, flag in enumerate(castling):
        bitplanes[12 + i, :, :] = float(flag)
    
    # Side to move (16)
    bitplanes[16, :, :] = float(board.turn)
    
    # En passant (17)
    if board.ep_square is not None:
        row, col = divmod(board.ep_square, 8)
        bitplanes[17, row, col] = 1.0
    
    # Halfmove and fullmove (18-19)
    bitplanes[18, :, :] = min(board.halfmove_clock / 50.0, 1.0)  # Cap at 50 moves
    bitplanes[19, :, :] = min(board.fullmove_number / 200.0, 1.0)  # Cap at 200 moves
    
    # Enhanced features (20-29)
    
    # King safety indicators (20-23)
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    if white_king:
        row, col = divmod(white_king, 8)
        # King exposure in 4 directions
        for i, (dr, dc) in enumerate([(0,1), (1,0), (0,-1), (-1,0)]):
            nr, nc = row + dr, col + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                if not board.piece_at(nr * 8 + nc):
                    bitplanes[20, nr, nc] = 1.0  # White king exposure
    
    if black_king:
        row, col = divmod(black_king, 8)
        for i, (dr, dc) in enumerate([(0,1), (1,0), (0,-1), (-1,0)]):
            nr, nc = row + dr, col + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                if not board.piece_at(nr * 8 + nc):
                    bitplanes[21, nr, nc] = 1.0  # Black king exposure
    
    # Material balance (24-25)
    piece_values = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0}
    white_material = sum(piece_values.get(piece.symbol().upper(), 0) 
                        for piece in board.piece_map().values() if piece.color == chess.WHITE)
    black_material = sum(piece_values.get(piece.symbol().upper(), 0) 
                        for piece in board.piece_map().values() if piece.color == chess.BLACK)
    
    total_material = white_material + black_material
    if total_material > 0:
        bitplanes[24, :, :] = white_material / total_material
        bitplanes[25, :, :] = black_material / total_material
    
    # Check indicator (26)
    if board.is_check():
        if board.turn == chess.WHITE:
            if white_king:
                row, col = divmod(white_king, 8)
                bitplanes[26, row, col] = 1.0
        else:
            if black_king:
                row, col = divmod(black_king, 8)
                bitplanes[26, row, col] = 1.0
    
    # Attack maps (28-29) - simplified version
    for square in chess.SQUARES:
        row, col = divmod(square, 8)
        # White attacks
        white_attackers = board.attackers(chess.WHITE, square)
        if white_attackers:
            bitplanes[28, row, col] = min(len(white_attackers) / 3.0, 1.0)
        
        # Black attacks  
        black_attackers = board.attackers(chess.BLACK, square)
        if black_attackers:
            bitplanes[29, row, col] = min(len(black_attackers) / 3.0, 1.0)
    
    return bitplanes

def fen_to_tokens(fen: str) -> list:
    """
    Converts a FEN string to a list of 77 ASCII integer tokens (padded/truncated as needed).
    """
    return [ord(c) for c in fen.ljust(77)[:77]]

# Dataset for value bin classification: (fen_tokens, uci_token, target_bin)
class ValueBinDataset(torch.utils.data.Dataset):
    """Dataset for (fen, score) tuples, generates bitplane tensor and bin index on-the-fly."""
    def __init__(self, fen_list, score_list, num_bins=128):
        self.fen_list = fen_list
        self.score_list = score_list
        self.num_bins = num_bins

    def __len__(self):
        return len(self.fen_list)

    def __getitem__(self, idx):
        fen = self.fen_list[idx]
        score = self.score_list[idx]
        bitplanes = torch.tensor(fen_to_bitplanes(fen), dtype=torch.float32)
        bin_idx = score_to_bin(float(score), num_bins=self.num_bins)
        bin_idx_tensor = torch.tensor(bin_idx, dtype=torch.long)
        return bitplanes, bin_idx_tensor


class HRMChess(nn.Module):
    def __init__(self, num_bins=128, hidden_dim=256, N=8, T=8, nhead=None, dim_feedforward=None):
        super().__init__()
        self.N = N
        self.T = T
        self.hidden_dim = hidden_dim
        self.num_bitplanes = 30
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