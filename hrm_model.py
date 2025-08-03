"""
HRM Chess Model - Hierarchical Reasoning Machine for Chess
Konvolúciós HRM implementáció
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
import chess
import numpy as np

# --- FEN to bitplane conversion ---
def fen_to_bitplanes(fen: str) -> np.ndarray:
    """
    Converts a FEN string to a [20, 8, 8] bitplane numpy array.
    Bitplanes: 12 for piece types, 4 for castling, 1 for side, 1 for ep, 1 for halfmove, 1 for fullmove.
    """
    board = chess.Board(fen)
    bitplanes = np.zeros((20, 8, 8), dtype=np.float32)
    # Piece bitplanes
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
    # Castling rights
    castling = [board.has_kingside_castling_rights(chess.WHITE),
                board.has_queenside_castling_rights(chess.WHITE),
                board.has_kingside_castling_rights(chess.BLACK),
                board.has_queenside_castling_rights(chess.BLACK)]
    for i, flag in enumerate(castling):
        bitplanes[12 + i, :, :] = float(flag)
    # Side to move
    bitplanes[16, :, :] = float(board.turn)
    # En passant
    bitplanes[17, :, :] = 0.0
    if board.ep_square is not None:
        row, col = divmod(board.ep_square, 8)
        bitplanes[17, row, col] = 1.0
    # Halfmove clock
    bitplanes[18, :, :] = board.halfmove_clock / 100.0
    # Fullmove number
    bitplanes[19, :, :] = board.fullmove_number / 100.0
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
    def __init__(self, num_bins=128, hidden_dim=256, N=8, T=8):
        super().__init__()
        self.N = N
        self.T = T
        self.hidden_dim = hidden_dim
        self.num_bitplanes = 20
        self.board_size = 8
        # 2D convolutional feature extractor (3 layers, decreasing channels)
        self.conv = nn.Sequential(
            nn.Conv2d(self.num_bitplanes, hidden_dim, kernel_size=3, padding=1),  # [B, 20, 8, 8] -> [B, hidden_dim, 8, 8]
            nn.ReLU(),  # [B, hidden_dim, 8, 8]
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),  # [B, hidden_dim, 8, 8] -> [B, hidden_dim//2, 8, 8]
            nn.ReLU(),  # [B, hidden_dim//2, 8, 8]
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, padding=1),  # [B, hidden_dim//2, 8, 8] -> [B, hidden_dim//4, 8, 8]
            nn.ReLU(),  # [B, hidden_dim//4, 8, 8]
            nn.Flatten()  # [B, hidden_dim//4, 8, 8] -> [B, (hidden_dim//4)*8*8]
        )
        # Board representation enhancer - további reprezentációs rétegek
        self.board_enhancer = nn.Sequential(
            nn.Linear((hidden_dim // 4) * 8 * 8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # L_net: Low-level module (zL, zH, board_enhanced) -> zL
        self.L_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        # H_net: High-level module (zH, zL) -> zH
        self.H_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        # Value head: bin classification
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_bins)
        )

    def forward(self, bitplanes, z_init=None):
        """
        bitplanes: [B, 20, 8, 8]  (20 bitplane sakk állás)
        """
        batch_size = bitplanes.size(0)
        # 2D convolutional feature extraction
        x = self.conv(bitplanes)  # [B, hidden_dim*8*8]
        board_enhanced = self.board_enhancer(x)  # [B, hidden_dim]

        # HRM hierarchical processing: N*T-1 steps without gradient
        if z_init is None:
            z_H = torch.zeros(batch_size, self.hidden_dim, device=x.device)
            z_L = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        else:
            z_H, z_L = z_init
        total_steps = self.N * self.T
        with torch.no_grad():
            for i in range(total_steps - 1):
                l_input = torch.cat([z_L, z_H, board_enhanced], dim=-1)
                z_L = self.L_net(l_input)
                if (i + 1) % self.T == 0:
                    h_input = torch.cat([z_H, z_L], dim=-1)
                    z_H = self.H_net(h_input)
        # Final step WITH gradient
        l_input = torch.cat([z_L, z_H, board_enhanced], dim=-1)
        z_L = self.L_net(l_input)
        h_input = torch.cat([z_H, z_L], dim=-1)
        z_H = self.H_net(h_input)
        value_logits = self.value_head(z_H)  # [B, num_bins]
        return value_logits

def train_step(model, batch, optimizer):
    """
    Training step for value bin classification.
    batch: (fen_tokens, target_bin)
    """
    fen_tokens, target_bin = batch[:2]
    optimizer.zero_grad()
    logits = model(fen_tokens)
    loss = F.cross_entropy(logits, target_bin)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return {'total_loss': loss.item()}

def train_loop(model, dataset, epochs=25, batch_size=16, lr=1e-4, warmup_epochs=3, device=None):
    """
    Training loop for value bin classification (cross-entropy)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n🏗️ TRAINING LOOP")
    print("   ⚖️ Value bin classification (cross-entropy)")
    print(f"   🔥 Warmup epochs: {warmup_epochs}")

    # Train/validation split
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

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

        for i, batch in enumerate(train_dataloader):
            batch = tuple(b.to(device) for b in batch)
            loss_info = train_step(model, batch, optimizer)
            train_loss += loss_info['total_loss']
            train_batches += 1
            scheduler.step()
            global_step += 1
            if i % 10000 == 0 and i > 0:
                current_lr = optimizer.param_groups[0]['lr']
                warmup_progress = min(1.0, global_step / warmup_steps) * 100
                print(f"Epoch {epoch}, Step {i}, Total loss: {loss_info['total_loss']:.4f}, "
                        f"LR: {current_lr:.2e}, Warmup: {warmup_progress:.1f}%")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_dataloader:
                batch = tuple(b.to(device) for b in batch)
                fen_tokens, target_bin = batch[:2]
                logits = model(fen_tokens)
                loss = F.cross_entropy(logits, target_bin)
                val_loss += loss.item()
                val_batches += 1

        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches

        current_lr = optimizer.param_groups[0]['lr']
        epoch_end_lr = current_lr

        warmup_status = "🔥 WARMUP" if epoch < warmup_epochs else "📈 COSINE"
        warmup_progress = min(1.0, global_step / warmup_steps) * 100

        print(f"📊 Epoch {epoch:2d}/{epochs} {warmup_status} | "
              f"Train loss: {avg_train_loss:.4f} | "
              f"Val loss: {avg_val_loss:.4f} | "
              f"LR: {epoch_start_lr:.2e}→{epoch_end_lr:.2e}")

        if epoch < warmup_epochs:
            print(f"    🔥 Warmup Progress: {warmup_progress:.1f}% ({global_step:,}/{warmup_steps:,} steps)")

        if epoch >= warmup_epochs:
            backup_scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'hyperparams': {
                    'hidden_dim': model.hidden_dim,
                    'N': model.N,
                    'T': model.T
                },
                'training_info': {
                    'epoch': epoch,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'lr': current_lr,
                    'model_type': 'HRM',
                    'warmup_epochs': warmup_epochs,
                    'global_step': global_step
                }
            }
            torch.save(checkpoint, "best_hrm_chess_model.pt")
            print(f"✅ Best model saved! (Val loss: {avg_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"⏹️ Early stopping at epoch {epoch} (patience exhausted)")
                break

    print("\n🎉 Training with warmup completed!")
    print("📁 Best model: best_hrm_chess_model.pt")
    print(f"📊 Best validation loss: {best_val_loss:.4f}")
    print(f"🔥 Total training steps: {global_step:,}")
    print(f"🔥 Warmup completed: {min(global_step, warmup_steps):,}/{warmup_steps:,} steps")

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