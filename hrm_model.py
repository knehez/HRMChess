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

def fen_to_tokens(fen: str) -> list:
    """
    Converts a FEN string to a list of 77 ASCII integer tokens (padded/truncated as needed).
    """
    return [ord(c) for c in fen.ljust(77)[:77]]

# Dataset for value bin classification: (fen_tokens, uci_token, target_bin)
class ValueBinDataset(torch.utils.data.Dataset):
    """Dataset for (fen_tokens, uci_token, target_bin) tuples"""
    def __init__(self, fen_tokens, uci_tokens, target_bins):
        self.fen_tokens = torch.as_tensor(fen_tokens).long()
        self.uci_tokens = torch.as_tensor(uci_tokens).long()
        self.target_bins = torch.as_tensor(target_bins).long()

    def __len__(self):
        return len(self.fen_tokens)

    def __getitem__(self, idx):
        return (
            self.fen_tokens[idx],           # [77]
            self.uci_tokens[idx],           # [1] or scalar
            self.target_bins[idx]           # int (bin label)
        )


class HRMChess(nn.Module):
    def __init__(self, vocab_size=128, uci_vocab_size=1968, num_bins=128, emb_dim=256, nhead=4, num_layers=4, N=8, T=8):
        super().__init__()
        self.N = N
        self.T = T
        self.hidden_dim = emb_dim
        self.seq_len = 78  # 77 FEN + 1 UCI

        # FEN karakter embedding (ASCII 0–127)
        self.fen_emb = nn.Embedding(vocab_size, emb_dim)
        # UCI lépés embedding (indexelt)
        self.uci_emb = nn.Embedding(uci_vocab_size, emb_dim)
        # Pozíció embedding
        self.pos_embedding = nn.Embedding(self.seq_len, emb_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # LayerNorm a transformer kimenetére
        self.post_transformer_norm = nn.LayerNorm(emb_dim)


        # L_net: Low-level module (zL, zH, transformer_out) -> zL
        self.L_net = nn.Sequential(
            nn.Linear(emb_dim * 3, emb_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.LayerNorm(emb_dim)
        )

        # H_net: High-level module (zH, zL) -> zH
        self.H_net = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.LayerNorm(emb_dim)
        )

        # Value head: bin classification (LayerNorm a fej előtt)
        self.value_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, num_bins)
        )

        # MLP pooling a teljes szekvenciára (FEN + UCI együtt)
        self.seq_mlp_pool = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU()
        )

    def forward(self, fen_tokens, uci_token, z_init=None):
        """
        fen_tokens: [B, 77]  (ASCII kódolt FEN karakterek)
        uci_token:  [B, 1]   (UCI lépés index)
        """
        fen_emb = self.fen_emb(fen_tokens)               # [B, 77, D]
        uci_emb = self.uci_emb(uci_token).squeeze(1)     # [B, D]
        uci_emb = uci_emb.unsqueeze(1)                   # [B, 1, D]
        x = torch.cat([fen_emb, uci_emb], dim=1)         # [B, 78, D]

        # Pozíciókódolás hozzáadása
        batch_size = x.size(0)
        positions = torch.arange(0, self.seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)          # [B, 78, D]
        x = x + pos_emb

        # Transformer
        x = self.transformer(x)                          # [B, 78, D]

        # LayerNorm a transformer kimenetére
        x = self.post_transformer_norm(x)

        # MLP pooling a teljes szekvenciára (FEN + UCI együtt)
        mlp_out = self.seq_mlp_pool(x)  # [B, 78, D]
        seq_agg = mlp_out.mean(dim=1)   # [B, D]
        out = seq_agg                   # [B, D]

        # HRM hierarchical processing: N*T-1 steps without gradient
        if z_init is None:
            z_H = torch.zeros(batch_size, self.hidden_dim, device=out.device)
            z_L = torch.zeros(batch_size, self.hidden_dim, device=out.device)
        else:
            z_H, z_L = z_init
        
        # HRM hierarchical processing: N*T-1 steps without gradient
        total_steps = self.N * self.T
        with torch.no_grad():
            for i in range(total_steps - 1):
                # L_net: (zL, zH, board_features) -> zL
                l_input = torch.cat([z_L, z_H, out], dim=-1)  # [B, 3D]
                z_L = self.L_net(l_input)
                
                # H_net: every T steps, (zH, zL) -> zH
                if (i + 1) % self.T == 0:
                    h_input = torch.cat([z_H, z_L], dim=-1)
                    z_H = self.H_net(h_input)
        # Final step WITH gradient
        l_input = torch.cat([z_L, z_H, out], dim=-1)
        z_L = self.L_net(l_input)
        h_input = torch.cat([z_H, z_L], dim=-1)
        z_H = self.H_net(h_input)
        # Value prediction from z_H
        value_logits = self.value_head(z_H)  # [B, num_bins]
        return value_logits

def train_step(model, batch, optimizer):
    """
    Training step for value bin classification.
    batch: (fen_tokens, uci_token, target_bin)
    """
    fen_tokens, uci_token, target_bin = batch[:3]
    optimizer.zero_grad()
    logits = model(fen_tokens, uci_token)
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

    print(f"\n🏗️ TRAINING LOOP")
    print(f"   ⚖️ Value bin classification (cross-entropy)")
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
            if i % 1000 == 0 and i > 0:
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
                fen_tokens, uci_token, target_bin = batch[:3]
                logits = model(fen_tokens, uci_token)
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
                    'T': model.T,
                    'seq_len': model.seq_len
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

def generate_all_possible_uci_moves():
    """
    Létrehozza az összes lehetséges (elméletileg létező) UCI lépést.
    Az UCI lépések formátuma például: e2e4, e7e8q stb.
    """
    all_moves = set()

    # Az összes lehetséges táblahely 64 mező: a1–h8
    squares = [chess.square(file, rank) for file in range(8) for rank in range(8)]

    # Létrehozunk mesterséges állásokat, ahol minden mezőn minden figura lehet
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    colors = [chess.WHITE, chess.BLACK]

    board = chess.Board(None)  # üres tábla

    for color in colors:
        for piece in piece_types:
            for from_sq in squares:
                board.clear_board()
                board.set_piece_at(from_sq, chess.Piece(piece, color))

                # Legyen egy király is, hogy ne legyen illegális állás
                king_sq = chess.E1 if color == chess.WHITE else chess.E8
                if from_sq != king_sq:
                    board.set_piece_at(king_sq, chess.Piece(chess.KING, color))

                board.turn = color
                legal_moves = list(board.legal_moves)

                for move in legal_moves:
                    all_moves.add(move.uci())

    # Explicit: minden lehetséges gyalog-promóciós lépés (a2-a1q, h7-h8n, stb.)
    files = 'abcdefgh'
    for color in [chess.WHITE, chess.BLACK]:
        if color == chess.WHITE:
            from_rank, to_rank = 6, 7  # 7. sor -> 8. sor
        else:
            from_rank, to_rank = 1, 0  # 2. sor -> 1. sor
        for file in range(8):
            from_sq = files[file] + str(from_rank+1)
            to_sq = files[file] + str(to_rank+1)
            for promo in 'qrbn':
                # sima előrelépés promócióval
                all_moves.add(f"{from_sq}{to_sq}{promo}")
            # ütés balra
            if file > 0:
                from_sq_left = files[file] + str(from_rank+1)
                to_sq_left = files[file-1] + str(to_rank+1)
                for promo in 'qrbn':
                    all_moves.add(f"{from_sq_left}{to_sq_left}{promo}")
            # ütés jobbra
            if file < 7:
                from_sq_right = files[file] + str(from_rank+1)
                to_sq_right = files[file+1] + str(to_rank+1)
                for promo in 'qrbn':
                    all_moves.add(f"{from_sq_right}{to_sq_right}{promo}")

    # Végül rendezzük, hogy stabil tokenizálás legyen
    return sorted(all_moves)

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