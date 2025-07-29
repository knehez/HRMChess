"""
HRM Chess Model - Hierarchical Reasoning Machine for Chess
Konvolúciós HRM implementáció
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PolicyDataset(torch.utils.data.Dataset):
    """Dataset Policy adatokhoz - Fixed for Stockfish evaluation data"""
    
    def __init__(self, states, policies):
        """
        Args:
            states: Board states (tensors)
            policies: List of move evaluations from Stockfish [(move_tuple, score), ...]
        """
        self.states = torch.FloatTensor(states)
        
        # Convert Stockfish evaluations to probability distributions
        policy_matrices = []
        for moves in policies:
            policy_matrix = torch.zeros(64, 64)
            
            if len(moves) > 0:
                # Extract scores and convert to probabilities
                move_tuples = []
                scores = []
                
                for move_tuple, score in moves:
                    move_tuples.append(move_tuple)
                    scores.append(score)                
                # Convert scores to probabilities using softmax with temperature
                if len(scores) > 0:
                    scores_tensor = torch.tensor(scores, dtype=torch.float32)
                    # Use temperature to control sharpness (lower = more focused)
                    temperature = 0.2
                    probabilities = torch.softmax(scores_tensor / temperature, dim=0)
                    
                    # Assign probabilities to policy matrix
                    for (from_sq, to_sq), prob in zip(move_tuples, probabilities):
                        if 0 <= from_sq < 64 and 0 <= to_sq < 64:
                            policy_matrix[from_sq, to_sq] = prob.item()
            
            # If no moves or all zero, create uniform distribution over legal moves
            if policy_matrix.sum() == 0:
                # This shouldn't happen with proper Stockfish data, but safety fallback
                policy_matrix.fill_(1.0 / (64 * 64))
            
            policy_matrices.append(policy_matrix)
        
        self.policies = torch.stack(policy_matrices)
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx]


class HRMChess(nn.Module):
    def __init__(self, input_dim=72, hidden_dim=192, N=8, T=8):  # Konvolúciós HRM
        super().__init__()
        self.N = N
        self.T = T
        self.hidden_dim = hidden_dim
        
        # 2D Convolutional input processor - egyszerűsített 8x8 sakktáblához
        # Input: 64 mező (8x8) + 8 extra info
        self.board_conv = nn.Sequential(
            # Egyszerű konvolúció 8x8 sakktáblához
            nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1),  # 8x8 → 8x8
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Flatten(),  # 8x8 → 64 * hidden_dim
        )
        
        # Extra info processor (8 dimenziós meta adatok)
        self.extra_processor = nn.Sequential(
            nn.Linear(8, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, hidden_dim//4)
        )
        
        # Combined features processor - frissített dimenzióval
        self.feature_combiner = nn.Sequential(
            nn.Linear(64 * hidden_dim + hidden_dim//4, hidden_dim),  # Egyszerűsített board_conv + extra
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Board representation enhancer - további reprezentációs rétegek
        self.board_enhancer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # L_net: Low-level module (zL, zH, board_features) -> zL
        self.L_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # zL + zH + board_features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # H_net: High-level module (zH, zL) -> zH  
        self.H_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # zH + zL
            nn.ReLU(), 
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Policy head - move prediction
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 64*64)
        )
        
        # Value head eltávolítva
    
    def forward(self, x, z_init=None):
        """
        A konvolúciós HRM modell forward pass-a.
        - Bemenet: 72 dimenziós vektor (64 mező + 8 extra info).
        - A 64 mezőt 2D konvolúcióval dolgozza fel.
        - A kimenet egy policy mátrix (64x64), ami a lépés-valószínűségeket tartalmazza.
        
        Args:
            x: Bemeneti tenzor (batch, 72).
            z_init: Kezdeti HRM állapotok (opcionális).
            
        Returns:
            move_logits: (batch, 64, 64) méretű policy mátrix.
        """
        batch_size = x.size(0)
        device = x.device
        
        # Split input: 64 board squares + 8 extra info
        board_squares = x[:, :64]  # (batch, 64) - sakktábla mezők
        extra_info = x[:, 64:]     # (batch, 8) - meta információk
        
        # Reshape board to 2D for convolution: (batch, 64) → (batch, 1, 8, 8)
        board_2d = board_squares.view(batch_size, 1, 8, 8)
        
        # Process board with simplified 2D convolution
        board_features = self.board_conv(board_2d)  # (batch, 64 * hidden_dim)
        
        # Process extra info
        extra_features = self.extra_processor(extra_info)  # (batch, hidden_dim//4)
        
        # Combine board and extra features
        combined_features = torch.cat([board_features, extra_features], dim=-1)
        board_features = self.feature_combiner(combined_features)  # (batch, hidden_dim)
        
        # Board representation enhancer
        board_features = self.board_enhancer(board_features)  # További feldolgozás
        
        # Initialize z_H and z_L
        if z_init is None:
            z_H = torch.zeros(batch_size, self.hidden_dim, device=device)
            z_L = torch.zeros(batch_size, self.hidden_dim, device=device)
        else:
            z_H, z_L = z_init
        
        # HRM hierarchical processing: N*T-1 steps without gradient
        total_steps = self.N * self.T
        with torch.no_grad():
            for i in range(total_steps - 1):
                # L_net: (zL, zH, board_features) -> zL
                l_input = torch.cat([z_L, z_H, board_features], dim=-1)
                z_L = self.L_net(l_input)
                
                # H_net: every T steps, (zH, zL) -> zH
                if (i + 1) % self.T == 0:
                    h_input = torch.cat([z_H, z_L], dim=-1)
                    z_H = self.H_net(h_input)
        
        # Final step WITH gradient
        l_input = torch.cat([z_L, z_H, board_features], dim=-1)
        z_L = self.L_net(l_input)
        
        h_input = torch.cat([z_H, z_L], dim=-1)
        z_H = self.H_net(h_input)
        
        # Policy prediction from z_H
        move_logits = self.policy_head(z_H).view(batch_size, 64, 64)
        return move_logits


def train_step(model, batch, optimizer, temperature=1.0, n_supervision=1):
    """
    HRM training step, kizárólag policy-tanításhoz.
    - Policy loss: Cross-entropy alapú.
    
    Args:
        model: HRMChess modell.
        batch: (x, pi_star) tuple, ahol x a bemeneti állapot, pi_star a cél policy.
        optimizer: Az optimalizáló.
        temperature: A policy eloszlás élesítésére szolgáló paraméter.
        n_supervision: Deep supervision lépések száma.
    """
    # Policy only training
    x, pi_star = batch[:2]
    
    optimizer.zero_grad()
    
    # Initialize z state
    batch_size = x.size(0)
    device = x.device
    z_H = torch.zeros(batch_size, model.hidden_dim, device=device)
    z_L = torch.zeros(batch_size, model.hidden_dim, device=device)
    z_state = (z_H, z_L)
    
    policy_losses = []
    # Deep Supervision: multiple steps
    for step in range(n_supervision):
        move_logits = model(x, z_state)
        effective_temp = temperature * (0.8 ** step)
        target_probs = pi_star.view(batch_size, -1)
        smoothed_targets = target_probs * 0.995 + 0.005 / target_probs.size(1)
        move_log_probs = F.log_softmax(move_logits.view(batch_size, -1) / effective_temp, dim=1)
        policy_loss = -torch.sum(smoothed_targets * move_log_probs) / batch_size
        policy_losses.append(policy_loss)
        if step < n_supervision - 1:
            with torch.no_grad():
                _ = model(x, z_state)
                z_state = (torch.zeros_like(z_H), torch.zeros_like(z_L))
        policy_loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    # Return loss info
    avg_policy_loss = sum(policy_losses) / n_supervision
    return {
        'total_loss': avg_policy_loss.item()
    }


def train_loop(model, dataset, epochs=25, batch_size=16, lr=2e-4, warmup_epochs=3, device=None):
    """
    Training loop for HRM model with learning rate warmup
    
    Args:
        warmup_epochs: Number of epochs for learning rate warmup (default: 3)
        device: torch device (cuda/cpu)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n🏗️ TRAINING LOOP")
    print(f"   ⚖️ Policy only training (value head removed)")
    print(f"   🔥 Warmup epochs: {warmup_epochs}")
    
    # Train/validation split
    train_size = int(0.85 * len(dataset))  # Nagyobb training set
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Learning rate scheduler with warmup support
    total_steps = len(train_dataloader) * epochs
    warmup_steps = len(train_dataloader) * warmup_epochs
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup from 0 to 1
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine annealing after warmup
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Backup scheduler for validation-based reduction (used only if warmup+cosine fails)
    backup_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=7, factor=0.7, min_lr=1e-6
    )
    
    model.to(device)
    model.train()
    
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 12  # Increased patience due to warmup
    
    print(f"📊 Training: {train_size:,} positions")
    print(f"📊 Validation: {val_size:,} positions")
    print(f"🔥 Total steps: {total_steps:,} (warmup: {warmup_steps:,})")
    
    global_step = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = {'total': 0, 'policy': 0}
        train_batches = 0
        epoch_start_lr = optimizer.param_groups[0]['lr']
        
        for i, batch in enumerate(train_dataloader):
            batch = tuple(b.to(device) for b in batch)
            
            # Policy only training step
            loss_info = train_step(
                model, batch, optimizer,
                temperature=0.9,
                n_supervision=1
            )
            train_losses['total'] += loss_info['total_loss']
            train_batches += 1
                
            # Update learning rate scheduler (step-based for warmup)
            scheduler.step()
            global_step += 1
            
            # Progress reporting
            if i % 1000 == 0 and i > 0:
                current_lr = optimizer.param_groups[0]['lr']
                warmup_progress = min(1.0, global_step / warmup_steps) * 100
                print(f"Epoch {epoch}, Step {i}, Total loss: {loss_info['total_loss']:.4f}, "
                        f"LR: {current_lr:.2e}, Warmup: {warmup_progress:.1f}%")
        
        # Validation phase
        model.eval()
        val_losses = {'total': 0, 'policy': 0}
        val_batches = 0
        with torch.no_grad():
            for batch in val_dataloader:
                batch = tuple(b.to(device) for b in batch)
                x, pi_star, *_ = batch  # ignore value targets if present
                move_logits = model(x)
                batch_size = x.size(0)
                target_probs = pi_star.view(batch_size, -1)
                move_log_probs = F.log_softmax(move_logits.view(batch_size, -1), dim=1)
                total_loss = -torch.sum(target_probs * move_log_probs) / batch_size
                val_losses['total'] += total_loss.item()
                val_batches += 1
        
        # Average losses
        avg_train_total = train_losses['total'] / train_batches
        avg_val_total = val_losses['total'] / val_batches
        
        current_lr = optimizer.param_groups[0]['lr']
        epoch_end_lr = current_lr
        
        # Warmup status
        warmup_status = "🔥 WARMUP" if epoch < warmup_epochs else "📈 COSINE"
        warmup_progress = min(1.0, global_step / warmup_steps) * 100
        
        print(f"📊 Epoch {epoch:2d}/{epochs} {warmup_status} | "
              f"Train loss: {avg_train_total:.4f} | "
              f"Val loss: {avg_val_total:.4f} | "
              f"LR: {epoch_start_lr:.2e}→{epoch_end_lr:.2e}")
        
        if epoch < warmup_epochs:
            print(f"    🔥 Warmup Progress: {warmup_progress:.1f}% ({global_step:,}/{warmup_steps:,} steps)")
        
        # Learning rate scheduling decisions
        if epoch >= warmup_epochs:
            # After warmup, use backup scheduler if needed for validation-based reduction
            backup_scheduler.step(avg_val_total)
        
        # Model saving and early stopping
        if avg_val_total < best_val_loss:
            best_val_loss = avg_val_total
            patience_counter = 0
            
            # Save best model
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'hyperparams': {
                    'hidden_dim': model.hidden_dim,
                    'N': model.N,
                    'T': model.T,
                    'input_dim': 72
                },
                'training_info': {
                    'epoch': epoch,
                    'train_loss': avg_train_total,
                    'val_loss': avg_val_total,
                    'lr': current_lr,
                    'model_type': 'HRM',
                    'warmup_epochs': warmup_epochs,
                    'global_step': global_step
                }
            }
            torch.save(checkpoint, "best_hrm_chess_model.pt")
            print(f"✅ Best model saved! (Val loss: {avg_val_total:.4f})")
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


def quick_train_eval(model, dataset, epochs=2, batch_size=32, lr=2e-4, subset_ratio=0.1, device=None):
    """
    Gyors training és evaluáció Optuna számára
    Csak kis adathalmaz részen, gyors feedback-ért
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Small subset for speed
    subset_size = int(len(dataset) * subset_ratio)
    subset_size = max(subset_size, 500)  # Minimum 500 sample
    
    indices = torch.randperm(len(dataset))[:subset_size]
    subset = torch.utils.data.Subset(dataset, indices)
    
    # Train/val split
    train_size = int(0.8 * len(subset))
    val_size = len(subset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(subset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    model.train()
    
    # Quick training
    for epoch in range(epochs):
        for batch in train_loader:
            batch = tuple(b.to(device) if hasattr(b, 'to') else b for b in batch)
            train_step(model, batch, optimizer, temperature=1.0, n_supervision=1)

    # Quick validation
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = tuple(b.to(device) for b in batch)
            x, pi_star = batch[:2]
            move_logits = model(x)
            batch_size = x.size(0)
            target_probs = pi_star.view(batch_size, -1)
            move_log_probs = F.log_softmax(move_logits.view(batch_size, -1), dim=1)
            policy_loss = -torch.sum(target_probs * move_log_probs) / batch_size
            total_loss += policy_loss.item()
            total_batches += 1

    return total_loss / total_batches if total_batches > 0 else float('inf')
