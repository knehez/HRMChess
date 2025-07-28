"""
HRM Chess Model - Hierarchical Reasoning Machine for Chess
Konvolúciós HRM implementáció Policy+Value head-del
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PolicyValueDataset(torch.utils.data.Dataset):
    """Dataset Policy + Value adatokhoz"""
    
    def __init__(self, states, policies, values):
        """
        Args:
            states: Board states (tensors)
            policies: Move policies (from_square, to_square)
            values: Position values [-1, 1]
        """
        self.states = torch.FloatTensor(states)
        self.values = torch.FloatTensor(values).unsqueeze(1)  # (N, 1)
        
        # Convert policies to one-hot matrix
        policy_matrices = []
        for from_sq, to_sq in policies:
            policy_matrix = torch.zeros(64, 64)
            policy_matrix[from_sq, to_sq] = 1.0
            policy_matrices.append(policy_matrix)
        
        self.policies = torch.stack(policy_matrices)
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]


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
        self.policy_head = nn.Linear(hidden_dim, 64*64)
        
        # Value head - position evaluation
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, 1),
            nn.Tanh()  # Output [-1, 1]
        )
    
    def forward(self, x, z_init=None, return_value=True):
        """
        Konvolúciós HRM implementation Policy+Value head-del:
        - Input: 72 dimenziós vektor (64 mező + 8 extra info)
        - 2D Konvolúció a sakktábla reprezentációhoz
        - Output: Policy (64x64) és opcionálisan Value (1)
        
        Args:
            x: Input tensor (batch, 72)
            z_init: Initial HRM states (optional)
            return_value: Ha True, visszaadja a value predikciót is
            
        Returns:
            move_logits: (batch, 64, 64) policy matrix
            values: (batch, 1) position values (ha return_value=True)
        """
        batch_size = x.size(0)
        device = x.device
        
        # Split input: 64 board squares + 8 extra info
        board_squares = x[:, :64]  # (batch, 64) - sakktábla mezők
        extra_info = x[:, 64:]     # (batch, 8) - meta információk
        
        # Reshape board to 2D for convolution: (batch, 64) → (batch, 1, 8, 8)
        board_2d = board_squares.view(batch_size, 1, 8, 8)
        
        # Process board with simplified 2D convolution
        board_features = self.board_conv(board_2d)  # (batch, 4 * hidden_dim//8)
        
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
        
        if return_value:
            # Value prediction from z_H
            values = self.value_head(z_H)  # (batch, 1)
            return move_logits, values
        else:
            return move_logits


def train_step(model, batch, optimizer, temperature=1.0, n_supervision=1, policy_weight=1.0, value_weight=0.5):
    """
    HRM training step Policy+Value támogatással:
    - Policy loss: Cross entropy
    - Value loss: MSE (opcionális)
    - Combined loss weighting
    
    Args:
        model: HRMChess model
        batch: (x, pi_star) vagy (x, pi_star, v_star) tuple
        optimizer: Optimizer
        temperature: Policy sharpening
        n_supervision: Deep supervision steps
        policy_weight: Policy loss weight
        value_weight: Value loss weight (0 = csak policy)
    """
    # Check if we have value targets
    has_value_targets = len(batch) == 3
    
    if has_value_targets:
        x, pi_star, v_star = batch
    else:
        x, pi_star = batch
        v_star = None
    
    optimizer.zero_grad()
    
    # Initialize z state
    batch_size = x.size(0)
    device = x.device
    z_H = torch.zeros(batch_size, model.hidden_dim, device=device)
    z_L = torch.zeros(batch_size, model.hidden_dim, device=device)
    z_state = (z_H, z_L)
    
    total_loss = 0.0
    policy_loss_total = 0.0
    value_loss_total = 0.0
    
    # Deep Supervision: multiple steps
    for step in range(n_supervision):
        # Forward pass - get both policy and value if available
        if has_value_targets:
            move_logits, values = model(x, z_state, return_value=True)
        else:
            move_logits = model(x, z_state, return_value=False)
        
        # Progressively lower temperature for sharper predictions
        effective_temp = temperature * (0.8 ** step)
        
        # Policy loss: cross entropy with label smoothing
        target_probs = pi_star.view(batch_size, -1)
        
        # Label smoothing for better generalization - ULTRA CSÖKKENTETT
        smoothed_targets = target_probs * 0.995 + 0.005 / target_probs.size(1)
        
        move_log_probs = F.log_softmax(move_logits.view(batch_size, -1) / effective_temp, dim=1)
        policy_loss = -torch.sum(smoothed_targets * move_log_probs) / batch_size
        
        # Combined loss
        step_loss = policy_weight * policy_loss
        policy_loss_total += policy_loss.item()
        
        # Value loss (if targets available)
        if has_value_targets and value_weight > 0:
            value_loss = F.mse_loss(values, v_star)
            step_loss += value_weight * value_loss
            value_loss_total += value_loss.item()
        
        total_loss += step_loss.item()
        
        # Detach z state for next supervision step
        if step < n_supervision - 1:
            with torch.no_grad():
                _ = model(x, z_state, return_value=False)
                z_state = (torch.zeros_like(z_H), torch.zeros_like(z_L))
        
        step_loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    # Return loss info
    loss_info = {
        'total_loss': total_loss / n_supervision,
        'policy_loss': policy_loss_total / n_supervision,
        'has_value': has_value_targets
    }
    
    if has_value_targets and value_weight > 0:
        loss_info['value_loss'] = value_loss_total / n_supervision
    
    return loss_info


def train_loop(model, dataset, epochs=25, batch_size=16, lr=2e-4, policy_weight=1.0, value_weight=0.5, warmup_epochs=3, device=None):
    """
    Training loop for HRM model with learning rate warmup
    
    Args:
        warmup_epochs: Number of epochs for learning rate warmup (default: 3)
        device: torch device (cuda/cpu)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n🏗️ TRAINING LOOP")
    print(f"   ⚖️ Policy weight: {policy_weight}")
    print(f"   ⚖️ Value weight: {value_weight}")
    print(f"   🔥 Warmup epochs: {warmup_epochs}")
    
    # Train/validation split
    train_size = int(0.85 * len(dataset))  # Nagyobb training set policy+value-hoz
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
        train_losses = {'total': 0, 'policy': 0, 'value': 0}
        train_batches = 0
        epoch_start_lr = optimizer.param_groups[0]['lr']
        
        for i, batch in enumerate(train_dataloader):
            batch = tuple(b.to(device) for b in batch)
            
            # Policy+Value training step
            loss_info = train_step(
                model, batch, optimizer,
                temperature=0.9,
                n_supervision=1,
                policy_weight=policy_weight,
                value_weight=value_weight
            )
            
            train_losses['total'] += loss_info['total_loss']
            train_losses['policy'] += loss_info['policy_loss']
            if 'value_loss' in loss_info:
                train_losses['value'] += loss_info['value_loss']
            train_batches += 1
            
            # Update learning rate scheduler (step-based for warmup)
            scheduler.step()
            global_step += 1
            
            # Progress reporting
            if i % 1000 == 0 and i > 0:
                current_lr = optimizer.param_groups[0]['lr']
                warmup_progress = min(1.0, global_step / warmup_steps) * 100
                print(f"Epoch {epoch}, Step {i}, Total: {loss_info['total_loss']:.4f}, "
                      f"Policy: {loss_info['policy_loss']:.4f}, "
                      f"Value: {loss_info.get('value_loss', 0):.4f}, "
                      f"LR: {current_lr:.2e}, Warmup: {warmup_progress:.1f}%")
        
        # Validation phase
        model.eval()
        val_losses = {'total': 0, 'policy': 0, 'value': 0}
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                batch = tuple(b.to(device) for b in batch)
                x, pi_star, v_star = batch
                
                move_logits, values = model(x, return_value=True)
                
                # Policy loss
                batch_size = x.size(0)
                target_probs = pi_star.view(batch_size, -1)
                move_log_probs = F.log_softmax(move_logits.view(batch_size, -1), dim=1)
                policy_loss = -torch.sum(target_probs * move_log_probs) / batch_size
                
                # Value loss
                value_loss = F.mse_loss(values, v_star)
                
                # Combined loss
                total_loss = policy_weight * policy_loss + value_weight * value_loss
                
                val_losses['total'] += total_loss.item()
                val_losses['policy'] += policy_loss.item()
                val_losses['value'] += value_loss.item()
                val_batches += 1
        
        # Average losses
        avg_train_total = train_losses['total'] / train_batches
        avg_train_policy = train_losses['policy'] / train_batches
        avg_train_value = train_losses['value'] / train_batches
        
        avg_val_total = val_losses['total'] / val_batches
        avg_val_policy = val_losses['policy'] / val_batches
        avg_val_value = val_losses['value'] / val_batches
        
        current_lr = optimizer.param_groups[0]['lr']
        epoch_end_lr = current_lr
        
        # Warmup status
        warmup_status = "🔥 WARMUP" if epoch < warmup_epochs else "📈 COSINE"
        warmup_progress = min(1.0, global_step / warmup_steps) * 100
        
        print(f"📊 Epoch {epoch:2d}/{epochs} {warmup_status} | "
              f"Train: {avg_train_total:.4f}(P:{avg_train_policy:.4f}+V:{avg_train_value:.4f}) | "
              f"Val: {avg_val_total:.4f}(P:{avg_val_policy:.4f}+V:{avg_val_value:.4f}) | "
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
                    'policy_weight': policy_weight,
                    'value_weight': value_weight,
                    'model_type': 'HRM_Policy_Value_Warmup',
                    'warmup_epochs': warmup_epochs,
                    'global_step': global_step
                }
            }
            torch.save(checkpoint, "best_hrm_policy_value_chess_model.pt")
            print(f"✅ Best Policy+Value model saved! (Val loss: {avg_val_total:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"⏹️ Early stopping at epoch {epoch} (patience exhausted)")
                break
    
    print(f"\n🎉 Policy+Value training with warmup completed!")
    print(f"📁 Best model: best_hrm_policy_value_chess_model.pt")
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
            loss = train_step(model, batch, optimizer, temperature=1.0, n_supervision=1)
    
    # Quick validation
    model.eval()
    total_loss = 0.0
    total_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = tuple(b.to(device) for b in batch)
            
            # Check if we have policy+value data (3 elements) or just policy data (2 elements)
            if len(batch) == 3:
                x, pi_star, v_star = batch
                move_logits, values = model(x, return_value=True)
                
                # Policy loss
                batch_size = x.size(0)
                target_probs = pi_star.view(batch_size, -1)
                move_log_probs = F.log_softmax(move_logits.view(batch_size, -1), dim=1)
                policy_loss = -torch.sum(target_probs * move_log_probs) / batch_size
                
                # Value loss
                value_loss = F.mse_loss(values, v_star)
                
                # Combined loss
                loss = policy_loss + 0.5 * value_loss
            else:
                x, pi_star = batch
                move_logits = model(x, return_value=False)
                target_probs = pi_star.view(x.size(0), -1)
                move_log_probs = F.log_softmax(move_logits.view(x.size(0), -1), dim=1)
                loss = -torch.sum(target_probs * move_log_probs) / x.size(0)
            
            total_loss += loss.item()
            total_batches += 1
    
    return total_loss / total_batches if total_batches > 0 else float('inf')
