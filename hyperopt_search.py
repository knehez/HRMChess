"""
Hyperparameter Optimization for HRMChess with Optuna
Progressive dataset size scaling and multi-objective optimization
"""
import os
import optuna
import torch
import numpy as np
import time
import json
import gc
from pathlib import Path
from hrm_model import PureViTChess, ValueBinDataset, train_step
import torch.nn.functional as F

# Set CUDA memory allocation to help prevent fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

class ProgressiveHyperOptimizer:
    def __init__(self, dataset_path='game_history_dataset.pt', 
                 min_samples=1024, max_samples=1000000, 
                 stages=5, max_epochs_per_stage=10):
        """
        Progressive hyperparameter optimization
        
        Args:
            dataset_path: Path to dataset
            min_samples: Starting dataset size
            max_samples: Maximum dataset size
            stages: Number of progressive stages
            max_epochs_per_stage: Max epochs per trial
        """
        self.dataset_path = dataset_path
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.stages = stages
        self.max_epochs_per_stage = max_epochs_per_stage
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load full dataset once
        print("🔄 Loading full dataset...")
        dataset_info = torch.load(dataset_path, weights_only=False)
        self.full_data = dataset_info['data']
        print(f"📊 Loaded {len(self.full_data):,} samples")
        
        # Progressive sample sizes
        self.sample_sizes = np.geomspace(min_samples, max_samples, stages).astype(int)
        print(f"📈 Progressive stages: {[f'{s:,}' for s in self.sample_sizes]}")
    
    def clear_gpu_memory(self):
        """Forcefully clear GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            time.sleep(1)  # Give GPU time to release memory
    
    def create_model_and_dataset(self, trial, sample_size):
        """Create model and dataset for trial"""
        
        # Clear GPU memory first
        self.clear_gpu_memory()
        
        # Hyperparameters to optimize with validation
        hidden_dim = trial.suggest_categorical('hidden_dim', [192, 256, 384])  # Nagyobb modellek is
        n_layers = trial.suggest_int('n_layers', 6, 10)  # Mélyebb modellek
        n_heads = trial.suggest_categorical('n_heads', [6, 8, 12, 16, 24])  # Több head
        
        # Validate hidden_dim % n_heads == 0
        if hidden_dim % n_heads != 0:
            raise ValueError(f"embed_dim ({hidden_dim}) must be divisible by num_heads ({n_heads})")
        
        ff_mult = trial.suggest_categorical('ff_mult', [4, 6])
        dropout = trial.suggest_float('dropout', 0.1, 0.25)  # Kisebb dropout range nagy adatnál
        
        # Training hyperparameters
        lr = trial.suggest_float('lr', 5e-6, 1e-4, log=True)  # Kisebb LR range nagy adatnál
        batch_size = trial.suggest_categorical('batch_size', [128])  # Nagyobb batch sizes
        weight_decay = trial.suggest_float('weight_decay', 1e-7, 5e-4, log=True)  # Kisebb weight decay
        
        # Memory check: skip very large models to avoid OOM (adjusted for larger datasets)
        model_memory_est = hidden_dim * n_layers * ff_mult * batch_size
        if model_memory_est > 5000000:  # Nagyobb threshold nagyobb adatokhoz
            raise ValueError(f"Model too large (estimated memory: {model_memory_est})")
        
        # Create model with mixed precision (not explicit float16)
        model = PureViTChess(
            num_bins=128,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            ff_mult=ff_mult,
            dropout=dropout
        ).to(self.device)  # Keep model in float32, use autocast for efficiency
        
        # Create dataset subset
        subset_data = self.full_data[:sample_size]
        dataset = ValueBinDataset(subset_data, num_bins=128)
        
        # Train/val split
        train_size = int(0.85 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # DataLoaders - reduced num_workers and use mixed precision
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=4, pin_memory=True
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True
        )
        
        # Optimizer and gradient scaler for float16
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scaler = torch.amp.GradScaler('cuda')
        
        return model, train_dataloader, val_dataloader, optimizer, scaler
    
    def train_and_evaluate(self, trial, sample_size, stage):
        """Train model and return validation metrics"""
        
        print(f"\n🔬 Trial {trial.number}, Stage {stage+1}/{self.stages}")
        print(f"📊 Sample size: {sample_size:,}")
        print(f"🎯 Parameters: {trial.params}")
        
        model, train_loader, val_loader, optimizer, scaler = self.create_model_and_dataset(
            trial, sample_size
        )
        
        # Training with early stopping and pruning
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 3
        
        # Adaptive epochs based on sample size - more epochs for small datasets
        if sample_size < 5000:
            max_epochs = 8
        elif sample_size < 50000:
            max_epochs = 6
        else:
            max_epochs = min(self.max_epochs_per_stage, 4)
        
        for epoch in range(max_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch in train_loader:
                batch = tuple(b.to(self.device) for b in batch)  # Keep original dtype
                
                # Use automatic mixed precision for memory efficiency
                loss_info = train_step(model, batch, optimizer, scaler=scaler, use_amp=True)
                
                train_loss += loss_info['total_loss']
                train_batches += 1
            
            avg_train_loss = train_loss / train_batches
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_batches = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = tuple(b.to(self.device) for b in batch)  # Keep original dtype
                    bitplanes, targets = batch[:2]
                    
                    with torch.amp.autocast('cuda'):  # Use autocast for validation too
                        logits = model(bitplanes)
                        loss = F.cross_entropy(logits, targets)
                    
                    val_loss += loss.item()
                    val_batches += 1
                    
                    # Accuracy
                    preds = torch.argmax(logits, dim=1)
                    val_correct += (preds == targets).sum().item()
                    val_total += targets.size(0)
            
            avg_val_loss = val_loss / val_batches
            val_accuracy = val_correct / val_total
            
            print(f"  Epoch {epoch+1}/{max_epochs}: "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, "
                  f"Val Acc: {val_accuracy:.3f}, {time.strftime('%H:%M:%S')}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
            
            # Optuna pruning (intermediate value reporting)
            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                print(f"  Trial pruned at epoch {epoch+1}")
                raise optuna.TrialPruned()
        
        # Cleanup - thorough memory cleanup
        del model, train_loader, val_loader, optimizer, scaler
        self.clear_gpu_memory()
        
        print(f"  ✅ Best Val Loss: {best_val_loss:.4f}")
        return best_val_loss, val_accuracy
    
    def objective(self, trial):
        """Optuna objective function - multi-stage progressive training with retry logic"""
        
        stage_losses = []
        stage_accuracies = []
        
        for stage, sample_size in enumerate(self.sample_sizes):
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    val_loss, val_acc = self.train_and_evaluate(trial, sample_size, stage)
                    stage_losses.append(val_loss)
                    stage_accuracies.append(val_acc)
                    break  # Success, exit retry loop
                    
                except optuna.TrialPruned:
                    # If pruned, use partial results
                    break
                    
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    retry_count += 1
                    if "CUDA out of memory" in str(e) or "out of memory" in str(e).lower():
                        print(f"  ⚠️ CUDA OOM error (attempt {retry_count}/{max_retries})")
                        if retry_count < max_retries:
                            print(f"  🔄 Retrying in {retry_count * 2} seconds...")
                            time.sleep(retry_count * 2)  # Progressive backoff
                            self.clear_gpu_memory()
                            continue
                        else:
                            print(f"  ❌ Max retries reached. Error in stage {stage+1}: {e}")
                            return float('inf')
                    else:
                        print(f"  ❌ Runtime error in stage {stage+1}: {e}")
                        return float('inf')
                        
                except ValueError as e:
                    print(f"  ❌ Architecture error in stage {stage+1}: {e}")
                    return float('inf')  # Don't retry architecture errors
                    
                except Exception as e:
                    retry_count += 1
                    print(f"  ❌ Unexpected error in stage {stage+1} (attempt {retry_count}/{max_retries}): {e}")
                    if retry_count < max_retries:
                        print(f"  🔄 Retrying in {retry_count * 2} seconds...")
                        time.sleep(retry_count * 2)
                        self.clear_gpu_memory()
                        continue
                    else:
                        print("  ❌ Max retries reached.")
                        return float('inf')
            
            # Early termination if loss is too high on smaller datasets
            if stage_losses:  # Only check if we have results
                # Lazább thresholdok nagyobb dataset optimalizáláshoz
                early_termination_thresholds = {0: 4.8, 1: 4.0, 2: 3.5, 3: 3.0}
                current_loss = stage_losses[-1]
                if stage in early_termination_thresholds and current_loss > early_termination_thresholds[stage]:
                    stage_name = ["very early", "early", "mid", "late"][stage] if stage < 4 else "final"
                    print(f"  ⚠️ {stage_name.title()} stage high loss ({current_loss:.3f}) on stage {stage+1}, terminating trial")
                    break
        
        if not stage_losses:
            return float('inf')
        
        # Multi-objective: focus on final stage loss but penalize early stage failures
        final_loss = stage_losses[-1]
        avg_loss = np.mean(stage_losses)
        
        # Weighted objective: 70% final stage, 30% average across stages
        objective_value = 0.7 * final_loss + 0.3 * avg_loss
        
        print(f"  🎯 Objective Value: {objective_value:.4f} "
              f"(Final: {final_loss:.4f}, Avg: {avg_loss:.4f})")
        
        return objective_value
    
    def optimize(self, n_trials=50, study_name="hrmchess_hyperopt"):
        """Run hyperparameter optimization"""
        
        print("🚀 Starting Progressive Hyperparameter Optimization")
        print(f"📊 Trials: {n_trials}")
        print(f"🎯 Target: Find best parameters for {self.max_samples:,} samples")
        
        # Create Optuna study
        study = optuna.create_study(
            direction='minimize',
            study_name=study_name,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=3
            ),
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Run optimization
        start_time = time.time()
        study.optimize(self.objective, n_trials=n_trials)
        elapsed_time = time.time() - start_time
        
        print(f"\n🎉 Optimization completed in {elapsed_time/3600:.1f} hours")
        
        # Results
        best_trial = study.best_trial
        print(f"\n🏆 Best Trial #{best_trial.number}")
        print(f"🎯 Best Objective Value: {best_trial.value:.4f}")
        print("🔧 Best Parameters:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")
        
        # Save results
        results = {
            'best_trial_number': best_trial.number,
            'best_value': best_trial.value,
            'best_params': best_trial.params,
            'optimization_time_hours': elapsed_time / 3600,
            'n_trials_completed': len(study.trials),
            'sample_sizes': self.sample_sizes.tolist()
        }
        
        results_path = f"hyperopt_results_{study_name}_{int(time.time())}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to: {results_path}")
        
        # Plot optimization history (if matplotlib available)
        try:
            import matplotlib.pyplot as plt
            optuna.visualization.matplotlib.plot_optimization_history(study)
            plt.savefig(f"hyperopt_history_{study_name}.png", dpi=150, bbox_inches='tight')
            
            optuna.visualization.matplotlib.plot_param_importances(study)
            plt.savefig(f"hyperopt_importance_{study_name}.png", dpi=150, bbox_inches='tight')
            print("📊 Plots saved: hyperopt_history.png, hyperopt_importance.png")
        except ImportError:
            print("📊 Install matplotlib for optimization plots")
        
        return study, best_trial.params


def main():
    """Run hyperparameter optimization"""
    
    # Configuration
    optimizer = ProgressiveHyperOptimizer(
        dataset_path='game_history_dataset.pt',
        min_samples=8192,      # Start small for quick feedback
        max_samples=10000000,  # Target size - full dataset
        stages=5,              # More progressive stages for better scaling
        max_epochs_per_stage=6 # Reduced epochs for larger datasets
    )
    
    # Run optimization
    _, best_params = optimizer.optimize(n_trials=20)
    
    print("\n🎯 Recommended hyperparameters for large dataset:")
    print(f"   Hidden Dim: {best_params['hidden_dim']}")
    print(f"   Layers: {best_params['n_layers']}")
    print(f"   Heads: {best_params['n_heads']}")
    print(f"   FF Mult: {best_params['ff_mult']}")
    print(f"   Dropout: {best_params['dropout']:.3f}")
    print(f"   Learning Rate: {best_params['lr']:.2e}")
    print(f"   Batch Size: {best_params['batch_size']}")
    print(f"   Weight Decay: {best_params['weight_decay']:.2e}")


if __name__ == "__main__":
    main()
