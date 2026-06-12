# pyrefly: ignore [missing-import]
import torch
import logging
import os
import argparse
from typing import Dict
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

from worlds import (
    SpatialShiftWorld, 
    TemporalDelayWorld, 
    ContextInversionWorld, 
    IQPatternWorld,
    AbstractWorld,
    FractalIQEngine
)
from model import UniversalMicroUnit, IQMicroUnit
from trainer import MultiTaskTrainer, IQMetaTrainer, ProceduralIQTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def plot_phase1_metrics(history: Dict, filepath: str) -> None:
    """
    Plots the separate loss and accuracy curves for Phase 1 and saves to file.
    """
    epochs = range(1, len(history['spatial_shift']['loss']) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Micro-World Intelligence Benchmark - Phase 1 Convergence Curves", fontsize=16, fontweight='bold')
    
    # Row 0: Losses
    axes[0, 0].plot(epochs, history['spatial_shift']['loss'], color='#1f77b4', linewidth=2)
    axes[0, 0].set_title('SpatialShiftWorld Loss (MSE)', fontsize=12, fontweight='semibold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, linestyle='--', alpha=0.6)
    
    axes[0, 1].plot(epochs, history['temporal_delay']['loss'], color='#2ca02c', linewidth=2)
    axes[0, 1].set_title('TemporalDelayWorld Loss (BCE)', fontsize=12, fontweight='semibold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, linestyle='--', alpha=0.6)
    
    axes[0, 2].plot(epochs, history['context_inversion']['loss'], color='#ff7f0e', linewidth=2)
    axes[0, 2].set_title('ContextInversionWorld Loss (BCE)', fontsize=12, fontweight='semibold')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].grid(True, linestyle='--', alpha=0.6)
    
    # Row 1: Accuracies
    axes[1, 0].plot(epochs, history['spatial_shift']['accuracy'], color='#1f77b4', linewidth=2)
    axes[1, 0].set_title('SpatialShiftWorld Accuracy (Err < 0.5)', fontsize=12, fontweight='semibold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].grid(True, linestyle='--', alpha=0.6)
    
    axes[1, 1].plot(epochs, history['temporal_delay']['accuracy'], label='Step Acc', color='#2ca02c', linewidth=2)
    axes[1, 1].plot(epochs, history['temporal_delay']['seq_accuracy'], label='Seq Acc', color='#1a5f1a', linewidth=2, linestyle='--')
    axes[1, 1].set_title('TemporalDelayWorld Accuracy', fontsize=12, fontweight='semibold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_ylim(0.0, 1.05)
    axes[1, 1].legend(loc='lower right')
    axes[1, 1].grid(True, linestyle='--', alpha=0.6)
    
    axes[1, 2].plot(epochs, history['context_inversion']['accuracy'], color='#ff7f0e', linewidth=2)
    axes[1, 2].set_title('ContextInversionWorld Accuracy', fontsize=12, fontweight='semibold')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Accuracy')
    axes[1, 2].set_ylim(0.0, 1.05)
    axes[1, 2].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(filepath, dpi=300)
    plt.close()
    logging.info(f"Convergence curves saved to {filepath}")


def plot_phase2_metrics(history: Dict, filepath: str) -> None:
    """
    Plots the separate loss and accuracy curves for Phase 2 and saves to file.
    """
    epochs = range(1, len(history['arithmetic']['loss']) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Meta-Learning IQ-Module - Phase 2 Convergence Curves", fontsize=16, fontweight='bold')
    
    # Row 0: Losses
    axes[0, 0].plot(epochs, history['arithmetic']['loss'], color='#1f77b4', linewidth=2)
    axes[0, 0].set_title('Arithmetic Progression Loss (MSE)', fontsize=12, fontweight='semibold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, linestyle='--', alpha=0.6)
    
    axes[0, 1].plot(epochs, history['cyclic']['loss'], color='#2ca02c', linewidth=2)
    axes[0, 1].set_title('Cyclic Bit Permutation Loss (BCE)', fontsize=12, fontweight='semibold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, linestyle='--', alpha=0.6)
    
    axes[0, 2].plot(epochs, history['bitwise']['loss'], color='#ff7f0e', linewidth=2)
    axes[0, 2].set_title('Bitwise Inversion Loss (BCE)', fontsize=12, fontweight='semibold')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].grid(True, linestyle='--', alpha=0.6)
    
    # Row 1: Accuracies
    axes[1, 0].plot(epochs, history['arithmetic']['accuracy'], color='#1f77b4', linewidth=2)
    axes[1, 0].set_title('Arithmetic Accuracy (Err < 0.25)', fontsize=12, fontweight='semibold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].grid(True, linestyle='--', alpha=0.6)
    
    axes[1, 1].plot(epochs, history['cyclic']['accuracy'], color='#2ca02c', linewidth=2)
    axes[1, 1].set_title('Cyclic Permutation Accuracy', fontsize=12, fontweight='semibold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_ylim(0.0, 1.05)
    axes[1, 1].grid(True, linestyle='--', alpha=0.6)
    
    axes[1, 2].plot(epochs, history['bitwise']['accuracy'], color='#ff7f0e', linewidth=2)
    axes[1, 2].set_title('Bitwise Inversion Accuracy', fontsize=12, fontweight='semibold')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Accuracy')
    axes[1, 2].set_ylim(0.0, 1.05)
    axes[1, 2].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(filepath, dpi=300)
    plt.close()
    logging.info(f"Phase 2 curves saved to {filepath}")


def save_model(model: torch.nn.Module, base_name: str) -> None:
    """
    Saves the model state dict to base_name_{number}.pt.
    If the file already exists, increments the number by 1.
    """
    import os
    i = 1
    while True:
        filename = f"{base_name}_{i}.pt"
        if not os.path.exists(filename):
            break
        i += 1
    torch.save(model.state_dict(), filename)
    logging.info(f"Model successfully saved to {filename}")


def run_phase1() -> None:
    # Hyperparameters
    hidden_dim = 64
    batch_size = 64
    steps_per_epoch = 10
    epochs = 1000
    learning_rate = 1e-3
    weight_decay = 1e-5
    
    logging.info("Initializing Micro-World Environments...")
    worlds: Dict[str, AbstractWorld] = {
        'spatial_shift': SpatialShiftWorld(),
        'temporal_delay': TemporalDelayWorld(sequence_length=10, delay=3),
        'context_inversion': ContextInversionWorld()
    }
    
    logging.info(f"Building UniversalMicroUnit Model (hidden_dim={hidden_dim})...")
    model = UniversalMicroUnit(hidden_dim=hidden_dim)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model created. Total trainable parameters: {total_params}")
    
    logging.info("Initializing Multi-Task Trainer...")
    trainer = MultiTaskTrainer(
        model=model,
        worlds=worlds,
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    logging.info(f"Starting Multi-Task training for {epochs} epochs...")
    
    completed = False
    try:
        for epoch in range(1, epochs + 1):
            train_losses = trainer.train_epoch(batch_size=batch_size, steps_per_epoch=steps_per_epoch)
            val_metrics = trainer.evaluate(val_batch_size=512)
            
            if epoch % 50 == 0 or epoch == 1 or epoch == epochs:
                logging.info(
                    f"Epoch {epoch:04d}/{epochs} | "
                    f"Losses -> Spatial: {train_losses['spatial_shift']:.4f}, "
                    f"Temporal: {train_losses['temporal_delay']:.4f}, "
                    f"Context: {train_losses['context_inversion']:.4f} | "
                    f"Val Acc -> Spatial: {val_metrics['spatial_shift']['accuracy']*100:.1f}%, "
                    f"Temporal (Step/Seq): {val_metrics['temporal_delay']['accuracy']*100:.1f}%/{val_metrics['temporal_delay']['seq_accuracy']*100:.1f}%, "
                    f"Context: {val_metrics['context_inversion']['accuracy']*100:.1f}%"
                )
        completed = True
    except KeyboardInterrupt:
        logging.info("Training interrupted by user. Evaluating current state...")
    finally:
        plot_filepath = os.path.join(os.getcwd(), "metrics.png")
        if len(trainer.history['spatial_shift']['loss']) > 0:
            plot_phase1_metrics(trainer.history, plot_filepath)
        save_model(model, "multitask_model")
            
    if len(trainer.history['spatial_shift']['loss']) > 0:
        logging.info("==================================================")
        logging.info("FINAL EVALUATION METRICS ON HOLDOUT VALIDATION SET")
        logging.info("==================================================")
        final_val = trainer.evaluate(val_batch_size=1000)
        logging.info(f"World 1: SpatialShiftWorld (Object Permanence) Accuracy : {final_val['spatial_shift']['accuracy']*100:.2f}%")
        logging.info(f"World 2: TemporalDelayWorld (Working Memory) Step Acc   : {final_val['temporal_delay']['accuracy']*100:.2f}%")
        logging.info(f"World 2: TemporalDelayWorld (Working Memory) Seq Acc    : {final_val['temporal_delay']['seq_accuracy']*100:.2f}%")
        logging.info(f"World 3: ContextInversionWorld (Rule Switching) Accuracy: {final_val['context_inversion']['accuracy']*100:.2f}%")
        logging.info("==================================================")
    else:
        logging.warning("No training history available. Skipping evaluation.")


def run_phase2() -> None:
    # Hyperparameters
    hidden_dim = 64
    batch_size = 64
    steps_per_epoch = 10
    epochs = 1000
    learning_rate = 1e-3
    weight_decay = 1e-5
    
    logging.info("Initializing IQPatternWorld Environments for Phase 2...")
    worlds: Dict[str, AbstractWorld] = {
        'arithmetic': IQPatternWorld('arithmetic'),
        'cyclic': IQPatternWorld('cyclic'),
        'bitwise': IQPatternWorld('bitwise')
    }
    
    logging.info(f"Building IQMicroUnit Model (hidden_dim={hidden_dim})...")
    model = IQMicroUnit(hidden_dim=hidden_dim)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model created. Total trainable parameters: {total_params}")
    
    logging.info("Initializing IQ Meta-Trainer...")
    trainer = IQMetaTrainer(
        model=model,
        worlds=worlds,
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    logging.info(f"Starting Meta-Learning training for {epochs} epochs...")
    
    completed = False
    try:
        for epoch in range(1, epochs + 1):
            train_losses = trainer.train_epoch(batch_size=batch_size, steps_per_epoch=steps_per_epoch)
            val_metrics = trainer.evaluate(val_batch_size=512)
            
            if epoch % 50 == 0 or epoch == 1 or epoch == epochs:
                logging.info(
                    f"Epoch {epoch:04d}/{epochs} | "
                    f"Losses -> Arithmetic: {train_losses['arithmetic']:.4f}, "
                    f"Cyclic: {train_losses['cyclic']:.4f}, "
                    f"Bitwise: {train_losses['bitwise']:.4f} | "
                    f"Val Acc -> Arithmetic: {val_metrics['arithmetic']['accuracy']*100:.1f}%, "
                    f"Cyclic: {val_metrics['cyclic']['accuracy']*100:.1f}%, "
                    f"Bitwise: {val_metrics['bitwise']['accuracy']*100:.1f}%"
                )
        completed = True
    except KeyboardInterrupt:
        logging.info("Training interrupted by user. Evaluating current state...")
    finally:
        # Save the convergence plots
        plot_filepath = os.path.join(os.getcwd(), "iq_meta_metrics.png")
        if len(trainer.history['arithmetic']['loss']) > 0:
            plot_phase2_metrics(trainer.history, plot_filepath)
        save_model(model, "iq_model")
            
    if len(trainer.history['arithmetic']['loss']) > 0:
        logging.info("==================================================")
        logging.info("FINAL EVALUATION METRICS ON HOLDOUT VALIDATION SET")
        logging.info("==================================================")
        final_val_normal = trainer.evaluate(val_batch_size=1000, ablate_context=False)
        logging.info(f"Rule 1: Arithmetic Progression Accuracy (Normal) : {final_val_normal['arithmetic']['accuracy']*100:.2f}%")
        logging.info(f"Rule 2: Cyclic Bit Permutation Accuracy (Normal) : {final_val_normal['cyclic']['accuracy']*100:.2f}%")
        logging.info(f"Rule 3: Bitwise Inversion Context Accuracy (Normal): {final_val_normal['bitwise']['accuracy']*100:.2f}%")
        
        logging.info("==================================================")
        logging.info("CONTEXT ABLATION TEST (UNIT TEST)")
        logging.info("==================================================")
        # The ablation test presents the model with mismatched/shuffled contexts.
        # If the model uses context for rule extraction, accuracy should drop to near 0%.
        final_val_ablated = trainer.evaluate(val_batch_size=1000, ablate_context=True)
        logging.info(f"Rule 1: Arithmetic Progression Accuracy (Ablated) : {final_val_ablated['arithmetic']['accuracy']*100:.2f}%")
        logging.info(f"Rule 2: Cyclic Bit Permutation Accuracy (Ablated) : {final_val_ablated['cyclic']['accuracy']*100:.2f}%")
        logging.info(f"Rule 3: Bitwise Inversion Context Accuracy (Ablated): {final_val_ablated['bitwise']['accuracy']*100:.2f}%")
        logging.info("==================================================")

        # Assertions for Unit Test
        is_meta_learning = True
        # Define task-specific ablated threshold constraints based on chance-level probability
        thresholds = {
            'arithmetic': 0.20,
            'cyclic': 0.35,  # 25% chance of random shift collision + constant vector matches (~29.7%)
            'bitwise': 0.20
        }
        
        for rule in ['arithmetic', 'cyclic', 'bitwise']:
            normal_acc = final_val_normal[rule]['accuracy']
            ablated_acc = final_val_ablated[rule]['accuracy']
            limit = thresholds[rule]
            logging.info(f"Rule '{rule}': normal={normal_acc*100:.1f}%, ablated={ablated_acc*100:.1f}% (threshold < {limit*100:.1f}%)")
            
            if normal_acc < 0.85:
                is_meta_learning = False
                logging.warning(f"Normal accuracy for {rule} is below threshold (0.85)")
            if ablated_acc > limit:
                is_meta_learning = False
                logging.warning(f"Ablated accuracy for {rule} is above threshold ({limit})")

        if is_meta_learning:
            logging.info("UNIT TEST RESULT: Context Ablation Check -> PASSED!")
        else:
            logging.error("UNIT TEST RESULT: Context Ablation Check -> FAILED!")
            if completed:
                raise RuntimeError("Model failed context ablation check. It may be memorizing inputs rather than inducing in-context rules.")
            else:
                logging.warning("Unit test assertion check was not passed, but exception is bypassed due to early termination.")
    else:
        logging.warning("No training history available. Skipping evaluation.")


def plot_phase3_metrics(history: Dict, filepath: str) -> None:
    """
    Plots the train loss, validation loss, and validation accuracy for the Procedural IQ Engine.
    Uses twinx() to plot loss on the left y-axis and validation accuracy on the right y-axis.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.suptitle("Procedural Fractal IQ-Engine - Convergence Curves", fontsize=16, fontweight='bold')
    
    # Plot Train and Validation Loss on left y-axis
    ax1.plot(epochs, history['train_loss'], color='#1f77b4', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], color='#ff7f0e', label='Val Loss', linewidth=2, linestyle='--')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('DSL CrossEntropy Loss', color='black', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(loc='upper left')
    
    # Plot Accuracy on right y-axis
    ax2 = ax1.twinx()
    ax2.plot(epochs, history['val_accuracy'], color='#2ca02c', label='Val Accuracy', linewidth=2)
    ax2.set_ylabel('Functional Sequence Accuracy', color='#2ca02c', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#2ca02c')
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    logging.info(f"Procedural metrics plot saved to {filepath}")


def run_phase3() -> None:
    # Hyperparameters
    hidden_dim = 256
    batch_size = 64
    steps_per_epoch = 10
    epochs = 3000
    learning_rate = 1e-3
    weight_decay = 1e-5
    
    logging.info("Initializing FractalIQEngine for Procedural Meta-Learning (Phase 3)...")
    world = FractalIQEngine()
    
    logging.info(f"Building Monolithic IQMicroUnit Model (hidden_dim={hidden_dim})...")
    model = IQMicroUnit(hidden_dim=hidden_dim)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model created. Total trainable parameters: {total_params}")
    
    logging.info("Initializing Procedural IQ Trainer...")
    trainer = ProceduralIQTrainer(
        model=model,
        world=world,
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_model_state = None
    
    from logger import CoTLogger
    cot_logger = CoTLogger()
    
    checkpoint_path = "cot_best_model.pt"
    if os.path.exists(checkpoint_path):
        logging.info(f"Loading existing checkpoint from {checkpoint_path}...")
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
            logging.info("Checkpoint loaded successfully. Evaluating initial model state...")
            best_val_loss, best_val_acc = trainer.evaluate(val_batch_size=512, chain_depth=2)
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            logging.info(f"Initial model state - Val DSL CrossEntropy Loss: {best_val_loss:.4f}, Val Functional Sequence Accuracy: {best_val_acc*100:.2f}%")
        except Exception as e:
            logging.warning(f"Failed to load checkpoint: {e}. Starting from scratch.")
            
    logging.info(f"Starting Procedural Meta-Learning training for {epochs} epochs...")
    
    completed = False
    
    try:
        for epoch in range(1, epochs + 1):
            train_loss = trainer.train_epoch(batch_size=batch_size, steps_per_epoch=steps_per_epoch, chain_depth=2)
            val_loss, val_acc = trainer.evaluate(val_batch_size=512, chain_depth=2)
            
            # Record history
            trainer.history['train_loss'].append(train_loss)
            trainer.history['val_loss'].append(val_loss)
            trainer.history['val_accuracy'].append(val_acc)
            
            # Track best model state based on validation Sequence Accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            
            if epoch % 50 == 0 or epoch == 1 or epoch == epochs:
                logging.info(
                    f"Epoch {epoch:04d}/{epochs} | "
                    f"Train DSL CrossEntropy Loss: {train_loss:.4f} | "
                    f"Val DSL CrossEntropy Loss: {val_loss:.4f} | "
                    f"Val Functional Sequence Accuracy: {val_acc*100:.2f}%"
                )
                
            # Live Plotting (every 50 epochs or epoch 1)
            if epoch % 50 == 0 or epoch == 1:
                plot_filepath = os.path.join(os.getcwd(), "ultimate_iq_metrics.png")
                plot_phase3_metrics(trainer.history, plot_filepath)
                
            # CoT Validation Logging (every 100 epochs or epoch 1)
            if epoch % 100 == 0 or epoch == 1:
                # Run and write deep breakdowns to cot_brain_debug.log
                with torch.no_grad():
                    X_ctx, Y_ctx, X_qry, Y_qry, chain_ops = world.generate_batch(64, chain_depth=2)
                    cot_logger.log_validation_episodes(epoch, model, X_ctx, Y_ctx, X_qry, Y_qry, chain_ops)
                
            # Checkpoint saving (every 100 epochs)
            if epoch % 100 == 0:
                if best_model_state is not None:
                    torch.save(best_model_state, "cot_best_model.pt")
                else:
                    torch.save(model.state_dict(), "cot_best_model.pt")
                logging.info(f"Epoch {epoch:04d}: saved best weights to cot_best_model.pt (validation Loss: {best_val_loss:.4f}, Functional Sequence Accuracy: {best_val_acc*100:.2f}%)")
                
        completed = True
    except KeyboardInterrupt:
        logging.info("Training interrupted by user. Evaluating current state...")
    finally:
        # Save the convergence plots at the end
        plot_filepath = os.path.join(os.getcwd(), "ultimate_iq_metrics.png")
        if len(trainer.history['train_loss']) > 0:
            plot_phase3_metrics(trainer.history, plot_filepath)
            
        # Save best model to cot_best_model.pt at the end (without duplication)
        if best_model_state is not None:
            torch.save(best_model_state, "cot_best_model.pt")
        else:
            torch.save(model.state_dict(), "cot_best_model.pt")
        logging.info("Final checkpoints and plots saved.")
 
    if len(trainer.history['train_loss']) > 0:
        logging.info("==================================================")
        logging.info("ZERO-SHOT HOLDOUT EXAM (OUT-OF-DISTRIBUTION)")
        logging.info("==================================================")
        # At the end of 3000 epochs, run a holdout test generating chains of chain_depth=3 
        # to evaluate true fluid intelligence
        print("Evaluating Zero-Shot Transfer to Chain Depth 3 (Unseen micro-worlds)...")
        ood_loss, ood_acc = trainer.evaluate(val_batch_size=1000, chain_depth=3)
        print(f"Zero-Shot Holdout Loss: {ood_loss:.4f}")
        print(f"Zero-Shot Holdout Functional Sequence Accuracy: {ood_acc*100:.2f}%")
        
        # Log zero-shot exam results to cot_brain_debug.log
        cot_logger.log_exam_summary(ood_acc, ood_loss)
        logging.info("==================================================")


def main() -> None:
    parser = argparse.ArgumentParser(description="Micro-World Intelligence Benchmark Framework")
    parser.add_argument(
        '--phase', 
        type=int, 
        default=3, 
        choices=[1, 2, 3], 
        help="Phase to execute (1: Multi-Task Core, 2: Meta-Learning IQ-Module, 3: Crystallized Meta-IQ Benchmark)"
    )
    args = parser.parse_args()

    # Set seed for reproducibility
    torch.manual_seed(42)
    import random
    random.seed(42)
    
    if args.phase == 1:
        logging.info("=== Running Phase 1 (Multi-Task Core) ===")
        run_phase1()
    elif args.phase == 2:
        logging.info("=== Running Phase 2 (Meta-Learning IQ-Module) ===")
        run_phase2()
    else:
        logging.info("=== Running Phase 3 (Crystallized Meta-IQ Benchmark) ===")
        run_phase3()

if __name__ == "__main__":
    main()
