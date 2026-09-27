import os

import matplotlib.pyplot as plt
import pandas as pd

from training_logger import TrainingLogger


def plot_training_metrics(run_id: str, output_dir: str):
    """
    Generate three plots from the training log of a specific run and save them
    in <output_dir>/plots/.
    Plots:
      1. Training Loss vs Step (or epoch average if step not available)
      2. Validation Loss vs Epoch
      3. ROUGE-1 and ROUGE-L vs Epoch
    Returns a dict with paths to the saved images.
    """
    logger = TrainingLogger()
    run_data = logger.get_run_data(run_id)
    if run_data is None:
        print(f"No run data found for run_id: {run_id}")
        return {}

    epochs_df = pd.DataFrame(run_data.get("epochs", []))
    if epochs_df.empty:
        print("No epoch data found.")
        return {}

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    saved_paths = {}

    # 1. Training Loss vs Step (if step column exists, else use epoch)
    if "train_loss" in epochs_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        if "step" in epochs_df.columns:
            # Use step
            ax.plot(epochs_df["step"], epochs_df["train_loss"], marker='o', linestyle='-', linewidth=1.5)
            ax.set_xlabel("Training Step")
        else:
            # Use epoch (average per epoch)
            epoch_loss = epochs_df.groupby("epoch")["train_loss"].mean().reset_index()
            ax.plot(epoch_loss["epoch"], epoch_loss["train_loss"], marker='o', linestyle='-', linewidth=1.5)
            ax.set_xlabel("Epoch")
        ax.set_ylabel("Training Loss")
        ax.set_title("Training Loss")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        loss_path = os.path.join(plots_dir, "training_loss.png")
        plt.savefig(loss_path, dpi=150)
        plt.close()
        saved_paths["training_loss"] = loss_path

    # 2. Validation Loss vs Epoch
    if "eval_loss" in epochs_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        # Use epoch
        val_loss = epochs_df.groupby("epoch")["eval_loss"].mean().reset_index()
        ax.plot(val_loss["epoch"], val_loss["eval_loss"], marker='s', linestyle='-', linewidth=1.5, color='orange')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Loss")
        ax.set_title("Validation Loss")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        val_path = os.path.join(plots_dir, "validation_loss.png")
        plt.savefig(val_path, dpi=150)
        plt.close()
        saved_paths["validation_loss"] = val_path

    # 3. ROUGE-1 and ROUGE-L vs Epoch
    rouge_cols = [c for c in epochs_df.columns if c.startswith("eval_rouge")]
    if rouge_cols:
        fig, ax = plt.subplots(figsize=(10, 6))
        # Use epoch average
        epoch_group = epochs_df.groupby("epoch")
        for col in rouge_cols:
            if col in epochs_df.columns:
                avg_rouge = epoch_group[col].mean().reset_index()
                label = col.replace("eval_", "").upper()
                ax.plot(avg_rouge["epoch"], avg_rouge[col], marker='o', linestyle='-', linewidth=1.5, label=label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("ROUGE Score")
        ax.set_title("ROUGE Scores")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        rouge_path = os.path.join(plots_dir, "rouge_scores.png")
        plt.savefig(rouge_path, dpi=150)
        plt.close()
        saved_paths["rouge_scores"] = rouge_path

    print(f"Plots saved to {plots_dir}")
    return saved_paths
