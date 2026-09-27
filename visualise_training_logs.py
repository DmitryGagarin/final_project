"""
visualise_training_logs.py

Read all training_log_*.csv files from a logs directory, parse training and evaluation
metrics (robustly handling variable row lengths), and generate comparison plots.

Interactive hover in Jupyter:
    - Install mplcursors: pip install mplcursors
    - Use %matplotlib notebook or %matplotlib widget in your notebook
    - Call run_visualisation(interactive=True)

Usage in Jupyter:
    from visualise_training_logs import run_visualisation
    %matplotlib notebook
    run_visualisation(logs_dir='logs', output_dir='plots', interactive=True)

Command line:
    python visualise_training_logs.py --logs_dir logs --output_dir plots
"""

import argparse
import csv
import glob
import os
import re
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set style
sns.set_style("whitegrid")
sns.set_context("talk", font_scale=1.1)


def parse_log_file(filepath):
    """
    Parse a CSV log file robustly, handling variable column counts.
    Returns two DataFrames: training rows and evaluation rows.
    """
    train_rows = []
    eval_rows = []

    # Extract model label from filename
    basename = os.path.basename(filepath)
    match = re.search(r'(\d{8}_\d{6})', basename)
    model_label = match.group(1) if match else basename.replace('training_log_', '').replace('.csv', '')

    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return pd.DataFrame(), pd.DataFrame()

        # Locate column indices
        try:
            loss_idx = header.index('loss')
        except ValueError:
            loss_idx = -1
        try:
            eval_loss_idx = header.index('eval_loss')
        except ValueError:
            eval_loss_idx = -1
        try:
            step_idx = header.index('step')
        except ValueError:
            step_idx = -1
        try:
            epoch_idx = header.index('epoch')
        except ValueError:
            epoch_idx = -1
        try:
            grad_norm_idx = header.index('grad_norm')
        except ValueError:
            grad_norm_idx = -1
        try:
            lr_idx = header.index('learning_rate')
        except ValueError:
            lr_idx = -1
        try:
            rouge1_idx = header.index('eval_rouge1')
        except ValueError:
            rouge1_idx = -1
        try:
            rougeL_idx = header.index('eval_rougeL')
        except ValueError:
            rougeL_idx = -1

        for row in reader:
            if len(row) < 2:
                continue

            # Check if this is an evaluation row (has eval_loss)
            if eval_loss_idx != -1 and len(row) > eval_loss_idx and row[eval_loss_idx].strip():
                eval_dict = {'model': model_label}
                if step_idx != -1 and len(row) > step_idx:
                    try:
                        eval_dict['step'] = float(row[step_idx]) if row[step_idx].strip() else None
                    except ValueError:
                        pass
                if epoch_idx != -1 and len(row) > epoch_idx:
                    try:
                        eval_dict['epoch'] = float(row[epoch_idx]) if row[epoch_idx].strip() else None
                    except ValueError:
                        pass
                if eval_loss_idx != -1 and len(row) > eval_loss_idx:
                    try:
                        eval_dict['eval_loss'] = float(row[eval_loss_idx]) if row[eval_loss_idx].strip() else None
                    except ValueError:
                        pass
                if rouge1_idx != -1 and len(row) > rouge1_idx:
                    try:
                        eval_dict['eval_rouge1'] = float(row[rouge1_idx]) if row[rouge1_idx].strip() else None
                    except ValueError:
                        pass
                if rougeL_idx != -1 and len(row) > rougeL_idx:
                    try:
                        eval_dict['eval_rougeL'] = float(row[rougeL_idx]) if row[rougeL_idx].strip() else None
                    except ValueError:
                        pass
                eval_rows.append(eval_dict)

            # Otherwise, if it has a loss, treat as training row
            elif loss_idx != -1 and len(row) > loss_idx and row[loss_idx].strip():
                train_dict = {'model': model_label}
                if step_idx != -1 and len(row) > step_idx:
                    try:
                        train_dict['step'] = float(row[step_idx]) if row[step_idx].strip() else None
                    except ValueError:
                        pass
                if epoch_idx != -1 and len(row) > epoch_idx:
                    try:
                        train_dict['epoch'] = float(row[epoch_idx]) if row[epoch_idx].strip() else None
                    except ValueError:
                        pass
                if loss_idx != -1 and len(row) > loss_idx:
                    try:
                        train_dict['loss'] = float(row[loss_idx]) if row[loss_idx].strip() else None
                    except ValueError:
                        pass
                if grad_norm_idx != -1 and len(row) > grad_norm_idx:
                    try:
                        train_dict['grad_norm'] = float(row[grad_norm_idx]) if row[grad_norm_idx].strip() else None
                    except ValueError:
                        pass
                if lr_idx != -1 and len(row) > lr_idx:
                    try:
                        train_dict['learning_rate'] = float(row[lr_idx]) if row[lr_idx].strip() else None
                    except ValueError:
                        pass
                train_rows.append(train_dict)

    train_df = pd.DataFrame(train_rows)
    eval_df = pd.DataFrame(eval_rows)

    # Drop rows where all relevant numeric columns are NaN
    if not train_df.empty:
        train_df = train_df.dropna(subset=['loss', 'step', 'epoch'], how='all')
    if not eval_df.empty:
        eval_df = eval_df.dropna(subset=['eval_loss', 'epoch'], how='all')

    return train_df, eval_df


def _add_hover_cursor(ax, label_field='model'):
    """Add mplcursors hover tooltip showing model name."""
    try:
        import mplcursors
        # Add cursor to all lines in the axes
        for line in ax.get_lines():
            # We want to show the model label for each point; but the label is constant per line.
            # We'll show the line label on hover.
            cursor = mplcursors.cursor(line, hover=True)
            cursor.connect("add", lambda sel: sel.annotation.set_text(sel.artist.get_label()))
        return True
    except ImportError:
        warnings.warn("mplcursors not installed. Install with 'pip install mplcursors' for hover tooltips.")
        return False


def plot_training_loss(train_dfs, output_dir, interactive=False):
    """Plot training loss (per step) for all models."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for df in train_dfs:
        df_sorted = df.sort_values('step')
        ax.plot(df_sorted['step'], df_sorted['loss'], label=df_sorted['model'].iloc[0], linewidth=2)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss per Step')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_loss_vs_step.png'), dpi=150)
    if interactive:
        _add_hover_cursor(ax)
    plt.show()
    plt.close()


def plot_training_loss_vs_epoch(train_dfs, output_dir, interactive=False):
    """Plot training loss vs epoch (averaged per epoch)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for df in train_dfs:
        epoch_loss = df.groupby('epoch')['loss'].mean().reset_index()
        ax.plot(epoch_loss['epoch'], epoch_loss['loss'], marker='o', label=df['model'].iloc[0])

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss (mean per epoch)')
    ax.set_title('Training Loss per Epoch')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_loss_vs_epoch.png'), dpi=150)
    if interactive:
        _add_hover_cursor(ax)
    plt.show()
    plt.close()


def plot_eval_loss(eval_dfs, output_dir, interactive=False):
    """Plot validation loss vs epoch."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for df in eval_dfs:
        df_sorted = df.sort_values('epoch')
        ax.plot(df_sorted['epoch'], df_sorted['eval_loss'], marker='o', label=df_sorted['model'].iloc[0])

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss per Epoch')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'validation_loss.png'), dpi=150)
    if interactive:
        _add_hover_cursor(ax)
    plt.show()
    plt.close()


def plot_rouge_scores(eval_dfs, output_dir, interactive=False):
    """Plot ROUGE-1 and ROUGE-L scores vs epoch."""
    has_rouge = any(all(col in df.columns for col in ['eval_rouge1', 'eval_rougeL']) for df in eval_dfs)
    if not has_rouge:
        print("No ROUGE columns found in evaluation logs. Skipping ROUGE plots.")
        return

    # ROUGE-1
    fig, ax = plt.subplots(figsize=(12, 6))
    for df in eval_dfs:
        if 'eval_rouge1' in df.columns:
            df_sorted = df.sort_values('epoch')
            ax.plot(df_sorted['epoch'], df_sorted['eval_rouge1'], marker='o', label=df_sorted['model'].iloc[0])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('ROUGE-1')
    ax.set_title('ROUGE-1 Score per Epoch')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rouge1.png'), dpi=150)
    if interactive:
        _add_hover_cursor(ax)
    plt.show()
    plt.close()

    # ROUGE-L
    fig, ax = plt.subplots(figsize=(12, 6))
    for df in eval_dfs:
        if 'eval_rougeL' in df.columns:
            df_sorted = df.sort_values('epoch')
            ax.plot(df_sorted['epoch'], df_sorted['eval_rougeL'], marker='o', label=df_sorted['model'].iloc[0])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('ROUGE-L')
    ax.set_title('ROUGE-L Score per Epoch')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rougeL.png'), dpi=150)
    if interactive:
        _add_hover_cursor(ax)
    plt.show()
    plt.close()


def plot_grad_norm(train_dfs, output_dir, interactive=False):
    """Plot gradient norm vs step."""
    has_grad = any('grad_norm' in df.columns for df in train_dfs)
    if not has_grad:
        print("No gradient norm data found. Skipping grad_norm plot.")
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    for df in train_dfs:
        if 'grad_norm' in df.columns:
            df_sorted = df.sort_values('step')
            ax.plot(df_sorted['step'], df_sorted['grad_norm'], label=df_sorted['model'].iloc[0])
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Norm per Step')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'grad_norm.png'), dpi=150)
    if interactive:
        _add_hover_cursor(ax)
    plt.show()
    plt.close()


def run_visualisation(logs_dir='logs', output_dir='plots', interactive=False):
    """
    Main function to parse logs and generate plots.
    If interactive=True, attempts to add hover tooltips (requires mplcursors).
    For full interactivity in Jupyter, use %matplotlib notebook or %matplotlib widget.
    """
    os.makedirs(output_dir, exist_ok=True)

    log_files = glob.glob(os.path.join(logs_dir, 'training_log_*.csv'))
    if not log_files:
        print(f"No training_log_*.csv files found in {logs_dir}")
        return

    print(f"Found {len(log_files)} log files.")

    train_dfs = []
    eval_dfs = []
    for f in log_files:
        try:
            train, eval_ = parse_log_file(f)
            if not train.empty:
                train_dfs.append(train)
            if not eval_.empty:
                eval_dfs.append(eval_)
        except Exception as e:
            print(f"Error parsing {f}: {e}")

    if not train_dfs and not eval_dfs:
        print("No valid data found.")
        return

    # Generate plots (saved to files and shown)
    if train_dfs:
        plot_training_loss(train_dfs, output_dir, interactive)
        plot_training_loss_vs_epoch(train_dfs, output_dir, interactive)
        plot_grad_norm(train_dfs, output_dir, interactive)
    if eval_dfs:
        plot_eval_loss(eval_dfs, output_dir, interactive)
        plot_rouge_scores(eval_dfs, output_dir, interactive)

    print(f"Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualise training logs from multiple models.')
    parser.add_argument('--logs_dir', type=str, default='logs',
                        help='Directory containing training_log_*.csv files (default: logs)')
    parser.add_argument('--output_dir', type=str, default='plots',
                        help='Directory to save plots (default: plots)')
    parser.add_argument('--interactive', action='store_true',
                        help='Enable interactive hover tooltips (requires mplcursors)')
    args = parser.parse_args()
    run_visualisation(logs_dir=args.logs_dir, output_dir=args.output_dir, interactive=args.interactive)


if __name__ == '__main__':
    main()
