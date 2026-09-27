import json
import os
from datetime import datetime
from typing import Dict, List, Any

import pandas as pd


class TrainingLogger:
    """Comprehensive training logger for summarisation models."""

    def __init__(self, log_dir: str = "./training_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.current_run = None

    def start_run(self, hyperparams: Dict[str, Any], dataset_size: int):
        """Start a new training run."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_run = {
            "run_id": timestamp,
            "hyperparameters": hyperparams,
            "dataset_size": dataset_size,
            "start_time": datetime.now().isoformat(),
            "epochs": [],
            "final_metrics": None,
            "sample_outputs": []
        }
        # Create run directory
        self.run_dir = os.path.join(self.log_dir, timestamp)
        os.makedirs(self.run_dir, exist_ok=True)
        return timestamp

    def log_epoch(self, epoch: int, metrics: Dict[str, float], sample_summary: str = None):
        """Log metrics for a single epoch."""
        if self.current_run is None:
            raise ValueError("No active training run")

        epoch_data = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        self.current_run["epochs"].append(epoch_data)

        if sample_summary:
            self.current_run["sample_outputs"].append({
                "epoch": epoch,
                "summary": sample_summary,
                "timestamp": datetime.now().isoformat()
            })

        # Write epoch log to CSV
        epoch_file = os.path.join(self.run_dir, f"epoch_{epoch:03d}.csv")
        pd.DataFrame([epoch_data]).to_csv(epoch_file, index=False)

        # Update the full log
        self._save_current_run()

    def log_final(self, metrics: Dict[str, float], model_path: str):
        """Log final metrics after training completes."""
        if self.current_run is None:
            raise ValueError("No active training run")

        self.current_run["final_metrics"] = metrics
        self.current_run["model_path"] = model_path
        self.current_run["end_time"] = datetime.now().isoformat()
        self.current_run["total_epochs"] = len(self.current_run["epochs"])

        # Calculate training time
        start = datetime.fromisoformat(self.current_run["start_time"])
        end = datetime.fromisoformat(self.current_run["end_time"])
        self.current_run["duration_seconds"] = (end - start).total_seconds()

        self._save_current_run()

        # Also save a summary JSON with just the final metrics for easy access
        summary_path = os.path.join(self.run_dir, "final_summary.json")
        with open(summary_path, "w") as f:
            json.dump({
                "run_id": self.current_run["run_id"],
                "final_metrics": metrics,
                "model_path": model_path,
                "hyperparameters": self.current_run["hyperparameters"],
                "duration_seconds": self.current_run["duration_seconds"]
            }, f, indent=2)

    def _save_current_run(self):
        """Save the current run data to JSON."""
        if self.current_run:
            run_file = os.path.join(self.run_dir, "full_log.json")
            with open(run_file, "w") as f:
                json.dump(self.current_run, f, indent=2, default=str)

    def get_run_data(self, run_id: str) -> Dict[str, Any]:
        """Retrieve data for a specific run."""
        run_dir = os.path.join(self.log_dir, run_id)
        run_file = os.path.join(run_dir, "full_log.json")
        if os.path.exists(run_file):
            with open(run_file, "r") as f:
                return json.load(f)
        return None

    def get_all_runs(self) -> List[Dict[str, Any]]:
        """Get summary of all training runs."""
        runs = []
        for run_dir in os.listdir(self.log_dir):
            run_path = os.path.join(self.log_dir, run_dir)
            if os.path.isdir(run_path):
                summary_file = os.path.join(run_path, "final_summary.json")
                if os.path.exists(summary_file):
                    with open(summary_file, "r") as f:
                        runs.append(json.load(f))
                else:
                    # Partial run - try to load from full log
                    full_file = os.path.join(run_path, "full_log.json")
                    if os.path.exists(full_file):
                        with open(full_file, "r") as f:
                            data = json.load(f)
                            runs.append({
                                "run_id": data.get("run_id", run_dir),
                                "hyperparameters": data.get("hyperparameters", {}),
                                "epochs_completed": len(data.get("epochs", [])),
                                "status": "incomplete"
                            })
        return runs

    def get_run_epochs_dataframe(self, run_id: str) -> pd.DataFrame:
        """Get all epoch data for a run as a DataFrame for plotting."""
        data = self.get_run_data(run_id)
        if data and "epochs" in data:
            return pd.DataFrame(data["epochs"])
        return pd.DataFrame()
