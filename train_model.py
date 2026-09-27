import csv
import json
import os
from datetime import datetime

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)

import model_registry
from plot_utils import plot_training_metrics
from training_logger import TrainingLogger


def train_english_lora(csv_path, hyperparams,
                       output_base_dir="./models", log_dir="./logs",
                       model_name=None, notes=""):
    """
    Train English T5-small with LoRA using given hyperparameters.

    If model_name is provided, the resulting model is added to
    models/registry.json so it can be referred to by name in the UI.

    Returns the path to the merged model directory.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path, sep=";")
    df.columns = df.columns.str.strip()
    required_cols = ["cleaned_text", "summary"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}. "
                         f"Found: {df.columns.tolist()}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base_dir, f"english_lora_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"Training run: {timestamp}")
    print(f"Hyperparameters: {hyperparams}")
    print(f"Output: {output_dir}")
    if model_name:
        print(f"Will register as: '{model_name}'")

    logger = TrainingLogger(log_dir="./training_logs")
    run_id = logger.start_run(hyperparams, len(df))

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if torch.backends.mps.is_available():
        print("MPS device")

    df = df.dropna(subset=["summary"])
    df = df[df["summary"].str.strip() != ""]
    if len(df) < 2:
        raise ValueError(f"Not enough labelled data: {len(df)} rows.")

    dataset = Dataset.from_pandas(df[["cleaned_text", "summary"]])
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_data = dataset["train"]
    val_data = dataset["test"]

    model_name_hf = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name_hf)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_hf)
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=hyperparams.get('r', 32),
        lora_alpha=hyperparams.get('alpha', 64),
        lora_dropout=hyperparams.get('dropout', 0.15),
        target_modules=["q", "v", "k", "o", "wi", "wo"],
    )
    model = get_peft_model(model, lora_config)
    model.to(device)
    model.print_trainable_parameters()

    def preprocess_function(examples):
        inputs = ["summarize: " + text for text in examples["cleaned_text"]]
        model_inputs = tokenizer(inputs, max_length=384, truncation=True, padding=False)
        labels = tokenizer(text_target=examples["summary"], max_length=64,
                           truncation=True, padding=False)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train = train_data.map(preprocess_function, batched=True,
                                     remove_columns=["cleaned_text", "summary"])
    tokenized_val = val_data.map(preprocess_function, batched=True,
                                 remove_columns=["cleaned_text", "summary"])

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.cpu().numpy() if hasattr(predictions, 'cpu') else np.array(predictions)
        labels = labels.cpu().numpy() if hasattr(labels, 'cpu') else np.array(labels)
        vocab_size = tokenizer.vocab_size
        predictions = np.clip(predictions, 0, vocab_size - 1).astype(np.int64)
        labels = np.clip(labels, 0, vocab_size - 1).astype(np.int64)

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [p if p.strip() else "no summary" for p in decoded_preds]
        decoded_labels = [l if l.strip() else "no summary" for l in decoded_labels]

        result = rouge.compute(predictions=decoded_preds, references=decoded_labels,
                               use_stemmer=True)
        return {k: round(v * 100, 2) for k, v in result.items()}

    class LogCallback(TrainerCallback):
        def __init__(self, csv_filename):
            self.csv_filename = csv_filename

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                logs["step"] = state.global_step
                logs["epoch"] = state.epoch
                with open(self.csv_filename, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(logs.keys()))
                    if f.tell() == 0:
                        writer.writeheader()
                    writer.writerow(logs)

    class SampleCallback(TrainerCallback):
        def __init__(self, sample_text, tokenizer, model_device):
            self.sample_text = sample_text
            self.tokenizer = tokenizer
            self.device = model_device
            self.bad_words_ids = self._get_bad_words_ids()

        def _get_bad_words_ids(self):
            phrases = [
                "Read more", "For more information", "Not financial advice",
                "Don’t forget to like and subscribe", "Click here", "Disclaimer",
                "This is not investment advice", "Subscribe to our newsletter",
            ]
            bad_ids = []
            for phrase in phrases:
                ids = self.tokenizer.encode(phrase, add_special_tokens=False)
                if ids:
                    bad_ids.append(ids)
            return bad_ids

        def on_epoch_end(self, args, state, control, **kwargs):
            model.eval()
            input_text = "summarize: " + self.sample_text
            inputs = self.tokenizer(input_text, return_tensors="pt",
                                    truncation=True, max_length=384).to(self.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=50, num_beams=5, early_stopping=True,
                    no_repeat_ngram_size=4, repetition_penalty=1.5,
                    bad_words_ids=self.bad_words_ids,
                )
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"--- Epoch {state.epoch:.2f} sample --- \n{summary}\n")
            model.train()

    class EpochLoggerCallback(TrainerCallback):
        def __init__(self, logger_obj, run_id):
            self.logger = logger_obj
            self.run_id = run_id

        def on_epoch_end(self, args, state, control, logs=None, **kwargs):
            if logs:
                metrics = {}
                for k_src, k_dst in [("loss", "train_loss"), ("eval_loss", "eval_loss"),
                                     ("eval_rouge1", "eval_rouge1"),
                                     ("eval_rougeL", "eval_rougeL"),
                                     ("learning_rate", "learning_rate")]:
                    if k_src in logs:
                        metrics[k_dst] = logs[k_src]
                self.logger.log_epoch(epoch=int(state.epoch), metrics=metrics)

    sample_callback = SampleCallback(val_data[0]["cleaned_text"], tokenizer, device)
    epoch_logger_callback = EpochLoggerCallback(logger, run_id)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=hyperparams.get('num_epochs', 15),
        per_device_train_batch_size=hyperparams.get('batch_size', 2),
        per_device_eval_batch_size=hyperparams.get('batch_size', 2),
        learning_rate=hyperparams.get('learning_rate', 5e-4),
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        predict_with_generate=True,
        logging_steps=5,
        logging_first_step=True,
        save_strategy="no",
        eval_strategy="epoch",
        report_to="none",
        dataloader_pin_memory=False,
        fp16=False,
        label_smoothing_factor=0.1,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            LogCallback(os.path.join(log_dir, f"training_log_{timestamp}.csv")),
            sample_callback,
            epoch_logger_callback,
        ],
    )

    print("Starting LoRA training...")
    trainer.train()

    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)

    merged_model = model.merge_and_unload()
    merged_output_dir = os.path.join(output_dir, "merged")
    os.makedirs(merged_output_dir, exist_ok=True)
    merged_model.save_pretrained(merged_output_dir)
    tokenizer.save_pretrained(merged_output_dir)

    metrics = {
        'hyperparams': hyperparams,
        'rouge': eval_results,
        'training_loss': [log.get('loss') for log in trainer.state.log_history
                          if 'loss' in log and log['loss'] is not None],
        'timestamp': timestamp,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    logger.log_final(eval_results, merged_output_dir)

    try:
        plot_paths = plot_training_metrics(run_id, output_dir)
        metrics["plot_paths"] = plot_paths
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
    except Exception as e:
        print(f"Could not generate plots: {e}")

    # ----- Register in the named-model registry -----
    if model_name:
        rouge_metrics = {k: v for k, v in eval_results.items() if k.startswith("eval_rouge")}
        # Strip eval_ prefix for cleaner keys
        rouge_clean = {k.replace("eval_", ""): v for k, v in rouge_metrics.items()}
        ok, msg = model_registry.register(
            name=model_name,
            path=merged_output_dir,
            hyperparams=hyperparams,
            metrics=rouge_clean,
            notes=notes,
            overwrite=False,
        )
        if ok:
            print(f"{msg}")
        else:
            print(f"Registry: {msg}. Model still usable by directory name.")

    print(f"Model saved to {merged_output_dir}")
    return merged_output_dir
