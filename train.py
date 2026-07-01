#!/usr/bin/env python3
"""
Train the log classifier model.

Usage:
    python train.py --data data/raw/log_entries.csv
    python train.py --data data/raw/log_entries.csv --model-type tfidf_rf
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold

from src.config import load_config
from src.preprocessing import LogPreprocessor
from src.training import LogClassifierTrainer
from src.evaluation import ModelEvaluator


def load_label_map(labels_path: str = "data/raw/root_cause_labels.csv") -> dict:
    """Load human-readable label names from the reference file."""
    import pandas as pd
    path = Path(labels_path)
    if path.exists():
        df = pd.read_csv(path)
        return dict(zip(df["id"], df["label"]))
    return {}


def main():
    parser = argparse.ArgumentParser(description="Train log classifier model")
    parser.add_argument("--data", required=True, help="Path to training CSV")
    parser.add_argument("--text-col", default="log_message", help="Name of the text column")
    parser.add_argument("--label-col", default="root_cause_label", help="Name of the label column")
    parser.add_argument("--labels-ref", default="data/raw/root_cause_labels.csv", help="Root cause labels reference CSV")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path")
    parser.add_argument("--model-type", default=None, help="Override model type (tfidf_lr | tfidf_rf)")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    if args.model_type:
        config.model_type = args.model_type

    # Setup logging
    logging.basicConfig(level=config.log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    logger.info(f"Starting training pipeline with model: {config.model_type}")

    # Resolve data path — convert xlsx to csv if needed
    data_path = Path(args.data)
    if data_path.suffix.lower() == ".xlsx":
        csv_path = data_path.with_suffix(".csv")
        logger.info(f"xlsx input detected — converting to {csv_path} ...")
        from convert_data import convert
        convert(xlsx_path=data_path, csv_path=csv_path)
        data_path = csv_path
    elif not data_path.exists():
        xlsx_path = data_path.parent / "data.xlsx"
        if xlsx_path.exists():
            logger.info(f"{data_path} not found — converting {xlsx_path} ...")
            from convert_data import convert
            convert(xlsx_path=xlsx_path, csv_path=data_path)
        else:
            logger.error(f"Data file not found: {data_path}")
            sys.exit(1)

    # Load human-readable label map
    label_map = load_label_map(args.labels_ref)

    # ---------- Preprocess ----------
    preprocessor = LogPreprocessor(config)
    df = preprocessor.load_data(str(data_path))
    df = preprocessor.preprocess(df, text_col=args.text_col, label_col=args.label_col)
    train_df, test_df = preprocessor.split_data(df)
    label_codes = preprocessor.get_label_names()  # e.g. ["RC-01", "RC-02", ...]

    # Build display names: "RC-01 (Authentication Failure)"
    label_names = [
        f"{code} ({label_map[code]})" if code in label_map else code
        for code in label_codes
    ]

    # ---------- Train ----------
    trainer = LogClassifierTrainer(config)
    trainer.train(
        X_train=train_df["cleaned_text"].values,
        y_train=train_df["label_encoded"].values,
    )

    # ---------- Cross-Validation (small dataset sanity check) ----------
    logger.info("Running 5-fold stratified cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.random_state)
    cv_scores = cross_val_score(
        trainer._build_pipeline(),
        df["cleaned_text"].values,
        df["label_encoded"].values,
        cv=cv,
        scoring="f1_macro",
    )
    print(f"\nCross-Validation F1 (macro): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"  Per-fold: {[f'{s:.4f}' for s in cv_scores]}")

    # ---------- Evaluate on held-out test set ----------
    y_pred = trainer.predict(test_df["cleaned_text"].values)
    y_true = test_df["label_encoded"].values

    evaluator = ModelEvaluator(config)
    results = evaluator.evaluate(y_true, y_pred, label_names)
    results["cross_validation"] = {
        "f1_macro_mean": round(float(cv_scores.mean()), 4),
        "f1_macro_std": round(float(cv_scores.std()), 4),
        "per_fold": [round(float(s), 4) for s in cv_scores],
    }
    evaluator.print_report(y_true, y_pred, label_names)
    evaluator.save_results()

    # ---------- Save model + metadata ----------
    trainer.save_model()

    metadata = {
        "label_codes": label_codes,
        "label_names": label_names,
        "label_map": label_map,
        "model_type": config.model_type,
        "train_size": len(train_df),
        "test_size": len(test_df),
    }
    meta_path = Path(config.model_save_dir) / "metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Backward compat: also save label_names.json
    label_path = Path(config.model_save_dir) / "label_names.json"
    with open(label_path, "w") as f:
        json.dump(label_codes, f)

    logger.info("Training pipeline complete.")


if __name__ == "__main__":
    main()
