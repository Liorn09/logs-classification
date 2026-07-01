"""Inference pipeline for classifying new log entries."""

import logging
import json
import joblib
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.config import Config, load_config
from src.preprocessing import LogPreprocessor
from src.summarizer import IssueSummarizer

logger = logging.getLogger(__name__)


class LogClassifierInference:
    """End-to-end inference: preprocess -> classify -> summarize."""

    def __init__(self, config: Config, model_path: str, label_names: List[str]):
        self.config = config
        self.preprocessor = LogPreprocessor(config)
        self.summarizer = IssueSummarizer(config)
        self.label_names = label_names

        # Load trained model
        self.pipeline = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")

    def classify_single(self, log_entry: str) -> Dict[str, Any]:
        """Classify a single log entry and generate a summary."""
        cleaned = self.preprocessor.clean_text(log_entry)

        # Predict
        pred_idx = self.pipeline.predict([cleaned])[0]
        probas = self.pipeline.predict_proba([cleaned])[0]
        confidence = float(probas.max())
        predicted_label = self.label_names[pred_idx]

        # Generate summary
        summary = self.summarizer.summarize(
            log_entry=log_entry,
            predicted_category=predicted_label,
            confidence=confidence,
        )

        return {
            "log_entry": log_entry,
            "predicted_category": predicted_label,
            "confidence": round(confidence, 4),
            "all_probabilities": {
                self.label_names[i]: round(float(p), 4)
                for i, p in enumerate(probas)
            },
            "summary": summary,
        }

    def classify_batch(self, log_entries: List[str]) -> List[Dict[str, Any]]:
        """Classify a batch of log entries."""
        results = []
        for i, entry in enumerate(log_entries):
            result = self.classify_single(entry)
            results.append(result)
            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(log_entries)} entries")

        logger.info(f"Classified {len(results)} entries total")
        return results

    def classify_from_file(self, filepath: str, text_col: str) -> List[Dict[str, Any]]:
        """Load log entries from a CSV and classify them."""
        df = pd.read_csv(filepath)
        entries = df[text_col].tolist()
        return self.classify_batch(entries)

    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """Save inference results to JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {output_path}")


def run_inference(
    model_path: str,
    label_names: List[str],
    input_source: str,
    text_col: str = "log_entry",
    output_path: str = "outputs/inference_results.json",
    config_path: str = "configs/config.yaml",
):
    """Convenience function to run inference from CLI or scripts."""
    config = load_config(config_path)
    engine = LogClassifierInference(config, model_path, label_names)

    if Path(input_source).exists():
        results = engine.classify_from_file(input_source, text_col)
    else:
        # Treat as a single log entry string
        results = [engine.classify_single(input_source)]

    engine.save_results(results, output_path)
    return results
