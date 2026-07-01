#!/usr/bin/env python3
"""
Run inference on new log entries.

Usage:
    # Classify entries from a CSV file
    python infer.py --input data/raw/new_logs.csv --text-col log_entry

    # Classify a single log entry
    python infer.py --entry "ERROR: Connection refused on port 5432 - database unreachable"

    # Use a specific model checkpoint
    python infer.py --input new_logs.csv --model outputs/models/tfidf_lr_pipeline.joblib
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from src.config import load_config
from src.inference import LogClassifierInference


def main():
    parser = argparse.ArgumentParser(description="Run log classifier inference")
    parser.add_argument("--input", default=None, help="Path to CSV with log entries")
    parser.add_argument("--entry", default=None, help="Single log entry string to classify")
    parser.add_argument("--text-col", default="log_entry", help="Column name for log text in CSV")
    parser.add_argument("--model", default=None, help="Path to trained model file")
    parser.add_argument("--labels", default=None, help="Path to label_names.json")
    parser.add_argument("--output", default="outputs/inference_results.json", help="Output file path")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path")
    args = parser.parse_args()

    config = load_config(args.config)

    logging.basicConfig(level=config.log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    # Resolve model path
    model_path = args.model or str(Path(config.model_save_dir) / f"{config.model_type}_pipeline.joblib")
    labels_path = args.labels or str(Path(config.model_save_dir) / "label_names.json")

    if not Path(model_path).exists():
        logger.error(f"Model not found at {model_path}. Run train.py first.")
        sys.exit(1)

    with open(labels_path) as f:
        label_names = json.load(f)

    # Initialize inference engine
    engine = LogClassifierInference(config, model_path, label_names)

    # Run inference
    if args.entry:
        results = [engine.classify_single(args.entry)]
    elif args.input:
        results = engine.classify_from_file(args.input, args.text_col)
    else:
        logger.error("Provide either --input (CSV path) or --entry (single log string)")
        sys.exit(1)

    # Save and display results
    engine.save_results(results, args.output)

    # Print a summary to console
    print(f"\nClassified {len(results)} log entries")
    print(f"Results saved to {args.output}\n")

    for i, r in enumerate(results[:5]):
        print(f"--- Entry {i + 1} ---")
        print(f"  Category:   {r['predicted_category']}")
        print(f"  Confidence: {r['confidence']}")
        print(f"  Severity:   {r['summary'].get('severity', 'N/A')}")
        print(f"  Component:  {r['summary'].get('component', 'N/A')}")
        print(f"  Action:     {r['summary'].get('recommended_action', 'N/A')}")
        print()

    if len(results) > 5:
        print(f"... and {len(results) - 5} more. See {args.output} for full results.")


if __name__ == "__main__":
    main()
