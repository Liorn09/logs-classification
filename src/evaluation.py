"""Model evaluation and metrics reporting."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from sklearn.model_selection import cross_val_score

from src.config import Config

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Compute, display, and persist evaluation metrics."""

    def __init__(self, config: Config):
        self.config = config
        self.results = {}

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        label_names: List[str],
    ) -> Dict[str, Any]:
        """Compute all evaluation metrics."""

        # Overall metrics
        self.results["accuracy"] = float(accuracy_score(y_true, y_pred))
        self.results["precision_macro"] = float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        )
        self.results["recall_macro"] = float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        )
        self.results["f1_macro"] = float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        )
        self.results["precision_weighted"] = float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        )
        self.results["recall_weighted"] = float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        )
        self.results["f1_weighted"] = float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        )

        # Per-class report
        report = classification_report(
            y_true, y_pred, target_names=label_names, output_dict=True, zero_division=0
        )
        self.results["per_class"] = {
            name: {
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "f1": round(metrics["f1-score"], 4),
                "support": int(metrics["support"]),
            }
            for name, metrics in report.items()
            if name in label_names
        }

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        self.results["confusion_matrix"] = {
            "matrix": cm.tolist(),
            "labels": label_names,
        }

        return self.results

    def print_report(self, y_true: np.ndarray, y_pred: np.ndarray, label_names: List[str]):
        """Print a formatted classification report to console."""
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)

        print(f"\nAccuracy:           {self.results['accuracy']:.4f}")
        print(f"Precision (macro):  {self.results['precision_macro']:.4f}")
        print(f"Recall (macro):     {self.results['recall_macro']:.4f}")
        print(f"F1 Score (macro):   {self.results['f1_macro']:.4f}")
        print(f"F1 Score (weighted):{self.results['f1_weighted']:.4f}")

        print("\n" + "-" * 60)
        print("PER-CLASS BREAKDOWN")
        print("-" * 60)
        print(
            classification_report(
                y_true, y_pred, target_names=label_names, zero_division=0
            )
        )

        print("-" * 60)
        print("CONFUSION MATRIX")
        print("-" * 60)
        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
        print(cm_df.to_string())
        print("=" * 60)

    def save_results(self, output_path: str = None):
        """Save evaluation metrics to JSON."""
        if output_path is None:
            output_dir = Path(self.config.eval_output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "evaluation_metrics.json"

        # Convert any numpy types for JSON serialization
        serializable = json.loads(json.dumps(self.results, default=str))

        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)

        logger.info(f"Evaluation results saved to {output_path}")
