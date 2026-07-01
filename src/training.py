"""Model training logic for log classification."""

import logging
import joblib
from pathlib import Path
from typing import Any, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.config import Config

logger = logging.getLogger(__name__)


class LogClassifierTrainer:
    """Trains and persists classification models."""

    def __init__(self, config: Config):
        self.config = config
        self.pipeline = None

    def _build_pipeline(self) -> Pipeline:
        """Construct sklearn pipeline based on config."""
        vectorizer = TfidfVectorizer(
            max_features=self.config.tfidf_max_features,
            ngram_range=self.config.tfidf_ngram_range,
            sublinear_tf=self.config.tfidf_sublinear_tf,
            strip_accents="unicode",
            stop_words="english",
        )

        if self.config.model_type == "tfidf_lr":
            classifier = LogisticRegression(
                max_iter=self.config.lr_max_iter,
                C=self.config.lr_C,
                class_weight="balanced",
                random_state=self.config.random_state,
                solver="lbfgs",
            )
        elif self.config.model_type == "tfidf_rf":
            classifier = RandomForestClassifier(
                n_estimators=self.config.rf_n_estimators,
                max_depth=self.config.rf_max_depth,
                class_weight="balanced",
                random_state=self.config.random_state,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")

        pipeline = Pipeline([
            ("tfidf", vectorizer),
            ("classifier", classifier),
        ])

        return pipeline

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
        """Train the classification pipeline."""
        logger.info(f"Training {self.config.model_type} model...")
        logger.info(f"Training samples: {len(X_train)}")

        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X_train, y_train)

        # Log feature info
        vectorizer = self.pipeline.named_steps["tfidf"]
        logger.info(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

        return self.pipeline

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        if self.pipeline is None:
            raise RuntimeError("Model not trained. Call train() first or load a saved model.")
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate prediction probabilities."""
        if self.pipeline is None:
            raise RuntimeError("Model not trained.")
        return self.pipeline.predict_proba(X)

    def save_model(self, filepath: str = None):
        """Persist trained model to disk."""
        if self.pipeline is None:
            raise RuntimeError("No model to save.")

        if filepath is None:
            save_dir = Path(self.config.model_save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            filepath = save_dir / f"{self.config.model_type}_pipeline.joblib"

        joblib.dump(self.pipeline, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a previously trained model."""
        self.pipeline = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
