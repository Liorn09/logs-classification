"""Data loading, cleaning, and preprocessing for log entries."""

import re
import logging
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import Config

logger = logging.getLogger(__name__)


class LogPreprocessor:
    """Handles loading, cleaning, and splitting log entry data."""

    def __init__(self, config: Config):
        self.config = config
        self.label_encoder = LabelEncoder()

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load raw log data from CSV."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} records from {filepath}")
        logger.info(f"Columns: {list(df.columns)}")
        return df

    def clean_text(self, text: str) -> str:
        """Normalize and clean a single log entry."""
        if not isinstance(text, str):
            return ""

        # Lowercase
        text = text.lower()

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Remove timestamps (common log patterns) - runs after lowercasing
        text = re.sub(
            r"\d{4}[-/]\d{2}[-/]\d{2}[t ]\d{2}:\d{2}:\d{2}[.\d]*z?", " ", text
        )

        # Normalize hex addresses and memory references
        text = re.sub(r"0x[0-9a-fA-F]+", "<HEX_ADDR>", text)

        # Normalize IP addresses
        text = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", "<IP_ADDR>", text)

        # Normalize file paths (keep the last segment for signal)
        text = re.sub(r"(/[\w.-]+)+/", "<PATH>/", text)

        # Normalize long numeric sequences but preserve short ones (error codes)
        text = re.sub(r"\b\d{6,}\b", "<NUM>", text)

        # Collapse repeated punctuation
        text = re.sub(r"[=\-]{3,}", " ", text)

        # Final whitespace cleanup
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def preprocess(
        self, df: pd.DataFrame, text_col: str, label_col: str
    ) -> pd.DataFrame:
        """Clean text and encode labels."""
        df = df.copy()

        # Drop rows with missing text or labels
        initial_len = len(df)
        df = df.dropna(subset=[text_col, label_col])
        dropped = initial_len - len(df)
        if dropped > 0:
            logger.warning(f"Dropped {dropped} rows with missing values")

        # Clean text
        df["cleaned_text"] = df[text_col].apply(self.clean_text)

        # Drop empty after cleaning
        df = df[df["cleaned_text"].str.len() > 0]

        # Encode labels
        df["label_encoded"] = self.label_encoder.fit_transform(df[label_col])

        logger.info(f"Preprocessed {len(df)} records")
        logger.info(f"Label distribution:\n{df[label_col].value_counts().to_string()}")

        return df

    def split_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Stratified train/test split."""
        train_df, test_df = train_test_split(
            df,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=df["label_encoded"],
        )
        logger.info(f"Train: {len(train_df)} | Test: {len(test_df)}")
        return train_df, test_df

    def get_label_names(self):
        """Return the mapping of encoded labels to original names."""
        return list(self.label_encoder.classes_)
