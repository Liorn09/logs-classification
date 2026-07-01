"""Configuration loader and validation."""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """Pipeline configuration container."""

    # Data
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    test_size: float = 0.2
    random_state: int = 42

    # Model
    model_type: str = "tfidf_lr"
    model_save_dir: str = "outputs/models"
    tfidf_max_features: int = 10000
    tfidf_ngram_range: tuple = (1, 2)
    tfidf_sublinear_tf: bool = True
    lr_max_iter: int = 1000
    lr_C: float = 1.0
    rf_n_estimators: int = 200
    rf_max_depth: Optional[int] = None

    # Summarizer
    summarizer_type: str = "template"
    llm_provider: str = "gemini"
    llm_model: str = "gemini-2.5-flash"
    llm_max_tokens: int = 256

    # Evaluation
    eval_output_dir: str = "outputs/evaluation"

    # Logging
    log_level: str = "INFO"


def load_config(config_path: str = "configs/config.yaml") -> Config:
    """Load configuration from YAML file, falling back to defaults."""
    path = Path(config_path)
    config = Config()

    if path.exists():
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        if raw:
            data = raw.get("data", {})
            model = raw.get("model", {})
            summarizer = raw.get("summarizer", {})
            evaluation = raw.get("evaluation", {})

            config.raw_dir = data.get("raw_dir", config.raw_dir)
            config.processed_dir = data.get("processed_dir", config.processed_dir)
            config.test_size = data.get("test_size", config.test_size)
            config.random_state = data.get("random_state", config.random_state)

            config.model_type = model.get("type", config.model_type)
            config.model_save_dir = model.get("save_dir", config.model_save_dir)

            tfidf = model.get("tfidf", {})
            config.tfidf_max_features = tfidf.get("max_features", config.tfidf_max_features)
            ngram = tfidf.get("ngram_range", list(config.tfidf_ngram_range))
            config.tfidf_ngram_range = tuple(ngram)
            config.tfidf_sublinear_tf = tfidf.get("sublinear_tf", config.tfidf_sublinear_tf)

            lr = model.get("logistic_regression", {})
            config.lr_max_iter = lr.get("max_iter", config.lr_max_iter)
            config.lr_C = lr.get("C", config.lr_C)

            rf = model.get("random_forest", {})
            config.rf_n_estimators = rf.get("n_estimators", config.rf_n_estimators)
            config.rf_max_depth = rf.get("max_depth", config.rf_max_depth)

            config.summarizer_type = summarizer.get("type", config.summarizer_type)
            llm = summarizer.get("llm", {})
            config.llm_provider = llm.get("provider", config.llm_provider)
            config.llm_model = llm.get("model", config.llm_model)
            config.llm_max_tokens = llm.get("max_tokens", config.llm_max_tokens)

            config.eval_output_dir = evaluation.get("output_dir", config.eval_output_dir)

    return config
