# Log Classifier Pipeline

An AI-powered prototype for classifying system error logs into root cause categories and generating structured issue summaries.

## Project Structure

```
log-classifier/
├── train.py                 # Training entry point
├── infer.py                 # Inference entry point
├── configs/
│   └── config.yaml          # Pipeline configuration
├── src/
│   ├── __init__.py
│   ├── config.py            # Config loader
│   ├── preprocessing.py     # Data cleaning and preparation
│   ├── training.py          # Model training logic
│   ├── inference.py         # Inference pipeline
│   ├── evaluation.py        # Metrics computation
│   └── summarizer.py        # Issue summary generation
├── data/
│   └── raw/                 # Original dataset(s)
├── outputs/
│   ├── models/              # Saved model artifacts
│   └── evaluation/          # Metrics and reports
├── tests/
│   └── test_core.py         # Unit tests
├── videolink.md
└── requirements.txt

```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py --data data/raw/log_entries.csv

# Run inference on a single log entry
python infer.py --entry "ERROR [db-pool] Connection timed out after 30000ms. All 20 pool slots exhausted."

# Run inference on a CSV file
python infer.py --input data/raw/log_entries.csv --text-col log_message

# Run unit tests
python -m pytest tests/ -v
```

## Model Approach

**Chosen model:** TF-IDF + Logistic Regression (`tfidf_lr`)

**Reasoning:**
- Log entries are short (~96 chars avg), domain-specific text where n-gram features capture error patterns effectively (e.g. "pool exhausted", "429 too many", "token expired")
- Logistic Regression with balanced class weights handles the mild class imbalance (12-18 samples per class) without synthetic oversampling artifacts
- Fast training (<1s), fully interpretable probability outputs, and no external dependencies beyond scikit-learn
- Multinomial LR provides calibrated confidence scores, useful for setting triage thresholds in production

**Alternatives evaluated:**
- TF-IDF + Random Forest: tested as comparison; similar accuracy (87.5%) but slightly lower cross-validation F1 (0.896 vs 0.909), higher variance across folds (+/- 0.078 vs 0.062), and less interpretable
- Fine-tuned DistilBERT: would likely improve recall on ambiguous entries (e.g. RC-07 vs RC-06 confusion) but adds significant training complexity, GPU dependency, and inference latency -- disproportionate for a 120-sample prototype

## Data Preprocessing

1. **Missing values:** rows with null text or labels are dropped (none in this dataset)
2. **Text normalization:**
   - Lowercased
   - Timestamps stripped (e.g. `2024-05-28T21:04:00Z`) -- these carry no root-cause signal
   - IP addresses normalized to `<IP_ADDR>` tokens
   - Hex memory addresses normalized to `<HEX_ADDR>`
   - File paths normalized to `<PATH>/filename`
   - Long numeric sequences (6+ digits) collapsed to `<NUM>` while preserving short error codes (e.g. 502, 429)
3. **Vectorization:** TF-IDF with unigram + bigram features, sublinear TF scaling, English stop words removed, max 5000 features
4. **Split:** 80/20 stratified train/test split (96 train / 24 test), preserving class proportions

## Evaluation Results

### Held-Out Test Set (n=24)

| Metric              | Score  |
|---------------------|--------|
| Accuracy            | 0.8750 |
| Precision (macro)   | 0.8958 |
| Recall (macro)      | 0.8750 |
| F1 Score (macro)    | 0.8714 |
| F1 Score (weighted) | 0.8798 |

### 5-Fold Stratified Cross-Validation (n=120)

| Metric         | Score              |
|----------------|--------------------|
| F1 macro (mean)| 0.9090 +/- 0.0623 |

Cross-validation on the full dataset provides a more robust estimate given the small test set size.

### Per-Class Performance

| Root Cause                         | Precision | Recall | F1    | Support |
|------------------------------------|-----------|--------|-------|---------|
| RC-01 (Authentication Failure)     | 1.00      | 1.00   | 1.00  | 3       |
| RC-02 (Database Connection Timeout)| 1.00      | 1.00   | 1.00  | 3       |
| RC-03 (Third-Party API Failure)    | 1.00      | 1.00   | 1.00  | 3       |
| RC-04 (Rate Limit Exceeded)        | 1.00      | 0.67   | 0.80  | 3       |
| RC-05 (Data Validation Error)      | 1.00      | 1.00   | 1.00  | 4       |
| RC-06 (Insufficient Permissions)   | 0.67      | 1.00   | 0.80  | 2       |
| RC-07 (Resource Exhaustion)        | 0.50      | 0.67   | 0.57  | 3       |
| RC-08 (Network / Connectivity Issue)| 1.00     | 0.67   | 0.80  | 3       |

### Error Analysis

The primary confusion occurs between RC-07 (Resource Exhaustion) and its neighbors:
- RC-07 entries misclassified as RC-06 (Insufficient Permissions) -- both involve system limits/access being denied, sharing vocabulary like "refused", "limit", "blocked"
- RC-04 (Rate Limit Exceeded) entry misclassified as RC-07 -- "limit" and "exceeded" appear in both categories
- RC-08 (Network / Connectivity Issue) entry misclassified as RC-07 -- resource-level language overlap

This is expected: with 12-18 training samples per class, the model has limited exposure to boundary-case vocabulary.

## Tradeoffs

- **Template vs LLM summarizer:** Defaulted to rule-based templates for zero external dependency, deterministic output, and zero latency. The LLM backend (configurable via `configs/config.yaml`) produces richer natural-language summaries but adds API cost (~$0.002/entry), 1-2s latency, and a failure mode requiring fallback handling.
- **Feature normalization depth:** Normalizing IPs, paths, and hex addresses trades away specificity (which exact IP failed) for better cross-service generalization. In production, the raw values would be preserved in structured fields alongside the classification.
- **Class weighting vs resampling:** Used `class_weight='balanced'` over SMOTE to avoid injecting synthetic feature combinations that could create false confidence in a 120-sample dataset.
- **Test set size:** 24 samples means individual misclassifications swing metrics significantly (each error = ~4% accuracy). Cross-validation provides the more reliable performance estimate.

## Limitations

- **Small dataset:** 120 samples across 8 classes is insufficient for production confidence. Performance will vary meaningfully with different random splits.
- **Single-label classification:** Real incidents often have multiple concurrent root causes (e.g. rate limiting + network issue). The current architecture assigns exactly one category.
- **Template summarizer heuristics:** Severity inference and component extraction use regex patterns that will miss non-standard log formats (e.g. structured JSON logs, custom delimiters).
- **No incremental learning:** Model must be fully retrained on new data; cannot adapt to emerging error patterns without a new training cycle.
- **Confidence calibration:** Prediction probabilities from a small-sample TF-IDF model are not well-calibrated; the absolute confidence values should not be used as-is for automated routing decisions without threshold tuning on a validation set.

## Productionization Roadmap

### Model Serving
- Serve behind a FastAPI endpoint with Pydantic request/response models
- Async batch processing endpoint for high-volume ingestion
- Model artifact versioning (MLflow or DVC) for reproducible rollbacks
- Containerized deployment (Docker) with health checks

### Monitoring
- Track prediction confidence distribution over time; alert when mean confidence drops below a tuned threshold (indicating distribution shift)
- Log all predictions with input hash, timestamps, and model version for audit trail
- Dashboard: classification volume by category/severity, confidence histograms, latency percentiles
- Track "low confidence" rate as a leading indicator of model staleness

### Drift Detection
- Monitor TF-IDF vocabulary coverage: flag when >N% of input tokens are OOV (out-of-vocabulary) relative to training data
- Compare weekly per-class prediction proportions against training distribution; alert on significant shifts (chi-squared test)
- Maintain a human-labeled sample stream; compute rolling precision/recall against it
- Alert on emergence of high-frequency new tokens (indicates new error types the model hasn't seen)

### Scaling
- Redis/RabbitMQ queue for decoupling ingestion from classification
- Horizontal autoscaling of worker pods based on queue depth
- For transformer-based models: GPU inference with batching and TorchServe/Triton
- Consider model distillation if upgrading to a larger model creates latency concerns

### Reliability
- Confidence-gated routing: entries below threshold go to human triage queue instead of auto-classification
- Circuit breaker on LLM summarizer path with automatic fallback to templates
- Graceful degradation: if the classifier service is down, raw logs are queued (not dropped)
- A/B testing framework for comparing model versions on live traffic before full rollover
- Automated retraining pipeline triggered by drift alerts, with human approval gate before deployment
