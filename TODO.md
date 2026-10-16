# TODO / Roadmap

## Phase 1: long-running jobs (code is ready, only compute time is needed)

- [ ] **Fine-tune the multi-task DistilBERT** (`python -m csi.models.train --epochs 3`)
  - About 2.5 h on a laptop CPU (2.5 s/step × 1,185 steps × 3 epochs) and about 10 min on a free Colab/Kaggle T4 GPU.
  - Fastest route: open the repo in Colab, `pip install -r requirements.txt`, run `python -m csi.data.pipeline` then `python -m csi.models.train --epochs 3`, and download `artifacts/models/multitask_distilbert/` and `reports/metrics_distilbert.json`.
  - Once the model folder exists, the API and dashboard use it automatically (INT8 dynamic quantisation for CPU serving).
  - Then update the README results table (DistilBERT vs baseline, macro F1 per task) and the coverage-precision table.
- [ ] **Pre-compute sentiment for all 23.7k tickets** so the Overview tab can show sentiment by queue and theme (RoBERTa on CPU takes ~20 min). Store it as a `sentiment` column in the `tickets` SQLite table.
- [ ] **Build and test the Docker images** (`docker compose up --build`). Docker Desktop was not running during development.
- [ ] **Benchmark LLM reply quality** with Groq/Ollama on ~50 test tickets: groundedness, citation validity and latency per provider. Local `gemma2:9b` on CPU takes ~30-60 s per reply.

## Phase 2: model quality

- [ ] Hyper-parameter sweep (learning rate, max length 256, task loss weights) and compare against focal loss for the rare queues.
- [ ] Probability calibration (temperature scaling) so confidence thresholds for auto-routing are reliable.
- [x] Label-noise audit: inspect the confusion between Technical Support, IT Support and Product Support, and consider merging or hierarchical routing.
- [x] Error analysis with misclassified examples per queue (`csi/models/error_analysis.py`, a script instead of a notebook).
- [ ] Urgency score validation: report AUROC against human "high" priority labels and tune the component weights.
- [ ] Cross-encoder re-ranker on top of hybrid retrieval, and a labelled golden set for KB retrieval (Recall@k).
- [x] Better offline reply fallback: filter out agent-facing policy sentences such as "agents must…" so offline drafts read as customer-facing.

## Phase 3: product and deployment

- [ ] Deploy a live demo on Hugging Face Spaces or Streamlit Community Cloud (baseline model and index fit in the free tier).
- [ ] Record a 2-minute walkthrough video: business problem, live triage, themes matrix, model metrics.
- [x] Agent feedback loop: thumbs up/down on drafts stored in SQLite and used as an evaluation set.
- [ ] Time-series trend monitoring once real timestamps exist (the public dataset has none): weekly theme volumes and alerting.
- [ ] Multilingual support (the dataset has 33k German tickets): multilingual MiniLM / XLM-R encoder.
- [x] Authentication and rate limiting on the API; CI workflow running `pytest` on every push.
- [ ] Scale-out data pipeline with PySpark for millions of tickets (current pandas pipeline handles ~60k rows in under a minute).
