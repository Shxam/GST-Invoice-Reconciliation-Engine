# GST Reconciliation OpenEnv

A **GST invoice reconciliation engine** built as an OpenEnv-compliant RL environment. This repo provides realistic benchmarks for AI agents to reconcile Indian purchase invoices against GSTR-2B (tax authority records) and compute claimable ITC (Input Tax Credit).

## What This Is

This is a **reinforcement learning environment** (`gst_env`) paired with a **FastAPI server** that exposes HTTP endpoints for agent interaction. Agents receive invoice data, classify mismatches, and submit reconciliation results. The environment grades submissions and computes rewards based on accuracy, ITC claims, and compliance.

**Core use case:** Train and benchmark LLM-based agents on GST compliance tasks without requiring live GSTR-2B access.

---

## Stack

- **Language:** Python 3.11
- **Framework / Runtime:** FastAPI + Uvicorn
- **Core Libraries:**
  - `openenv-core` — RL environment scaffold
  - `Pydantic` — strict data validation
  - `Faker` — synthetic invoice generation
  - `Groq` / OpenAI SDK — LLM agents (optional)

---

## How It's Organized

```
.
├── gst_env/                    # Main environment package (95% of repo)
│   ├── env.py                  # GSTReconciliationEnv: core RL env
│   ├── models.py               # Pydantic schemas (Invoice, GSTR2BEntry, etc.)
│   ├── data_generator.py        # Task-specific synthetic data generators (task1–6)
│   ├── agent.py                # Reference agent (Groq LLM-based)
│   ├── baseline.py             # Baseline: deterministic pre-filter only
│   ├── main.py                 # FastAPI app with /reset, /step, /health endpoints
│   └── graders/                # Task-specific scoring logic
│
├── server/                     # ASGI server entry point
│   └── app.py                  # Wraps gst_env.main:app for Uvicorn
│
├── inference.py                # Standalone agent runner (60-40 hybrid)
├── pyproject.toml              # Project metadata
├── requirements.txt            # Dependencies
├── Dockerfile                  # Docker image (Python 3.11-slim)
└── openenv.yaml                # OpenEnv metadata (task descriptions, etc.)
```

### How It Fits Together

1. **Reset phase:** Agent POST to `/reset` with `task_id` → receives `Observation` (invoices + GSTR-2B entries)
2. **Processing:** Agent classifies each invoice (`MATCHED` / `MISMATCH` / `MISSING_IN_2B` / `EXTRA_IN_2B`)
3. **Submit phase:** Agent POST to `/step` with `Action` (classifications + claimable ITC)
4. **Grading:** Server compares against ground truth, returns `Reward` with match score + ITC accuracy
5. **Next task:** Agent can call `/reset` with another task_id (task1_easy through task6_mixed_docs)

Each task is a **deterministic synthetic scenario** with 6 difficulty levels and ~10–500 invoices per task.

---

## How to Run It

### Prerequisites
- Python 3.11+
- Virtual environment (recommended)

### Quick Start (Local)

```bash
# 1. Clone and install
git clone https://github.com/Shxam/openenv.git
cd openenv
pip install -r requirements.txt

# 2. Start server on port 7860
python -m uvicorn gst_env.main:app --host 0.0.0.0 --port 7860 --reload

# 3. In another terminal, run the baseline agent
python -c "from gst_env.baseline import main; main()"

# or run the reference Groq LLM agent (requires GROQ_API_KEY):
export GROQ_API_KEY=gsk_your_key_here
python gst_env/agent.py
```

### Docker

```bash
docker build -t gst-recon .
docker run -p 7860:7860 -e GROQ_API_KEY=gsk_... gst-recon
```

### Environment Variables

| Variable       | Default                                  | Purpose                   |
|----------------|------------------------------------------|---------------------------|
| `GROQ_API_KEY` | (empty)                                  | LLM API key (Groq)        |
| `BASE_URL`     | `http://localhost:7860`                  | Environment server URL    |
| `GROQ_MODEL`   | `llama-3.3-70b-versatile`                | LLM model name            |
| `API_BASE_URL` | `https://api.groq.com/openai/v1`        | LLM API endpoint          |

### Quick Test

```bash
curl http://localhost:7860/health
# → {"status": "ok"}
```

---

## Task Descriptions

Each task is a self-contained scenario with increasing complexity:

| Task              | Invoices | Scenario                                                         | Challenge                    |
|-------------------|----------|------------------------------------------------------------------|------------------------------|
| **task1_easy**    | 10       | Perfect match expected; all well-formed                         | Baseline accuracy            |
| **task2_medium**  | 50       | ~8 mismatches; amount diffs >15%, date shifts, GSTIN errors    | Noise tolerance              |
| **task3_hard**    | 200      | Adversarial: near-miss amounts (±Rs 1–5), OCR GSTIN, date shifts | Precision                    |
| **task4_credit_notes** | 75  | Credit notes, debit notes, advance receipts; ITC eligibility   | ITC rules                    |
| **task5_stress**  | 500      | All mismatch types; high volume                                 | Scalability                  |
| **task6_mixed_docs** | 150   | Mixed document types + OCR errors + near-miss amounts          | Robustness                   |

---

## Agent Architecture

### Baseline (Deterministic)
```python
# gst_env/baseline.py
1. Pre-filter all invoices deterministically:
   - MISSING_IN_2B: invoice_number not in GSTR-2B
   - EXTRA_IN_2B: invoice_number appears 2+ times
   - MATCHED / MISMATCH: field-by-field comparison (tolerance: ±Rs 1)
2. Compute ITC = sum(cgst + sgst + igst) for MATCHED invoices
3. Submit action
```

### Hybrid LLM Agent (Reference)
```python
# gst_env/agent.py (Groq)
Phase 1: Deterministic pre-pass on all invoices (~1s)
         → high_confidence cases locked in
         → low_confidence cases → LLM batch

Phase 2: Single LLM call per batch of ≤10 ambiguous invoices (~15–20s per batch)
         - Hard timeout: 25s per call
         - Smart prompt includes deterministic_label so LLM refines, not re-classifies

Phase 3: Merge deterministic + LLM results
         - High-confidence cases always keep deterministic label
         - Low-confidence: prefer LLM if valid JSON returned, else deterministic fallback
```

### Standalone Inference Runner
```bash
# inference.py — 60-40 hybrid for standalone use
# Designed for Hugging Face Spaces with 30-min task budget
python inference.py
```

---

## API Endpoints

### POST `/reset`

**Request:**
```json
{
  "task_id": "task1_easy"
}
```

**Response:**
```json
{
  "task_id": "task1_easy",
  "episode_id": "uuid...",
  "invoices": [
    {
      "invoice_id": "inv_001",
      "invoice_number": "INV-2024-001",
      "vendor_gstin": "27AABCS9991M1Z0",
      "invoice_date": "2024-01-15",
      "taxable_value": 10000.00,
      "cgst": 900.00,
      "sgst": 900.00,
      "igst": 0.00
    }
    ...
  ],
  "gstr2b_entries": [
    {
      "invoice_number": "INV-2024-001",
      "supplier_gstin": "27AABCS9991M1Z0",
      "invoice_date": "2024-01-15",
      "taxable_value": 10000.00,
      "cgst": 900.00,
      "sgst": 900.00,
      "igst": 0.00,
      "itc_available": true,
      "document_type": "invoice"
    }
    ...
  ],
  "max_itc_possible": 123456.78,
  "tax_period": "2024-25"
}
```

### POST `/step`

**Request:**
```json
{
  "reconciliation_result": [
    {
      "invoice_id": "inv_001",
      "status": "MATCHED",
      "correction_note": null,
      "mismatch_fields": []
    },
    {
      "invoice_id": "inv_002",
      "status": "MISMATCH",
      "correction_note": "GSTIN differs",
      "mismatch_fields": ["supplier_gstin"]
    }
  ],
  "claimable_itc": 900.00,
  "confidence": 0.92
}
```

**Response:**
```json
{
  "done": true,
  "reward": {
    "total": 0.85,
    "match_score": 0.90,
    "itc_score": 0.95,
    "coverage_score": 1.0,
    "false_positive_penalty": 0.0,
    "penalty_day_penalty": 1.0
  },
  "info": {
    "correct_matches": 9,
    "total_invoices": 10,
    "itc_error": 0.0432,
    "coverage": 1.0,
    "episode_id": "uuid...",
    "fraud_count": 0,
    "task_score": 0.85
  }
}
```

### GET `/health`

```json
{
  "status": "ok"
}
```

---

## Scoring Logic

**Match Score** = correct_matches / total_invoices

**ITC Score** = 1 - abs(predicted_itc - true_itc) / (true_itc + ε)

**Final Reward** = weighted combination (grader-specific per task)

**Fraud Penalty** = -0.05 per fraudulent MATCHED claim on MISSING_IN_2B invoice (capped at -0.30)

---

## Try Asking

1. **"How do I run a custom agent against task3_hard?"**
   - See `gst_env/agent.py` for Groq example; adapt the `run_task()` function.

2. **"What makes an invoice MISMATCH vs. MATCHED?"**
   - See `gst_env/data_generator.py` → `_compute_mismatch_fields()` and the task instructions in `gst_env/env.py`.

3. **"How do I deploy this as an OpenEnv benchmark?"**
   - The env is OpenEnv-compliant. See `openenv.yaml` for HF Spaces metadata; Docker image is ready.

---

## License

MIT
