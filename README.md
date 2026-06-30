# GST Reconciliation OpenEnv

A **GST invoice reconciliation engine** built as an OpenEnv-compliant RL environment. This provides realistic benchmarks for AI agents to reconcile Indian purchase invoices against GSTR-2B without requiring live tax authority access.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Agent (LLM or Deterministic)                    │
└────────────────┬────────────────────────────────────────────────────────┘
                 │
         ┌───────┴────────┐
         │                │
    HTTP POST          HTTP POST
   /reset              /step
         │                │
         ▼                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      FastAPI Server (main.py)                           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Route Handlers:                                                 │  │
│  │  • POST /reset → Initialize episode, generate synthetic data     │  │
│  │  • POST /step  → Score classifications, compute reward           │  │
│  │  • GET /health → Server status check                             │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────┬───────────────────┬───────────────────┬───────────────────────┘
         │                   │                   │
         ▼                   ▼                   ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│  GSTReconcilia   │ │   Data           │ │  Graders/        │
│  tionEnv         │ │   Generator      │ │  Scoring         │
│  (env.py)        │ │   (data_gen.py)  │ │  Logic           │
│                  │ │                  │ │  (graders/)      │
│ • State mgmt     │ │ • Synthetic      │ │                  │
│ • Validation     │ │   invoices       │ │ • Match score    │
│ • Ground truth   │ │ • GSTR-2B data   │ │ • ITC accuracy   │
│   comparison     │ │ • Task 1–6       │ │ • Fraud penalty  │
│                  │ │                  │ │                  │
└──────────────────┘ └──────────────────┘ └──────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       Pydantic Models (models.py)                        │
│                                                                          │
│  • Invoice: vendor_gstin, invoice_date, taxable_value, cgst, sgst, igst │
│  • GSTREntry: supplier_gstin, invoice_date, itc_available, doc_type    │
│  • Action: reconciliation_result[], claimable_itc, confidence           │
│  • Observation: invoices[], gstr2b_entries[], max_itc_possible          │
│  • Reward: total, match_score, itc_score, coverage_score, penalties    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
openenv/
├── gst_env/                          # Main environment package (95% of codebase)
│   ├── env.py                        # GSTReconciliationEnv: core RL logic
│   ├── models.py                     # Pydantic schemas
│   ├── data_generator.py             # Synthetic data generators (tasks 1–6)
│   ├── agent.py                      # Hybrid LLM agent (Groq reference)
│   ├── baseline.py                   # Deterministic baseline
│   ├── main.py                       # FastAPI app
│   └── graders/                      # Task-specific scoring logic
│
├── server/
│   └── app.py                        # ASGI entry point
│
├── inference.py                      # Standalone agent runner
├── openenv.yaml                      # OpenEnv metadata
├── Dockerfile                        # Python 3.11-slim
├── requirements.txt                  # Dependencies
└── pyproject.toml                    # Project metadata
```

---

## Workflow

**Agent Interaction Loop:**

1. **Reset phase** → Agent POSTs `/reset` with `task_id`  
   Returns: invoices + GSTR-2B entries

2. **Processing** → Agent classifies each invoice:
   - `MATCHED` — invoice matches GSTR-2B entry
   - `MISMATCH` — data discrepancies detected
   - `MISSING_IN_2B` — invoice not in GSTR-2B
   - `EXTRA_IN_2B` — extra GSTR-2B entries

3. **Submit phase** → Agent POSTs `/step` with classifications + ITC claim  
   Returns: reward (match_score, itc_score, penalties)

4. **Next task** → Agent can reset with another task_id (task1_easy → task6_mixed_docs)

---

## Task Hierarchy

| Task | Invoices | Scenario | Challenge |
|------|----------|----------|-----------|
| **task1_easy** | 10 | Perfect matches | Baseline accuracy |
| **task2_medium** | 50 | ~8 mismatches; >15% amount diffs | Noise tolerance |
| **task3_hard** | 200 | Adversarial near-miss amounts (±Rs 1–5) | Precision |
| **task4_credit_notes** | 75 | Credit/debit notes, ITC rules | ITC eligibility |
| **task5_stress** | 500 | All mismatch types; high volume | Scalability |
| **task6_mixed_docs** | 150 | Mixed docs + OCR + near-miss amounts | Robustness |

---

## Agent Strategies

### Baseline (Deterministic)
- Pre-filter invoices with hard rules (tolerance: ±Rs 1)
- Compute ITC from MATCHED invoices only
- Fast, no LLM calls

### Hybrid LLM Agent (Reference)
- **Phase 1:** Deterministic pre-pass (1s) → high-confidence cases locked in
- **Phase 2:** LLM batch refinement (15–20s) for ambiguous cases (≤10 per batch)
- **Phase 3:** Merge results with fallback to deterministic if LLM fails

---

## Stack

- **Language:** Python 3.11
- **Framework:** FastAPI + Uvicorn
- **Data validation:** Pydantic
- **Synthetic data:** Faker
- **LLM:** Groq / OpenAI SDK (optional)
- **Deployment:** Docker, OpenEnv-compliant

---

## Quick Start

```bash
# Install
git clone https://github.com/Shxam/openenv.git && cd openenv
pip install -r requirements.txt

# Run server
python -m uvicorn gst_env.main:app --host 0.0.0.0 --port 7860

# In another terminal, run baseline
python -c "from gst_env.baseline import main; main()"

# Or run Groq LLM agent
export GROQ_API_KEY=gsk_...
python gst_env/agent.py
```

**Test:** `curl http://localhost:7860/health` → `{"status": "ok"}`
