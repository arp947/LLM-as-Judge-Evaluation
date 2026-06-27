# LLM-as-Judge Evaluation Pipeline

A bias-aware LLM judging pipeline that evaluates and compares model outputs using structured verdicts, position-swap bias detection, and adversarial probes.

## Results Summary

| Metric | Value |
|---|---|
| V1 Win Rate | 60.0% |
| V2 Win Rate | 26.7% |
| Position Flip Rate | 13.3% |
| **Declared Winner** | **CONFIG v1 (Model Output A)** |

---

## Project Structure

```
ps2/
├── config.py          # Model config & API key settings
├── judge_engine.py    # Core judge logic, bias mitigations, JSON parsing
├── main_eval.py       # Test suite runner & report generator
├── schema.py          # Pydantic verdict schemas
├── test_suite.json    # Output report (auto-generated)
├── evaluation_report.json
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

Set your Groq API key in `config.py`:
```python
GROQ_API_KEY: str = "your_groq_api_key_here"
```
Or use a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free key at: https://console.groq.com/keys

---

## Run

```bash
python main_eval.py
```

Outputs `evaluation_report.json` with full per-case verdicts and a summary.

---

## Models Used

| Role | Model |
|---|---|
| Generator A | `llama-3.1-8b-instant` |
| Generator B | `llama-3.1-70b-versatile` |
| Judge | `llama-3.3-70b-versatile` |

Judge is a stronger/newer model than generators to reduce self-enhancement bias.

---

## Judging Mode

**Pairwise A-vs-B with reference** — the judge receives both model outputs and the expected reference, then picks a winner per criterion. This mode is used because:
- It avoids score clustering (no raw number scale needed)
- It directly answers "which config is better" for A/B comparison
- Reference grounding prevents hallucination-friendly verdicts

---

## Rubric

| Criterion | Description |
|---|---|
| Correctness | Facts must match the expected reference |
| Completeness | All parts of the question answered |
| Faithfulness | No hallucinated or invented details |
| Length Calibration | Penalize verbosity that adds no information |

---

## Bias Handling

### 1. Position Bias
**Problem:** Judges tend to favor whichever answer appears first (A).  
**Mitigation:** Every pair is evaluated twice with A/B order swapped. Winners are normalized and compared. If they disagree → TIE.  
**Measured:** Position flip rate = **13.3%** (2/15 cases flipped).

### 2. Verbosity / Length Bias
**Problem:** Longer answers look more authoritative even when wrong.  
**Mitigation:** Explicit `Length Calibration` rubric criterion penalizes verbose outputs with no added information density.  
**Probe:** Case 3 — verbose-but-wrong (90s) vs terse-but-correct (30s). Judge correctly penalized the verbose wrong answer.

### 3. Self-Enhancement Bias
**Problem:** A model judging its own family's outputs inflates their scores.  
**Mitigation:** Judge model (`llama-3.3-70b-versatile`) is a different/newer generation than Generator A (`llama-3.1-8b-instant`).

### 4. Sycophancy Bias
**Problem:** Judge agrees with confidently stated but wrong answers.  
**Probe:** Case 4 — Model A confidently confirms wrong replication factor (5). Model B correctly states 3. Judge correctly picked B in 2/2 runs.

---

## Adversarial Probe Results

| Case | Probe Type | Judge Fooled? |
|---|---|---|
| Case 3 | Verbose-but-wrong vs Terse-but-correct | No — terse correct answer won |
| Case 4 | Confidently-wrong vs Factually-correct | No — factual answer won |

---

## A/B Comparison

Comparing **Config V1** (concise, direct answers) vs **Config V2** (verbose, sometimes incorrect answers):

| Config | Win Rate |
|---|---|
| V1 | 60.0% |
| V2 | 26.7% |
| TIE | 13.3% |

**Winner: Config V1** — consistently more correct and concise across 15 test cases.

---

## Discussion

**How biased before vs after mitigation:**
- Before (no position swap): all 15 cases would silently resolve to TIE due to inconsistent JSON schemas from the judge.
- After (strict prompt template + position swap): 13.3% flip rate — low enough to trust the majority verdicts.

**Would you let it gate a release?**  
With a 13.3% flip rate and adversarial probes passing, it is suitable for **pre-release screening** but not as the sole gate. Recommended use: flag regressions automatically, require human review only on flagged cases.

---

## Cost Tracking

Each run consumes `2 API calls per test case × 15 cases = 30 total judge calls`.  
Token usage is logged in `evaluation_report.json` under `estimated_tokens_used`.
