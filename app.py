import json
import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="LLM-as-Judge Evaluation", page_icon="⚖️", layout="wide")

# ── Load report ──────────────────────────────────────────────────────────────
REPORT_PATH = "evaluation_report.json"

def load_report():
    if not os.path.exists(REPORT_PATH):
        return None
    with open(REPORT_PATH) as f:
        return json.load(f)

# ── Run pipeline ─────────────────────────────────────────────────────────────
def run_pipeline():
    # Reset engine state so counts start fresh
    from judge_engine import LLMJudgeEngine
    from main_eval import TEST_SUITE
    import json as _json

    engine = LLMJudgeEngine()
    results, flip_count, v1_wins, v2_wins, ties = [], 0, 0, 0, 0
    total = len(TEST_SUITE)
    bar = st.progress(0, text="Running evaluation…")

    for i, case in enumerate(TEST_SUITE):
        verdict = engine.evaluate_with_position_swap(case)
        if not verdict["is_consistent"]:
            flip_count += 1
        w = verdict["final_winner"]
        if w == "A": v1_wins += 1
        elif w == "B": v2_wins += 1
        else: ties += 1
        results.append({
            "case_id": case["id"], "input": case["input"],
            "consistent_order_agreement": verdict["is_consistent"],
            "assigned_winner": w, "raw_logs": verdict
        })
        bar.progress((i + 1) / total, text=f"Case {i+1}/{total} done")

    report = {
        "summary": {
            "total_evaluated_cases": total,
            "v1_prompt_win_rate": v1_wins / total,
            "v2_prompt_win_rate": v2_wins / total,
            "tie_or_position_conflict_rate": ties / total,
            "measured_position_flip_rate": flip_count / total,
            "judge_total_api_calls": engine.call_count,
            "estimated_tokens_used": engine.total_tokens_consumed,
        },
        "details": results,
    }
    with open(REPORT_PATH, "w") as f:
        _json.dump(report, f, indent=2)
    bar.empty()
    return report

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("⚖️ LLM-as-Judge Evaluation Pipeline")
st.caption("Pairwise A-vs-B judging with position-swap bias detection · Groq · llama-3.3-70b-versatile")

report = load_report()

col_run, _ = st.columns([1, 5])
with col_run:
    if st.button("▶ Run Evaluation", type="primary"):
        report = run_pipeline()
        st.success("Evaluation complete!")
        st.rerun()

if report is None:
    st.info("No report found. Click **Run Evaluation** to generate one.")
    st.stop()

s = report["summary"]
v1  = s["v1_prompt_win_rate"]
v2  = s["v2_prompt_win_rate"]
tie = s["tie_or_position_conflict_rate"]
flip = s["measured_position_flip_rate"]
winner_label = "CONFIG V1 🏆" if v1 > v2 else ("CONFIG V2 🏆" if v2 > v1 else "TIE")

# ── Summary metrics ───────────────────────────────────────────────────────────
st.divider()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("V1 Win Rate",   f"{v1*100:.1f}%")
c2.metric("V2 Win Rate",   f"{v2*100:.1f}%")
c3.metric("TIE Rate",      f"{tie*100:.1f}%")
c4.metric("Position Flip", f"{flip*100:.1f}%")
c5.metric("Declared Winner", winner_label)

st.caption(f"Total API calls: **{s['judge_total_api_calls']}** · Tokens used: **{s['estimated_tokens_used']:,}**")

# ── Win-rate bar chart ────────────────────────────────────────────────────────
st.divider()
st.subheader("A/B Win Rate Comparison")
fig_bar = go.Figure(go.Bar(
    x=["V1 (Config A)", "V2 (Config B)", "TIE"],
    y=[v1*100, v2*100, tie*100],
    marker_color=["#2ecc71", "#e74c3c", "#95a5a6"],
    text=[f"{v:.1f}%" for v in [v1*100, v2*100, tie*100]],
    textposition="outside",
))
fig_bar.update_layout(yaxis_title="Win Rate (%)", yaxis_range=[0, 100],
                      plot_bgcolor="rgba(0,0,0,0)", height=350)
st.plotly_chart(fig_bar, use_container_width=True)

# ── Per-case results table ────────────────────────────────────────────────────
st.divider()
st.subheader("Per-Case Results")

rows = []
for d in report["details"]:
    r1 = d["raw_logs"].get("run_1_verdict", {})
    r2 = d["raw_logs"].get("run_2_verdict", {})
    criteria = r1.get("per_criterion_eval", [])
    avg_a = round(sum(c.get("score_a", 0) for c in criteria) / len(criteria), 2) if criteria else "-"
    avg_b = round(sum(c.get("score_b", 0) for c in criteria) / len(criteria), 2) if criteria else "-"
    rows.append({
        "Case": d["case_id"],
        "Question": d["input"],
        "Winner": d["assigned_winner"],
        "Consistent": "✅" if d["consistent_order_agreement"] else "🔄 Flip",
        "Avg Score A": avg_a,
        "Avg Score B": avg_b,
    })

df = pd.DataFrame(rows)

def color_winner(val):
    if val == "A":  return "background-color:#d5f5e3;color:#1e8449"
    if val == "B":  return "background-color:#fadbd8;color:#922b21"
    return "background-color:#f0f0f0;color:#555"

st.dataframe(
    df.style.map(color_winner, subset=["Winner"]),
    use_container_width=True, hide_index=True
)

# ── Per-case detail expanders ─────────────────────────────────────────────────
st.divider()
st.subheader("Detailed Verdict Breakdown")

for d in report["details"]:
    winner = d["assigned_winner"]
    icon = "🟢" if winner == "A" else ("🔴" if winner == "B" else "🟡")
    with st.expander(f"{icon} Case {d['case_id']} — {d['input']}"):
        st.markdown(f"**Final Winner:** `{winner}` &nbsp;|&nbsp; **Consistent:** {'✅ Yes' if d['consistent_order_agreement'] else '🔄 Flipped (TIE assigned)'}")

        criteria = d["raw_logs"].get("run_1_verdict", {}).get("per_criterion_eval", [])
        if criteria:
            cdf = pd.DataFrame([{
                "Criterion": c["criterion"],
                "Score A": c.get("score_a", "-"),
                "Score B": c.get("score_b", "-"),
                "Rationale": c.get("rationale", ""),
            } for c in criteria])
            st.dataframe(cdf, use_container_width=True, hide_index=True)

        r1_rationale = d["raw_logs"].get("run_1_verdict", {}).get("global_rationale", "")
        if r1_rationale:
            st.caption(f"**Judge rationale (run 1):** {r1_rationale}")

# ── Bias summary ──────────────────────────────────────────────────────────────
st.divider()
st.subheader("Bias Analysis")

b1, b2 = st.columns(2)
with b1:
    st.markdown("**Position Bias**")
    fig_flip = go.Figure(go.Pie(
        labels=["Consistent", "Flipped"],
        values=[1 - flip, flip],
        marker_colors=["#2ecc71", "#e74c3c"],
        hole=0.5,
    ))
    fig_flip.update_layout(height=250, margin=dict(t=10, b=10))
    st.plotly_chart(fig_flip, use_container_width=True)
    st.caption(f"Flip rate: **{flip*100:.1f}%** — {'✅ Low, verdicts trustworthy' if flip < 0.2 else '⚠️ High, review manually'}")

with b2:
    st.markdown("**Bias Mitigations Applied**")
    st.markdown("""
| Bias | Mitigation |
|---|---|
| Position | Every pair run twice with A/B swapped |
| Verbosity | Length Calibration rubric criterion |
| Self-Enhancement | Judge = different model generation |
| Sycophancy | Per-criterion grounding + adversarial probe |
| Score Clustering | Pairwise mode — no raw scale needed |
""")

st.divider()
st.caption("Built with Streamlit · Groq API · llama-3.3-70b-versatile judge")
