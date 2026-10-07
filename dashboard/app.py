from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DATA = ROOT / "data" / "processed"

st.set_page_config(page_title="Provider Intelligence", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#142337; --muted:#758397; --line:#e8edf2; --teal:#087e8b; --mint:#dbf3ed; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:#f5f7fa; color:var(--ink); }
.block-container { padding:1.5rem 2.2rem 3rem; max-width:1600px; }
[data-testid="stSidebar"] { background:#10263a; }
[data-testid="stSidebar"] * { color:#eef5f8 !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { padding:8px 10px; border-radius:8px; }
h1,h2,h3 { font-family:'Manrope',sans-serif; letter-spacing:-.035em; }
.hero { background:linear-gradient(120deg,#10263a,#174e5c); color:white; padding:25px 30px; border-radius:18px; margin:0 0 18px; }
.hero h1 { color:white; margin:0; font-size:30px; }
.hero p { color:#c8d9df; margin:6px 0 0; font-size:14px; }
.eyebrow { color:#4dd3ba; text-transform:uppercase; letter-spacing:.13em; font-weight:700; font-size:11px; margin-bottom:5px; }
.metric { background:white; border:1px solid #e8edf2; border-radius:14px; padding:17px 19px; min-height:105px; box-shadow:0 3px 14px #162a3a08; }
.metric-label { color:#758397; font-size:12px; font-weight:600; }
.metric-value { color:#142337; font-family:'Manrope',sans-serif; font-size:27px; font-weight:800; margin-top:7px; }
.metric-note { color:#97a3b1; font-size:11px; margin-top:2px; }
.panel { background:white; border:1px solid #e8edf2; border-radius:14px; padding:18px 20px; box-shadow:0 3px 14px #162a3a08; }
.small-note { color:#7c8999; font-size:12px; }
div[data-testid="stDataFrame"] { border:1px solid #e8edf2; border-radius:12px; }
</style>
""", unsafe_allow_html=True)


def csv(name):
    path = OUT / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def json_file(name):
    path = OUT / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

queue = csv("investigation_queue.csv")
features = pd.read_csv(DATA / "provider_features.csv") if (DATA / "provider_features.csv").exists() else pd.DataFrame()
importance = csv("global_importance.csv")
metrics = csv("final_test_results.csv")
ranking = csv("topk_ranking.csv")
threshold = json_file("threshold_choice.json")
selection = json_file("selection.json")
audit = json_file("audit_raw.json")
explanations = json_file("explanations.json")

if queue.empty:
    st.error("No investigation queue found. Run the project pipeline first so outputs/investigation_queue.csv exists.")
    st.stop()

queue["p_fraud"] = pd.to_numeric(queue["p_fraud"], errors="coerce").fillna(0)
queue["exposure_amount"] = pd.to_numeric(queue["exposure_amount"], errors="coerce").fillna(0)
queue["expected_exposure"] = pd.to_numeric(queue["expected_exposure"], errors="coerce").fillna(0)

with st.sidebar:
    st.markdown("<div style='font-size:11px;letter-spacing:.16em;color:#69d7c1;font-weight:700'>◈ PROVIDER INTELLIGENCE</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:21px;font-weight:800;margin:7px 0 22px'>Investigation<br>workspace</div>", unsafe_allow_html=True)
    page = st.radio("WORKSPACE", ["Overview", "Investigation queue", "Model & data"], label_visibility="visible")
    st.markdown("---")
    st.caption(f"Model: {selection.get('selected_model', threshold.get('selected_model', 'Available model'))}")
    st.caption("Historical provider-level decision support")
    st.markdown("---")
    st.caption("Scores prioritize review. They do not establish fraud or support adverse action on their own.")

st.markdown("<div class='hero'><div class='eyebrow'>Healthcare integrity · provider-level review</div><h1>Provider Intelligence</h1><p>Find the next case to review, understand the signal, and keep investigators in control.</p></div>", unsafe_allow_html=True)

fraud_count = int((queue.get("PotentialFraud", pd.Series(dtype=float)) == 1).sum()) if "PotentialFraud" in queue else 0
n_providers = int(audit.get("n_providers_labelled", len(features) or len(queue)))
fraud_rate = float(audit.get("fraud_rate", fraud_count / max(len(queue), 1)))
exposure = float(queue["exposure_amount"].sum())

if page == "Overview":
    st.markdown("### Portfolio snapshot")
    k1,k2,k3,k4 = st.columns(4)
    for col, label, value, note in [
        (k1,"Providers in review set",f"{n_providers:,}","Provider-level records"),
        (k2,"Flagged in labeled data",f"{fraud_count:,}",f"Historical label rate {fraud_rate:.1%}"),
        (k3,"Queue exposure",f"${exposure/1e6:,.1f}M","Reimbursement represented in current queue"),
        (k4,"Selected model",selection.get("selected_model", threshold.get("selected_model","—")),"Prioritization support")]:
        with col: st.markdown(f"<div class='metric'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div><div class='metric-note'>{note}</div></div>",unsafe_allow_html=True)
    st.write("")
    left,right = st.columns([1.15,.85])
    with left:
        st.markdown("<div class='panel'>",unsafe_allow_html=True)
        st.markdown("#### Risk and exposure landscape")
        fig=px.scatter(queue, x="p_fraud", y="exposure_amount", color="risk_band" if "risk_band" in queue else None,
            size="expected_exposure", hover_name="Provider", hover_data={"p_fraud":":.1%","exposure_amount":":$,.0f","expected_exposure":":$,.0f"},
            color_discrete_map={"HIGH":"#e06b5c","MEDIUM":"#e8aa46","LOW":"#148b89"},
            labels={"p_fraud":"Model score","exposure_amount":"Total reimbursement ($)","risk_band":"Queue band"})
        fig.update_layout(height=330,margin=dict(l=0,r=0,t=8,b=0),paper_bgcolor="white",plot_bgcolor="white",font_color="#536276",legend_title_text="")
        fig.update_xaxes(tickformat=".0%",gridcolor="#edf1f4")
        fig.update_yaxes(tickprefix="$",tickformat="~s",gridcolor="#edf1f4")
        st.plotly_chart(fig,use_container_width=True)
        st.markdown("</div>",unsafe_allow_html=True)
    with right:
        st.markdown("<div class='panel'>",unsafe_allow_html=True)
        st.markdown("#### Leading model signals")
        if not importance.empty:
            imp=importance.head(8).sort_values("importance")
            imp["label"]=imp["feature"].str.replace("_"," ").str.capitalize()
            fig=px.bar(imp,x="importance",y="label",orientation="h",color_discrete_sequence=["#168b88"],labels={"importance":"Relative importance","label":""})
            fig.update_layout(height=330,margin=dict(l=0,r=5,t=8,b=0),paper_bgcolor="white",plot_bgcolor="white",font_color="#536276",showlegend=False)
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False)
            st.plotly_chart(fig,use_container_width=True)
        else: st.info("Feature importance output is not available.")
        st.markdown("</div>",unsafe_allow_html=True)
    st.markdown("#### First cases to review")
    show=queue.sort_values("priority").head(8).copy()
    show["p_fraud"]=show["p_fraud"].map(lambda x:f"{x:.1%}")
    show["exposure_amount"]=show["exposure_amount"].map(lambda x:f"${x:,.0f}")
    show["expected_exposure"]=show["expected_exposure"].map(lambda x:f"${x:,.0f}")
    st.dataframe(show[[c for c in ["priority","Provider","risk_band","p_fraud","exposure_amount","expected_exposure"] if c in show]],hide_index=True,use_container_width=True)
    st.markdown("<div class='small-note'>Historical label: PotentialFraud. A high score is a prioritization signal for human review, not a finding.</div>",unsafe_allow_html=True)

elif page == "Investigation queue":
    st.markdown("### Investigation queue")
    st.markdown("Search and filter the ranked providers. Select a provider below to open a review brief.")
    f1,f2,f3=st.columns([1.3,1,1])
    with f1: query=st.text_input("Search provider",placeholder="e.g. PRV56560")
    with f2: bands=st.multiselect("Queue band",sorted(queue["risk_band"].dropna().unique()) if "risk_band" in queue else [],default=sorted(queue["risk_band"].dropna().unique()) if "risk_band" in queue else [])
    with f3: min_score=st.slider("Minimum score",0.0,1.0,0.0,0.05,format="%.0f%%")
    filtered=queue.copy()
    if query: filtered=filtered[filtered["Provider"].astype(str).str.contains(query,case=False,na=False)]
    if bands and "risk_band" in filtered: filtered=filtered[filtered["risk_band"].isin(bands)]
    filtered=filtered[filtered["p_fraud"]>=min_score].sort_values("priority")
    st.caption(f"Showing {len(filtered):,} providers")
    table=filtered.copy()
    table["p_fraud"]=table["p_fraud"].map(lambda x:f"{x:.1%}")
    for c in ["exposure_amount","expected_exposure"]:
        if c in table: table[c]=table[c].map(lambda x:f"${x:,.0f}")
    st.dataframe(table[[c for c in ["priority","Provider","risk_band","p_fraud","exposure_amount","expected_exposure"] if c in table]],hide_index=True,use_container_width=True,height=390)
    ids=filtered["Provider"].astype(str).tolist()
    if ids:
        selected=st.selectbox("Open provider review brief",ids)
        rec=filtered.loc[filtered["Provider"].astype(str)==selected].iloc[0]
        st.markdown("---")
        st.markdown(f"#### Review brief · {selected}")
        a,b,c=st.columns(3)
        a.metric("Model score",f"{rec['p_fraud']:.1%}")
        b.metric("Queue exposure",f"${rec['exposure_amount']:,.0f}")
        c.metric("Expected exposure",f"${rec['expected_exposure']:,.0f}")
        if not features.empty and "Provider" in features:
            detail=features[features["Provider"].astype(str)==selected]
            if not detail.empty:
                row=detail.iloc[0]
                d1,d2,d3,d4=st.columns(4)
                for col,label,key,fmt in [(d1,"Claims","n_claims",",.0f"),(d2,"Beneficiaries","n_unique_beneficiaries",",.0f"),(d3,"Inpatient share","inpatient_share",".1%"),(d4,"Mean reimbursement","mean_reimbursed","$,.0f")]:
                    if key in row and pd.notna(row[key]): col.metric(label,format(row[key],fmt))
        st.info("Review the underlying records and context before taking action. Model scores estimate alignment with historical PotentialFraud labels.")
    else: st.info("No providers match these filters.")

else:
    st.markdown("### Model & data")
    st.markdown("Review how the model was evaluated and what data supports this queue.")
    a,b,c=st.columns(3)
    a.metric("Selected model",selection.get("selected_model",threshold.get("selected_model","—")))
    a.metric("Model justified vs rule", "Yes" if selection.get("model_justified") else "Review")
    a.metric("Gain over best rule (PR-AUC)",f"{selection.get('gain_over_best_rule',0):.3f}")
    b.metric("Cost threshold (OOF)",f"{threshold.get('chosen_threshold',0):.0%}")
    tr=threshold.get("optimal_threshold_range_across_assumptions",[0,0])
    b.metric("Threshold range",f"{tr[0]:.0%}–{tr[1]:.0%}")
    b.caption("Threshold uses illustrative assumptions; see sensitivity analysis.")
    b.metric("Providers labeled",f"{n_providers:,}")
    if not metrics.empty:
        st.markdown("#### One-time test evaluation")
        st.dataframe(metrics,hide_index=True,use_container_width=True)
    if not ranking.empty:
        st.markdown("#### Investigation capacity trade-off")
        test_rank=ranking[ranking["dataset"].astype(str).str.contains("test",case=False)]
        st.dataframe(test_rank if not test_rank.empty else ranking.head(20),hide_index=True,use_container_width=True)
    st.markdown("#### Data notes")
    st.markdown(f"- Labeled providers: **{n_providers:,}**; claims: **{audit.get('n_claims',0):,}**; historical fraud label rate: **{fraud_rate:.1%}**.\n- Claim dates in source data: **{(audit.get('date_span') or ['—','—'])[0]}** through **{(audit.get('date_span') or ['—','—'])[-1]}**.\n- The label is `PotentialFraud`, not adjudicated fraud. Results reflect historical Medicare data and require revalidation before another use.")
    st.warning("Decision support only. This model ranks providers for human review; it does not determine fraud, deny claims, or justify sanctions.")
