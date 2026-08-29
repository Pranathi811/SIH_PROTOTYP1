"""Streamlit operations dashboard for the synthetic Mission 1 dataset."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import plotly.express as px
import requests as r
import streamlit as st

from src.data_loader import CHANNELS, load_dataset
from src.models import build_models, channel_deviations, per_type_metrics
from src.validate import validate_dataset
from src.substations import load_substations

st.set_page_config(page_title="Mission Control | Satellite Health", page_icon="S", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;600;700;800&display=swap');
:root { --ink:#102a43; --muted:#627d98; --teal:#00a896; --orange:#f28f3b; --paper:#f4f7f5; }
html, body, [class*="css"] { font-family: Manrope, sans-serif; }
.stApp { background: linear-gradient(135deg,#f4f7f5 0%,#e8f3f0 55%,#fff8ed 100%); color:var(--ink); }
[data-testid="stSidebar"] { background:#102a43; }
[data-testid="stSidebar"] * { color:#f4f7f5; }
.eyebrow { color:#00a896; font:500 12px 'DM Mono'; letter-spacing:1.4px; text-transform:uppercase; }
h1 { font-weight:800; letter-spacing:-1px; color:#102a43; }
.metric-card { background:#fff; border:1px solid #d9e2ec; border-radius:8px; padding:18px 20px; box-shadow:0 4px 18px rgba(16,42,67,.06); }
.metric-label { color:#627d98; font-size:12px; text-transform:uppercase; letter-spacing:1px; }
.metric-value { color:#102a43; font-size:27px; font-weight:800; margin-top:5px; }
.note { color:#627d98; font-size:13px; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_frames():
    return load_dataset()


@st.cache_resource
def get_models():
    return build_models(get_frames())


@st.cache_data
def get_substations():
    return load_substations()


frames = get_frames()
models = get_models()
substations = get_substations()
errors = validate_dataset(frames)
test = frames["test"].copy()
substation_id = st.sidebar.selectbox("Substation", list(substations), format_func=lambda key: substations[key]["display_name"])
substation = substations[substation_id]
model_name = st.sidebar.selectbox("Detection model", ["isolation_forest", "autoencoder"], format_func=lambda x: x.replace("_", " ").title())
channel = st.sidebar.selectbox("Telemetry view", ["All channels", *CHANNELS])
show = st.sidebar.radio("Records", ["All records", "Anomalies only", "Normal only"])
date_min, date_max = test.timestamp.min().date(), test.timestamp.max().date()
dates = st.sidebar.date_input("Time window", (date_min, date_max), min_value=date_min, max_value=date_max)

predicted = models["predictions"][model_name]
scores = models["scores"][model_name]
test["predicted_label"] = predicted
test["anomaly_score"] = scores
test["anomaly_type"] = test.timestamp.map(frames["labels"].set_index("timestamp")["anomaly_type"])
start, end = dates if isinstance(dates, tuple) and len(dates) == 2 else (date_min, date_max)
visible = test[test.timestamp.dt.date.between(start, end)]
if show == "Anomalies only": visible = visible[visible.predicted_label.eq(1)]
if show == "Normal only": visible = visible[visible.predicted_label.eq(0)]

st.markdown(f'<div class="eyebrow">{substation["display_name"].upper()} / ESA-ADB PROTOTYPE</div>', unsafe_allow_html=True)
st.title(f'{substation["display_name"]} health, at a glance')
st.markdown('<div class="note">Synthetic telemetry operations console · 1-minute cadence · results are prototype-only</div>', unsafe_allow_html=True)
if errors:
    st.error("Dataset validation failed: " + " | ".join(errors))

anomaly_count = int(test.predicted_label.sum())
cards = [("Records monitored", f"{len(test):,}"), ("Flagged anomalies", f"{anomaly_count:,}"), ("Anomaly rate", f"{anomaly_count / len(test):.1%}"), ("Model F1", f"{models['metrics'][model_name]['f1_score']:.3f}")]
cols = st.columns(4)
for col, (label, value) in zip(cols, cards):
    col.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)

st.subheader("Telemetry stream")
st.caption("Parameter labels are configurable per substation. The supplied synthetic dataset remains channel-based until an approved data dictionary is provided.")
plot_channels = CHANNELS if channel == "All channels" else [channel]
long = visible[["timestamp", *plot_channels]].melt("timestamp", var_name="channel", value_name="reading")
long["channel"] = long["channel"].map(substation["parameters"])
fig = px.line(long, x="timestamp", y="reading", color="channel", template="simple_white", height=400, color_discrete_sequence=["#00a896", "#f28f3b", "#3478b9", "#7b61a8", "#d1495b"])
anomalies = visible[visible.predicted_label.eq(1)]
for _, row in anomalies.iterrows():
    fig.add_vline(x=row.timestamp, line_width=0.6, line_color="#f28f3b", opacity=0.28)
fig.update_layout(margin=dict(l=10,r=10,t=20,b=10), legend_title_text="")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Channel diagnostics")
diagnostic_timestamp = st.selectbox("Inspect a test reading", test.timestamp.dt.strftime("%Y-%m-%d %H:%M:%S"), index=0)
diagnostic = test[test.timestamp.dt.strftime("%Y-%m-%d %H:%M:%S").eq(diagnostic_timestamp)].iloc[0]
diagnostic_rows = channel_deviations(models, [float(diagnostic[channel_name]) for channel_name in CHANNELS])
diagnostic_frame = pd.DataFrame(diagnostic_rows).set_index("channel")
diagnostic_frame.index = diagnostic_frame.index.map(substation["parameters"])
st.dataframe(diagnostic_frame[["value", "z_score"]].style.format({"value":"{:.4f}", "z_score":"{:.3f}"}), use_container_width=True)
st.caption("Absolute z-score shows how far each channel is from the training baseline; the largest value is the strongest contributor.")

left, right = st.columns([1.05, 1])
with left:
    st.subheader("Anomaly types")
    type_counts = frames["anomaly_types"]["anomaly_type"].value_counts().rename_axis("type").reset_index(name="count")
    st.plotly_chart(px.bar(type_counts, x="type", y="count", color="type", template="simple_white", height=300, color_discrete_sequence=["#00a896", "#f28f3b", "#3478b9", "#d1495b"]), use_container_width=True)
with right:
    st.subheader("Model performance")
    metrics = pd.DataFrame(models["metrics"]).T[["precision", "recall", "f1_score", "roc_auc", "pr_auc"]]
    st.dataframe(metrics.style.format("{:.3f}"), use_container_width=True)
    st.caption("Metrics are calculated against the supplied timestamp-level labels.")

st.subheader("Per-type detection")
st.dataframe(pd.DataFrame(per_type_metrics(frames, predicted)).set_index("type").style.format({"precision":"{:.3f}","recall":"{:.3f}","f1":"{:.3f}"}), use_container_width=True)
st.caption(f"Showing {len(visible):,} of {len(test):,} test records · threshold {models['thresholds'][model_name]:.4f} · synthetic data, not official ESA-ADB benchmark results")
