from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from automatic_parking import analyse_automatic, load_yolo
from core import annotate, read_image, summarize

ROOT = Path(__file__).parent
CUSTOM_WEIGHTS = ROOT / "models" / "parking_best.onnx"
WEIGHTS = CUSTOM_WEIGHTS if CUSTOM_WEIGHTS.exists() else ROOT / "models" / "yolo11n-obb.onnx"

st.set_page_config(page_title="ParkVision AI", page_icon="🅿️", layout="wide")
st.markdown("""
<style>
:root{--navy:#071a34;--blue:#2864e8;--teal:#08a88f;--ink:#14213d;--muted:#6e7b91;--line:#dce5f1;--bg:#f3f7fc}
.stApp{background:var(--bg);color:var(--ink)}[data-testid="stSidebar"]{background:#fff;border-right:1px solid var(--line)}
.block-container{max-width:1450px;padding-top:1.5rem}.hero{padding:2.2rem 2.4rem;border-radius:26px;color:white;background:linear-gradient(120deg,#071a34,#173b7b 70%,#087f78);box-shadow:0 18px 46px #1c376322;margin-bottom:1.25rem}
.hero small{color:#72e3d1;font-weight:800;letter-spacing:.14em}.hero h1{font-size:clamp(2rem,4vw,3.3rem);line-height:1.02;letter-spacing:-.04em;margin:.6rem 0}.hero p{color:#d9e4f5;max-width:850px;font-size:1.03rem;line-height:1.6}
.pill{display:inline-block;padding:.38rem .65rem;border:1px solid #ffffff30;border-radius:99px;margin:.65rem .3rem 0 0;color:#eef5ff;font-size:.78rem}
.card{background:#fff;border:1px solid var(--line);border-radius:18px;padding:1rem 1.1rem;box-shadow:0 7px 22px #24395a0c}.card b{color:var(--navy)}.card span{color:var(--muted);font-size:.84rem}
[data-testid="stMetric"]{background:#fff;border:1px solid var(--line);border-radius:17px;padding:1rem 1.1rem;box-shadow:0 7px 20px #1e2e500d}[data-testid="stMetricValue"]{color:var(--navy);font-weight:850}
[data-testid="stFileUploader"]{background:#fff;border:1.5px dashed #9fb2ce;border-radius:18px;padding:.45rem}[data-testid="stFileUploader"] section{background:#f8faff;border-radius:13px}
.stButton>button{border:0;border-radius:12px;min-height:3.05rem;font-weight:750;color:#fff;background:linear-gradient(90deg,#2458df,#3f79ef);box-shadow:0 8px 18px #3157e233}.stButton>button:hover{color:white}
.legend{font-size:.84rem;color:#6d7890}.good{color:#09a67e}.bad{color:#ed5264}.footer{text-align:center;color:#8b97a9;font-size:.75rem;padding:2rem}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_detector():
    return load_yolo(WEIGHTS)


def result_table(results):
    return pd.DataFrame([{
        "Space": item.slot_id, "Status": item.status,
        "Confidence": f"{item.confidence:.1%}",
        "Occupied probability": f"{item.occupied_probability:.1%}",
    } for item in results])


with st.sidebar:
    st.markdown("## 🅿️ ParkVision AI")
    st.caption("UrbanFlow · Smart mobility")
    st.divider()
    st.markdown("#### Automatic analysis")
    detection_confidence = st.slider("Aerial YOLO fallback confidence", .01, .30, .045, .005,
        help="Used only when painted parking bays cannot be detected. Lower values find more vehicles but may add false detections.")
    st.caption("No rows, columns, margins or layout JSON are required.")
    st.divider()
    with st.expander("What is automatic?"):
        st.write("YOLO locates cars, motorcycles, buses and trucks. ParkVision groups them into parking rows and estimates visible empty gaps between vehicles.")
    with st.expander("Important accuracy note"):
        st.write("Upload the original photograph. Screenshots with old boxes, labels or app controls reduce accuracy. Empty capacity outside detected vehicle rows is not invented.")
    with st.expander("Kaggle dataset connection"):
        try:
            kaggle_ready = bool(st.secrets.get("KAGGLE_USERNAME") and st.secrets.get("KAGGLE_KEY"))
        except Exception:
            kaggle_ready = False
        if kaggle_ready:
            st.success("Kaggle API configured securely")
        else:
            st.caption("Kaggle is used only for retraining. Add credentials through Streamlit Secrets; never place a private key in app.py or GitHub.")

st.markdown("""
<section class="hero"><small>URBANFLOW AI · AUTOMATIC PARKING INTELLIGENCE</small>
<h1>Upload. Detect. Park.</h1><p>ParkVision automatically locates vehicles, discovers parking rows, estimates visible spaces and converts the result into live availability and urban-impact insights.</p>
<span class="pill">Automatic YOLO detection</span><span class="pill">No manual grid</span><span class="pill">Parking forecast</span><span class="pill">Human review loop</span></section>
""", unsafe_allow_html=True)

st.markdown("### Analyse any clear parking-lot photograph")
st.caption("Best results: original JPG/PNG, full parking area visible, good lighting, and vehicles not heavily hidden by trees or roofs.")
upload = st.file_uploader("Upload parking image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if upload is None:
    st.session_state.pop("automatic_analysis", None)
    a, b, c = st.columns(3)
    a.markdown('<div class="card"><b>Automatic layout discovery</b><br><span>Parking rows are inferred from vehicle positions—no fixed 12-slot template.</span></div>', unsafe_allow_html=True)
    b.markdown('<div class="card"><b>Honest confidence</b><br><span>Low-confidence detections are clearly reported for human review.</span></div>', unsafe_allow_html=True)
    c.markdown('<div class="card"><b>More than detection</b><br><span>Forecast demand, estimate time saved and export an operational report.</span></div>', unsafe_allow_html=True)
else:
    try:
        raw = upload.getvalue()
        image_key = hashlib.sha256(raw + str(detection_confidence).encode()).hexdigest()
        image = read_image(raw)
        h, w = image.shape[:2]
        left, right = st.columns([1.45, 1], gap="large")
        with left:
            st.image(image, caption="Original uploaded photograph", use_container_width=True)
        with right:
            st.markdown("### Ready for automatic analysis")
            st.write(f"**Image:** {w} × {h} pixels")
            st.info("The detector identifies vehicles and constructs parking rows automatically. No grid calibration is required.", icon="✨")
            analyse = st.button("Detect parking automatically", type="primary", use_container_width=True)

        if analyse:
            with st.spinner("Loading YOLO and analysing the complete parking area..."):
                model = get_detector()
                results, diagnostics = analyse_automatic(image, model, detection_confidence)
                if not results:
                    raise ValueError("No parked vehicles were detected. Use a clearer, uncropped parking photograph or reduce detection sensitivity. ParkVision will not invent spaces without visual evidence.")
                summary = summarize(results)
                marked = annotate(image, results)
                png = io.BytesIO()
                Image.fromarray(marked).save(png, format="PNG")
                table = result_table(results)
                st.session_state["automatic_analysis"] = {
                    "key": image_key, "summary": summary, "output": marked, "png": png.getvalue(),
                    "table": table, "diagnostics": diagnostics,
                    "review": sum(item.confidence < .70 for item in results),
                    "average": sum(item.confidence for item in results) / len(results),
                }

        data = st.session_state.get("automatic_analysis")
        if data and data["key"] == image_key:
            summary, diagnostics = data["summary"], data["diagnostics"]
            st.markdown("---\n## Parking intelligence overview")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Estimated spaces", summary["total"])
            m2.metric("Available", summary["available"])
            m3.metric("Occupied", summary["occupied"])
            m4.metric("Utilisation", f'{summary["occupancy"]:.1f}%')
            st.info(f'**{summary["level"]} congestion:** {summary["message"]}')
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Vehicles detected", diagnostics["vehicles"])
            q2.metric("Parking rows", diagnostics["rows"])
            q3.metric("Inferred empty gaps", diagnostics["inferred_empty"])
            q4.metric("Scene reliability", diagnostics["reliability"])
            st.caption(f'Analysis engine: {diagnostics.get("engine", "automatic")}')

            map_tab, report_tab, forecast_tab, impact_tab, review_tab = st.tabs(["Parking map", "Space report", "Capacity forecast", "Urban impact", "Review predictions"])
            with map_tab:
                st.markdown('<div class="legend"><span class="good">● Available inferred gap</span> &nbsp; <span class="bad">● YOLO-detected occupied space</span></div>', unsafe_allow_html=True)
                st.image(data["output"], caption="Automatic ParkVision result", use_container_width=True)
                st.download_button("Download annotated result", data["png"], "parkvision_automatic_result.png", "image/png", use_container_width=True)
            with report_tab:
                st.dataframe(data["table"], hide_index=True, use_container_width=True)
                st.download_button("Download CSV report", data["table"].to_csv(index=False).encode(), "parkvision_report.csv", "text/csv", use_container_width=True)
                st.caption("Confidence describes an individual prediction. It is not test-set accuracy.")
            with forecast_tab:
                x, y = st.columns(2)
                arrivals = x.slider("Expected arrivals in 30 minutes", 0, 30, 5)
                departures = y.slider("Expected departures in 30 minutes", 0, 30, 2)
                projected = max(0, min(summary["total"], summary["occupied"] + arrivals - departures))
                available = summary["total"] - projected
                f1, f2 = st.columns(2)
                f1.metric("Projected occupied", projected, projected - summary["occupied"])
                f2.metric("Projected available", available, available - summary["available"])
                if available == 0:
                    st.error("Parking may become full—redirect incoming drivers.")
                elif projected / summary["total"] > .75:
                    st.warning("High congestion is likely—prepare overflow guidance.")
                else:
                    st.success("Capacity should remain available.")
            with impact_tab:
                drivers = st.slider("Drivers guided per day", 10, 500, 100, 10)
                minutes = st.slider("Search time avoided per driver", 1., 10., 4., .5)
                hours = drivers * minutes / 60
                i1, i2, i3 = st.columns(3)
                i1.metric("Time saved/day", f"{hours:.1f} h")
                i2.metric("Fuel saved/day", f"{hours*.8:.1f} L")
                i3.metric("CO₂ avoided/day", f"{hours*.8*2.31:.1f} kg")
                st.caption("Planning estimate, not a measured emissions result.")
            with review_tab:
                reviewed = data["table"][["Space", "Status", "Confidence"]].copy()
                reviewed["Corrected status"] = reviewed["Status"]
                edited = st.data_editor(reviewed, disabled=["Space", "Status", "Confidence"], hide_index=True, use_container_width=True,
                    column_config={"Corrected status": st.column_config.SelectboxColumn(options=["Available", "Occupied"], required=True)})
                corrections = (edited["Corrected status"] != edited["Status"]).sum()
                st.metric("Corrections marked", int(corrections))
                st.download_button("Download training feedback", edited.to_csv(index=False).encode(), "parkvision_feedback.csv", "text/csv", use_container_width=True)
            if diagnostics["reliability"] != "Strong":
                st.warning("This scene has limited evidence. Review the map before using its counts operationally.", icon="⚠️")
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
    except Exception as exc:
        st.error("Automatic analysis could not start. Install the project requirements and confirm models/yolo11n-obb.onnx is present.", icon="🚫")
        with st.expander("Technical details"):
            st.code(f"{type(exc).__name__}: {exc}")

st.markdown('<div class="footer">ParkVision AI · Automatic multi-layout parking analytics</div>', unsafe_allow_html=True)
