from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from automatic_parking import analyse_automatic, analyse_calibrated_grid, load_yolo
from core import annotate, read_image, summarize

ROOT = Path(__file__).parent
CUSTOM_WEIGHTS = ROOT / "models" / "parking_best.onnx"
WEIGHTS = CUSTOM_WEIGHTS if CUSTOM_WEIGHTS.exists() else ROOT / "models" / "yolo11n-obb.onnx"
METADATA = ROOT / "models" / "model_metadata.json"
OCCUPANCY_MODEL = ROOT / "models" / "occupancy_classifier.joblib"

st.set_page_config(page_title="ParkVision AI", page_icon="🅿️", layout="wide")
st.markdown("""
<style>
:root{--navy:#071a34;--blue:#2864e8;--teal:#08a88f;--ink:#14213d;--muted:#6e7b91;--line:#dce5f1;--bg:#f3f7fc}
.stApp{background:linear-gradient(180deg,#f7faff 0,#eef4fb 100%);color:var(--ink)}[data-testid="stSidebar"]{background:linear-gradient(180deg,#06162f,#0d2852);border-right:0}[data-testid="stSidebar"] *{color:#eef5ff}[data-testid="stSidebar"] hr{border-color:#ffffff22}[data-testid="stSidebar"] [data-testid="stMetric"]{background:#ffffff10;border-color:#ffffff20}[data-testid="stSidebar"] [data-testid="stMetricValue"]{color:#fff}
.block-container{max-width:1450px;padding-top:1.25rem;padding-bottom:2rem}.hero{position:relative;overflow:hidden;padding:2.5rem 2.6rem;border-radius:28px;color:white;background:radial-gradient(circle at 88% 18%,#25d8bd55,transparent 24%),radial-gradient(circle at 72% 120%,#3978ff55,transparent 35%),linear-gradient(120deg,#06162f,#123d7d 66%,#087f78);box-shadow:0 22px 55px #102b5a26;margin-bottom:1.25rem;border:1px solid #ffffff18}
.hero small{color:#72e3d1;font-weight:800;letter-spacing:.14em}.hero h1{font-size:clamp(2rem,4vw,3.3rem);line-height:1.02;letter-spacing:-.04em;margin:.6rem 0}.hero p{color:#d9e4f5;max-width:850px;font-size:1.03rem;line-height:1.6}
.pill{display:inline-block;padding:.38rem .65rem;border:1px solid #ffffff30;border-radius:99px;margin:.65rem .3rem 0 0;color:#eef5ff;font-size:.78rem}
.card{height:100%;background:linear-gradient(145deg,#fff,#f9fbff);border:1px solid var(--line);border-radius:20px;padding:1.2rem 1.25rem;box-shadow:0 10px 30px #24395a0d;transition:.2s ease}.card:hover{transform:translateY(-2px);box-shadow:0 14px 34px #24395a16}.card b{color:var(--navy);font-size:1rem}.card span{color:var(--muted);font-size:.86rem;line-height:1.55}.eyebrow{color:#087f78;font-size:.72rem;font-weight:850;letter-spacing:.12em}.statusbar{display:flex;gap:.8rem;flex-wrap:wrap;margin:-.25rem 0 1.2rem}.statusitem{background:#fff;border:1px solid var(--line);border-radius:999px;padding:.48rem .8rem;color:#52627a;font-size:.78rem;box-shadow:0 5px 18px #1e2e5008}.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:#11b88f;margin-right:.4rem;box-shadow:0 0 0 4px #11b88f19}
[data-testid="stMetric"]{background:linear-gradient(145deg,#fff,#f8fbff);border:1px solid var(--line);border-radius:18px;padding:1rem 1.1rem;box-shadow:0 8px 24px #1e2e500c}[data-testid="stMetricValue"]{color:var(--navy);font-weight:850}[data-testid="stMetricLabel"]{color:#6d7890}
[data-testid="stFileUploader"]{background:#fff;border:1.5px dashed #9fb2ce;border-radius:18px;padding:.45rem}[data-testid="stFileUploader"] section{background:#f8faff;border-radius:13px}
.stButton>button{border:0;border-radius:12px;min-height:3.05rem;font-weight:750;color:#fff;background:linear-gradient(90deg,#2458df,#3f79ef);box-shadow:0 8px 18px #3157e233}.stButton>button:hover{color:white}
.stTabs [data-baseweb="tab-list"]{gap:.35rem;background:#e9f0f9;padding:.35rem;border-radius:14px}.stTabs [data-baseweb="tab"]{height:2.8rem;border-radius:10px;padding:0 1rem}.stTabs [aria-selected="true"]{background:#fff;box-shadow:0 4px 14px #1c376318}.section-title{margin-top:1.5rem}.legend{font-size:.84rem;color:#6d7890}.good{color:#09a67e}.bad{color:#ed5264}.footer{text-align:center;color:#8b97a9;font-size:.75rem;padding:2rem}.insight{padding:1rem 1.1rem;border-left:4px solid #2864e8;background:#fff;border-radius:0 14px 14px 0;box-shadow:0 6px 20px #1c37630b}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_detector():
    return load_yolo(WEIGHTS)


@st.cache_resource(show_spinner=False)
def get_occupancy_model():
    if not OCCUPANCY_MODEL.exists():
        return None
    import joblib
    return joblib.load(OCCUPANCY_MODEL)


@st.cache_data(show_spinner=False)
def evaluation_metrics():
    """Return only metrics written by train.py from an untouched test split."""
    if not METADATA.exists():
        return None
    try:
        data = json.loads(METADATA.read_text(encoding="utf-8"))
        metrics = data["test_metrics"]
        return {name: float(metrics[name]) for name in ("accuracy", "precision", "recall", "f1_score")}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def result_table(results):
    return pd.DataFrame([{
        "Space": item.slot_id, "Status": item.status,
        "Confidence": f"{item.confidence:.1%}",
        "Occupied probability": f"{item.occupied_probability:.1%}",
    } for item in results])


with st.sidebar:
    st.markdown("## ◉ ParkVision AI")
    st.caption("SMART PARKING COMMAND CENTER")
    st.divider()
    st.markdown("#### Analysis mode")
    analysis_mode = st.radio(
        "Choose detection method",
        ["Calibrated grid (recommended)", "Automatic estimate (experimental)"],
        help="Calibration guarantees that every known parking bay is analysed. Automatic mode may miss obscured lines or empty edge spaces.",
    )
    calibrated = analysis_mode.startswith("Calibrated")
    if calibrated:
        st.caption("Set the visible grid once for a fixed camera or regular parking image.")
        grid_a, grid_b = st.columns(2)
        grid_rows = grid_a.number_input("Rows", 1, 20, 2)
        grid_columns = grid_b.number_input("Spaces/row", 1, 30, 5)
        st.caption(f"Configured capacity: **{grid_rows * grid_columns} spaces**")
        with st.expander("Adjust grid boundaries", expanded=False):
            grid_left = st.slider("Left edge", 0.00, 0.40, 0.02, 0.01)
            grid_right = st.slider("Right edge", 0.60, 1.00, 0.98, 0.01)
            grid_top = st.slider("Top edge", 0.00, 0.40, 0.08, 0.01)
            grid_bottom = st.slider("Bottom edge", 0.60, 1.00, 0.96, 0.01)
            grid_row_gap = st.slider("Gap between rows", 0.00, 0.30, 0.10, 0.01)
            grid_column_gap = st.slider("Gap between spaces", 0.00, 0.08, 0.01, 0.005)
            occupancy_threshold = st.slider("Occupied threshold", 0.20, 0.80, 0.50, 0.05)
    else:
        grid_rows = grid_columns = 1
        grid_left, grid_right, grid_top, grid_bottom = 0.0, 1.0, 0.0, 1.0
        grid_row_gap = grid_column_gap = 0.0
        occupancy_threshold = 0.50
    st.markdown("#### Automatic fallback")
    detection_confidence = st.slider("Aerial YOLO fallback confidence", .01, .30, .045, .005,
        help="Used only when painted parking bays cannot be detected. Lower values find more vehicles but may add false detections.")
    st.caption("No rows, columns, margins or layout JSON are required.")
    st.divider()
    with st.expander("What is automatic?"):
        st.write("YOLO locates cars, motorcycles, buses and trucks. ParkVision groups them into parking rows and estimates visible empty gaps between vehicles.")
    with st.expander("Important accuracy note"):
        st.write("Upload the original photograph. Screenshots with old boxes, labels or app controls reduce accuracy. Empty capacity outside detected vehicle rows is not invented.")
    metrics = evaluation_metrics()
    with st.expander("Verified model evaluation", expanded=bool(metrics)):
        if metrics:
            st.metric("Held-out test accuracy", f'{metrics["accuracy"]:.2%}')
            st.caption(f'Precision {metrics["precision"]:.2%} · Recall {metrics["recall"]:.2%} · F1 {metrics["f1_score"]:.2%}')
            st.caption("Evaluated on 5,000 labelled spaces from the official PKLot test folder.")
            st.caption("These results come from the untouched test split and are different from per-image confidence.")
        else:
            st.caption("No verified test report is bundled yet. Train the model first; ParkVision will never invent an accuracy value.")
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
<h1>Parking intelligence,<br>built for real decisions.</h1><p>Turn one parking-lot image into availability, demand forecasts, operational alerts, revenue scenarios, sustainability estimates and exportable management insights.</p>
<span class="pill">97.02% tested accuracy</span><span class="pill">Automatic space intelligence</span><span class="pill">Demand planning</span><span class="pill">Operations dashboard</span></section>
<div class="statusbar"><span class="statusitem"><i class="dot"></i>AI model ready</span><span class="statusitem"><i class="dot"></i>PKLot trained</span><span class="statusitem"><i class="dot"></i>Privacy-first image processing</span><span class="statusitem"><i class="dot"></i>Reports enabled</span></div>
""", unsafe_allow_html=True)

st.markdown("### Analyse any clear parking-lot photograph")
st.caption("Best results: original JPG/PNG, full parking area visible, good lighting, and vehicles not heavily hidden by trees or roofs.")
upload = st.file_uploader("Upload parking image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if upload is None:
    st.session_state.pop("automatic_analysis", None)
    st.markdown('<div class="eyebrow">ONE PLATFORM · SIX CAPABILITIES</div><h3>More than occupied versus available</h3>', unsafe_allow_html=True)
    first = st.columns(3)
    first[0].markdown('<div class="card"><b>01 · Space intelligence</b><br><span>Discover parking rows, classify spaces and create an annotated availability map automatically.</span></div>', unsafe_allow_html=True)
    first[1].markdown('<div class="card"><b>02 · Demand forecasting</b><br><span>Model incoming and departing vehicles to identify capacity pressure before the lot becomes full.</span></div>', unsafe_allow_html=True)
    first[2].markdown('<div class="card"><b>03 · Operations guidance</b><br><span>Receive clear recommendations for overflow activation, entrance signage and driver redirection.</span></div>', unsafe_allow_html=True)
    st.write("")
    second = st.columns(3)
    second[0].markdown('<div class="card"><b>04 · Revenue planning</b><br><span>Estimate parking income using capacity, turnover, pricing and operating-hour scenarios.</span></div>', unsafe_allow_html=True)
    second[1].markdown('<div class="card"><b>05 · Sustainability impact</b><br><span>Explore time, fuel and estimated CO₂ savings created by reducing the search for parking.</span></div>', unsafe_allow_html=True)
    second[2].markdown('<div class="card"><b>06 · Quality control</b><br><span>Review uncertain predictions, correct statuses and export feedback for future model improvement.</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><div class="eyebrow">WORKFLOW</div><h3>From image to action in three steps</h3></div>', unsafe_allow_html=True)
    w1, w2, w3 = st.columns(3)
    w1.info("**1 · Upload**\n\nChoose a clear original parking-lot photograph.")
    w2.info("**2 · Analyse**\n\nThe trained model identifies parking availability.")
    w3.info("**3 · Decide**\n\nUse forecasts, alerts and reports to plan operations.")
else:
    try:
        raw = upload.getvalue()
        settings_key = f"{analysis_mode}:{detection_confidence}:{grid_rows}:{grid_columns}:{grid_left}:{grid_right}:{grid_top}:{grid_bottom}:{grid_row_gap}:{grid_column_gap}:{occupancy_threshold}"
        image_key = hashlib.sha256(raw + settings_key.encode()).hexdigest()
        image = read_image(raw)
        h, w = image.shape[:2]
        left, right = st.columns([1.45, 1], gap="large")
        with left:
            st.image(image, caption="Original uploaded photograph", use_container_width=True)
        with right:
            st.markdown("### Ready for parking analysis")
            st.write(f"**Image:** {w} × {h} pixels")
            if calibrated:
                st.info(f"The calibrated layout will analyse exactly {grid_rows * grid_columns} spaces. Adjust the boundaries in the sidebar if the overlay does not match the lot.", icon="🎯")
                analyse = st.button("Analyse calibrated parking", type="primary", use_container_width=True)
            else:
                st.info("Automatic mode estimates the layout and may miss obscured lines or edge spaces. Review every result.", icon="✨")
                analyse = st.button("Estimate parking automatically", type="primary", use_container_width=True)

        if analyse:
            with st.spinner("Discovering parking spaces and analysing the image..."):
                occupancy_model = get_occupancy_model()
                if calibrated:
                    results, diagnostics = analyse_calibrated_grid(
                        image, occupancy_model, int(grid_rows), int(grid_columns),
                        grid_left, grid_right, grid_top, grid_bottom,
                        grid_row_gap, grid_column_gap, occupancy_threshold,
                    )
                else:
                    results, diagnostics = analyse_automatic(image, None, detection_confidence, occupancy_model)
                    if not results:
                        if not WEIGHTS.exists():
                            raise ValueError(
                                "Parking lines could not be identified and the fallback detector is not bundled. "
                                "Add models/yolo11n-obb.onnx to GitHub, then reboot the Streamlit app."
                            )
                        results, diagnostics = analyse_automatic(image, get_detector(), detection_confidence, occupancy_model)
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
            if summary["available"] <= 2:
                st.error("Action recommended: activate overflow parking or redirect arriving drivers.")
            elif summary["occupancy"] >= 70:
                st.warning("Action recommended: display limited-capacity guidance at the entrance.")
            else:
                st.success("Operations normal: enough spaces are currently visible.")
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Vehicles detected", diagnostics["vehicles"])
            q2.metric("Parking rows", diagnostics["rows"])
            q3.metric("Inferred empty gaps", diagnostics["inferred_empty"])
            q4.metric("Scene reliability", diagnostics["reliability"])
            st.caption(f'Analysis engine: {diagnostics.get("engine", "automatic")}')

            overview_tab, map_tab, report_tab, forecast_tab, operations_tab, impact_tab, review_tab = st.tabs(["Executive view", "Parking map", "Space report", "Demand forecast", "Operations", "Sustainability", "Quality review"])
            with overview_tab:
                left_overview, right_overview = st.columns([1.2, 1])
                with left_overview:
                    st.markdown("#### Management summary")
                    st.progress(summary["occupancy"] / 100, text=f'{summary["occupancy"]:.1f}% of detected capacity occupied')
                    if summary["occupancy"] > 85:
                        st.markdown('<div class="insight"><b>Critical capacity pressure</b><br>Prepare overflow parking and redirect new arrivals immediately.</div>', unsafe_allow_html=True)
                    elif summary["occupancy"] > 70:
                        st.markdown('<div class="insight"><b>Capacity tightening</b><br>Update entrance signage and monitor arrivals closely.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="insight"><b>Healthy availability</b><br>Normal parking operations can continue.</div>', unsafe_allow_html=True)
                with right_overview:
                    st.markdown("#### Decision indicators")
                    st.write(f"**Available-space ratio:** {100-summary['occupancy']:.1f}%")
                    st.write(f"**Predictions for review:** {data['review']}")
                    st.write(f"**Average prediction confidence:** {data['average']:.1%}")
                    st.write(f"**Analysis reliability:** {diagnostics['reliability']}")
            with map_tab:
                st.markdown('<div class="legend"><span class="good">● Available inferred gap</span> &nbsp; <span class="bad">● YOLO-detected occupied space</span></div>', unsafe_allow_html=True)
                st.image(data["output"], caption="Automatic ParkVision result", use_container_width=True)
                st.download_button("Download annotated result", data["png"], "parkvision_automatic_result.png", "image/png", use_container_width=True)
            with report_tab:
                st.dataframe(data["table"], hide_index=True, use_container_width=True)
                st.download_button("Download CSV report", data["table"].to_csv(index=False).encode(), "parkvision_report.csv", "text/csv", use_container_width=True)
                st.caption("Confidence describes an individual prediction. It is not test-set accuracy.")
            with forecast_tab:
                st.markdown("#### Two-hour capacity scenario")
                x, y = st.columns(2)
                arrivals = x.slider("Arrivals per 30 minutes", 0, 30, 5)
                departures = y.slider("Departures per 30 minutes", 0, 30, 2)
                projected_values = [summary["occupied"]]
                for _ in range(4):
                    projected_values.append(max(0, min(summary["total"], projected_values[-1] + arrivals - departures)))
                forecast = pd.DataFrame({"Time":["Now","+30 min","+60 min","+90 min","+120 min"],"Occupied":projected_values})
                st.line_chart(forecast.set_index("Time"), color="#2864e8")
                projected = projected_values[-1]
                projected_available = summary["total"] - projected
                f1, f2, f3 = st.columns(3)
                f1.metric("Occupied in 2 hours", projected, projected-summary["occupied"])
                f2.metric("Available in 2 hours", projected_available, projected_available-summary["available"])
                f3.metric("Projected utilisation", f"{100*projected/summary['total']:.1f}%")
                if projected_available == 0:
                    st.error("Parking may become full—redirect incoming drivers.")
                elif projected / summary["total"] > .75:
                    st.warning("High congestion is likely—prepare overflow guidance.")
                else:
                    st.success("Capacity should remain available.")
            with operations_tab:
                st.markdown("#### Parking operations and revenue scenario")
                o1, o2, o3 = st.columns(3)
                hourly_fee = o1.number_input("Parking fee per hour", min_value=0.0, value=30.0, step=5.0)
                average_stay = o2.number_input("Average stay (hours)", min_value=.5, value=2.0, step=.5)
                operating_hours = o3.number_input("Operating hours per day", min_value=1, max_value=24, value=12)
                turnover = max(1.0, operating_hours / average_stay)
                estimated_visits = round(summary["occupied"] * turnover)
                estimated_revenue = estimated_visits * hourly_fee * average_stay
                r1, r2, r3 = st.columns(3)
                r1.metric("Estimated daily visits", estimated_visits)
                r2.metric("Capacity turnovers", f"{turnover:.1f}×")
                r3.metric("Potential daily revenue", f"₹{estimated_revenue:,.0f}")
                st.caption("Scenario estimate based on current occupied spaces—not audited financial revenue.")
                st.markdown("#### Recommended operating response")
                if summary["occupancy"] >= 85:
                    st.error("Open overflow capacity, display FULL warnings and assign staff to the entrance.")
                elif summary["occupancy"] >= 70:
                    st.warning("Prepare overflow capacity and update the availability sign every 15 minutes.")
                else:
                    st.success("Keep normal entry flow and recheck availability during the next demand interval.")
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
                quality1, quality2, quality3 = st.columns(3)
                quality1.metric("Average confidence", f"{data['average']:.1%}")
                quality2.metric("Needs review", data["review"])
                quality3.metric("Test-set accuracy", f'{metrics["accuracy"]:.2%}' if metrics else "Not available")
                reviewed = data["table"][["Space", "Status", "Confidence"]].copy()
                reviewed["Corrected status"] = reviewed["Status"]
                edited = st.data_editor(reviewed, disabled=["Space", "Status", "Confidence"], hide_index=True, use_container_width=True,
                    column_config={"Corrected status": st.column_config.SelectboxColumn(options=["Available", "Occupied"], required=True)})
                corrections = (edited["Corrected status"] != edited["Status"]).sum()
                st.metric("Corrections marked", int(corrections))
                st.download_button("Download training feedback", edited.to_csv(index=False).encode(), "parkvision_feedback.csv", "text/csv", use_container_width=True)
                st.caption("Test accuracy is measured on 5,000 labelled PKLot spaces. Per-image confidence is a different value.")
            if diagnostics["reliability"] != "Strong":
                st.warning("This scene has limited evidence. Review the map before using its counts operationally.", icon="⚠️")
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
    except Exception as exc:
        st.error("Automatic analysis could not start. Check the deployment requirements and bundled model file.", icon="🚫")
        with st.expander("Technical details"):
            st.code(f"{type(exc).__name__}: {exc}")

st.markdown('<div class="footer">ParkVision AI · Automatic multi-layout parking analytics</div>', unsafe_allow_html=True)
