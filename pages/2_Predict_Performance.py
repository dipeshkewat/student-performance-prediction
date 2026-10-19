"""Page 2: Predict Performance - Individual Student Grade Forecaster & Batch CSV Generator."""

from __future__ import annotations

import io
from pathlib import Path
import streamlit as st
import pandas as pd

from src.dashboard_utils import (
    apply_modern_theme,
    render_top_navbar,
    generate_sample_csv,
    create_donut_gauge,
)
from src.predict import (
    predict_single,
    predict_batch,
    classify_performance_tier,
)
from src.preprocessing import ALL_FEATURES

st.set_page_config(
    page_title="Predict Performance | Smart Dashboard",
    page_icon="🔮",
    layout="wide",
)

apply_modern_theme()
render_top_navbar(user_name="Grace Stanley", role="Academic Advisor")

st.title("🔮 Student Performance Predictor & Simulator")
st.caption("Forecast final academic achievement (G3) using trained machine learning intelligence.")

model_path = Path("models") / "best_pipeline.joblib"
if not model_path.exists():
    st.error(
        "⚠️ The production model pipeline is not yet trained. "
        "Please run `python -m src.train` in your terminal to train and serialize the champion pipeline."
    )
    st.stop()

tab_single, tab_batch = st.tabs(["👤 Interactive Student Predictor", "📁 Batch Class CSV Generator"])

# ==============================================================================
# TAB 1: INDIVIDUAL PREDICTION
# ==============================================================================
with tab_single:
    st.markdown(
        """
        <div style="background: #EFF6FF; border-left: 4px solid #3B82F6; padding: 14px 18px; border-radius: 12px; margin-bottom: 20px;">
            <strong style="color: #1E3A8A;">Forecasting Note:</strong>
            <span style="color: #1E40AF; font-size: 0.9rem;">
                This simulator estimates the student's <strong>final grade (G3)</strong> after preliminary period grades
                (<strong>G1</strong> and <strong>G2</strong>) are recorded. Use this tool to simulate what happens if a student increases study hours or reduces absences!
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("single_prediction_form"):
        st.subheader("1. Preliminary Period Grades")
        c1, c2 = st.columns(2)
        with c1:
            g1_val = st.number_input(
                "First Period Grade (G1) [0 to 20]:",
                min_value=0.0,
                max_value=20.0,
                value=13.0,
                step=0.5,
                help="Student mark in first trimester",
            )
        with c2:
            g2_val = st.number_input(
                "Second Period Grade (G2) [0 to 20]:",
                min_value=0.0,
                max_value=20.0,
                value=14.0,
                step=0.5,
                help="Student mark in second trimester",
            )

        st.subheader("2. Study Habits & Attendance")
        c3, c4, c5 = st.columns(3)
        with c3:
            studytime_val = st.selectbox(
                "Weekly Study Time:",
                options=[1, 2, 3, 4],
                format_func=lambda x: {
                    1: "Level 1: < 2 hrs / week",
                    2: "Level 2: 2 to 5 hrs / week",
                    3: "Level 3: 5 to 10 hrs / week",
                    4: "Level 4: > 10 hrs / week",
                }[x],
                index=1,
            )
        with c4:
            failures_val = st.number_input(
                "Past Retentions (failures) [0 to 4]:",
                min_value=0,
                max_value=4,
                value=0,
                step=1,
                help="Previous course retentions",
            )
        with c5:
            absences_val = st.number_input(
                "Course Absences (days) [0 to 93]:",
                min_value=0,
                max_value=93,
                value=3,
                step=1,
                help="Total days absent",
            )

        st.subheader("3. Support & Extracurriculars")
        c6, c7, c8, c9, c10 = st.columns(5)
        with c6:
            schoolsup_val = st.selectbox("School Tutoring:", options=["no", "yes"], index=0)
        with c7:
            famsup_val = st.selectbox("Family Support:", options=["yes", "no"], index=0)
        with c8:
            activities_val = st.selectbox("Extracurriculars:", options=["yes", "no"], index=0)
        with c9:
            higher_val = st.selectbox("Aims Higher Ed:", options=["yes", "no"], index=0)
        with c10:
            internet_val = st.selectbox("Home Internet:", options=["yes", "no"], index=0)

        submitted = st.form_submit_button("🚀 Run Performance Forecast", use_container_width=True)

    if submitted:
        input_payload = {
            "G1": g1_val,
            "G2": g2_val,
            "studytime": studytime_val,
            "failures": failures_val,
            "absences": absences_val,
            "schoolsup": schoolsup_val,
            "famsup": famsup_val,
            "activities": activities_val,
            "higher": higher_val,
            "internet": internet_val,
        }

        result = predict_single(input_payload)

        if not result["success"]:
            st.error("Validation error:")
            for err in result["errors"]:
                st.write(f"- {err}")
        else:
            pred_score = result["predicted_grade"]
            band = result["performance_band"]
            badge = result["badge"]
            guidance = result["guidance"]
            risks = result["risk_factors"]

            st.markdown("---")
            st.subheader("🎯 Forecast Summary")

            c_res1, c_res2 = st.columns([5, 7])
            with c_res1:
                badge_bg = {"success": "#ECFDF5", "info": "#EFF6FF", "warning": "#FFFBEB", "danger": "#FEF2F2"}.get(badge, "#F1F5F9")
                badge_fg = {"success": "#065F46", "info": "#1E40AF", "warning": "#92400E", "danger": "#991B1B"}.get(badge, "#334155")
                st.markdown(
                    f"""
                    <div style="background: #ffffff; border-radius: 20px; padding: 22px; border: 1px solid #EEF2F6; box-shadow: 0 4px 20px rgba(0,0,0,0.04);">
                        <div style="font-size: 0.85rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Estimated Final Mark</div>
                        <div style="font-size: 2.8rem; font-weight: 800; color: #1E3A8A; margin: 4px 0;">
                            {pred_score:.2f} <span style="font-size: 1.2rem; color: #94A3B8;">/ 20</span>
                        </div>
                        <span style="background: {badge_bg}; color: {badge_fg}; font-weight: 700; padding: 4px 14px; border-radius: 20px; font-size: 0.85rem;">
                            {band}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c_res2:
                st.markdown(
                    f"""
                    <div style="background: #ffffff; border-radius: 20px; padding: 22px; border: 1px solid #EEF2F6; box-shadow: 0 4px 20px rgba(0,0,0,0.04);">
                        <div style="font-weight: 700; color: #1E293B; font-size: 1rem; margin-bottom: 8px;">Pedagogical Recommendation:</div>
                        <div style="color: #475569; font-size: 0.92rem; line-height: 1.5;">{guidance}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if risks:
                st.markdown("<h5 style='color: #DC2626; margin-top: 16px;'>⚠️ Behavioral Risk Warnings:</h5>", unsafe_allow_html=True)
                for r in risks:
                    st.warning(f"• {r}")


# ==============================================================================
# TAB 2: BATCH CSV PREDICTOR
# ==============================================================================
with tab_batch:
    st.subheader("Bulk Grade Generation for Full Classroom")
    st.write(
        "Upload a roster of students. The platform will batch-predict all final grades and generate a downloadable, evaluated roster."
    )

    b_col1, b_col2 = st.columns([4, 6])
    with b_col1:
        st.download_button(
            label="📥 Download Clean CSV Template",
            data=generate_sample_csv(),
            file_name="student_batch_roster_template.csv",
            mime="text/csv",
        )
    with b_col2:
        st.caption(f"**Required Features:** `{', '.join(ALL_FEATURES)}`")

    uploaded_batch = st.file_uploader("Select Classroom CSV:", type=["csv"], key="batch_uploader")

    if uploaded_batch is not None:
        try:
            content = uploaded_batch.getvalue()
            delim = ";" if content[:1024].count(b";") > content[:1024].count(b",") else ","
            batch_df = pd.read_csv(io.StringIO(content.decode("utf-8")), sep=delim)
            results_df, errs = predict_batch(batch_df)

            if errs:
                st.error("Validation issues found:")
                for e in errs:
                    st.write(f"- {e}")
            else:
                st.success(f"Generated predictions for all {len(results_df)} student records!")
                st.dataframe(results_df, use_container_width=True)

                out_buf = io.StringIO()
                results_df.to_csv(out_buf, index=False)
                st.download_button(
                    label="💾 Download Evaluated Results CSV",
                    data=out_buf.getvalue().encode("utf-8"),
                    file_name="evaluated_student_predictions.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")
