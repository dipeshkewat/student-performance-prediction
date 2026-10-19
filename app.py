"""Smart Academic Analytics & Student Dashboard.

Inspired by modern SaaS ed-tech design (Image 1).
Supports interactive Whole Class Analytics, Specific Student Deep-Dive, and Custom Student CSV Uploads.
"""

from __future__ import annotations

import io
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.dashboard_utils import (
    apply_modern_theme,
    render_top_navbar,
    get_active_dataset,
    set_custom_dataset,
    reset_to_default_dataset,
    create_donut_gauge,
    safe_scatter_trendline,
    generate_sample_csv,
)
from src.predict import classify_performance_tier, analyze_risk_factors

st.set_page_config(
    page_title="Smart Student Performance Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_modern_theme()

# --- Top Navigation Bar ---
render_top_navbar(user_name="Grace Stanley", role="Academic Advisor")

# --- Dataset Management in Sidebar ---
st.sidebar.markdown("### 📁 Dataset Control")
df, dataset_name = get_active_dataset()

st.sidebar.info(f"**Active Cohort:**\n{dataset_name}")

uploaded_file = st.sidebar.file_uploader(
    "Upload Student Records (CSV):",
    type=["csv"],
    help="Upload your own student class dataset to see the whole analytics dynamically update!",
)

if uploaded_file is not None:
    try:
        content = uploaded_file.getvalue()
        sample_text = content[:2048].decode("utf-8", errors="ignore")
        delim = ";" if sample_text.count(";") > sample_text.count(",") else ","
        raw_uploaded = pd.read_csv(io.StringIO(content.decode("utf-8")), sep=delim)
        success, msg = set_custom_dataset(raw_uploaded, uploaded_file.name)
        if success:
            st.sidebar.success(msg)
            df, dataset_name = get_active_dataset()
        else:
            st.sidebar.error(msg)
    except Exception as ex:
        st.sidebar.error(f"Error parsing file: {str(ex)}")

col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    if st.button("🔄 Reset Default", help="Reload the standard 395-student benchmark dataset"):
        reset_to_default_dataset()
        st.rerun()
with col_s2:
    sample_csv_bytes = generate_sample_csv()
    st.download_button(
        label="📥 Sample CSV",
        data=sample_csv_bytes,
        file_name="sample_student_roster.csv",
        mime="text/csv",
        help="Download ready-to-use student CSV template",
    )

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧭 Navigation & Filters")
analysis_mode = st.sidebar.radio(
    "Select Analytics View:",
    options=["🏫 Whole Class Analysis", "👤 Specific Student Deep-Dive"],
    index=0,
)

if df.empty:
    st.error("No student data available. Please upload a student CSV or click 'Reset Default'.")
    st.stop()

# Compute key class statistics
total_students = len(df)
avg_g3 = float(df["G3"].mean())
pass_count = int((df["G3"] >= 10).sum())
pass_rate = (pass_count / total_students) * 100
at_risk_count = int((df["G3"] < 10).sum())
high_achievers = int((df["G3"] >= 14).sum())
avg_absences = float(df["absences"].mean())
attendance_health = max(0.0, 100.0 - (avg_absences / 25.0 * 100.0))

# --- Hero Greeting Banner (Inspired by Image 1) ---
hero_html = f"""
<div class="hero-card">
    <div style="max-width: 65%;">
        <div class="hero-title">Hello Grace! 👋</div>
        <div class="hero-subtitle">
            You are managing <strong>{total_students} students</strong>. Currently, <strong>{at_risk_count} students</strong> 
            require academic advisory, while <strong>{high_achievers} students</strong> are maintaining honors marks. Let's inspect the cohort progress!
        </div>
        <div class="hero-badge">Cohort: {dataset_name} &bull; Class Pass Rate: {pass_rate:.1f}%</div>
    </div>
    <div style="text-align: right; z-index: 1;">
        <div style="font-size: 3.8rem; line-height: 1;">🧑‍💻</div>
    </div>
</div>
"""
st.markdown(hero_html, unsafe_allow_html=True)


# ==============================================================================
# MODE 1: WHOLE CLASS ANALYTICS
# ==============================================================================
if analysis_mode == "🏫 Whole Class Analysis":
    # 3-Column Top Grid: Performance Card (Left), Donut Rings (Middle), Schedule (Right)
    col_left, col_mid, col_right = st.columns([5, 4, 3])

    # Left Panel: Performance Overview (Bar distribution across milestones)
    with col_left:
        st.markdown(
            """
            <div class="dashboard-card">
                <div class="card-header-flex">
                    <div>
                        <div class="card-title">Class Academic Milestones</div>
                        <div class="card-subtitle">Term 1 (G1), Term 2 (G2) vs Final Outcome (G3)</div>
                    </div>
                    <span class="stat-pill">Academic Year 2026</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        milestone_means = pd.DataFrame({
            "Milestone": ["Period 1 (G1)", "Period 2 (G2)", "Final Exam (G3)"],
            "Average Grade": [df["G1"].mean(), df["G2"].mean(), df["G3"].mean()],
            "Passing Cutoff": [10.0, 10.0, 10.0],
        })

        fig_bars = px.bar(
            milestone_means,
            x="Milestone",
            y="Average Grade",
            text="Average Grade",
            color="Milestone",
            color_discrete_sequence=["#6366F1", "#3B82F6", "#10B981"],
        )
        fig_bars.add_hline(y=10, line_dash="dash", line_color="#EF4444", annotation_text="Passing Threshold (10/20)")
        fig_bars.update_traces(texttemplate="%{text:.1f} / 20", textposition="outside")
        fig_bars.update_layout(
            template="plotly_white",
            height=260,
            margin=dict(l=10, r=10, t=20, b=10),
            showlegend=False,
            yaxis=dict(range=[0, 20]),
        )
        st.plotly_chart(fig_bars, use_container_width=True)

    # Middle Panel: Donut Progress Rings (Inspired by "My visit" in Image 1)
    with col_mid:
        st.markdown(
            """
            <div class="dashboard-card">
                <div class="card-header-flex">
                    <div>
                        <div class="card-title">Cohort Health Metrics</div>
                        <div class="card-subtitle">Key behavioral & attainment indicators</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        d1, d2 = st.columns(2)
        with d1:
            st.caption("<center><b>Class Pass Rate</b></center>", unsafe_allow_html=True)
            st.plotly_chart(create_donut_gauge(pass_rate, "Pass Rate", color="#3B82F6"), use_container_width=True)
        with d2:
            st.caption("<center><b>Attendance Health</b></center>", unsafe_allow_html=True)
            st.plotly_chart(create_donut_gauge(attendance_health, "Attendance", color="#10B981"), use_container_width=True)

        d3, d4 = st.columns(2)
        with d3:
            high_study_pct = (df["studytime"] >= 2).mean() * 100
            st.caption("<center><b>Study Habit (≥2h/wk)</b></center>", unsafe_allow_html=True)
            st.plotly_chart(create_donut_gauge(high_study_pct, "Study Time", color="#6366F1"), use_container_width=True)
        with d4:
            zero_fail_pct = (df["failures"] == 0).mean() * 100
            st.caption("<center><b>Zero Failures Record</b></center>", unsafe_allow_html=True)
            st.plotly_chart(create_donut_gauge(zero_fail_pct, "No Failures", color="#F59E0B"), use_container_width=True)

    # Right Panel: Upcoming Academic Events & Mentors (Matching Image 1)
    with col_right:
        st.markdown(
            """
            <div class="dashboard-card">
                <div class="card-title" style="margin-bottom: 12px;">📅 Academic Calendar</div>
                <div style="background: #F8FAFC; border-radius: 12px; padding: 12px; margin-bottom: 10px; border-left: 4px solid #3B82F6;">
                    <div style="font-weight: 700; font-size: 0.88rem; color: #1E293B;">Final Math Exam (G3)</div>
                    <div style="font-size: 0.78rem; color: #64748B;">Term-end standardized test</div>
                </div>
                <div style="background: #F8FAFC; border-radius: 12px; padding: 12px; margin-bottom: 14px; border-left: 4px solid #F59E0B;">
                    <div style="font-weight: 700; font-size: 0.88rem; color: #1E293B;">Remedial Mentoring Lab</div>
                    <div style="font-size: 0.78rem; color: #64748B;">For students with &le; 10 in G2</div>
                </div>

                <div class="card-title" style="margin-bottom: 10px; font-size: 0.95rem;">👥 Faculty Advisors</div>
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                    <div style="width: 32px; height: 32px; border-radius: 50%; background: #E0E7FF; color: #4338CA; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.8rem;">MJ</div>
                    <div>
                        <div style="font-size: 0.85rem; font-weight: 600; color: #1E293B;">Mary Johnson</div>
                        <div style="font-size: 0.75rem; color: #64748B;">Senior Math Lead</div>
                    </div>
                </div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="width: 32px; height: 32px; border-radius: 50%; background: #ECFDF5; color: #047857; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.8rem;">JB</div>
                    <div>
                        <div style="font-size: 0.85rem; font-weight: 600; color: #1E293B;">James Brown</div>
                        <div style="font-size: 0.75rem; color: #64748B;">Academic Counselor</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Lower Grid: Class Distribution & Behavioral Correlations
    c_dist, c_corr = st.columns(2)

    with c_dist:
        st.subheader("📊 Final Grade Distribution Curve")
        fig_hist = px.histogram(
            df,
            x="G3",
            nbins=21,
            title="Distribution of Class Grades (0 to 20 Scale)",
            labels={"G3": "Final Grade (G3)", "count": "Student Count"},
            color_discrete_sequence=["#3B82F6"],
        )
        fig_hist.add_vline(x=10, line_dash="dash", line_color="#EF4444", annotation_text="Passing Cutoff (10)")
        fig_hist.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_hist, use_container_width=True)

    with c_corr:
        st.subheader("📈 Milestone Progression (G2 vs Final G3)")
        # Safe scatter trendline avoiding statsmodels crash
        fig_scatter = safe_scatter_trendline(
            df=df,
            x_col="G2",
            y_col="G3",
            title="Correlation: Midterm Grade (G2) vs Final Grade (G3)",
            x_label="Period 2 Grade (G2)",
            y_label="Final Grade (G3)",
            color_col="failures",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")

    # High Priority Intervention Table & High Achievers
    st.subheader("📋 Student Action Roster")
    tab_risk, tab_top = st.tabs(["⚠️ At-Risk Students (Priority Intervention)", "🏆 Top Performing Students"])

    with tab_risk:
        risk_df = df[df["G3"] < 10][["Student_ID", "G1", "G2", "G3", "failures", "absences", "studytime", "schoolsup"]].copy()
        if risk_df.empty:
            st.success("🎉 Outstanding! No students in this cohort currently fall below the passing grade of 10.")
        else:
            st.dataframe(risk_df.sort_values(by="G3"), use_container_width=True)

    with tab_top:
        top_df = df[df["G3"] >= 14][["Student_ID", "G1", "G2", "G3", "failures", "absences", "studytime", "higher"]].copy()
        st.dataframe(top_df.sort_values(by="G3", ascending=False), use_container_width=True)


# ==============================================================================
# MODE 2: SPECIFIC STUDENT DEEP-DIVE ANALYSIS
# ==============================================================================
elif analysis_mode == "👤 Specific Student Deep-Dive":
    st.subheader("🔍 Individual Student Performance Profile")
    st.write("Select any student from the cohort below to inspect their detailed diagnostic card, marks trajectory, and personalized recommendations.")

    # Student Selector
    student_options = df["Student_ID"].tolist()
    student_labels = {
        sid: f"{sid}  |  G1: {df.loc[df['Student_ID'] == sid, 'G1'].values[0]}  |  G2: {df.loc[df['Student_ID'] == sid, 'G2'].values[0]}  |  G3: {df.loc[df['Student_ID'] == sid, 'G3'].values[0]}"
        for sid in student_options
    }

    selected_student_id = st.selectbox(
        "Choose Student:",
        options=student_options,
        format_func=lambda sid: student_labels[sid],
        index=0,
    )

    stu_row = df[df["Student_ID"] == selected_student_id].iloc[0]

    # Performance Band & Status
    g3_score = float(stu_row["G3"])
    band_name, badge_type, guidance = classify_performance_tier(g3_score)
    risks = analyze_risk_factors(stu_row.to_dict())

    # Individual Profile Header Card
    badge_colors = {
        "success": ("#ECFDF5", "#065F46", "Outstanding"),
        "info": ("#EFF6FF", "#1E40AF", "Good Standing"),
        "warning": ("#FFFBEB", "#92400E", "Needs Attention"),
        "danger": ("#FEF2F2", "#991B1B", "Critical Advisory"),
    }
    bg_col, txt_col, status_lbl = badge_colors.get(badge_type, ("#F1F5F9", "#334155", "Normal"))

    profile_card_html = f"""
    <div class="student-profile-badge">
        <div class="student-avatar-lg">🎓</div>
        <div style="flex-grow: 1;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <h3 style="margin: 0; color: #0F172A; font-weight: 800;">{selected_student_id}</h3>
                <span style="background: {bg_col}; color: {txt_col}; padding: 4px 12px; border-radius: 20px; font-weight: 700; font-size: 0.82rem;">
                    {band_name}
                </span>
            </div>
            <div style="color: #64748B; font-size: 0.88rem; margin-top: 4px;">
                Weekly Study Time: <strong>Level {stu_row['studytime']}</strong> &bull; Absences: <strong>{stu_row['absences']} days</strong> &bull; Past Failures: <strong>{stu_row['failures']}</strong>
            </div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 0.78rem; color: #94A3B8; text-transform: uppercase; font-weight: 700;">Final Mark (G3)</div>
            <div style="font-size: 2.2rem; font-weight: 800; color: #1E3A8A;">{g3_score:.1f} <span style="font-size: 1rem; color: #64748B;">/ 20</span></div>
        </div>
    </div>
    """
    st.markdown(profile_card_html, unsafe_allow_html=True)

    # 3-Column Detail Grid
    col_g1, col_g2, col_g3 = st.columns([4, 4, 4])

    with col_g1:
        st.markdown(
            """
            <div class="dashboard-card">
                <div class="card-title">📈 Milestone Grade Trajectory</div>
                <div class="card-subtitle">Comparison against class average</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        traj_df = pd.DataFrame({
            "Period": ["Term 1 (G1)", "Term 2 (G2)", "Final (G3)"],
            "Student Score": [stu_row["G1"], stu_row["G2"], stu_row["G3"]],
            "Class Average": [df["G1"].mean(), df["G2"].mean(), df["G3"].mean()],
        })

        fig_traj = go.Figure()
        fig_traj.add_trace(go.Bar(
            x=traj_df["Period"],
            y=traj_df["Student Score"],
            name="Student",
            marker_color="#3B82F6",
            text=traj_df["Student Score"],
            textposition="outside",
        ))
        fig_traj.add_trace(go.Scatter(
            x=traj_df["Period"],
            y=traj_df["Class Average"],
            name="Class Mean",
            mode="lines+markers",
            line=dict(color="#EF4444", width=2.5, dash="dash"),
        ))
        fig_traj.update_layout(
            template="plotly_white",
            height=250,
            margin=dict(l=10, r=10, t=20, b=10),
            yaxis=dict(range=[0, 22]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_traj, use_container_width=True)

    with col_g2:
        st.markdown(
            """
            <div class="dashboard-card">
                <div class="card-title">🎯 Individual Readiness Gauges</div>
                <div class="card-subtitle">Behavioral habit percentages</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st_c1, st_c2 = st.columns(2)
        with st_c1:
            stu_att_pct = max(0.0, 100.0 - (float(stu_row["absences"]) / 20.0 * 100.0))
            st.caption("<center><b>Attendance Health</b></center>", unsafe_allow_html=True)
            st.plotly_chart(create_donut_gauge(stu_att_pct, "Attendance", color="#10B981", size=130), use_container_width=True)
        with st_c2:
            stu_study_pct = (float(stu_row["studytime"]) / 4.0) * 100.0
            st.caption("<center><b>Study Intensity</b></center>", unsafe_allow_html=True)
            st.plotly_chart(create_donut_gauge(stu_study_pct, "Study", color="#6366F1", size=130), use_container_width=True)

    with col_g3:
        st.markdown(
            """
            <div class="dashboard-card">
                <div class="card-title">💡 Personalized Pedagogical Advisory</div>
                <div class="card-subtitle">Automated recommendations for this student</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div style="background: #F8FAFC; border-radius: 12px; padding: 14px; border: 1px solid #E2E8F0; margin-top: 10px;">
                <p style="margin: 0; font-size: 0.9rem; color: #334155;"><strong>Advisor Guidance:</strong> {guidance}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if risks:
            st.markdown("<div style='margin-top: 12px; font-weight: 700; color: #DC2626; font-size: 0.85rem;'>Detected Risk Factors:</div>", unsafe_allow_html=True)
            for r in risks:
                st.markdown(f"<div style='font-size: 0.82rem; color: #B91C1C;'>• {r}</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                """
                <div style="margin-top: 12px; color: #047857; font-size: 0.85rem;">
                    ✅ No major behavioral risk flags detected. Student is on track.
                </div>
                """,
                unsafe_allow_html=True,
            )
