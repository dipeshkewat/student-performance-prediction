"""Smart Academic Analytics & Student Performance Dashboard.

Inspired by modern SaaS ed-tech design (Image 1).
Features:
- Clean, uncluttered sidebar with active cohort status
- Main-page Dataset Upload & Management Hub
- Dual-Mode: Whole Class Overview & Specific Student Deep-Dive
- 100% clean HTML rendering (no raw code leakage)
"""

from __future__ import annotations

import io
import textwrap
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

# --- Active Dataset ---
df, dataset_name = get_active_dataset()

# --- Clean, Minimal Sidebar (Inspired by Image 1) ---
st.sidebar.markdown(
    """
    <div style="padding: 10px 0 20px 0; border-bottom: 1px solid #EEF2F6; margin-bottom: 20px;">
        <div style="font-size: 1.3rem; font-weight: 800; color: #1E3A8A; display: flex; align-items: center; gap: 8px;">
            <span>⬡</span> Smart
        </div>
        <div style="font-size: 0.78rem; color: #64748B; margin-top: 4px;">Learning Analytics System</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("#### 📌 Active Roster")
st.sidebar.markdown(
    f"""
    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 12px; margin-bottom: 20px;">
        <div style="font-size: 0.75rem; text-transform: uppercase; color: #94A3B8; font-weight: 700;">Loaded Cohort</div>
        <div style="font-size: 0.88rem; font-weight: 700; color: #1E293B; margin-top: 3px;">{dataset_name}</div>
        <div style="font-size: 0.78rem; color: #3B82F6; font-weight: 600; margin-top: 4px;">● {len(df)} Student Records</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("#### ⚡ Quick Actions")
if st.sidebar.button("🔄 Reset to Default Cohort", use_container_width=True):
    reset_to_default_dataset()
    st.rerun()

sample_csv_data = generate_sample_csv()
st.sidebar.download_button(
    label="📥 Download Sample CSV",
    data=sample_csv_data,
    file_name="sample_student_roster.csv",
    mime="text/csv",
    use_container_width=True,
)

st.sidebar.markdown("---")
st.sidebar.caption("© 2026 Smart Educational Analytics | Version 2.0")

if df.empty:
    st.error("No student data found. Please click 'Reset to Default Cohort' in the sidebar.")
    st.stop()

# Key class statistics
total_students = len(df)
avg_g3 = float(df["G3"].mean())
pass_count = int((df["G3"] >= 10).sum())
pass_rate = (pass_count / total_students) * 100
at_risk_count = int((df["G3"] < 10).sum())
high_achievers = int((df["G3"] >= 14).sum())
avg_absences = float(df["absences"].mean())
attendance_health = max(0.0, 100.0 - (avg_absences / 25.0 * 100.0))

# --- Hero Greeting Banner (Inspired by Image 1) ---
hero_html = textwrap.dedent(f"""
<div class="hero-card">
    <div style="max-width: 68%;">
        <div class="hero-title">Hello Grace! 👋</div>
        <div class="hero-subtitle">
            You are managing <strong>{total_students} students</strong>. Currently, <strong>{at_risk_count} students</strong> 
            require academic advisory, while <strong>{high_achievers} students</strong> are maintaining honors marks. Let's inspect the cohort progress!
        </div>
        <div class="hero-badge">Active: {dataset_name} &bull; Pass Rate: {pass_rate:.1f}%</div>
    </div>
    <div style="text-align: right; z-index: 1;">
        <div style="font-size: 3.6rem; line-height: 1;">🧑‍💻</div>
    </div>
</div>
""")
st.markdown(hero_html, unsafe_allow_html=True)

# --- Main Page Dataset Upload & Management Hub ---
with st.expander("📂 **Dataset Management: Upload Custom Student Records (CSV)**", expanded=False):
    st.write(
        "Upload a new student CSV file to instantly evaluate your own classroom data. "
        "The whole analytics, distributions, and predictions will update dynamically in real time!"
    )
    col_u1, col_u2 = st.columns([7, 3])
    with col_u1:
        uploaded_csv = st.file_uploader(
            "Choose Student CSV File:",
            type=["csv"],
            key="main_uploader",
            help="Upload CSV containing student academic records",
        )
        if uploaded_csv is not None:
            try:
                content = uploaded_csv.getvalue()
                delim = ";" if content[:1024].count(b";") > content[:1024].count(b",") else ","
                new_df = pd.read_csv(io.StringIO(content.decode("utf-8")), sep=delim)
                success, msg = set_custom_dataset(new_df, uploaded_csv.name)
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    with col_u2:
        st.markdown("**Need a Template?**")
        st.download_button(
            label="📥 Download Template CSV",
            data=sample_csv_data,
            file_name="student_template.csv",
            mime="text/csv",
            use_container_width=True,
        )
        if st.button("🔄 Reset Default Roster", use_container_width=True, key="reset_btn_main"):
            reset_to_default_dataset()
            st.rerun()

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# --- Dual Analysis View Tabs ---
view_tab1, view_tab2 = st.tabs(["🏫 Whole Class Analysis", "👤 Specific Student Deep-Dive"])


# ==============================================================================
# TAB 1: WHOLE CLASS ANALYTICS
# ==============================================================================
with view_tab1:
    # 3-Column Top Grid: Performance Card (Left), Donut Rings (Middle), Schedule & Mentors (Right)
    col_left, col_mid, col_right = st.columns([5, 4, 3])

    # Left: Class Milestones Card
    with col_left:
        st.markdown(
            textwrap.dedent("""
            <div class="dashboard-card">
                <div class="card-header-flex">
                    <div>
                        <div class="card-title">Class Academic Milestones</div>
                        <div class="card-subtitle">Term 1 (G1), Term 2 (G2) vs Final Exam (G3)</div>
                    </div>
                    <span style="background: #F1F5F9; color: #475569; padding: 4px 10px; border-radius: 8px; font-size: 0.82rem; font-weight: 600;">Academic Year 2026</span>
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )

        milestone_means = pd.DataFrame({
            "Milestone": ["Period 1 (G1)", "Period 2 (G2)", "Final Exam (G3)"],
            "Average Grade": [df["G1"].mean(), df["G2"].mean(), df["G3"].mean()],
        })

        fig_bars = px.bar(
            milestone_means,
            x="Milestone",
            y="Average Grade",
            text="Average Grade",
            color="Milestone",
            color_discrete_sequence=["#6366F1", "#3B82F6", "#10B981"],
        )
        fig_bars.add_hline(y=10, line_dash="dash", line_color="#EF4444", annotation_text="Passing Cutoff (10/20)")
        fig_bars.update_traces(texttemplate="%{text:.1f} / 20", textposition="outside")
        fig_bars.update_layout(
            template="plotly_white",
            height=260,
            margin=dict(l=10, r=10, t=20, b=10),
            showlegend=False,
            yaxis=dict(range=[0, 21]),
        )
        st.plotly_chart(fig_bars, use_container_width=True)

    # Middle: Donut Progress Rings (Inspired by Image 1)
    with col_mid:
        st.markdown(
            textwrap.dedent("""
            <div class="dashboard-card">
                <div class="card-header-flex">
                    <div>
                        <div class="card-title">Cohort Health Metrics</div>
                        <div class="card-subtitle">Key behavioral & attainment indicators</div>
                    </div>
                </div>
            </div>
            """),
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

    # Right: Upcoming Academic Events & Mentors (100% Valid Clean HTML)
    with col_right:
        calendar_and_mentors_html = textwrap.dedent("""
        <div class="dashboard-card">
            <div class="card-title" style="margin-bottom: 12px;">📅 Academic Calendar</div>
            <div style="background: #F8FAFC; border-radius: 12px; padding: 12px; margin-bottom: 10px; border-left: 4px solid #3B82F6;">
                <div style="font-weight: 700; font-size: 0.88rem; color: #1E293B;">Final Math Exam (G3)</div>
                <div style="font-size: 0.78rem; color: #64748B;">Term-end standardized test</div>
            </div>
            <div style="background: #F8FAFC; border-radius: 12px; padding: 12px; margin-bottom: 16px; border-left: 4px solid #F59E0B;">
                <div style="font-weight: 700; font-size: 0.88rem; color: #1E293B;">Remedial Mentoring Lab</div>
                <div style="font-size: 0.78rem; color: #64748B;">For students with &le; 10 in G2</div>
            </div>
            <div class="card-title" style="margin-bottom: 10px; font-size: 0.95rem;">👥 Faculty Advisors</div>
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                <div style="width: 34px; height: 34px; border-radius: 50%; background: #E0E7FF; color: #4338CA; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.82rem;">MJ</div>
                <div>
                    <div style="font-size: 0.86rem; font-weight: 700; color: #1E293B;">Mary Johnson</div>
                    <div style="font-size: 0.75rem; color: #64748B;">Senior Mathematics Lead</div>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 34px; height: 34px; border-radius: 50%; background: #ECFDF5; color: #047857; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.82rem;">JB</div>
                <div>
                    <div style="font-size: 0.86rem; font-weight: 700; color: #1E293B;">James Brown</div>
                    <div style="font-size: 0.75rem; color: #64748B;">Academic Guidance Counselor</div>
                </div>
            </div>
        </div>
        """)
        st.markdown(calendar_and_mentors_html, unsafe_allow_html=True)

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
    sub_tab_risk, sub_tab_top = st.tabs(["⚠️ At-Risk Students (Priority Intervention)", "🏆 Top Performing Students"])

    with sub_tab_risk:
        risk_df = df[df["G3"] < 10][["Student_ID", "G1", "G2", "G3", "failures", "absences", "studytime", "schoolsup"]].copy()
        if risk_df.empty:
            st.success("🎉 Outstanding! No students in this cohort currently fall below the passing grade of 10.")
        else:
            st.dataframe(risk_df.sort_values(by="G3"), use_container_width=True)

    with sub_tab_top:
        top_df = df[df["G3"] >= 14][["Student_ID", "G1", "G2", "G3", "failures", "absences", "studytime", "higher"]].copy()
        st.dataframe(top_df.sort_values(by="G3", ascending=False), use_container_width=True)


# ==============================================================================
# TAB 2: SPECIFIC STUDENT DEEP-DIVE ANALYSIS
# ==============================================================================
with view_tab2:
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

    badge_colors = {
        "success": ("#ECFDF5", "#065F46"),
        "info": ("#EFF6FF", "#1E40AF"),
        "warning": ("#FFFBEB", "#92400E"),
        "danger": ("#FEF2F2", "#991B1B"),
    }
    bg_col, txt_col = badge_colors.get(badge_type, ("#F1F5F9", "#334155"))

    profile_card_html = textwrap.dedent(f"""
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
                Weekly Study Time: <strong>Level {stu_row['studytime']}</strong> &bull; Absences: <strong>{stu_row['absences']} days</strong> &bull; Past Retentions: <strong>{stu_row['failures']}</strong>
            </div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 0.78rem; color: #94A3B8; text-transform: uppercase; font-weight: 700;">Final Mark (G3)</div>
            <div style="font-size: 2.2rem; font-weight: 800; color: #1E3A8A;">{g3_score:.1f} <span style="font-size: 1rem; color: #64748B;">/ 20</span></div>
        </div>
    </div>
    """)
    st.markdown(profile_card_html, unsafe_allow_html=True)

    # 3-Column Detail Grid
    col_g1, col_g2, col_g3 = st.columns([4, 4, 4])

    with col_g1:
        st.markdown(
            textwrap.dedent("""
            <div class="dashboard-card">
                <div class="card-title">📈 Milestone Grade Trajectory</div>
                <div class="card-subtitle">Comparison against class average</div>
            </div>
            """),
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
            textwrap.dedent("""
            <div class="dashboard-card">
                <div class="card-title">🎯 Individual Readiness Gauges</div>
                <div class="card-subtitle">Behavioral habit percentages</div>
            </div>
            """),
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
            textwrap.dedent("""
            <div class="dashboard-card">
                <div class="card-title">💡 Personalized Pedagogical Advisory</div>
                <div class="card-subtitle">Automated recommendations for this student</div>
            </div>
            """),
            unsafe_allow_html=True,
        )

        st.markdown(
            textwrap.dedent(f"""
            <div style="background: #F8FAFC; border-radius: 12px; padding: 14px; border: 1px solid #E2E8F0; margin-top: 10px;">
                <p style="margin: 0; font-size: 0.9rem; color: #334155;"><strong>Advisor Guidance:</strong> {guidance}</p>
            </div>
            """),
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
