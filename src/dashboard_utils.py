"""Dashboard UI utilities, styles inspired by modern SaaS ed-tech dashboards, and stateful dataset accessors."""

from __future__ import annotations

import io
import textwrap
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import load_data
from src.preprocessing import clean_dataset, ALL_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.predict import load_production_model

DATA_RAW_PATH = Path("data") / "raw" / "student-mat.csv"
DATA_CLEANED_PATH = Path("data") / "processed" / "student_cleaned.csv"


# ==============================================================================
# PREMIUM MODERN SAAS CSS (Inspired by Image 1: Smart Student Dashboard)
# ==============================================================================
MODERN_DASHBOARD_CSS = textwrap.dedent("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #F4F6FB;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
        max-width: 1440px;
    }

    /* Top Navigation Simulation */
    .top-navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #ffffff;
        padding: 12px 24px;
        border-radius: 18px;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.03);
        margin-bottom: 18px;
        border: 1px solid #EEF2F6;
    }
    .top-navbar .brand {
        display: flex;
        align-items: center;
        gap: 10px;
        font-weight: 800;
        font-size: 1.25rem;
        color: #1E3A8A;
    }
    .top-navbar .profile {
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 0.9rem;
        font-weight: 600;
        color: #334155;
    }
    .avatar-circle {
        width: 38px;
        height: 38px;
        border-radius: 50%;
        background: linear-gradient(135deg, #3B82F6, #1D4ED8);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.95rem;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.25);
    }

    /* Welcome Hero Card */
    .hero-card {
        background: #ffffff;
        border-radius: 22px;
        padding: 24px 28px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
        border: 1px solid #EEF2F6;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 18px;
        position: relative;
        overflow: hidden;
    }
    .hero-title {
        font-size: 1.6rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        font-size: 0.93rem;
        color: #64748B;
        line-height: 1.5;
    }
    .hero-badge {
        display: inline-block;
        background: #EFF6FF;
        color: #2563EB;
        font-weight: 600;
        font-size: 0.8rem;
        padding: 4px 12px;
        border-radius: 20px;
        margin-top: 10px;
    }

    /* Card Panels */
    .dashboard-card {
        background: #ffffff;
        border-radius: 20px;
        padding: 20px 22px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
        border: 1px solid #EEF2F6;
        margin-bottom: 18px;
        height: 100%;
    }
    .card-header-flex {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 14px;
    }
    .card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1E293B;
    }
    .card-subtitle {
        font-size: 0.82rem;
        color: #94A3B8;
    }

    /* Student Profile Header in Deep-Dive */
    .student-profile-badge {
        display: flex;
        align-items: center;
        gap: 16px;
        background: #F8FAFC;
        padding: 18px 22px;
        border-radius: 18px;
        border: 1px solid #E2E8F0;
        margin-bottom: 18px;
    }
    .student-avatar-lg {
        width: 58px;
        height: 58px;
        border-radius: 50%;
        background: linear-gradient(135deg, #6366F1, #3B82F6);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        font-weight: 700;
    }

    /* Clean Modern Sidebar Matching Image 1 */
    section[data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid #EEF2F6;
    }
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] span, 
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #1E293B !important;
    }
    section[data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
        border-radius: 10px !important;
        border: 1px solid #E2E8F0 !important;
    }
    
    .stButton>button {
        border-radius: 12px;
        font-weight: 600;
        padding: 8px 18px;
        transition: all 0.2s ease-in-out;
    }

    /* Hide ugly default streamlit header clutter */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""")


def apply_modern_theme() -> None:
    """Inject modern SaaS dashboard CSS styling."""
    st.markdown(MODERN_DASHBOARD_CSS, unsafe_allow_html=True)


def render_top_navbar(user_name: str = "Grace Stanley", role: str = "Academic Advisor") -> None:
    """Render the top search and user profile bar inspired by Image 1."""
    initials = "".join([part[0] for part in user_name.split() if part])[:2]
    html = textwrap.dedent(f"""
    <div class="top-navbar">
        <div class="brand">
            <span style="font-size: 1.4rem;">🎓</span> Smart Analytics
        </div>
        <div style="flex-grow: 1; max-width: 450px; margin: 0 30px;">
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 20px; padding: 7px 16px; display: flex; align-items: center; color: #94A3B8; font-size: 0.88rem;">
                <span style="margin-right: 8px;">🔍</span> Search students, courses, alerts...
            </div>
        </div>
        <div class="profile">
            <span style="cursor: pointer; font-size: 1.1rem; margin-right: 8px;">🔔</span>
            <span style="cursor: pointer; font-size: 1.1rem; margin-right: 12px;">✉️</span>
            <div class="avatar-circle">{initials}</div>
            <div>
                <div style="font-size: 0.9rem; font-weight: 700; color: #0F172A; line-height: 1.2;">{user_name}</div>
                <div style="font-size: 0.75rem; color: #64748B;">{role}</div>
            </div>
        </div>
    </div>
    """)
    st.markdown(html, unsafe_allow_html=True)


# ==============================================================================
# DATASET MANAGEMENT (Supports default cohort and user CSV uploads)
# ==============================================================================
def get_active_dataset() -> Tuple[pd.DataFrame, str]:
    """Retrieve active dataset from session state or load default student-mat.csv."""
    if "active_df" not in st.session_state:
        if DATA_CLEANED_PATH.exists():
            df = pd.read_csv(DATA_CLEANED_PATH)
            st.session_state["active_df"] = df
            st.session_state["dataset_name"] = "Default Cohort (student-mat.csv)"
        elif DATA_RAW_PATH.exists():
            raw_df = load_data(DATA_RAW_PATH)
            cleaned_df, _ = clean_dataset(raw_df)
            st.session_state["active_df"] = cleaned_df
            st.session_state["dataset_name"] = "Default Cohort (student-mat.csv)"
        else:
            st.session_state["active_df"] = pd.DataFrame()
            st.session_state["dataset_name"] = "Empty"

    df = st.session_state["active_df"]
    if not df.empty and "Student_ID" not in df.columns:
        df["Student_ID"] = [f"STU-{1001 + i}" for i in range(len(df))]
        st.session_state["active_df"] = df

    return st.session_state["active_df"], st.session_state.get("dataset_name", "Default Cohort")


def set_custom_dataset(df: pd.DataFrame, filename: str) -> Tuple[bool, str]:
    """Validate and set a user-uploaded dataset as the active cohort."""
    try:
        cleaned_df, report = clean_dataset(df)
        if cleaned_df.empty:
            return False, "Uploaded file contains no valid records after cleaning."

        try:
            pipeline = load_production_model()
            X = cleaned_df[ALL_FEATURES]
            preds = np.clip(pipeline.predict(X), 0.0, 20.0).round(2)
            cleaned_df["Predicted_G3"] = preds
        except Exception:
            pass

        cleaned_df["Student_ID"] = [f"STU-{1001 + i}" for i in range(len(cleaned_df))]
        st.session_state["active_df"] = cleaned_df
        st.session_state["dataset_name"] = f"Uploaded: {filename} ({len(cleaned_df)} students)"
        return True, f"Successfully loaded {len(cleaned_df)} student records from {filename}."
    except Exception as e:
        return False, f"Failed to ingest CSV: {str(e)}"


def reset_to_default_dataset() -> None:
    """Reset session state back to standard UCI dataset."""
    if DATA_CLEANED_PATH.exists():
        df = pd.read_csv(DATA_CLEANED_PATH)
    else:
        raw_df = load_data(DATA_RAW_PATH)
        df, _ = clean_dataset(raw_df)

    df["Student_ID"] = [f"STU-{1001 + i}" for i in range(len(df))]
    st.session_state["active_df"] = df
    st.session_state["dataset_name"] = "Default Cohort (student-mat.csv)"


# ==============================================================================
# VISUAL DONUT METERS (Matching Image 1 Donut Rings)
# ==============================================================================
def create_donut_gauge(
    percentage: float,
    title: str,
    color: str = "#3B82F6",
    size: int = 140,
) -> go.Figure:
    """Create a sleek circular donut meter matching Image 1 'My visit' style."""
    pct = max(0.0, min(100.0, float(percentage)))
    remainder = 100.0 - pct

    fig = go.Figure(
        data=[
            go.Pie(
                values=[pct, remainder],
                hole=0.74,
                sort=False,
                direction="clockwise",
                showlegend=False,
                marker=dict(colors=[color, "#E2E8F0"]),
                hoverinfo="none",
                textinfo="none",
            )
        ]
    )

    fig.update_layout(
        width=size,
        height=size,
        margin=dict(l=5, r=5, t=5, b=5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[
            dict(
                text=f"<b>{int(round(pct))}%</b>",
                x=0.5,
                y=0.5,
                font_size=16,
                font_color="#0F172A",
                font_family="Plus Jakarta Sans, sans-serif",
                showarrow=False,
            )
        ],
    )
    return fig


def safe_scatter_trendline(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
    color_col: Optional[str] = None,
) -> go.Figure:
    """Build a scatter plot with robust trendline that NEVER crashes if statsmodels is missing."""
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        title=title,
        labels={x_col: x_label, y_col: y_label},
        template="plotly_white",
        color_continuous_scale="Blues" if color_col else None,
    )

    valid = df[[x_col, y_col]].dropna()
    if len(valid) > 2:
        x_vals = valid[x_col].values
        y_vals = valid[y_col].values
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 50)
        y_line = slope * x_line + intercept

        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name=f"Trendline (slope={slope:.2f})",
                line=dict(color="#2563EB", width=2.5, dash="solid"),
            )
        )

    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def generate_sample_csv() -> bytes:
    """Generate a sample CSV template for testing batch predictions and uploads."""
    sample_data = pd.DataFrame([
        {
            "G1": 14,
            "G2": 15,
            "studytime": 3,
            "failures": 0,
            "absences": 2,
            "schoolsup": "no",
            "famsup": "yes",
            "activities": "yes",
            "higher": "yes",
            "internet": "yes",
            "G3": 15,
        },
        {
            "G1": 9,
            "G2": 8,
            "studytime": 1,
            "failures": 1,
            "absences": 12,
            "schoolsup": "yes",
            "famsup": "no",
            "activities": "no",
            "higher": "yes",
            "internet": "no",
            "G3": 8,
        },
        {
            "G1": 17,
            "G2": 18,
            "studytime": 4,
            "failures": 0,
            "absences": 0,
            "schoolsup": "no",
            "famsup": "yes",
            "activities": "yes",
            "higher": "yes",
            "internet": "yes",
            "G3": 18,
        },
        {
            "G1": 7,
            "G2": 6,
            "studytime": 1,
            "failures": 2,
            "absences": 18,
            "schoolsup": "yes",
            "famsup": "no",
            "activities": "no",
            "higher": "no",
            "internet": "yes",
            "G3": 6,
        },
    ])
    csv_buf = io.StringIO()
    sample_data.to_csv(csv_buf, index=False)
    return csv_buf.getvalue().encode("utf-8")
