from scipy import stats
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Unified Distribution Explorer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Unified Distribution Explorer")
st.markdown("**Interactively explore how mean, standard deviation, and skew affect both continuous and discrete distributions.**")

# =============================
# Helper functions (shared)
# =============================

def calculate_parameters_for_targets(skew_param: float, target_mean: float, target_std: float):
    """Given desired skew (shape), mean, and std for a skew-normal, return loc and scale.
    We use the standard skew-normal relationships to back out loc/scale.
    """
    if skew_param == 0:
        delta = 0.0
    else:
        delta = skew_param / np.sqrt(1.0 + skew_param**2)

    correction = np.sqrt(max(1e-4, 1 - 2 * delta**2 / np.pi))  # guard near-zero
    scale = target_std / correction
    loc = target_mean - scale * delta * np.sqrt(2 / np.pi)
    return loc, scale

# -----------------------------
# Continuous: stats + plot
# -----------------------------

def continuous_stats(skew_a: float, target_mean: float, target_std: float):
    loc, scale = calculate_parameters_for_targets(skew_a, target_mean, target_std)
    dist = stats.skewnorm(skew_a, loc=loc, scale=scale)

    # Mode (approximate via dense grid evaluate PDF)
    x_mode_grid = np.linspace(loc - 4*scale, loc + 4*scale, 2000)
    y_mode_grid = dist.pdf(x_mode_grid)
    mode_val = x_mode_grid[np.argmax(y_mode_grid)]

    q1 = dist.ppf(0.25)
    q3 = dist.ppf(0.75)

    return {
        'mean': float(dist.mean()),
        'median': float(dist.median()),
        'mode': float(mode_val),
        'variance': float(dist.var()),
        'std_dev': float(dist.std()),
        'target_mean': target_mean,
        'target_std': target_std,
        'loc_used': float(loc),
        'scale_used': float(scale),
        'skewness': float(dist.stats(moments='s')),
        'kurtosis': float(dist.stats(moments='k')),
        'iqr': float(q3 - q1),
        'dist': dist
    }


def plot_continuous(stats_dict, x_range=(-10, 10), y_pad=0.1):
    dist = stats_dict['dist']
    x = np.linspace(x_range[0], x_range[1], 1000)
    y = dist.pdf(x)
    y_max = float(np.max(y)) if len(y) else 0.6

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='lines', name='PDF', line=dict(width=3), fill='tozeroy'
    ))

    # vertical markers
    fig.add_vline(x=stats_dict['mean'], line_dash="solid", line_color="red",
                  annotation_text=f"Mean: {stats_dict['mean']:.3f}", annotation_position="top",
                  annotation=dict(yshift=10))
    fig.add_vline(x=stats_dict['median'], line_dash="dash", line_color="green",
                  annotation_text=f"Median: {stats_dict['median']:.3f}", annotation_position="top",
                  annotation=dict(yshift=-5))
    fig.add_vline(x=stats_dict['mode'], line_dash="dot", line_color="orange",
                  annotation_text=f"Mode: {stats_dict['mode']:.3f}", annotation_position="top",
                  annotation=dict(yshift=-20))

    fig.update_layout(
        xaxis_title="Value",
        yaxis_title="Probability Density",
        xaxis=dict(range=x_range, showgrid=True),
        yaxis=dict(range=(0, 0.6), showgrid=False),
        height=500,
        showlegend=False
    )
    return fig

# -----------------------------
# Discrete: build from continuous, stats + plot
# -----------------------------

def discrete_from_continuous(skew_a: float, target_mean: float, target_std: float, x_range=(-10, 10)):
    loc, scale = calculate_parameters_for_targets(skew_a, target_mean, target_std)
    dist = stats.skewnorm(skew_a, loc=loc, scale=scale)

    x_vals = np.arange(x_range[0], x_range[1] + 1)
    probs = dist.pdf(x_vals)
    probs = probs / probs.sum()  # normalize to sum to 1
    return x_vals, probs


def discrete_stats(skew_a: float, target_mean: float, target_std: float, x_range=(-10, 10)):
    x_vals, probs = discrete_from_continuous(skew_a, target_mean, target_std, x_range)

    mean = float(np.sum(x_vals * probs))
    var = float(np.sum((x_vals - mean) ** 2 * probs))
    std = np.sqrt(var)

    cum = np.cumsum(probs)
    median = float(x_vals[np.argmax(cum >= 0.5)])
    mode = float(x_vals[np.argmax(probs)])

    mean_centered = x_vals - mean
    skewness = float(np.sum((mean_centered ** 3) * probs) / (std ** 3)) if std > 0 else 0.0
    kurtosis = float(np.sum((mean_centered ** 4) * probs) / (std ** 4) - 3) if std > 0 else -3.0

    q1 = float(x_vals[np.argmax(cum >= 0.25)])
    q3 = float(x_vals[np.argmax(cum >= 0.75)])

    return {
        'mean': mean,
        'median': median,
        'mode': mode,
        'variance': var,
        'std_dev': float(std),
        'target_mean': target_mean,
        'target_std': target_std,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'iqr': float(q3 - q1),
        'x_values': x_vals,
        'probabilities': probs
    }


def plot_discrete(stats_dict, x_range=(-10, 10), y_pad=0.05):
    x_vals = stats_dict['x_values']
    probs = stats_dict['probabilities']
    y_max = float(probs.max()) if len(probs) else 0.5

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_vals,
        y=probs,
        name='PMF (normalized from PDF)',
        marker_line_width=1,
        width=0.8
    ))

    fig.add_vline(x=stats_dict['mean'], line_dash="solid", line_color="red",
                  annotation_text=f"Mean: {stats_dict['mean']:.3f}", annotation_position="top",
                  annotation=dict(yshift=10))
    fig.add_vline(x=stats_dict['median'], line_dash="dash", line_color="green",
                  annotation_text=f"Median: {stats_dict['median']:.0f}", annotation_position="top",
                  annotation=dict(yshift=-5))
    fig.add_vline(x=stats_dict['mode'], line_dash="dot", line_color="orange",
                  annotation_text=f"Mode: {stats_dict['mode']:.0f}", annotation_position="top",
                  annotation=dict(yshift=-20))

    fig.update_layout(
        xaxis_title="Value",
        yaxis_title="Probability Mass",
        xaxis=dict(range=[x_range[0]-0.5, x_range[1]+0.5], showgrid=True, dtick=1),
        yaxis=dict(range=(0, 0.6), showgrid=False),
        height=500,
        showlegend=False
    )
    return fig

# =============================
# Sidebar Controls (shared)
# =============================
mode = st.sidebar.radio(
    "Mode",
    options=["Continuous", "Discrete"],
    index=0,
    help="Choose which view to display"
)

st.sidebar.subheader("Central Tendency")
target_mean = st.sidebar.slider("Mean", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)

st.sidebar.subheader("Dispersion")
target_std = st.sidebar.slider("Standard Deviation", min_value=0.1, max_value=3.0, value=1.0, step=0.1)

st.sidebar.subheader("Shape")
skew_param = st.sidebar.slider("Skew (shape)", min_value=-15.0, max_value=15.0, value=0.0, step=0.1)

st.sidebar.markdown("---")
xr_min, xr_max = -10, 10
y_pad_pdf = 0.6
y_pad_pmf = 0.2

# =============================
# Compute & Render
# =============================
# =============================
# Compute & Render
# =============================
if mode == "Continuous":
    c_stats = continuous_stats(skew_param, target_mean, target_std)
    st.header("📈 Continuous Distribution")

    # chart and metrics side by side
    col1, col2 = st.columns([1.5, 1])  # wider for chart
    with col1:
        fig_c = plot_continuous(c_stats, x_range=(xr_min, xr_max), y_pad=y_pad_pdf)
        st.plotly_chart(fig_c, use_container_width=True)

    with col2:
        with st.expander("Descriptive Statistics — Continuous", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.subheader("M. of Central Tendency")
                st.metric("Mean", f"{c_stats['mean']:.4f}")
                st.metric("Median", f"{c_stats['median']:.4f}")
                st.metric("Mode", f"{c_stats['mode']:.4f}")
            with c2:
                st.subheader("Measures of Dispersion")
                st.metric("Std Dev", f"{c_stats['std_dev']:.4f}")
                st.metric("Variance", f"{c_stats['variance']:.4f}")
                st.metric("IQR", f"{c_stats['iqr']:.4f}")
            with c3:
                st.subheader("Measures of Shape")
                st.metric("Skewness", f"{c_stats['skewness']:.4f}")
                st.metric("Kurtosis", f"{c_stats['kurtosis']:.4f}")

if mode == "Discrete":
    d_stats = discrete_stats(skew_param, target_mean, target_std, x_range=(xr_min, xr_max))
    st.header("🔢 Discrete Distribution")

    # chart and metrics side by side
    col1, col2 = st.columns([1.5, 1])
    with col1:
        fig_d = plot_discrete(d_stats, x_range=(xr_min, xr_max), y_pad=y_pad_pmf)
        st.plotly_chart(fig_d, use_container_width=True)

    with col2:
        with st.expander("Descriptive Statistics — Discrete", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.subheader("M. of Central Tendency")
                st.metric("Mean", f"{d_stats['mean']:.4f}")
                st.metric("Median", f"{d_stats['median']:.0f}")
                st.metric("Mode", f"{d_stats['mode']:.0f}")
            with c2:
                st.subheader("Measures of Dispersion")
                st.metric("Std Dev", f"{d_stats['std_dev']:.4f}")
                st.metric("Variance", f"{d_stats['variance']:.4f}")
                st.metric("IQR", f"{d_stats['iqr']:.0f}")
            with c3:
                st.subheader("Measures of Shape")
                st.metric("Skewness", f"{d_stats['skewness']:.4f}")
                st.metric("Kurtosis", f"{d_stats['kurtosis']:.4f}")

st.markdown("---")
st.markdown("**AJ Averett** ")
