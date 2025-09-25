# app.py
import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt

st.set_page_config(page_title="Linear Regression Animator", layout="wide")

# ---------------- Sidebar ----------------
st.sidebar.header("Data & Training Settings")

mode_data = st.sidebar.radio("Data source", ["Synthetic", "Upload CSV"])
n_points = st.sidebar.slider("Number of points (Synthetic)", 10, 400, 80, 5)
true_a = st.sidebar.slider("True slope (Synthetic)", -5.0, 5.0, 2.0, 0.1)
true_b = st.sidebar.slider("True intercept (Synthetic)", -50.0, 50.0, 10.0, 1.0)
noise = st.sidebar.slider("Noise σ (Synthetic)", 0.0, 20.0, 4.0, 0.5)

uploaded = None
if mode_data == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload CSV with columns x,y", type=["csv"])

st.sidebar.markdown("---")
fix_intercept = st.sidebar.checkbox("Fix intercept b (animate 1D parabola J(a))", value=False)
fixed_b_value = st.sidebar.slider("Fixed b value", -50.0, 50.0, 0.0, 1.0, disabled=not fix_intercept)

lr = st.sidebar.slider("Learning rate (α)", 1e-4, 1.0, 0.05, 0.0001, format="%.4f")
iters = st.sidebar.slider("Iterations", 10, 1000, 200, 10)
pause = st.sidebar.slider("Frame delay (sec)", 0.0, 0.2, 0.02, 0.01)
seed = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)

st.sidebar.markdown("---")
init_a = st.sidebar.number_input("Init a", value=0.0, step=0.1)
init_b = st.sidebar.number_input("Init b", value=0.0, step=0.5, disabled=fix_intercept)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: 절편 고정 체크 시, 오른쪽 위 패널이 **J(a)** 포물선이 되어 하강을 직관적으로 볼 수 있어요. 절편 미고정 시 등고선(볼록그릇)에서 (a,b)이 내려갑니다.")

start = st.sidebar.button("Start / Reset")

# ---------------- Data ----------------
rng = np.random.default_rng(seed)

if mode_data == "Synthetic":
    x = np.linspace(-10, 10, n_points)
    y = true_a * x + true_b + rng.normal(0, noise, size=x.shape)
else:
    import io, csv
    if uploaded is not None:
        raw = uploaded.read().decode("utf-8")
        data = np.loadtxt(io.StringIO(raw), delimiter=",", skiprows=1)
        if data.ndim == 1:  # single row
            data = data.reshape(1, -1)
        x, y = data[:, 0], data[:, 1]
    else:
        st.stop()

x = x.astype(float)
y = y.astype(float)
n = len(x)

# 1/(2n) MSE 정의 (경사하강의 수식을 깔끔하게)
def mse_12n(a, b, x, y):
    e = y - (a * x + b)
    return (e @ e) / (2.0 * len(x))

# 그라디언트 (1/2n 스케일)
def grad_12n(a, b, x, y):
    n = len(x)
    e = y - (a * x + b)
    # ∂J/∂a = -(1/n) * sum(x*e)
    # ∂J/∂b = -(1/n) * sum(e)
    d_a = -(np.sum(x * e) / n)
    d_b = -(np.sum(e) / n)
    return d_a, d_b

# ---------------- Layout ----------------
st.title("Linear Regression — Animated Gradient Descent")

col_left, col_right = st.columns([3, 4])
col_r_top, col_r_bottom = st.columns([1, 1])

# placeholders
ph_line = col_left.empty()
ph_surface = col_r_top.empty()
ph_mse = col_r_bottom.empty()

# 초기값
a, b = float(init_a), (float(init_b) if not fix_intercept else float(fixed_b_value))

# Precompute for 1D parabola J(a) if b fixed
if fix_intercept:
    a_grid = np.linspace(a - 6.0, a + 6.0, 300)
    J_grid = np.array([mse_12n(A, b, x, y) for A in a_grid])

# Precompute for 2D contour if learning (a,b)
if not fix_intercept:
    a_span = np.linspace(a - 6.0, a + 6.0, 100)
    b_center = b
    b_span = np.linspace(b_center - 60.0, b_center + 60.0, 100)
    AA, BB = np.meshgrid(a_span, b_span)
    # Efficient J over grid
    # J(a,b) = (1/2n) * sum((y - a x - b)^2)
    Y = y.reshape(1, 1, -1)
    X = x.reshape(1, 1, -1)
    AA3 = AA[..., None]
    BB3 = BB[..., None]
    E = Y - (AA3 * X + BB3)
    J_surf = np.mean(E ** 2, axis=-1) / 2.0

# 버튼을 누를 때만 애니메이션
if start:
    J_hist = []

    for t in range(iters):
        # -------- Draw scatter + line (matplotlib) --------
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(x, y, s=16, alpha=0.8)
        xs = np.array([x.min() - 1, x.max() + 1])
        ys = a * xs + b
        ax.plot(xs, ys, linewidth=2.5)
        ax.set_title(f"Data & Line  (iter={t})  a={a:.4f}, b={b:.4f}")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.grid(True, alpha=0.3)
        ph_line.pyplot(fig, clear_figure=True)
        plt.close(fig)

        # -------- Update surface / parabola panel (plotly) --------
        if fix_intercept:
            # 포물선 J(a) 위에서 현재 a 위치
            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(x=a_grid, y=J_grid, mode="lines", name="J(a)"))
            fig_p.add_trace(go.Scatter(x=[a], y=[mse_12n(a, b, x, y)],
                                       mode="markers", marker=dict(size=10),
                                       name="current a"))
            fig_p.update_layout(title=f"MSE parabola J(a) with b={b:.3f}",
                                xaxis_title="a", yaxis_title="J(a)")
            ph_surface.plotly_chart(fig_p, use_container_width=True)
        else:
            # 등고선에서 (a,b) 점 위치
            fig_c = go.Figure(
                data=go.Contour(
                    z=J_surf, x=a_span, y=b_span,
                    contours_coloring="heatmap", showscale=True, coloraxis=None
                )
            )
            fig_c.add_trace(go.Scatter(x=[a], y=[b], mode="markers",
                                       marker=dict(size=10),
                                       name="(a,b)"))
            fig_c.update_layout(title="MSE contour J(a,b)",
                                xaxis_title="a", yaxis_title="b")
            ph_surface.plotly_chart(fig_c, use_container_width=True)

        # -------- Compute loss + plot J history --------
        J = mse_12n(a, b, x, y)
        J_hist.append(J)
        fig_m, axm = plt.subplots(figsize=(5.5, 3.5))
        axm.plot(J_hist, linewidth=2)
        axm.set_title(f"MSE vs Iter (current: {J:.6f})")
        axm.set_xlabel("iteration"); axm.set_ylabel("J")
        axm.grid(True, alpha=0.3)
        ph_mse.pyplot(fig_m, clear_figure=True)
        plt.close(fig_m)

        # -------- Gradient step (1/2n scaling) --------
        d_a, d_b = grad_12n(a, b, x, y)
        if fix_intercept:
            b = fixed_b_value
            a = a - lr * d_a
        else:
            a = a - lr * d_a
            b = b - lr * d_b

        time.sleep(pause)

else:
    # 첫 로딩 시 정적 프레임 표시
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, s=16, alpha=0.8, label="data")
    xs = np.array([x.min() - 1, x.max() + 1])
    ax.plot(xs, init_a * xs + (fixed_b_value if fix_intercept else init_b),
            linewidth=2.0, label="initial line")
    ax.legend()
    ax.set_title("Click 'Start / Reset' to animate")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ph_line.pyplot(fig, clear_figure=True)
    plt.close(fig)

    if fix_intercept:
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=[], y=[]))
        fig_p.update_layout(title="MSE parabola J(a) will appear here",
                            xaxis_title="a", yaxis_title="J(a)")
        ph_surface.plotly_chart(fig_p, use_container_width=True)
    else:
        fig_c = go.Figure()
        fig_c.update_layout(title="MSE contour J(a,b) will appear here",
                            xaxis_title="a", yaxis_title="b")
        ph_surface.plotly_chart(fig_c, use_container_width=True)

    fig_m, axm = plt.subplots(figsize=(5.5, 3.5))
    axm.plot([])
    axm.set_title("MSE vs Iter will appear here")
    axm.grid(True, alpha=0.3)
    ph_mse.pyplot(fig_m, clear_figure=True)
    plt.close(fig_m)
