import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1) Synthetic tractography streamline ---
t = np.linspace(0, 2 * np.pi, 200)
streamline = np.column_stack([
    60 + 12 * np.cos(t) + 4 * np.cos(3 * t),
    80 + 40 * t / (2 * np.pi) + 5 * np.sin(2 * t),
    70 + 10 * np.sin(t) + 3 * np.sin(5 * t)
])

com = np.array([80.0, 100.0, 80.0])

# --- 2) Centered streamline ---
streamline_centered = streamline - com
x_c, y_c, z_c = streamline_centered[:, 0], streamline_centered[:, 1], streamline_centered[:, 2]

# --- 3) Spherical coords ---
epsilon = 1e-6

r = np.linalg.norm(streamline_centered, axis=1)
r_min, r_max = r.min(), r.max()
r_normalized = (r - r_min) / (r_max - r_min + epsilon)

theta = np.arccos(np.clip(z_c / (r + epsilon), -1, 1))  # [0, pi]
phi = np.arctan2(y_c, x_c)                               # [-pi, pi]

theta_sin = np.sin(theta)
phi_cos = np.cos(phi)

# --- r vectors: subsample ---
step = 12
idx = np.arange(0, len(x_c), step)

rx, ry, rz = [], [], []
for i in idx:
    rx += [x_c[i], 0, None]
    ry += [y_c[i], 0, None]
    rz += [z_c[i], 0, None]

# --- Axis ranges (make panels 1 & 2 comparable/print-consistent) ---
pad_world = 6
x1_min, x1_max = streamline[:, 0].min() - pad_world, streamline[:, 0].max() + pad_world
y1_min, y1_max = streamline[:, 1].min() - pad_world, streamline[:, 1].max() + pad_world
z1_min, z1_max = streamline[:, 2].min() - pad_world, streamline[:, 2].max() + pad_world

pad_centered = 3
x2_min, x2_max = x_c.min() - pad_centered, x_c.max() + pad_centered
y2_min, y2_max = y_c.min() - pad_centered, y_c.max() + pad_centered
z2_min, z2_max = z_c.min() - pad_centered, z_c.max() + pad_centered

# --- Figure: 1 row × 3 cols ---
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=[
        "1. Raw streamline + CoM",
        "2. Centered streamline + r vectors",
        "3. Feature space (cosφ, sinθ, r_norm)",
    ],
    specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}, {"type": "scatter3d"}]],
    horizontal_spacing=0.04,
)

# Panel 1: raw streamline + CoM
fig.add_trace(
    go.Scatter3d(
        x=streamline[:, 0], y=streamline[:, 1], z=streamline[:, 2],
        mode="lines",
        line=dict(width=5, color="rgba(50,70,170,0.95)"),
        name="Streamline",
        showlegend=True,
    ),
    row=1, col=1,
)
fig.add_trace(
    go.Scatter3d(
        x=[com[0]], y=[com[1]], z=[com[2]],
        mode="markers",
        marker=dict(size=8, symbol="diamond", color="rgba(200,60,60,1)"),
        name="Center of mass (CoM)",
        showlegend=True,
    ),
    row=1, col=1,
)

# Panel 2: centered streamline
fig.add_trace(
    go.Scatter3d(
        x=x_c, y=y_c, z=z_c,
        mode="lines",
        line=dict(width=5, color="rgba(30,160,120,0.95)"),
        name="Centered streamline",
        showlegend=True,
    ),
    row=1, col=2,
)

# Panel 2: r vectors (lighter + fewer visual priority)
fig.add_trace(
    go.Scatter3d(
        x=rx, y=ry, z=rz,
        mode="lines",
        line=dict(width=1.5, color="rgba(230,150,60,0.35)", dash="dot"),
        name="r vectors",
        showlegend=True,
    ),
    row=1, col=2,
)

# Panel 2: anchor points (shared coloraxis)
fig.add_trace(
    go.Scatter3d(
        x=x_c[idx], y=y_c[idx], z=z_c[idx],
        mode="markers",
        marker=dict(
            size=6,
            symbol="circle",
            color=r_normalized[idx],
            coloraxis="coloraxis",
            line=dict(color="white", width=1),
        ),
        name="Anchor points",
        showlegend=True,
    ),
    row=1, col=2,
)

# Panel 2: origin marker (no 3D text labels)
fig.add_trace(
    go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers",
        marker=dict(size=8, symbol="diamond", color="rgba(200,60,60,1)"),
        name="Origin (CoM)",
        showlegend=True,
    ),
    row=1, col=2,
)

# Panel 3: feature-space line (neutral, no competing colorscale)
fig.add_trace(
    go.Scatter3d(
        x=phi_cos,
        y=theta_sin,
        z=r_normalized,
        mode="lines",
        line=dict(width=4, color="rgba(120,120,120,0.55)"),
        name="Trajectory",
        showlegend=True,
    ),
    row=1, col=3,
)

# Panel 3: mapped points (shared coloraxis)
fig.add_trace(
    go.Scatter3d(
        x=phi_cos[idx],
        y=theta_sin[idx],
        z=r_normalized[idx],
        mode="markers",
        marker=dict(
            size=6,
            symbol="circle",
            color=r_normalized[idx],
            coloraxis="coloraxis",
            line=dict(color="white", width=1),
        ),
        name="Mapped points",
        showlegend=True,
    ),
    row=1, col=3,
)

# --- Consistent scenes: aspect + camera + subtle grid (thesis-friendly) ---
camera = dict(eye=dict(x=1.6, y=1.6, z=1.05))

axis_mm = dict(
    showbackground=False,
    showgrid=True,
    gridcolor="rgba(0,0,0,0.15)",
    zeroline=False,
    ticks="outside",
    ticklen=4,
    tickcolor="rgba(0,0,0,0.5)",
)

axis_feat = dict(
    showbackground=False,
    showgrid=True,
    gridcolor="rgba(0,0,0,0.15)",
    zeroline=False,
    ticks="outside",
    ticklen=4,
    tickcolor="rgba(0,0,0,0.5)",
)

fig.update_layout(
    template="plotly_white",
    title=dict(
        text="Tractography streamline transformations",
        x=0.01, xanchor="left",
    ),
    font=dict(family="Times New Roman", size=16),
    height=650,
    margin=dict(l=10, r=10, t=80, b=105),
    legend=dict(orientation="h", yanchor="bottom", y=-0.20, xanchor="center", x=0.5),
    # Shared colorbar for r_norm across panels 2 & 3 (via coloraxis) [shared coloraxis approach]
    coloraxis=dict(
        colorscale="Viridis",
        cmin=0, cmax=1,
        colorbar=dict(
            title="r_norm",
            len=0.72,
            y=0.52,
            x=1.02,
            thickness=16,
        ),
    ),
)

# Scene 1
fig.update_layout(
    scene=dict(
        aspectmode="data",
        camera=camera,
        xaxis=dict(title="X (mm)", range=[x1_min, x1_max], **axis_mm),
        yaxis=dict(title="Y (mm)", range=[y1_min, y1_max], **axis_mm),
        zaxis=dict(title="Z (mm)", range=[z1_min, z1_max], **axis_mm),
    )
)

# Scene 2
fig.update_layout(
    scene2=dict(
        aspectmode="data",
        camera=camera,
        xaxis=dict(title="X (mm)", range=[x2_min, x2_max], **axis_mm),
        yaxis=dict(title="Y (mm)", range=[y2_min, y2_max], **axis_mm),
        zaxis=dict(title="Z (mm)", range=[z2_min, z2_max], **axis_mm),
    )
)

# Scene 3
fig.update_layout(
    scene3=dict(
        aspectmode="data",
        camera=camera,
        xaxis=dict(title="cos(φ)", **axis_feat),
        yaxis=dict(title="sin(θ)", **axis_feat),
        zaxis=dict(title="r_norm", range=[0, 1], **axis_feat),
    )
)

fig.show()

# --- Static export (recommended for thesis PDF/SVG) ---
# Requires: pip install -U kaleido
import kaleido
kaleido.get_chrome_sync()
fig.write_image("tractography_transform.pdf", width=1800, height=650, scale=2)
fig.write_image("tractography_transform.svg", width=1800, height=650)
