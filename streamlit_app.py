"""
DepthFX — Streamlit Interface

Clean, full-width, always-on real-time AI monocular depth estimation
and depth-aware visual effects dashboard with integrated controls.
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# Prevent Streamlit module watcher from crashing when inspecting torch.classes
try:
    import torch
    if hasattr(torch, "_classes") and hasattr(torch._classes, "__getattr__"):
        try:
            torch.classes.__path__ = []
        except Exception:
            pass
except Exception:
    pass

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ── Page Configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DepthFX Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── Clean, Full-Width Professional Stylesheet ─────────────────────────────────

st.markdown("""
<style>
    /* Dark minimalist palette */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        overflow-y: hidden;
    }

    /* Hide sidebar completely */
    [data-testid="stSidebar"], section[data-testid="stSidebar"] {
        display: none !important;
    }

    /* Page container: extra top space so nothing is hidden under Streamlit's top bar */
    .block-container {
        padding-top: 4.5rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }

    footer, #MainMenu { visibility: hidden; }


    /* Dashboard Header — centered, large, accented */
    .app-header-wrap {
        text-align: center;
        margin-bottom: 14px;
    }
    .app-title {
        font-size: 34px;
        font-weight: 800;
        letter-spacing: 1.2px;
        line-height: 1.15;
    }
    .app-title-depth { color: #3fb950; }
    .app-title-fx    { color: #f0f6fc; }
    .app-title-dash  {
        font-size: 18px;
        font-weight: 600;
        color: #c9d1d9;
        letter-spacing: 1px;
        margin-top: 3px;
    }
    .app-subtitle {
        font-size: 14px;
        color: #8b949e;
        margin-top: 4px;
        letter-spacing: 0.2px;
    }

    /* ── Video panel cards ─────────────────────────────────────────── */
    /* View header: flush border, spans full column width */
    .view-header {
        background: #161b22;
        border: 1px solid #30363d;
        border-bottom: none;
        border-radius: 6px 6px 0 0;
        padding: 7px 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 13px;
        font-weight: 600;
        color: #c9d1d9;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        /* Full-bleed: cancel column padding */
        margin-left: 0; margin-right: 0;
    }
    .view-tag {
        font-size: 12px;
        font-family: Consolas, monospace;
        color: #8b949e;
        font-weight: 400;
    }

    /* Strip Streamlit wrappers around st.image so the frame is flush */
    [data-testid="stElementContainer"]:has([data-testid="stImage"]) {
        padding: 0 !important;
        margin-top: -6px !important; /* close gap between header and image */
    }
    [data-testid="stElementContainer"]:has([data-testid="stImage"]) > div,
    [data-testid="stImage"] > div,
    [data-testid="stImage"] > div > div {
        padding: 0 !important;
        margin: 0 !important;
        width: 100% !important;
    }

    /*
     * KEY FIX: aspect-ratio:4/3 makes the container height = width × 0.75
     * so the 640×480 (4:3) camera image exactly fills the box with ZERO letterboxing.
     * object-fit:contain preserves aspect ratio without distortion.
     */
    [data-testid="stImage"] {
        width: 100% !important;
        aspect-ratio: 4 / 3 !important;
        display: block !important;
        box-sizing: border-box !important;
        overflow: hidden !important;
        border-left: 1px solid #30363d !important;
        border-right: 1px solid #30363d !important;
        border-bottom: 1px solid #30363d !important;
        border-top: none !important;
        border-radius: 0 0 6px 6px !important;
        background-color: #010409 !important;
        line-height: 0 !important;
    }
    [data-testid="stImage"] img {
        width: 100% !important;
        height: 100% !important;
        object-fit: contain !important;
        display: block !important;
        padding: 0 !important;
        margin: 0 !important;
        border-radius: 0 !important;
    }


    /* Telemetry grid cards in main dashboard */
    .telemetry-grid-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 10px 14px;
        height: 100%;
        margin-top: 10px;
    }
    .telemetry-card-title {
        font-size: 13px;
        font-weight: 700;
        color: #f0f6fc;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
        border-bottom: 1px solid #21262d;
        padding-bottom: 4px;
    }
    .data-row {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        padding: 3px 0;
        font-size: 13px;
    }
    .data-label { color: #8b949e; }
    .data-val {
        color: #58a6ff;
        font-family: Consolas, monospace;
        font-weight: 600;
        font-size: 13px;
    }

    /* Unique action buttons at the bottom — style real Streamlit buttons */
    .action-col button {
        font-size: 14px !important;
        font-weight: 700 !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        padding: 12px 48px !important;
        border-radius: 6px !important;
        width: 100% !important;
        min-height: 48px !important;
        transition: all 0.18s ease !important;
        box-shadow: 0 2px 8px rgba(88,166,255,0.15) !important;
        border: 2px solid #58a6ff !important;
        background: transparent !important;
        color: #58a6ff !important;
    }
    .action-col button:hover {
        background: #58a6ff !important;
        color: #0d1117 !important;
        box-shadow: 0 4px 16px rgba(88,166,255,0.35) !important;
    }
    .reset-col button {
        font-size: 14px !important;
        font-weight: 700 !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        padding: 12px 48px !important;
        border-radius: 6px !important;
        width: 100% !important;
        min-height: 48px !important;
        transition: all 0.18s ease !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.35) !important;
        border: 2px solid #30363d !important;
        background: transparent !important;
        color: #8b949e !important;
    }
    .reset-col button:hover {
        background: #30363d !important;
        color: #f0f6fc !important;
    }
</style>
""", unsafe_allow_html=True)


# ── System Info Discovery ────────────────────────────────────────────────────

@st.cache_data
def get_system_info():
    import torch
    info = {
        "cuda": torch.cuda.is_available(),
        "pytorch": torch.__version__,
        "gpu": "Unavailable",
        "cuda_version": "Unavailable",
        "compute": "Unavailable",
    }
    if info["cuda"]:
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = str(torch.version.cuda) if torch.version.cuda else "N/A"
        cap = torch.cuda.get_device_capability(0)
        info["compute"] = f"{cap[0]}.{cap[1]}"
    return info


SYS = get_system_info()


def _clean_gpu_name(name):
    for prefix in ("NVIDIA GeForce ", "NVIDIA ", "AMD Radeon ", "Intel(R) "):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


# ── Model Loading ────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(input_size=320):
    from depth_estimator import DepthEstimator
    return DepthEstimator(input_size=input_size)


# ── CPU Effects and Colormaps ────────────────────────────────────────────────

PRESETS = {
    "LIGHT":  {"fog": 0.30, "blur": 0.25, "fs": 0.55, "fe": 0.95},
    "MEDIUM": {"fog": 0.55, "blur": 0.50, "fs": 0.35, "fe": 0.90},
    "STRONG": {"fog": 0.85, "blur": 0.85, "fs": 0.20, "fe": 0.80},
}

COLORMAPS = {
    "Inferno": cv2.COLORMAP_INFERNO,
    "Turbo": cv2.COLORMAP_TURBO,
    "Jet": cv2.COLORMAP_JET,
    "Magma": cv2.COLORMAP_MAGMA,
    "Plasma": cv2.COLORMAP_PLASMA,
}


def apply_fog(frame, depth, strength, start, end):
    dist = 1.0 - depth
    t = np.clip((dist - start) / max(end - start, 1e-6), 0.0, 1.0) * strength
    fog_color = np.array([184, 196, 214], dtype=np.float32)
    return np.clip(
        frame.astype(np.float32) * (1.0 - t[..., None]) + fog_color * t[..., None],
        0, 255,
    ).astype(np.uint8)


def apply_blur(frame, depth, strength):
    blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=10)
    dist = 1.0 - depth
    mask = np.clip((dist - 0.5) / 0.5, 0.0, 1.0) * strength
    mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=3)[..., None]
    return np.clip(
        frame.astype(np.float32) * (1.0 - mask) + blurred.astype(np.float32) * mask,
        0, 255,
    ).astype(np.uint8)


def apply_bg_blur(frame, depth, strength):
    blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=14)
    dist = 1.0 - depth
    mask = (np.clip((dist - 0.25) / 0.50, 0.0, 1.0) * strength)[..., None]
    return np.clip(
        frame.astype(np.float32) * (1.0 - mask) + blurred.astype(np.float32) * mask,
        0, 255,
    ).astype(np.uint8)


def depth_to_heatmap(depth, colormap_id=cv2.COLORMAP_INFERNO):
    d_u8 = (np.clip(depth, 0.0, 1.0) * 255.0).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(d_u8, colormap_id)
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)


def depth_to_grayscale(depth):
    g = (np.clip(depth, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)


def render_frame_image(ph, img_rgb):
    """Render image using JPEG encoding for fast, error-free browser rendering."""
    ph.image(img_rgb, channels="RGB", output_format="JPEG", width="stretch")


# ── Render Dashboard Telemetry Grid HTML Helper ──────────────────────────────

def generate_telemetry_grid_html(fps=0.0, ai_ms=0.0):
    """Generates the clean horizontal 4-column telemetry grid for the bottom dashboard space."""
    gpu_label = _clean_gpu_name(SYS.get("gpu", "Unavailable"))
    cuda_status = "Active" if SYS["cuda"] else "Unavailable"

    cards = f"""
<div style="display: flex; gap: 12px; flex-wrap: nowrap;">
  <div style="flex: 1; min-width: 180px;">
    <div class="telemetry-grid-card">
      <div class="telemetry-card-title">Performance</div>
      <div class="data-row"><span class="data-label">Frame Rate</span><span class="data-val">{fps:.1f} FPS</span></div>
      <div class="data-row"><span class="data-label">AI Latency</span><span class="data-val">{ai_ms:.1f} ms</span></div>
      <div class="data-row"><span class="data-label">Inference Rate</span><span class="data-val">Every Frame (1:1)</span></div>
      <div class="data-row"><span class="data-label">AI Resolution</span><span class="data-val">320×320</span></div>
      <div class="data-row"><span class="data-label">Motion Delay</span><span class="data-val">Instant (0.00)</span></div>
    </div>
  </div>

  <div style="flex: 1; min-width: 180px;">
    <div class="telemetry-grid-card">
      <div class="telemetry-card-title">System Status</div>
      <div class="data-row"><span class="data-label">Camera</span><span class="data-val">Connected</span></div>
      <div class="data-row"><span class="data-label">AI Model</span><span class="data-val">Ready</span></div>
      <div class="data-row"><span class="data-label">CUDA Engine</span><span class="data-val">{cuda_status}</span></div>
      <div class="data-row"><span class="data-label">Pipeline</span><span class="data-val">Running</span></div>
      <div class="data-row"><span class="data-label">Renderer</span><span class="data-val">CUDA / OpenCV</span></div>
    </div>
  </div>

  <div style="flex: 1; min-width: 180px;">
    <div class="telemetry-grid-card">
      <div class="telemetry-card-title">Hardware</div>
      <div class="data-row"><span class="data-label">GPU</span><span class="data-val">{gpu_label}</span></div>
      <div class="data-row"><span class="data-label">CUDA Version</span><span class="data-val">{SYS.get("cuda_version", "N/A")}</span></div>
      <div class="data-row"><span class="data-label">PyTorch</span><span class="data-val">{SYS.get("pytorch", "N/A")}</span></div>
      <div class="data-row"><span class="data-label">Compute Cap</span><span class="data-val">{SYS.get("compute", "N/A")}</span></div>
      <div class="data-row"><span class="data-label">Host OS</span><span class="data-val">Windows</span></div>
    </div>
  </div>

  <div style="flex: 1; min-width: 180px;">
    <div class="telemetry-grid-card">
      <div class="telemetry-card-title">AI Model</div>
      <div class="data-row"><span class="data-label">Architecture</span><span class="data-val">Depth Anything V2</span></div>
      <div class="data-row"><span class="data-label">Backbone</span><span class="data-val">DINOv2 ViT-S</span></div>
      <div class="data-row"><span class="data-label">Decoder Head</span><span class="data-val">DPT</span></div>
      <div class="data-row"><span class="data-label">Precision</span><span class="data-val">FP16 Autocast</span></div>
      <div class="data-row"><span class="data-label">Weights</span><span class="data-val">vits.pth</span></div>
    </div>
  </div>
</div>
"""
    return cards


# ── Top Dashboard Header ─────────────────────────────────────────────────────

st.markdown("""
<div class="app-header-wrap">
  <div class="app-title">
    <span class="app-title-depth">Depth</span><span class="app-title-fx">FX</span>
  </div>
  <div class="app-title-dash">Dashboard</div>
  <div class="app-subtitle">Real-Time Monocular Depth Estimation &amp; Depth-Aware Visual Effects</div>
</div>
""", unsafe_allow_html=True)


# ── Integrated Control Toolbar (On Dashboard) ────────────────────────────────

ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4, ctrl_c5, ctrl_c6 = st.columns(
    [1.5, 1.3, 1.3, 1.4, 2.0, 1.0], vertical_alignment="center"
)

with ctrl_c1:
    heatmap_name = st.selectbox(
        "Heatmap Palette",
        list(COLORMAPS.keys()),
        index=0,
        label_visibility="collapsed",
    )
    active_colormap = COLORMAPS[heatmap_name]

with ctrl_c2:
    fog_on = st.toggle("Atmospheric Fog", value=True)

with ctrl_c3:
    blur_on = st.toggle("Depth Blur", value=True)

with ctrl_c4:
    bg_blur_on = st.toggle("Background Blur", value=False)

with ctrl_c5:
    preset = st.radio(
        "Preset",
        ["LIGHT", "MEDIUM", "STRONG"],
        index=1,
        label_visibility="collapsed",
        horizontal=True,
    )

with ctrl_c6:
    camera_index = st.selectbox(
        "Camera",
        options=list(range(11)),
        index=0,
        label_visibility="collapsed",
    )


# ── Always-On Live Stream Execution ──────────────────────────────────────────

# 1. Model Initialization
try:
    with st.spinner("Loading AI model..."):
        model = load_model()
except FileNotFoundError:
    st.error("Model checkpoint not found in checkpoints/depth_anything_v2_vits.pth")
    st.stop()
except Exception as e:
    st.error(f"Model initialization error: {e}")
    st.stop()

# 2. Camera Connection
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    cap.release()
    cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    st.error(f"Cannot access camera index {camera_index}. Please check camera connection.")
    st.stop()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

import torch

# Triple View Columns (Full Width)
c1, c2, c3 = st.columns(3, gap="small")
with c1:
    st.markdown(
        f'<div class="view-header">'
        f'<span>Normal + Effects</span>'
        f'<span class="view-tag">{cam_w}×{cam_h}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    norm_ph = st.empty()

with c2:
    st.markdown(
        f'<div class="view-header">'
        f'<span>Depth Map</span>'
        f'<span class="view-tag">320×320</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    depth_ph = st.empty()

with c3:
    st.markdown(
        f'<div class="view-header">'
        f'<span>Depth Heatmap</span>'
        f'<span class="view-tag">{heatmap_name}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    heat_ph = st.empty()

# Dashboard Telemetry Placeholder in the bottom space
telemetry_ph = st.empty()
telemetry_ph.markdown(generate_telemetry_grid_html(0.0, 0.0), unsafe_allow_html=True)

# Action buttons at the very bottom — centered with unique CSS-styled design
st.markdown('<div style="margin-top: 18px;"></div>', unsafe_allow_html=True)
_gap1, _snap_col, _reset_col, _gap2 = st.columns([3.5, 1, 1, 3.5])
with _snap_col:
    st.markdown('<div class="action-col">', unsafe_allow_html=True)
    if st.button("Snapshot", use_container_width=True, key="snap_btn"):
        st.session_state["snapshot_pending"] = True
    st.markdown('</div>', unsafe_allow_html=True)
with _reset_col:
    st.markdown('<div class="reset-col">', unsafe_allow_html=True)
    if st.button("Reset", use_container_width=True, key="reset_btn"):
        for k in list(st.session_state.keys()):
            if k not in ("snap_btn", "reset_btn"):
                del st.session_state[k]
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


fps_counter = 0
fps = 0.0
fps_time = time.perf_counter()
ai_ms = 0.0
perf_time = 0.0
settings = PRESETS[preset]

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        fps_counter += 1

        # Depth inference (computed every frame for real-time responsiveness)
        t0 = time.perf_counter()
        raw_depth = model.estimate(frame)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ai_ms = (time.perf_counter() - t0) * 1000.0

        raw_depth = np.asarray(raw_depth, dtype=np.float32)
        if raw_depth.shape[:2] != frame.shape[:2]:
            raw_depth = cv2.resize(raw_depth, (frame.shape[1], frame.shape[0]))

        depth = raw_depth  # Zero ghosting / instantaneous tracking

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1. Normal frame with applied effects
        output_normal = rgb.copy()
        if blur_on:
            output_normal = apply_blur(output_normal, depth, settings["blur"])
        if bg_blur_on:
            output_normal = apply_bg_blur(output_normal, depth, settings["blur"])
        if fog_on:
            output_normal = apply_fog(output_normal, depth, settings["fog"], settings["fs"], settings["fe"])

        # 2. Grayscale Depth Map
        output_depth = depth_to_grayscale(depth)

        # 3. Thermal Heatmap
        output_heat = depth_to_heatmap(depth, colormap_id=active_colormap)

        # Render video streams
        render_frame_image(norm_ph, output_normal)
        render_frame_image(depth_ph, output_depth)
        render_frame_image(heat_ph, output_heat)

        # Snapshot handler
        if st.session_state.get("snapshot_pending"):
            out_dir = ROOT_DIR / "outputs"
            out_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            combined = np.hstack([output_normal, output_depth, output_heat])
            fpath = out_dir / f"depthfx_snapshot_{timestamp}.png"
            cv2.imwrite(str(fpath), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            st.session_state["snapshot_pending"] = False
            st.toast(f"Snapshot saved: outputs/{fpath.name}")

        # FPS calculation
        now = time.perf_counter()
        elapsed = now - fps_time
        if elapsed >= 0.8:
            fps = fps_counter / elapsed
            fps_counter = 0
            fps_time = now

        # Dashboard Telemetry update (~2 Hz)
        if now - perf_time >= 0.5:
            telemetry_ph.markdown(
                generate_telemetry_grid_html(fps, ai_ms),
                unsafe_allow_html=True,
            )
            perf_time = now

        time.sleep(0.004)

finally:
    cap.release()
