# DepthFX — Complete Technical Analysis Report

> **Report generated from:** direct source code inspection of every file in the repository.
> **Date:** 2026-09-02 (updated)
> **Original report date:** 2026-08-31
> **Scope:** All source, shaders, model code, archive, assets, config, documentation, and the new Streamlit web dashboard added 2026-09-01.

---

# PART 1 — COMPLETE PROJECT UNDERSTANDING

---

## 1. Complete Project Overview

DepthFX is a real-time computer vision and GPU rendering application written in Python. It captures live webcam video, estimates per-pixel depth using a deep neural network (Depth Anything V2 Small), and applies GPU-accelerated visual effects (fog, blur, lighting, edge enhancement) that are driven by the depth map — all running at interactive frame rates on a single machine with an NVIDIA GPU.

The entire pipeline runs in a single Python process: OpenCV captures frames, PyTorch + CUDA runs the AI model, and OpenGL + GLSL renders the final output with depth-aware effects.

## 2. Main Purpose

DepthFX demonstrates how to pair a real-time AI inference workload (monocular depth estimation) with a GPU graphics rendering pipeline (OpenGL/GLSL effects). It is a portfolio/demonstration project that proves the author can build a system where two distinct GPU workloads — neural network inference and fragment shader rendering — run concurrently in one application.

## 3. What Problem It Solves

Standard webcam feeds have no depth information — every pixel is just RGB with no notion of distance. DepthFX recovers approximate depth from a single 2D camera image in real time, then uses that depth to drive visual effects that normally require dedicated depth sensors (LiDAR, stereo cameras, structured-light scanners, etc.).

## 4. Why AI Depth Estimation Is Used

Because the input is a monocular (single-lens) webcam, there is no geometric baseline for triangulation. Depth must be inferred from visual cues alone. A deep neural network trained on millions of images learns to recognise cues like perspective, occlusion, texture gradients, and relative object sizes — enabling per-pixel relative depth estimation from a single RGB image.

## 5. Why Depth Anything V2

Depth Anything V2 is a state-of-the-art monocular depth estimation model from TikTok/ByteDance. It was chosen because:

- **Small variant (ViT-S)** is fast enough for real-time use (~20 ms on an RTX 4070)
- Uses a **DINOv2 backbone** pre-trained via self-supervised learning, giving strong visual features without labelled depth data
- Produces **dense, smooth depth maps** with good edge preservation
- The **DPT (Dense Prediction Transformer) head** provides multi-scale feature fusion for high-quality depth
- Pre-trained checkpoints are publicly available
- The architecture supports variable input sizes (the project uses 320x320)

## 6. Complete Folder and File Structure

```
DepthFX/
├── .git/                                   # Git repository
├── .gitignore                              # Ignore rules (572 bytes)
├── .streamlit/                             # Streamlit config directory
├── .venv/                                  # Python virtual environment
├── .vscode/
│   └── settings.json                       # VS Code interpreter + extra paths
├── README.md                               # Project documentation (11.7 KB)
├── PROJECT_REPORT.md                       # This complete technical report
├── requirements.txt                        # Pinned Python dependencies (591 bytes)
├── run_app.bat                             # Windows batch launcher for the Streamlit dashboard
├── streamlit_app.py                        # Streamlit web dashboard (627 lines) — NEW
│
├── assets/
│   ├── images/
│   │   ├── DepthFX_CoverImage.png          # High-resolution project cover banner (1.47 MB)
│   │   ├── depth_test.jpeg                 # Real-time triple-view screenshot (3456×2160, 1.18 MB)
│   │   ├── depth_test_.png                 # Depth heatmap visualization example (114 KB)
│   │   └── sample.png                      # Visual effects & depth estimation sample (1.57 MB)
│   └── videos/
│       ├── demo.gif                        # Animated live demonstration preview (1.57 MB)
│       ├── demo.webp                       # Lightweight WebP animation preview (672 KB)
│       └── InShot_20260902_015848586.mp4   # Real-time live demo recording (31.2 MB)
│
├── checkpoints/
│   └── depth_anything_v2_vits.pth          # Model weights (~94.6 MB, git-ignored)
│
├── outputs/                                # Runtime screenshots (git-ignored)
│   ├── depthfx_20260825_163723.png
│   └── depthfx_20260825_164254.png
│
├── archive/
│   └── old_experiments/                    # Superseded earlier implementations
│       ├── effects.py                      # CPU-based fog + blur (NumPy/OpenCV)
│       ├── gpu_depth_blur.py               # Earlier GPU blur experiment
│       ├── gpu_depth_camera.py             # Earlier GPU camera experiment
│       ├── gpu_depth_fog.py                # Earlier GPU fog experiment
│       ├── gpu_depth_lighting.py           # Earlier GPU lighting experiment
│       ├── gpu_depth_test.py               # Earlier GPU depth test
│       ├── gpu_renderer.py                 # Earlier modular GPU renderer
│       ├── gpu_test.py                     # Minimal OpenGL init test
│       ├── performance.py                  # CPU-side PerformanceProfiler
│       ├── test_depth.py                   # Single-frame depth test script
│       ├── test_depth_utils.py             # Manual normalize_depth test
│       ├── test_effects.py                 # Manual fog test
│       ├── webcam.py                       # Minimal webcam viewer (OpenCV)
│       └── shaders/                        # Old separate GLSL fragments
│           ├── camera.frag
│           ├── depth_blur.frag
│           ├── depth_fog.frag
│           ├── depth_fx.frag               # 577-line combined shader (old)
│           ├── depth_lighting.frag
│           ├── depth_view.frag
│           └── overlay.frag
│
├── scripts/                                # Empty — reserved for future
├── tests/                                  # Empty — no automated tests
│
└── src/
    ├── __init__.py                         # Empty package marker
    ├── depth_utils.py                      # normalize_depth() function
    ├── depth_estimator.py                  # DepthEstimator class + benchmark
    ├── gpu_depth_fx.py                     # OpenGL/GLFW application (55,569 bytes)
    ├── shaders/
    │   └── fullscreen.vert                 # External vertex shader file (not used by main)
    └── depth_anything_v2/                  # Model implementation
        ├── dpt.py                          # DPTHead + DepthAnythingV2 class
        ├── dinov2.py                       # DinoVisionTransformer + factory functions
        ├── dinov2_layers/
        │   ├── __init__.py                 # Exports Mlp, PatchEmbed, etc.
        │   ├── attention.py                # MemEffAttention
        │   ├── block.py                    # NestedTensorBlock
        │   ├── drop_path.py                # DropPath regularization
        │   ├── layer_scale.py              # LayerScale
        │   ├── mlp.py                      # Mlp (MLP head)
        │   ├── patch_embed.py              # PatchEmbed
        │   └── swiglu_ffn.py               # SwiGLU FFN variants
        └── util/
            ├── blocks.py                   # FeatureFusionBlock, _make_scratch
            └── transform.py               # Resize, NormalizeImage, PrepareForNet
```

## 7. Purpose of Every Important File and Folder

| File / Folder | Purpose |
|---|---|
| `streamlit_app.py` | **NEW — Streamlit web dashboard.** Full-width dark-theme browser UI for the DepthFX pipeline. Contains all CSS styling, effect controls, triple-view video panels, live telemetry grid, and Snapshot/Reset actions. Replaces the GLFW window for web-accessible demonstration. |
| `run_app.bat` | **NEW — Windows batch launcher.** Activates the `.venv` virtual environment and runs `streamlit run streamlit_app.py`. Single-click entry point for the Streamlit dashboard. |
| `gpu_depth_fx.py` | **Main OpenGL application.** Contains inline GLSL shaders, `HUDRenderer`, `GPUTimer`, `save_screenshot()`, `create_*` texture/quad functions, `key_callback`, `cursor_callback`, and the `main()` render loop. This is the original GLFW/OpenGL entry point. |
| `depth_estimator.py` | Wraps Depth Anything V2 into a `DepthEstimator` class. Handles model loading, CUDA device selection, FP16 autocast, `estimate()` for inference, `warmup()` and `benchmark()` methods. Also has a standalone `main()` for benchmarking. |
| `depth_utils.py` | Single utility function `normalize_depth()` — min-max normalisation to [0,1]. |
| `dpt.py` | `DPTHead` (Dense Prediction Transformer head) and `DepthAnythingV2` nn.Module. Contains `infer_image()` which handles preprocessing, forward, output interpolation. |
| `dinov2.py` | `DinoVisionTransformer` — the backbone encoder. Factory functions: `vit_small`, `vit_base`, `vit_large`, `vit_giant2`, and `DINOv2()` dispatcher. |
| `dinov2_layers/` | Standard ViT building blocks: `MemEffAttention`, `NestedTensorBlock`, `Mlp`, `PatchEmbed`, `SwiGLUFFNFused`, `DropPath`, `LayerScale`. |
| `util/transform.py` | Image preprocessing transforms: `Resize` (ensure_multiple_of=14), `NormalizeImage` (ImageNet mean/std), `PrepareForNet` (HWC to CHW). |
| `util/blocks.py` | `FeatureFusionBlock` and `ResidualConvUnit` — DPT decoder building blocks with residual convolutions and bilinear upsampling. |
| `fullscreen.vert` | External GLSL vertex shader (present but **not loaded** by the main application, which uses inline shaders). |
| `checkpoints/` | Holds the ~94.6 MB model weights file `depth_anything_v2_vits.pth`. |
| `outputs/` | Runtime screenshots saved by `save_screenshot()` (OpenGL) or Snapshot button (Streamlit). |
| `assets/images/` | Contains `depth_test.jpg` — real-time triple-view live pipeline screenshot used in the README. |
| `archive/old_experiments/` | Superseded code from development iteration — CPU-based effects, modular GPU experiments, standalone test scripts, old separate GLSL shader files. |
| `tests/` | **Empty.** No automated tests exist. |
| `scripts/` | **Empty.** Reserved for future utility scripts. |
| `requirements.txt` | Pinned dependencies including PyTorch 2.13+cu130 and Streamlit. |
| `.gitignore` | Excludes `.pth` weights, `.venv`, `outputs/`, `__pycache__/`, media files, etc. |
| `README.md` | Comprehensive documentation with architecture diagram, feature list, controls, performance data, installation instructions. |

## 8. How the Project Starts and Runs

**Two entry points now exist:**

### Entry Point A — Streamlit Web Dashboard (NEW, primary demo interface)
```
.\run_app.bat          (Windows launcher)
# or directly:
streamlit run streamlit_app.py
```
1. Streamlit starts a local web server (default: http://localhost:8501)
2. The page loads and immediately auto-starts the AI pipeline (no button needed)
3. `load_model()` is called via `@st.cache_resource` — loads Depth Anything V2 onto CUDA
4. `cv2.VideoCapture(camera_index)` opens the webcam
5. Triple-view column layout renders: Normal + Effects | Depth Map | Depth Heatmap
6. A `while True` loop runs the pipeline, updating `st.empty()` placeholders each frame
7. Telemetry grid is updated every 0.5s
8. Snapshot and Reset buttons are rendered at the bottom

### Entry Point B — OpenGL/GLFW Application (original)
```
python src/gpu_depth_fx.py
```
1. `main()` is called from `if __name__ == "__main__":`
2. `DepthEstimator()` is instantiated — loads Depth Anything V2 ViT-S checkpoint onto CUDA, sets `model.eval()`, enables `cudnn.benchmark` and TF32
3. GLFW initialises an OpenGL 3.3 Core Profile context, creates a 1280x720 window with vsync disabled (`swap_interval(0)`)
4. The inline GLSL vertex+fragment shaders are compiled into a program
5. A fullscreen quad VAO/VBO is created
6. Webcam is opened (tries `CAMERA_INDEX=4` with DirectShow, falls back to auto-detect indices 0-5)
7. Camera actual resolution is read; textures (RGB8 + R32F) are created at that resolution
8. `GPUTimer` and `HUDRenderer` are initialised
9. All shader uniform locations are queried
10. Key and cursor callbacks are registered
11. **Main loop** runs until `Q` is pressed or window is closed

## 9. Complete End-to-End Data Flow

```
Webcam -> BGR frame (OpenCV) -> [every 2nd frame] -> DepthEstimator.estimate()
  -> infer_image() at 320px with FP16 autocast
  -> raw float32 depth -> normalize_depth() [0,1]
  -> resize to camera resolution if needed
  -> EWMA temporal smoothing with previous depth
  -> current_depth float32 array

BGR frame -> cv2.cvtColor(BGR to RGB) -> np.flipud() -> RGB uint8 array
current_depth -> np.flipud() -> float32 array

RGB -> glTexSubImage2D -> GL_TEXTURE0 (RGB8 texture)
depth -> glTexSubImage2D -> GL_TEXTURE1 (R32F texture)

GPU timer begin ->
  Fullscreen quad drawn -> fragment shader executes for every pixel:
    -> sample u_depth (R32F) -> clamp [0,1]
    -> if DEPTH mode: output grayscale depth
    -> if HEATMAP mode: output blue to green to red color
    -> if NORMAL mode:
      -> sample u_color (RGB8) -> get base color
      -> depthAwareBlur() -> multi-tap depth-driven blur
      -> applyBackgroundBlur() -> wider-band background blur
      -> applyLighting() -> mouse-driven virtual point light x depth
      -> depthEdge() -> depth-discontinuity edge enhancement
      -> applyFog() -> smoothstep fog ramp
      -> clamp and output
GPU timer end ->

HUD overlay rendered (PIL -> RGBA textures -> alpha-blended quads)
Screenshot captured if requested
Buffers swapped
FPS/timing updated
```

## 10. Webcam / Phone Camera Input

- **Webcam:** OpenCV `VideoCapture` with DirectShow backend (`cv2.CAP_DSHOW`) on Windows. Targets 640x480, buffer size 1.
- **Auto-detection:** If the configured `CAMERA_INDEX = 4` fails, tries indices 0-5 with validation: minimum 160x120, non-black frame, std > 5.0.
- **Phone camera:** Supported indirectly — if a phone is connected via DroidCam, Iriun, or similar virtual webcam software that registers as a system camera, it works with no code change. `CAMERA_INDEX` just needs to match the virtual device index.
- The camera verification (`_open_camera()`) discards 5 stale frames before reading a test frame and validates that the frame contains actual image content (not a dummy/black/noise stream).

## 11. OpenCV Processing

- Reads BGR frames from the camera each iteration
- Resizes to `CAMERA_WIDTH x CAMERA_HEIGHT` if the frame dimensions don't match (using `cv2.INTER_AREA`)
- `cv2.cvtColor(BGR to RGB)` for OpenGL upload
- `np.flipud()` to flip vertically (OpenGL origin is bottom-left, OpenCV is top-left)
- Also used inside `DepthAnythingV2.infer_image()` for BGR to RGB conversion and inside `save_screenshot()` for RGB to BGR conversion before `cv2.imwrite()`

## 12. Depth Anything V2 Model

The model is the **Small** variant:
- **Encoder:** DINOv2 ViT-S — 12 transformer blocks, embed_dim=384, 6 attention heads, patch_size=14
- **Decoder:** DPT head with feature fusion — extracts intermediate features from blocks [2, 5, 8, 11], projects through 1x1 convolutions, resizes via transposed convolutions, fuses via `FeatureFusionBlock` residual units, and outputs a single-channel depth map
- **Config:** `encoder="vits"`, `features=64`, `out_channels=[48, 96, 192, 384]`
- **Inference:** `infer_image(raw_image, input_size=320)` — preprocesses (resize to 320px lower bound with ensure_multiple_of=14, ImageNet normalise, HWC to CHW), runs forward pass, bilinear-interpolates output back to original resolution

## 13. PyTorch Usage

- **Model definition:** `nn.Module` subclasses (`DepthAnythingV2`, `DPTHead`, `DinoVisionTransformer`, all layer modules)
- **Inference:** `torch.inference_mode()` context (disables autograd graph construction)
- **FP16:** `torch.autocast(device_type="cuda", dtype=torch.float16)` — mixed-precision without explicit `.half()` conversion
- **Device management:** `.to(device)`, `torch.cuda.synchronize()` for accurate timing
- **Optimizations:** `cudnn.benchmark = True`, TF32 enabled for both matmul and cuDNN
- **Checkpoint loading:** `torch.load(path, map_location="cpu")` then `model.load_state_dict()`

## 14. CUDA Usage

- All model inference runs on CUDA GPU (`device = "cuda"`)
- `torch.cuda.synchronize()` is called after inference for accurate wall-clock timing
- `torch.cuda.get_device_name(0)`, `.get_device_capability(0)` for device info
- `torch.cuda.memory_allocated()` / `memory_reserved()` for VRAM reporting in benchmark
- `torch.cuda.empty_cache()` on benchmark exit
- CUDA version reported via `torch.version.cuda`

## 15. FP16 Inference

- **Method:** `torch.autocast(device_type="cuda", dtype=torch.float16)` — PyTorch's automatic mixed-precision
- The model weights stay in FP32. The input tensor is FP32. Autocast selects FP16 for eligible operations (linear layers, convolutions, attention) automatically. Accumulation stays in FP32 where needed.
- The code explicitly notes: "Do NOT call self.model.half()" — because `infer_image()` produces FP32 tensors in preprocessing, and autocast handles the type conversion safely.
- FP16 is only enabled when `device == "cuda"` and `USE_FP16 = True`.

## 16. Depth Normalization

**Two normalization passes:**

1. **Inside `DepthEstimator.estimate()`:** calls `normalize_depth()` from `depth_utils.py` — min-max normalization to [0,1], handles degenerate case (range < 1e-6 returns zeros), clips to [0,1], replaces NaN/inf.
2. **Inside the main loop (redundant):** performs a second min-max normalization on the returned depth. This is technically unnecessary since `estimate()` already normalizes, but acts as a safety net.

## 17. Temporal Depth Update

- `DEPTH_UPDATE_INTERVAL = 2` — depth inference runs only on every 2nd frame (and always on frame 1)
- Condition: `total_frames == 1 or (total_frames % DEPTH_UPDATE_INTERVAL == 0)`
- Between depth updates, the last `current_depth` is reused for rendering
- This halves the AI compute cost (~50% update ratio)

## 18. Temporal Smoothing

- **EWMA (Exponentially Weighted Moving Average):**
  ```python
  current_depth = DEPTH_SMOOTHING * previous_depth + (1.0 - DEPTH_SMOOTHING) * new_depth
  ```
- `DEPTH_SMOOTHING = 0.65` — 65% weight on previous frame, 35% weight on new prediction
- On the very first frame, no blending is applied (uses `new_depth` directly)
- `previous_depth = current_depth.copy()` retains the blended result for the next update
- **Purpose:** reduces per-frame flicker and jitter in the depth map without perceptible lag

## 19. CPU Responsibilities

| Task | Details |
|---|---|
| Camera capture | OpenCV reads BGR frames |
| Image conversion | BGR to RGB, vertical flip, contiguous array |
| Depth preprocessing | Input transform pipeline (resize, normalize, transpose) |
| Depth post-processing | min-max normalization, temporal blending, resize, nan handling |
| Texture upload | `glTexSubImage2D` calls (driver copies CPU to GPU) |
| HUD rendering | PIL/Pillow draws text/shapes, uploads RGBA textures |
| FPS calculation | `time.perf_counter()` based counting |
| Input handling | GLFW keyboard/mouse callbacks |
| Screenshot save | `glReadPixels` (GPU to CPU), `cv2.imwrite()` |

## 20. GPU Responsibilities

| Task | Details |
|---|---|
| AI inference | Depth Anything V2 forward pass on CUDA (PyTorch) |
| Fragment shader | All visual effects (fog, blur, lighting, edge, heatmap) |
| Texture storage | RGB8 + R32F textures in VRAM |
| HUD blending | Alpha-composited RGBA quad rendering |
| Timing | `GL_TIME_ELAPSED` query measures shader execution |
| Display | Framebuffer presentation via GLFW swap |

## 21. OpenGL Architecture

- **Profile:** OpenGL 3.3 Core Profile
- **Window:** GLFW-managed, 1280x720 initial size, vsync disabled
- **Rendering model:** Immediate-mode-style single fullscreen quad per frame
- **Shader programs:** Two — main effects shader (inline GLSL) and HUD blit shader (inline GLSL)
- **Geometry:** Fullscreen quad = 2 triangles, 6 vertices, stored in a VAO/VBO
- **Textures:** RGB8 (unit 0), R32F (unit 1), plus 3 RGBA8 HUD panel textures
- **Framebuffer:** Default (on-screen) — no FBO/offscreen rendering
- **Blending:** Enabled only for HUD overlay (`GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA`)
- **Resize handling:** `framebuffer_size_callback` adjusts viewport and repositions HUD

## 22. Texture Formats

| Texture | Internal Format | Upload Format | Type | Purpose |
|---|---|---|---|---|
| RGB camera | `GL_RGB8` | `GL_RGB` | `GL_UNSIGNED_BYTE` | Webcam colour data |
| Depth map | `GL_R32F` | `GL_RED` | `GL_FLOAT` | Depth values [0,1] |
| HUD panels (x3) | `GL_RGBA8` | `GL_RGBA` | `GL_UNSIGNED_BYTE` | UI overlay with alpha |

## 23. RGB Texture Flow

1. OpenCV reads BGR frame, `cv2.cvtColor(BGR to RGB)`, `np.flipud()`, contiguous uint8 array
2. `glPixelStorei(GL_UNPACK_ALIGNMENT, 1)` (RGB has no padding)
3. `glActiveTexture(GL_TEXTURE0)`, `glBindTexture`, `glTexSubImage2D` uploads at camera resolution
4. Fragment shader samples via `texture(u_color, uv).rgb`

## 24. Depth Texture Flow

1. `current_depth` float32 array, `np.flipud()`, contiguous float32 array
2. `glPixelStorei(GL_UNPACK_ALIGNMENT, 4)` (float32 is 4-byte aligned)
3. `glActiveTexture(GL_TEXTURE1)`, `glBindTexture`, `glTexSubImage2D` uploads at camera resolution
4. Fragment shader samples via `texture(u_depth, uv).r`

## 25. R32F Depth Texture

- **GL_R32F:** single-channel, 32-bit floating-point internal format
- **Why R32F?** Depth values are continuous floats in [0,1]. Using R32F preserves full precision without quantisation artifacts that would occur with 8-bit (GL_R8) textures. The shader reads `texture(u_depth, uv).r` and gets the exact float32 value.
- Each texel = 4 bytes. At 640x480, the texture is ~1.2 MB in VRAM.

## 26. GLSL Shader Pipeline

**Vertex Shader (inline):**
- Passes through position and texcoord
- Maps NDC [-1,1] quad to full viewport
- Texcoords [0,1] passed to fragment shader

**Fragment Shader (inline, ~235 lines):**
- `getDepth(uv)` — samples R32F depth texture
- `getColor(uv)` — samples RGB8 colour texture
- `depthHeatmap(depth)` — blue to green to red false-colour mapping
- `depthAwareBlur(uv, original, depth)` — 4-tap weighted blur, radius scales with distance
- `applyBackgroundBlur(uv, color, depth)` — similar blur with wider depth band for background
- `depthEdge(uv)` — Sobel-like edge detection on depth map (4 neighbours)
- `applyLighting(color, depth, uv)` — point light at mouse position, attenuated by distance and modulated by depth
- `applyFog(color, depth)` — smoothstep fog ramp, configurable start/end/strength
- `main()` — dispatches display mode, then chains: blur, bg blur, lighting, edge, fog, clamp, output

## 27. Every Implemented Visual Effect

1. **Depth-aware blur** — multi-tap (4 iterations x 4 directions = 16 samples + center) weighted blur. Blur radius increases with distance from camera. Uses `smoothstep(u_depth_threshold, 1.0, 1.0 - depth)` to ramp.
2. **Background blur** — similar kernel but with a wider `smoothstep(0.25, 0.75, 1.0 - depth)` band, specifically targeting background separation.
3. **Depth-aware fog** — blends scene colour toward `vec3(0.72, 0.77, 0.84)` (light blue-grey) using `smoothstep(u_fog_start, u_fog_end, 1.0 - depth)`.
4. **Mouse-controlled virtual lighting** — screen-space point light positioned at mouse cursor. Light intensity = `(1.0 - smoothstep(0.0, 0.70, distanceToLight))` x depth factor x strength. Ambient term ensures unlit areas are not fully black.
5. **Depth edge enhancement** — computes depth discontinuities by sampling 4 neighbours (left/right/up/down), summing absolute differences x 5.0, adding a subtle `edge * 0.035` to the colour.
6. **Depth grayscale** — `vec3(depth)` — simple grayscale visualisation.
7. **Depth heatmap** — false-colour: depth < 0.5 blends blue to green, depth >= 0.5 blends green to red.

## 28. NORMAL / DEPTH / HEATMAP Modes

| Mode | `u_display_mode` | Rendering |
|---|---|---|
| NORMAL (0) | 0 | Full RGB with all effects applied |
| DEPTH (1) | 1 | `vec3(depth)` — greyscale, close=bright, far=dark |
| HEATMAP (2) | 2 | False-colour (`depthHeatmap()`) + bottom legend bar |

Cycled by pressing `D`. The HUD shows a colour legend bar only in HEATMAP mode.

## 29. Background Blur

- Independent from depth-aware blur — can be toggled separately with `P`
- Uses `smoothstep(0.25, 0.75, 1.0 - depth)` — wider transition band than standard blur
- Same multi-tap kernel (4 iterations x 4 directions)
- Applies after standard blur but before lighting
- Simulates a video-conferencing-style "blur my background" effect
- **Note:** separation is purely depth-based, not semantic (no person segmentation)

## 30. Fog

- Blends scene colour toward `vec3(0.72, 0.77, 0.84)` (light blue-grey fog colour)
- `fog = smoothstep(u_fog_start, u_fog_end, 1.0 - depth) * u_fog_strength`
- Configurable start depth, end depth, and overall strength via presets
- Applied last in the effect chain (after lighting and edge)

## 31. Blur (Depth-Aware)

- Blurs pixels based on their depth: farther pixels get more blur
- `blurFactor = smoothstep(u_depth_threshold, 1.0, 1.0 - depth) * u_blur_strength`
- Early exit: if `blurFactor <= 0.001`, returns original colour (optimization)
- Radius: `1.0 + blurFactor * 4.0` pixels
- 4-tap kernel sampling in horizontal and vertical directions
- Weight decay: `1.0 / (1.0 + i * 0.45)` — closer samples get more weight

## 32. Mouse Lighting

- Cursor position tracked via `cursor_callback`, normalised to [0,1] UV space
- Light attenuation: `1.0 - smoothstep(0.0, 0.70, distanceToLight)`
- Depth modulation: `mix(0.45, 1.0, depth)` — near objects receive more light
- Final: `color * (u_ambient_strength + light * depthFactor * u_light_strength)`
- Mouse Y is inverted: `1.0 - ypos/height` (OpenGL origin is bottom)

## 33. Effect Presets

Three presets control six shader uniforms:

| Preset | fog_str | blur_str | light_str | ambient | fog_start | fog_end |
|---|---|---|---|---|---|---|
| 1 (LIGHT) | 0.30 | 0.25 | 0.35 | 0.80 | 0.55 | 0.95 |
| 2 (MEDIUM) | 0.55 | 0.50 | 0.65 | 0.65 | 0.35 | 0.90 |
| 3 (STRONG) | 0.85 | 0.85 | 0.95 | 0.50 | 0.20 | 0.80 |

Selected via keys `1`, `2`, `3`. Default is MEDIUM. `R` resets to MEDIUM.

## 34. Screenshot Capture

- Triggered by pressing `S`
- `glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)` reads the final framebuffer
- Result is flipped vertically (`np.flipud`), converted RGB to BGR, saved as PNG to `outputs/depthfx_YYYYMMDD_HHMMSS.png`
- Handles filename collision by appending `_1`, `_2`, etc.
- Captured **after** HUD is rendered, so screenshots include the HUD overlay

## 35. GPU Timing / Profiling

- `GPUTimer` class wraps OpenGL `GL_TIME_ELAPSED` timer queries
- `begin()` before fullscreen quad draw, `end()` after
- `read()` checks `GL_QUERY_RESULT_AVAILABLE` and reads nanosecond result, converts to milliseconds
- Maintains a rolling window of up to 300 samples
- On exit, reports average, minimum, and maximum GPU shader time
- Gracefully handles unsupported hardware: disables and reports errors only once

## 36. FPS / Performance Monitoring

- FPS counted over 1-second windows: `fps = fps_counter / elapsed`
- AI inference time: `time.perf_counter()` wall-clock with `torch.cuda.synchronize()`
- GPU shader time: OpenGL timer query (nanosecond precision)
- Window title updated every 250ms: `DepthFX | {fps} FPS | AI {ms} ms | GPU {ms} ms | {mode}`
- HUD stats panel (top-right) shows FPS, AI inference time, GPU shader time, GPU name, CUDA version
- HUD stats throttled to ~4 Hz to avoid performance overhead

## 37. Keyboard Controls

| Key | Action |
|---|---|
| `D` | Cycle display mode: NORMAL, DEPTH, HEATMAP |
| `F` | Toggle fog ON/OFF |
| `B` | Toggle blur ON/OFF |
| `P` | Toggle background blur ON/OFF |
| `1` | Light preset |
| `2` | Medium preset (default) |
| `3` | Strong preset |
| `R` | Reset to defaults (medium, fog+blur on, bg blur off, NORMAL mode) |
| `S` | Screenshot |
| `Q` | Quit |

All key actions trigger on `glfw.PRESS` only (not repeat/release).

## 38. Mouse Controls

- Cursor position continuously tracked via GLFW callback
- Normalized to [0,1] x [0,1] UV space (Y inverted for OpenGL)
- Drives the virtual light position (`u_light_position` uniform)
- Light follows mouse in real time — creates an interactive flashlight/spotlight effect

## 39. Camera Configuration

| Constant | Value | Purpose |
|---|---|---|
| `CAMERA_INDEX` | 4 | Default camera device index |
| `CAMERA_WIDTH` | 640 | Target capture width |
| `CAMERA_HEIGHT` | 480 | Target capture height |
| Buffer size | 1 | Minimise latency |
| Backend | `cv2.CAP_DSHOW` | DirectShow (Windows) |

The actual camera resolution is read back after opening and can override the defaults.

## 40. AI Configuration

| Constant | Value | Location |
|---|---|---|
| `AI_SIZE` / `INFERENCE_SIZE` | 320 | Inference input resolution |
| `USE_FP16` | True | Enable FP16 autocast |
| `WARMUP_FRAMES` | 10 | GPU warmup iterations |
| `BENCHMARK_FRAMES` | 100 | Benchmark iterations |
| `DEPTH_UPDATE_INTERVAL` | 2 | Frames between depth updates |
| `DEPTH_SMOOTHING` | 0.65 | EWMA blend factor |
| Model | `vits` | ViT-Small encoder |
| Features | 64 | DPT head features |
| Out channels | [48, 96, 192, 384] | Multi-scale channel sizes |

## 41. GPU Configuration

| Setting | Value | Purpose |
|---|---|---|
| `cudnn.benchmark` | True | Auto-tune cuDNN kernels |
| `cuda.matmul.allow_tf32` | True | Allow TF32 in matmul |
| `cudnn.allow_tf32` | True | Allow TF32 in cuDNN |
| `float32_matmul_precision` | "high" | Prefer faster imprecise matmul |
| OpenGL profile | 3.3 Core | Shader model 3.30 |
| VSync | Disabled | `swap_interval(0)` — uncapped FPS |

## 42. Dependencies

From `requirements.txt` (key packages):

| Package | Version | Role |
|---|---|---|
| `torch` | 2.13.0+cu130 | AI inference framework |
| `torchvision` | 0.28.0+cu130 | Vision transforms support |
| `opencv-python` | 5.0.0.93 | Camera capture, image ops |
| `PyOpenGL` | 3.1.10 | OpenGL Python bindings |
| `PyOpenGL-accelerate` | 3.1.10 | Optimised OpenGL bindings |
| `glfw` | 2.10.2 | Window/input management |
| `numpy` | 2.4.6 | Array operations |
| `Pillow` | 12.3.0 | HUD text/shape rendering |
| `timm` | 1.0.28 | ViT backbone support |
| `einops` | 0.8.2 | Tensor reshaping utilities |

## 43. Model Checkpoint Requirements

- **File:** `checkpoints/depth_anything_v2_vits.pth` (~94.6 MB)
- **Not distributed** — `.pth` files are git-ignored
- Application raises `FileNotFoundError` if missing (explicit check in `DepthEstimator.__init__()`)
- Must be downloaded separately from the Depth Anything V2 repository

## 44. Git/GitHub Files

- `.git/` — active Git repository
- `.gitignore` — configured to exclude:
  - Python: `__pycache__/`, `*.py[cod]`, virtual environments
  - Models: `*.pth`, `*.pt`, `*.onnx`, `*.engine`, `*.safetensors`, `*.bin`
  - Media: `*.mp4`, `*.avi`, `*.mov`, `*.mkv`
  - Generated: `outputs/`, `logs/`, `results/`, `benchmarks/`
  - IDE: `.vscode/`, `.idea/`
  - OS: `.DS_Store`, `Thumbs.db`

## 45. .gitignore Behaviour

The `.gitignore` correctly prevents large binary files (model weights, videos) and generated outputs from being committed. `.vscode/` is listed and the directory exists locally with `settings.json` — this means the VS Code config is **not tracked in Git** (correct for personal settings).

## 46. Assets

- `assets/images/DepthFX_CoverImage.png` (1.47 MB) — high-resolution project cover banner
- `assets/images/depth_test.jpeg` (1.18 MB) — real-time triple-view live pipeline screenshot used in README documentation
- `assets/images/depth_test_.png` (114 KB) — depth heatmap visualization example
- `assets/images/sample.png` (1.57 MB) — visual effects and depth estimation sample showcase
- `assets/videos/demo.gif` (1.57 MB) — animated live demonstration preview looping directly in README
- `assets/videos/demo.webp` (672 KB) — lightweight WebP animation preview
- `assets/videos/InShot_20260902_015848586.mp4` (31.2 MB) — live demonstration video showing real-time AI depth estimation and depth-driven effects

## 47. Outputs

- `outputs/` directory — contains runtime screenshots
- Two existing screenshots: `depthfx_20260825_163723.png` (412 KB) and `depthfx_20260825_164254.png` (416 KB)
- Git-ignored — not tracked in repository

## 48. Archive / Old Experiments

The `archive/old_experiments/` directory contains the project's evolutionary history:

- **`webcam.py`** — earliest experiment: minimal OpenCV webcam viewer with FPS overlay
- **`test_depth.py`** — single-frame depth inference test (no temporal smoothing, no effects)
- **`test_depth_utils.py`** — manual test of `normalize_depth()` with a 3x3 array
- **`effects.py`** — CPU-based depth fog and blur using NumPy/OpenCV (replaced by GLSL)
- **`test_effects.py`** — manual test of CPU-based fog effect
- **`gpu_test.py`** — minimal GLFW/OpenGL init test (clear screen only)
- **`gpu_renderer.py`** — earlier modular GPU renderer that loaded external shader files
- **`gpu_depth_blur.py`**, **`gpu_depth_camera.py`**, **`gpu_depth_fog.py`**, **`gpu_depth_lighting.py`**, **`gpu_depth_test.py`** — individual GPU effect experiments before consolidation
- **`performance.py`** — CPU-side `PerformanceProfiler` class (replaced by GPU timer queries)
- **`shaders/`** — 7 separate GLSL fragment shaders from the modular phase (now consolidated into inline code in `gpu_depth_fx.py`)

This archive shows clear iterative development: webcam test, depth test, CPU effects, individual GPU effects, consolidated GPU application.

## 49. Tests

- `tests/` directory is **empty** — no automated tests exist
- Old manual test scripts are in `archive/old_experiments/` (not pytest-based)
- **STATUS: NOT IMPLEMENTED**

## 50. Scripts

- `scripts/` directory is **empty**
- **STATUS: NOT IMPLEMENTED**

## 51. README / Documentation

The README.md is comprehensive and well-structured:
- Architecture diagram (Mermaid flowchart)
- Feature list
- "How It Works" section explaining the 7-step pipeline
- Display modes and effect levels tables
- Controls table
- Performance measurements (from actual hardware)
- Installation instructions
- Model setup instructions
- Running instructions
- Project structure diagram
- Tech stack table
- Limitations section
- Future work section

**Quality:** The README accurately reflects the actual codebase. No claims of features that don't exist in the code.

## 52. Current Strengths

1. **Clean single-file architecture** — the main application is self-contained and readable
2. **Proper separation of AI and rendering** — PyTorch inference and OpenGL rendering are cleanly separated
3. **Real-time performance** — temporal skipping + EWMA smoothing achieves interactive frame rates
4. **Professional HUD** — dark-theme overlay with PIL rendering, throttled updates, state-driven redraw
5. **Robust camera detection** — auto-fallback with frame validation
6. **GPU timing** — hardware-level shader profiling with `GL_TIME_ELAPSED`
7. **Clean resource cleanup** — `try/finally` block releases all GL resources, camera, and window
8. **Well-documented** — README accurately matches the implementation
9. **Iterative development visible** — archive shows engineering process
10. **Modern PyTorch best practices** — `inference_mode()`, autocast, cudnn.benchmark, TF32

## 53. Current Limitations

1. **Relative depth only** — normalized per-frame to [0,1], not metric distance
2. **NVIDIA-only** — no AMD/Intel GPU support; CPU fallback exists but is untested/unoptimised
3. **Depth-based background separation** — not semantic; depth ambiguity between foreground and background at similar distances
4. **Streamlit mode: no temporal smoothing** — the Streamlit pipeline runs every frame directly without EWMA blending, which was removed to improve responsiveness. Fast motion may show slight flicker.
5. **No automated tests** — `tests/` is empty
6. **No error recovery for camera loss** — if camera disconnects mid-run, the loop prints a warning but continues attempting reads
7. **Single-threaded** — AI inference blocks the render loop; no asynchronous inference
8. **No configuration file** — all settings are Python constants
9. **Windows-only** — DirectShow backend, Windows font paths in HUD (OpenGL mode); Streamlit mode works cross-platform
10. **Streamlit mode: no mouse-controlled lighting** — the interactive lighting effect is exclusive to the OpenGL/GLFW application

## 54. Potential Bugs or Weaknesses That Can Be Verified

1. **Double normalisation:** `DepthEstimator.estimate()` normalises to [0,1], then the main loop normalises again (lines 1429-1435). The second normalisation is redundant but not harmful — it will always map [0,1] to [0,1].

2. **`fullscreen.vert` unused:** The external shader file in `src/shaders/fullscreen.vert` flips the Y texcoord (`1.0 - a_texcoord.y`), but the main application uses inline vertex shaders that do NOT flip. The external file is never loaded by `gpu_depth_fx.py`.

3. **Lighting always enabled:** `lighting_enabled = True` is set at init but there is no keyboard toggle to disable it. The `key_callback` handles F (fog), B (blur), P (bg blur) — but not L (lighting). Lighting is always on.

4. **`u_time` unused in shader:** The uniform `u_time` is uploaded every frame (`float(time.perf_counter())`) but the fragment shader never reads it. The declaration exists but no shader code references it.

5. **Camera index hardcoded to 4:** If the auto-detect loop fails (no camera on 0-5), the application crashes. Index 4 is unusual — most systems use 0 or 1.

6. **Transform recreated per frame:** Inside `DepthAnythingV2.image2tensor()`, the `Compose([Resize, NormalizeImage, PrepareForNet])` transform pipeline is recreated on every call. This is a minor inefficiency (allocates new objects each frame).

7. **HUD font path is Windows-only:** The font candidates list (`C:/Windows/Fonts/...`) will fail silently on Linux/macOS, falling back to PIL default fonts.

## 55. What Is Complete

- End-to-end real-time pipeline (camera, AI, GPU effects, display)
- All 7 visual effects (depth blur, bg blur, fog, lighting, edge, grayscale, heatmap)
- Three display modes
- Three effect presets
- Mouse-controlled lighting (OpenGL mode)
- Screenshot / Snapshot capture
- GPU timer profiling (OpenGL mode)
- FPS monitoring
- HUD overlay with stats/controls (OpenGL mode)
- Temporal smoothing
- Depth update skipping
- Camera auto-detection
- Window resize handling (OpenGL mode)
- Clean resource cleanup
- Standalone depth benchmark
- README documentation
- Model weights loaded and working (verified by existing screenshots in outputs/)
- **[NEW]** Demo video recording (`assets/videos/InShot_20260902_015848586.mp4`) showing live depth estimation and visual effects
- **[NEW]** Streamlit web dashboard (`streamlit_app.py`) with full-width dark-theme UI
- **[NEW]** Auto-starting always-on pipeline (no start button required in Streamlit mode)
- **[NEW]** Integrated control toolbar (heatmap palette, fog/blur/bg-blur toggles, effect preset, camera index)
- **[NEW]** Triple-view live video panel (Normal + Effects, Depth Map, Depth Heatmap)
- **[NEW]** Real-time telemetry grid (Performance, System Status, Hardware, AI Model cards)
- **[NEW]** Browser-accessible Snapshot and Reset actions at bottom of page
- **[NEW]** Windows batch launcher (`run_app.bat`)
- **[NEW]** JPEG-encoded frame streaming for fast browser updates
- **[NEW]** Torch watcher patch (prevents Streamlit module watcher crash on torch.classes)

## 56. What Is Incomplete

- Automated tests (`tests/` is empty)
- Utility scripts (`scripts/` is empty)
- Lighting toggle key in OpenGL mode (always enabled)
- Configuration file (all constants are hardcoded)
- Cross-platform support (Windows-only)
- TensorRT export
- Asynchronous AI inference
- License file
- Phone camera specific support (works indirectly but no dedicated code)
- Streamlit mode: no mouse-controlled lighting (that is OpenGL-only)
- Streamlit mode: no direct GPU timer query display (telemetry is computed via time.perf_counter)

## 57. Recommended Future Development Stages (Ordered by Importance)

1. **Add automated unit tests** — test `normalize_depth()`, shader uniform handling, effect toggle state, screenshot naming logic. The `tests/` directory already exists.

2. **Add a lighting toggle key** — `L` to toggle lighting on/off, consistent with the existing F/B/P pattern.

3. **Remove redundant double normalisation** — eliminate the second min-max normalisation in the main loop since `estimate()` already returns [0,1].

4. **Externalise configuration** — move constants like `CAMERA_INDEX`, `AI_SIZE`, `DEPTH_UPDATE_INTERVAL`, `DEPTH_SMOOTHING` to a config file (YAML/JSON/TOML).

5. **Add asynchronous AI inference** — run depth estimation in a background thread to decouple AI latency from render frame rate.

6. **Add TensorRT / ONNX export** — for further inference speedup beyond FP16 autocast.

7. **Cache the transform pipeline** — move `Compose(...)` creation out of `image2tensor()` into `__init__()`.

8. **Record demo videos** — populate `assets/videos/` for portfolio presentation.

9. **Add an open-source license** — MIT or similar.

10. **Cross-platform support** — abstract Windows-specific code (DirectShow, font paths).

## 58. What Should NOT Be Changed

- **The GLSL fragment shader** — all effects are working, well-structured, and performant (~0.6-0.9 ms)
- **The `DepthEstimator` class** — clean interface, proper FP16 handling, good error handling
- **The temporal smoothing logic** — EWMA at 0.65 is a good balance for visual quality
- **The `GPUTimer` class** — robust error handling, rolling window, graceful degradation
- **The `HUDRenderer` class** — professional design, throttled updates, state-driven redraw
- **The `save_screenshot()` function** — handles edge cases, collision avoidance
- **The resource cleanup in `finally`** — thorough, handles exceptions per-resource
- **The effect presets dictionary** — well-tuned values
- **The camera auto-detection logic** — robust validation with frame content checking
- **The Depth Anything V2 model code** — upstream reference implementation, should not be modified

---

# PART 2 — WHY DID I BUILD IT THIS WAY?

---

## Depth Anything V2

**Q: Why did you choose Depth Anything V2 for depth estimation?**
It produces state-of-the-art monocular depth quality with a small (ViT-S) variant fast enough for real-time use. The DINOv2 backbone gives strong visual features from self-supervised pre-training, and the DPT head provides multi-scale feature fusion for smooth, edge-aware depth maps.

**Q: What would happen if you removed the AI depth estimation entirely?**
The entire project would lose its core functionality. Without depth, the fog, blur, lighting, and edge effects would have no distance information and would degrade to flat screen-space effects — essentially a basic camera filter app.

**Q: What alternatives could you have used?**
MiDaS (earlier generation), ZoeDepth (metric depth), Marigold (diffusion-based — too slow for real-time), stereo cameras (requires special hardware), or structured light depth sensors (LiDAR, Kinect). Depth Anything V2 was chosen for the best balance of quality, speed, and single-camera compatibility.

**Q: What trade-off does this decision make?**
We get real-time monocular depth without special hardware, but the depth is **relative** (not metric), normalised per-frame, and can be ambiguous in flat scenes with similar foreground/background textures.

---

## DINOv2 Backbone

**Q: Why does Depth Anything V2 use a DINOv2 backbone?**
DINOv2 is a self-supervised vision transformer pre-trained on a massive dataset without depth labels. It learns rich, general-purpose visual features (edges, textures, object boundaries, spatial structure) that transfer exceptionally well to dense prediction tasks like depth estimation.

**Q: What would happen if you used a CNN backbone instead?**
CNNs (like ResNet) are faster but produce less globally coherent features. ViTs with self-attention capture long-range spatial relationships that are critical for depth estimation — understanding that a distant building is farther than a nearby person requires global context.

---

## PyTorch

**Q: Why use PyTorch as the inference framework?**
PyTorch has the strongest ecosystem for research models. Depth Anything V2's reference implementation is in PyTorch. It provides `inference_mode()`, `autocast()`, and CUDA integration out of the box.

**Q: What alternative framework could you have used?**
ONNX Runtime or TensorRT for faster inference, but with more complex model export. TensorFlow/JAX would require model reimplementation.

---

## CUDA

**Q: Why is CUDA used?**
Depth Anything V2 is a transformer with ~25M parameters. Running all those matrix multiplications and attention computations on CPU would be 10-50x slower. CUDA enables GPU-parallel computation for real-time performance.

**Q: What would happen if CUDA is unavailable?**
The code falls back to CPU (`device = "cpu"`). The application would still run but inference would take ~200-500ms per frame instead of ~20ms, making real-time operation impossible.

---

## FP16

**Q: Why use FP16 inference?**
FP16 halves the memory bandwidth requirements and enables Tensor Core acceleration on NVIDIA GPUs. For depth estimation, the slight precision reduction (FP32 to FP16) has negligible impact on depth quality.

**Q: Why use autocast instead of .half()?**
`torch.autocast()` is safer — it automatically decides which operations run in FP16 vs FP32. Calling `.half()` directly on the model would require all inputs to be FP16 too, but `infer_image()` produces FP32 tensors in preprocessing, which would cause type mismatch errors.

**Q: What are the disadvantages?**
In rare cases, FP16 can cause numerical instability (underflow in very small gradients). For inference-only depth estimation, this is not a practical concern.

---

## 320x320 AI Inference

**Q: Why run inference at 320x320 instead of the full 640x480?**
The model's quality plateaus at 320px for this use case. Running at 640x480 would roughly quadruple inference time (~80ms vs ~20ms) with minimal depth quality improvement. The output is bilinear-interpolated back to camera resolution anyway.

**Q: Why not go even smaller, like 160x160?**
At very low resolutions, the model loses fine detail and edge accuracy. 320 is the sweet spot between speed and quality for ViT-S.

---

## Webcam Input

**Q: Why use a webcam instead of pre-recorded video?**
The project demonstrates **real-time** capabilities. A live webcam makes the interactive effects (mouse lighting, keyboard controls) meaningful. Pre-recorded video would make it a batch processor, not a real-time demo.

---

## OpenCV

**Q: Why use OpenCV for camera capture?**
OpenCV is the standard Python library for camera access. It provides cross-platform webcam capture, BGR to RGB conversion, image resizing, and file I/O. It also handles DirectShow on Windows.

**Q: What would happen if OpenCV were removed?**
Camera capture, colour conversion, image resizing, and screenshot saving would all need replacement. GLFW cannot capture cameras. You would need a platform-specific camera library.

---

## Depth Normalization

**Q: Why normalise depth to [0,1]?**
The raw model output has arbitrary scale (different per frame). Normalising to [0,1] makes the values directly usable as shader uniforms and texture values. The GLSL shader expects depth in [0,1] for `smoothstep()` and `mix()` operations.

**Q: What would happen if you skipped normalisation?**
The shader effects would produce inconsistent results. A fog start of 0.55 means nothing if the depth range changes from [0.2, 3.5] to [0.1, 7.8] between frames.

---

## Temporal Depth Updates (DEPTH_UPDATE_INTERVAL = 2)

**Q: Why update depth only every 2 frames?**
AI inference (~20ms) is the bottleneck. By skipping every other frame, the effective AI load is halved, leaving more time for rendering and reducing total frame time. The reused depth is still valid because scenes change slowly between consecutive frames.

**Q: What would happen if you updated every frame?**
FPS would drop because every frame would wait for ~20ms of AI inference. With a 2-frame interval, alternate frames render at near-zero AI cost.

**Q: What would happen if the interval were 10?**
The depth map would become visibly stale during motion. Objects would appear to "slide" relative to their depth-driven effects for ~10 frames before the depth catches up.

---

## DEPTH_SMOOTHING = 0.65

**Q: Why use EWMA smoothing at 0.65?**
A factor of 0.65 retains 65% of the previous depth and blends in 35% of the new prediction. This suppresses per-frame flicker and jitter from the neural network without introducing perceptible lag.

**Q: What happens if smoothing is set to 0.0?**
No temporal smoothing — each new depth map is used directly. This would show raw model output with visible flicker, especially in textureless regions where the model is uncertain.

**Q: What happens if smoothing is set to 0.95?**
Very heavy smoothing — only 5% of each new prediction is blended in. Depth would update very slowly, causing ghosting artifacts during fast motion (your hand would blur through its own depth shadow).

---

## OpenGL

**Q: Why use OpenGL for rendering instead of computing effects in Python?**
OpenGL fragment shaders run per-pixel in parallel on thousands of GPU cores. At 640x480 (307,200 pixels), the GPU completes all effects in ~0.6ms. The same operations on the CPU with NumPy would take 10-100ms because they are not parallelised at the pixel level. OpenGL also provides direct GPU texture storage — depth and colour data stay on the GPU for rendering without CPU round-trips.

**Q: Why OpenGL 3.3 specifically?**
3.3 Core Profile is the lowest version that supports all required features (VAOs, uniform blocks, sampler2D, layout qualifiers) while being universally supported on modern GPUs. It does not need Vulkan's complexity for a single fullscreen quad.

---

## GLSL

**Q: Why write effects in GLSL instead of using a game engine?**
GLSL gives direct control over every GPU operation. A game engine would add massive overhead, abstractions, and dependencies for what is fundamentally a single fullscreen fragment shader. GLSL also demonstrates the author's understanding of GPU programming at the hardware interface level.

---

## R32F Depth Texture

**Q: Why use R32F instead of R8 for the depth texture?**
R32F stores depth as a 32-bit float, preserving the full precision of the normalised [0,1] depth values. R8 (8-bit unsigned) would quantise depth into 256 levels, creating visible banding artifacts in smooth depth gradients — especially visible in fog ramps and blur transitions.

**Q: What is the VRAM cost?**
R32F costs 4x more than R8: 640x480x4 = 1.2MB vs 300KB. This is negligible on modern GPUs with gigabytes of VRAM.

---

## RGB Texture

**Q: Why use a separate RGB8 texture?**
The colour data needs to be on the GPU for the fragment shader to sample it. Uploading once per frame via `glTexSubImage2D` is efficient. RGB8 (3 bytes per texel) matches the OpenCV output format.

---

## Fullscreen Quad

**Q: Why render a fullscreen quad instead of individual shapes?**
The entire scene is the camera frame. A fullscreen quad ensures every pixel on screen is processed by the fragment shader exactly once. It is the standard approach for image-space post-processing effects in GPU rendering.

---

## GPU-Side Effects

**Q: Why perform all effects on the GPU?**
The CPU would process pixels sequentially. The GPU processes all 307,200 pixels simultaneously via thousands of shader cores. This is why GPU shader time is ~0.6ms while CPU-based effects (in the archived `effects.py`) would take 10-100ms.

---

## Separation Between AI Processing and Rendering

**Q: Why separate AI inference from GPU rendering?**
They are fundamentally different workloads using different GPU APIs (CUDA vs OpenGL). PyTorch owns the CUDA context for AI compute. OpenGL owns the rendering context for visual effects. Separating them allows each to be optimised independently and prevents resource contention.

---

## Mouse-Controlled Lighting

**Q: Where would this be useful in a real application?**
Interactive lighting demonstrates depth-aware 3D-like effects from a 2D camera. In AR applications, a virtual light source that responds to depth creates convincing relighting effects without explicit 3D reconstruction.

---

## Screenshot Capture

**Q: Why use glReadPixels instead of capturing from OpenCV?**
`glReadPixels` captures the **final rendered output** including all GPU effects and HUD overlay. OpenCV only has access to the raw camera frame before any processing.

---

## GPU Timer Queries

**Q: Why use GL_TIME_ELAPSED instead of Python's time.perf_counter()?**
`time.perf_counter()` measures CPU wall-clock time, but GPU operations are asynchronous. The GPU might still be executing shaders when the CPU moves on. `GL_TIME_ELAPSED` measures actual GPU execution time at the hardware level, providing accurate shader profiling.

---

## Effect Presets

**Q: Why have three presets instead of individual sliders?**
Presets provide a curated user experience — the six parameters are interdependent (fog start/end must be coordinated with fog strength, ambient must balance with light strength). Individual sliders would require the user to understand shader parameter interactions.

---

## Phone Camera Input

**Q: Why support phone cameras?**
Phone cameras often have higher quality sensors than laptop webcams. Supporting them (via virtual webcam software) gives better input quality for the depth model, improving the overall effect quality.

---

# PART 3 — PURPOSE AND REAL-WORLD APPLICATIONS

---

## Q: What is the actual purpose of DepthFX?
**A:** DepthFX is a **portfolio/demonstration project** that proves the author can build a system where real-time AI inference (monocular depth estimation) and GPU graphics rendering (OpenGL/GLSL effects) run concurrently in a single application. It demonstrates knowledge of computer vision, deep learning, GPU programming, real-time systems, and performance optimization.

## Q: Who could use a system like this?
**A:** Computer vision researchers, real-time graphics engineers, video conferencing product teams, AR/VR developers, creative tools developers, game engine developers working on depth-based effects, and mobile camera application teams.

---

### IMPLEMENTED NOW

- Real-time monocular depth estimation from a live webcam
- GPU-accelerated depth-driven fog, blur, background blur, lighting, edge enhancement
- Three display modes (normal, depth grayscale, depth heatmap)
- Interactive mouse-controlled virtual lighting
- Screenshot capture with GPU readback
- GPU shader profiling
- Professional HUD overlay
- Three effect intensity presets
- Phone camera support via virtual webcam software

### POSSIBLE FUTURE APPLICATIONS

| Application | How DepthFX Architecture Applies | What Would Need to Change |
|---|---|---|
| **Video conferencing background blur** | Background blur effect already demonstrates depth-based separation | Would need semantic person segmentation for robust separation, not just depth thresholding |
| **Virtual backgrounds** | Depth map enables foreground/background separation | Would need matting/alpha blending and a replacement background pipeline |
| **AR/VR** | Depth map could drive occlusion, object placement, relighting | Would need metric depth, 3D reconstruction, head tracking, and stereo rendering |
| **Camera applications** | Depth-driven bokeh (portrait mode) | Would need fine-grained alpha matting at hair/edge boundaries |
| **Real-time graphics / games** | Depth-aware post-processing effects | The shader architecture is directly applicable; would need integration with a scene graph |
| **Interactive installations** | Mouse lighting demonstrates interactive depth-aware effects | Would need multi-touch, gesture recognition, and projection mapping |
| **Computer vision research** | Provides a real-time depth estimation testbed | Would need metric depth, evaluation metrics, and dataset comparison |
| **Production application** | Core architecture is sound | Would need async inference, error recovery, configuration system, deployment packaging, testing, and security |

---

### Which parts are demonstration/portfolio quality?
- The end-to-end pipeline concept
- The shader effects
- The HUD design
- The depth estimation integration
- The performance profiling

### Which parts would need improvement before production use?
- Asynchronous AI inference (currently blocks render loop)
- Error recovery and resilience (camera disconnect, GPU errors)
- Configuration system (currently hardcoded constants)
- Automated test suite
- Cross-platform support
- Packaging and deployment
- Semantic segmentation for robust background separation
- Metric depth for AR/VR applications

---

# PART 4 — TECHNICAL INTERVIEW QUESTIONS

---

## Beginner

**Q: What is DepthFX?**
A real-time computer vision application that estimates depth from a live webcam feed using AI and applies GPU-accelerated visual effects driven by that depth map.

**Q: What does the project do?**
It captures live video, runs a neural network to estimate how far away each pixel is, then uses that distance information to add effects like fog, blur, and lighting — all in real time.

**Q: What is monocular depth estimation?**
Estimating the distance of every pixel in an image from a single camera (one lens), using only visual cues like perspective, occlusion, and object size — without stereo cameras or depth sensors.

**Q: What is a depth map?**
A 2D image where each pixel's brightness represents its distance from the camera. In DepthFX, bright pixels are near and dark pixels are far.

**Q: What does relative depth mean?**
The depth values show which pixels are closer or farther relative to each other, but not the actual metric distance (e.g., "2.5 meters"). Values are normalised to [0,1] per frame.

**Q: What is CUDA?**
NVIDIA's parallel computing platform that allows GPU cores to run general-purpose computations. DepthFX uses CUDA for AI inference — running the neural network on the GPU.

**Q: What is OpenGL?**
A graphics API for rendering 2D and 3D content on the GPU. DepthFX uses OpenGL 3.3 to render the camera feed with depth-aware effects.

**Q: What is GLSL?**
OpenGL Shading Language — the programming language for writing GPU shaders. DepthFX's fog, blur, lighting, and heatmap effects are all written in GLSL.

**Q: What is OpenCV?**
An open-source computer vision library. DepthFX uses it for webcam capture, colour conversion (BGR and RGB), image resizing, and screenshot saving.

**Q: What is FP16?**
16-bit floating-point (half precision). Using FP16 instead of FP32 halves memory usage and enables faster computation on GPU Tensor Cores, with negligible quality loss for inference.

**Q: What is a GPU shader?**
A small program that runs on the GPU for each vertex or pixel. DepthFX's fragment shader processes every pixel of the camera frame, reading depth and colour to compute effects in parallel.

---

## Intermediate

**Q: Why is depth estimated from a single camera?**
Because most webcams and phones have a single lens. Stereo depth requires two cameras with known baseline. Monocular depth estimation uses a neural network to infer depth from visual cues that humans also use (perspective, occlusion, relative size).

**Q: Why use Depth Anything V2 specifically?**
It provides the best balance of quality and speed for real-time monocular depth. The ViT-S variant runs at ~20ms per frame on an RTX 4070 while producing smooth, edge-aware depth maps. The DINOv2 backbone gives strong pre-trained features.

**Q: Why run inference at 320x320?**
Quality plateaus at 320px for this use case. Running at 640x480 would quadruple compute time with minimal quality improvement. The output is bilinear-interpolated back to camera resolution.

**Q: Why use temporal smoothing?**
Neural network depth predictions have per-frame jitter — small variations even in static scenes. EWMA smoothing (65% old + 35% new) reduces flicker while maintaining responsiveness.

**Q: Why update depth every two frames?**
AI inference (~20ms) is the bottleneck. Skipping alternate frames halves AI compute cost. The reused depth is still valid because scenes change slowly between consecutive frames.

**Q: Why use R32F?**
32-bit float preserves full depth precision. 8-bit (R8) would quantise depth into 256 levels, creating visible banding in fog/blur transitions.

**Q: Why upload depth to an OpenGL texture?**
So the GLSL fragment shader can sample it per-pixel in parallel. Without GPU texture storage, depth data would need to be uploaded to the shader per-draw-call, which is much slower.

**Q: Why perform effects in GLSL?**
GPU fragment shaders process all 307,200 pixels simultaneously in ~0.6ms. The same operations on the CPU would take 10-100ms because they are sequential.

**Q: Why separate AI inference from rendering?**
They use different GPU APIs (CUDA vs OpenGL) and different GPU hardware units. Separating them allows independent optimisation and prevents resource contention.

**Q: Why use a fullscreen quad?**
The entire scene is the camera frame. A fullscreen quad ensures every screen pixel is processed by the fragment shader exactly once — the standard approach for image-space post-processing.

**Q: How does depth-aware blur work?**
The GLSL shader calculates a blur factor from depth using `smoothstep()`. Farther pixels get more blur. The kernel samples 16 neighbouring texels (4 iterations x 4 directions) with distance-decaying weights, then blends with the original based on the blur factor.

**Q: How does background blur work?**
Similar to depth-aware blur but with a wider depth transition band (`smoothstep(0.25, 0.75, ...)`), specifically targeting the "background" depth range. It simulates video-conferencing-style background blur.

**Q: How does the heatmap work?**
The GLSL function `depthHeatmap()` maps depth [0,1] to a blue to green to red gradient. Depth < 0.5 blends blue to green; depth >= 0.5 blends green to red. Far objects appear blue, near objects appear red.

**Q: How does mouse lighting use depth?**
The shader computes distance from each pixel's UV to the mouse position. Light intensity falls off with distance (`smoothstep`). A depth factor `mix(0.45, 1.0, depth)` modulates brightness — near objects receive more light, simulating 3D-like illumination.

**Q: How is screenshot capture implemented?**
`glReadPixels()` reads the final framebuffer (RGB, unsigned byte). The result is vertically flipped (`np.flipud`), converted RGB to BGR, and saved as PNG via `cv2.imwrite()` to `outputs/`.

**Q: How is GPU shader time measured?**
OpenGL `GL_TIME_ELAPSED` timer queries wrap the fullscreen quad draw call. `glBeginQuery` before draw, `glEndQuery` after. The result (in nanoseconds) is read next frame after checking `GL_QUERY_RESULT_AVAILABLE`.

---

## Advanced

**Q: Explain the complete GPU/CPU pipeline.**
**CPU:** OpenCV reads BGR frame, converts to RGB, flips vertically, uploads to GPU as RGB8 texture. Every 2nd frame: BGR frame goes to DepthEstimator.estimate(), PyTorch/CUDA forward pass at 320px with FP16 autocast, raw depth, normalise to [0,1], EWMA blend with previous depth, flips vertically, uploads to GPU as R32F texture.
**GPU (CUDA):** Runs the ViT-S transformer (patch embed, 12 attention blocks, DPT head, bilinear interpolation) in FP16 mixed precision.
**GPU (OpenGL):** Fragment shader samples both textures per pixel, applies blur, bg blur, lighting, edge, fog, output. HUD renderer uploads PIL-generated RGBA textures and blits them with alpha blending.

**Q: Where are the major performance bottlenecks?**
1. AI inference (~20ms) dominates total frame time. 2. CPU to GPU texture upload for depth (307K float32 values). 3. The fullscreen quad fragment shader is well under 1ms and is not a bottleneck.

**Q: What happens between OpenCV and the neural network?**
The BGR frame from OpenCV goes to `DepthAnythingV2.infer_image()` which: converts BGR to RGB, normalises to [0,1], resizes to 320px (lower bound, ensure_multiple_of=14), applies ImageNet mean/std normalisation, transposes HWC to CHW, converts to PyTorch tensor, moves to CUDA device.

**Q: What happens between the depth model and OpenGL?**
The model outputs a CUDA tensor, `.cpu().numpy()` moves it to CPU, `normalize_depth()` maps to [0,1], `np.nan_to_num()` handles edge cases, `cv2.resize()` to camera resolution, EWMA blend, `np.flipud()`, `glTexSubImage2D()` uploads to R32F texture.

**Q: Why is R32F appropriate for the depth texture?**
Depth values are continuous floats in [0,1]. R32F stores the exact value. R8 quantises to 256 levels (0.004 per step), creating visible banding in `smoothstep()` fog ramps. R16F (half float) would also work but R32F matches the source data type.

**Q: What happens if depth inference runs every frame?**
FPS drops because every frame waits ~20ms for inference. At 60 FPS target (16.7ms per frame), AI alone exceeds the budget. With interval=2, alternate frames have near-zero AI cost.

**Q: What happens if the smoothing factor is changed?**
Lower values (e.g., 0.2): more responsive but more flicker. Higher values (e.g., 0.9): smoother but more ghosting during motion. 0.65 is a balance.

**Q: What are the trade-offs of FP16?**
Advantages: ~2x faster on Tensor Cores, halved memory bandwidth. Disadvantages: reduced dynamic range (max representable: 65504), potential precision loss in accumulation. For inference (no gradients), these disadvantages are negligible.

**Q: Why not run the entire pipeline in PyTorch?**
PyTorch is not designed for real-time rendering. It would require rendering to a PyTorch tensor, then displaying with matplotlib or similar — adding latency and losing access to OpenGL's efficient display pipeline, vsync, and input handling.

**Q: Why not perform the visual effects on the CPU?**
The archived `effects.py` shows CPU-based fog/blur using NumPy/OpenCV. It was replaced because CPU processing of 307K pixels per effect is 10-100x slower than GPU parallel execution. The GPU shader does all effects in ~0.6ms.

**Q: How would you reduce latency?**
1. Async AI inference in a background thread. 2. TensorRT or ONNX Runtime for faster inference. 3. Reduce AI input size (256px). 4. Use PBO (Pixel Buffer Object) for async texture upload.

**Q: How would you increase FPS?**
1. Increase `DEPTH_UPDATE_INTERVAL` (e.g., 4). 2. Async inference. 3. Lower render resolution. 4. Simplify shader (fewer blur taps). 5. TensorRT.

**Q: How would you handle higher-resolution cameras?**
The texture and shader scale naturally. AI inference stays at 320px regardless. The bottleneck shifts to texture upload bandwidth and shader fill rate. At 1080p, the R32F texture is ~8MB per upload.

**Q: How would you support multiple cameras?**
Multiple `VideoCapture` instances, multiple RGB textures, either separate depth estimators or round-robin inference. Shader would need a multi-view mode or split-screen rendering.

**Q: How would you support CPU fallback?**
The code already sets `device = "cpu"` when CUDA is unavailable. But inference would take 200-500ms per frame. Reducing input size, increasing update interval (e.g., 10), and simplifying temporal processing would be needed.

**Q: How would you support another GPU vendor (AMD/Intel)?**
Replace CUDA with ROCm (AMD) or OpenVINO (Intel) for AI inference. OpenGL rendering works on any vendor. Alternatively, export the model to ONNX and use ONNX Runtime with DirectML or OpenVINO backends.

**Q: How would you improve depth quality?**
Use ViT-B or ViT-L model (larger, more accurate, but slower). Increase inference resolution to 518px. Use metric depth models (ZoeDepth) for absolute distance. Apply bilateral filtering to the depth map for edge preservation.

**Q: How would you reduce temporal artifacts?**
Optical-flow-guided depth warping between inference frames. Bilateral temporal filtering. Kalman filtering per-pixel. Higher inference frequency (smaller, faster model).

**Q: How would you improve background separation?**
Add semantic person segmentation (e.g., MediaPipe, SAM) alongside depth. Use both signals: depth for approximate layering, segmentation for precise boundaries. This is how production video conferencing (Zoom, Teams) works.

**Q: How would you redesign this for production?**
Async inference pipeline (producer-consumer). Configuration system. Error recovery. Health monitoring. Modular effect system. Plugin architecture. Automated testing. Deployment packaging (Docker/Electron). Telemetry.

**Q: What parts of the system scale poorly?**
1. Single-threaded AI inference blocks rendering. 2. CPU to GPU texture upload is synchronous. 3. The fullscreen quad shader is O(pixels x taps) — higher resolution increases cost linearly.

**Q: What happens if the GPU becomes the bottleneck?**
If shader time exceeds 16ms (unlikely with current effects), reduce blur tap count, disable edge detection, or render at a lower resolution then upscale.

**Q: What happens if AI inference becomes the bottleneck?**
It already is (~20ms). Mitigate by: increasing update interval, async inference, smaller model, TensorRT, or lower input resolution.

**Q: How would you profile the complete pipeline?**
CPU: `time.perf_counter()` for each stage. GPU (CUDA): `torch.cuda.Event` for inference timing. GPU (OpenGL): `GL_TIME_ELAPSED` queries. End-to-end: frame time = 1/FPS. Tools: NVIDIA Nsight Systems, Nsight Graphics, py-spy.

**Q: What would you change if the target was 60 FPS?**
60 FPS = 16.7ms per frame. AI inference alone is ~20ms. Solutions: async inference thread (AI runs at its own pace, rendering uses latest available depth), TensorRT (cut inference to ~5ms), increase update interval to 4+ frames.

---

# PART 5 — "WHAT IF?" QUESTIONS

---

**Q: What if the webcam resolution becomes 1920x1080?**
The texture upload (`glTexSubImage2D`) would transfer more data (~6MB RGB, ~8MB R32F). AI inference stays at 320px. The shader processes more pixels but stays well under 1ms. `cv2.resize` output to 1080p is fast. Overall impact: moderate — mainly texture upload bandwidth.

**Q: What if AI inference takes 50ms?**
At interval=2, the effective frame time on inference frames would be ~51ms (50ms AI + ~1ms render). FPS would drop to ~20. Solution: increase update interval to 4+ or use async inference.

**Q: What if the GPU shader takes 10ms?**
This would indicate an extremely complex shader or very high resolution. At 60 FPS target, 10ms shader + 10ms AI (async) = 20ms total, limiting to 50 FPS. Solution: reduce blur taps, disable edge detection, render at lower resolution.

**Q: What if the camera drops frames?**
The `camera.read()` returns `success=False`. The current code prints a warning and `continue`s to the next loop iteration. The depth map and display are not updated. If drops are frequent, FPS drops and the HUD shows stale data. Solution: add a frame timeout and error counter.

**Q: What if depth prediction flickers?**
This is why temporal smoothing exists. At alpha=0.65, 65% of the previous depth is retained. If flicker is severe, increase alpha to 0.8-0.9. If it is still bad, the scene may be adversarial for the model (flat textures, reflections, transparent surfaces).

**Q: What if the depth map contains noisy edges?**
The depth edge enhancement (`depthEdge()`) would amplify the noise as visible edge artifacts. Solution: apply a bilateral filter to the depth map before upload, or reduce the edge enhancement coefficient (currently `edge * 0.035`).

**Q: What if the background is very similar to the foreground?**
The depth model relies on visual cues. If foreground and background have similar textures, colours, and scales, depth estimation becomes ambiguous. Background blur would be inaccurate. Solution: supplement with semantic segmentation.

**Q: What if lighting changes suddenly?**
The depth model may produce temporarily inconsistent depth maps. Temporal smoothing helps absorb the transition. If the change is extreme (e.g., lights turning off), the model may produce poor depth for several frames until it adapts.

**Q: What if the phone camera disconnects?**
`camera.read()` returns `False`. The loop continues attempting reads. There is no automatic reconnection logic. The application would eventually need to be restarted. Solution: add a reconnection mechanism with exponential backoff.

**Q: What if CUDA is unavailable?**
`DepthEstimator.__init__()` sets `device = "cpu"`. FP16 autocast is disabled. Inference runs on CPU at 10-50x slower (~200-500ms per frame). The application would be essentially non-interactive.

**Q: What if the model checkpoint is missing?**
`DepthEstimator.__init__()` raises `FileNotFoundError` with a clear message showing the expected path. The application exits immediately.

**Q: What if OpenGL initialization fails?**
`glfw.init()` returns False, raising `RuntimeError("GLFW initialization failed")`. If the window cannot be created, the application terminates. GLFW requires a display — headless environments fail.

**Q: What if the shader compilation fails?**
`shaders.compileShader()` raises an exception with the GLSL compiler error message. This would indicate a driver bug or unsupported GLSL feature. The application crashes at startup.

**Q: What if the application needs to run at 60 FPS?**
Current architecture cannot achieve 60 FPS with AI inference on every frame (~20ms alone). Solutions: async inference thread, TensorRT (~5ms), increase depth update interval, or accept that AI runs at 30 FPS while rendering runs at 60 FPS with interpolated depth.

**Q: What if we want 4K input?**
Texture upload bandwidth becomes significant (~32MB R32F per frame). AI still runs at 320px. Shader fill rate increases 9x vs 720p. Solution: render effects at a lower resolution and upscale, or use FBO-based multi-resolution rendering.

**Q: What if we want multiple people in the scene?**
The depth model handles this naturally — it estimates depth for the entire scene, including multiple people at different distances. The effects work per-pixel regardless of scene content. Background blur would still separate by depth, not by person identity.

**Q: What if we want semantic person segmentation instead of depth-based separation?**
This would require adding a segmentation model (MediaPipe, SAM, or a custom model). The depth map alone cannot distinguish "person" from "object at the same depth". The shader would need an additional mask texture and alpha-based compositing.

---

# PART 6 — CODE-LEVEL QUESTIONS

---

## DepthEstimator class (depth_estimator.py)

**What does it do?** Wraps the Depth Anything V2 model into a reusable inference interface.

**`__init__(self, input_size=320)`**
- Detects CUDA availability, loads checkpoint, moves model to device, enables optimisations (cudnn.benchmark, TF32)
- Sets `model.eval()` for inference mode
- Does NOT call `.half()` — uses autocast instead
- **Performance impact:** One-time cost at startup (~2-5 seconds)

**`estimate(self, frame)`**
- Input: BGR OpenCV image (np.ndarray)
- Output: float32 depth map normalised to [0,1]
- Uses `torch.inference_mode()` + `torch.autocast(cuda, fp16)`
- Calls `model.infer_image(frame, input_size=self.input_size)`
- Post-processes: `normalize_depth()`, `nan_to_num()`, `clip(0,1)`
- **Performance impact:** ~20ms per call on RTX 4070

**`warmup(self, frame, iterations=10)`**
- Runs 10 dummy inferences to warm up GPU and cuDNN autotune
- **Purpose:** First inference is slower due to kernel compilation; warmup amortises this

**`benchmark(self, frame, iterations=100)`**
- Runs warmup, then 100 timed inferences with `cuda.synchronize()`
- Reports average/fastest/slowest latency, estimated FPS, VRAM usage
- **Purpose:** Standalone performance measurement tool

---

## normalize_depth() (depth_utils.py)

**What does it do?** Min-max normalises a depth array to [0,1].
**Input:** np.ndarray (any shape, any float range)
**Output:** np.ndarray float32, clipped to [0,1]
**Edge case:** If range < 1e-6 (flat depth), returns zeros
**Dependencies:** Used by `DepthEstimator.estimate()`
**If removed:** Depth values would be in arbitrary ranges, breaking all shader effects

---

## GPUTimer class (gpu_depth_fx.py lines 884-984)

**What does it do?** Wraps OpenGL timer queries for GPU shader profiling.
**`begin()`** — starts GL_TIME_ELAPSED query
**`end()`** — ends the query
**`read()`** — checks availability, reads nanosecond result, converts to ms, maintains 300-sample rolling window
**`cleanup()`** — deletes the GL query object
**Important:** Gracefully disables on error (reports once, then silently degrades)
**If removed:** No GPU shader time measurement. FPS still works, but you cannot separate AI time from shader time.

---

## HUDRenderer class (gpu_depth_fx.py lines 332-881)

**What does it do?** Renders a professional dark-theme overlay HUD using PIL to OpenGL textures.
**Three panels:** Left sidebar (controls/status), top-right stats, bottom legend (heatmap mode only)
**`update(state)`** — regenerates panel images as needed: stats at ~4 Hz, controls on state change, legend on mode change
**`render(state)`** — blits panel textures with alpha blending
**`resize(w, h)`** — repositions panels and recreates geometry for window resize
**Important parameters:** C_ACCENT = teal/green (0x00D2AA), all font sizes hardcoded
**If removed:** Application still runs but has no visual feedback for settings, performance, or controls

---

## save_screenshot() (gpu_depth_fx.py lines 987-1017)

**What does it do?** Captures the current OpenGL framebuffer to a PNG file.
**Input:** GLFW window handle
**Process:** `glReadPixels`, `np.flipud`, `cv2.cvtColor(RGB to BGR)`, `cv2.imwrite`
**Output:** `outputs/depthfx_YYYYMMDD_HHMMSS.png`
**If removed:** No screenshot functionality. The `S` key would do nothing.

---

## create_fullscreen_quad() (gpu_depth_fx.py lines 1029-1062)

**What does it do?** Creates a VAO/VBO with 6 vertices (2 triangles) covering the entire viewport.
**Vertex format:** 4 floats per vertex: (x, y, u, v) — position and texcoord interleaved
**NDC coordinates:** [-1,-1] to [1,1] — full viewport coverage
**UV coordinates:** [0,0] to [1,1] — full texture sampling
**If removed:** Nothing renders on screen — no geometry to draw.

---

## create_rgb_texture() / create_depth_texture() (gpu_depth_fx.py lines 1065-1114)

**What do they do?** Allocate GPU textures for camera colour (RGB8) and depth (R32F).
**Parameters:** Linear filtering, clamp-to-edge wrapping, sized at camera resolution
**If removed:** No GPU storage for input data — shader has nothing to sample.

---

## key_callback() (gpu_depth_fx.py lines 1278-1315)

**What does it do?** GLFW keyboard handler that toggles effects, cycles modes, and triggers actions.
**Uses `nonlocal`** to modify closure variables (`fog_enabled`, `blur_enabled`, `effect_level`, `display_mode`, `bg_blur_enabled`, `take_screenshot`)
**Only fires on `glfw.PRESS`** — ignores key repeat and release events.

---

## cursor_callback() (gpu_depth_fx.py lines 1317-1328)

**What does it do?** Tracks mouse position and normalises to [0,1] UV space.
**Y inversion:** `mouse_y = 1.0 - ypos/height` (OpenGL origin is bottom-left)
**Clipping:** `np.clip(0.0, 1.0)` prevents out-of-bounds values
**If removed:** Light stays at initial position (0.5, 0.5) — no mouse interaction.

---

## main() render loop (gpu_depth_fx.py lines 1117-1663)

**What does it do?** The core application — initialisation, render loop, and cleanup.
**Loop body (~200 lines):**
1. `glfw.poll_events()` — process input
2. `camera.read()` — capture frame
3. Conditional depth estimation (every 2nd frame)
4. Temporal smoothing (EWMA blend)
5. Texture upload (RGB + depth)
6. Set uniforms
7. GPU timer, draw fullscreen quad, GPU timer
8. HUD update/render
9. Screenshot if requested
10. Swap buffers, update FPS/title

**Cleanup (`finally` block):** Releases camera, HUD, timer, textures, VBO, VAO, shader program, GLFW window. Reports total frames, depth updates, and GPU timing statistics.

---

## GLSL depthAwareBlur() (gpu_depth_fx.py lines 141-182)

**What does it do?** Multi-tap weighted blur with depth-dependent radius.
**Key parameters:**
- `u_depth_threshold` (0.50) — depth below which no blur is applied
- `u_blur_strength` — maximum blur intensity
- `u_texel_size` — pixel size for offset calculation
**Early exit:** Returns original if `blurFactor <= 0.001`
**Performance impact:** 16 texture samples per pixel when active

---

## GLSL applyFog() (gpu_depth_fx.py lines 259-273)

**What does it do?** Blends scene colour toward fog colour based on depth.
**Key calculation:** `fog = smoothstep(u_fog_start, u_fog_end, 1.0 - depth) * u_fog_strength`
**Fog colour:** `vec3(0.72, 0.77, 0.84)` — light blue-grey
**Performance impact:** Minimal — single smoothstep + mix per pixel

---

## GLSL applyLighting() (gpu_depth_fx.py lines 239-257)

**What does it do?** Applies a virtual point light at the mouse position, modulated by depth.
**Key calculations:**
- Distance attenuation: `1.0 - smoothstep(0.0, 0.70, distanceToLight)`
- Depth modulation: `mix(0.45, 1.0, depth)` — near objects get more light
- Final: `color * (ambient + light * depthFactor * strength)`

---

## DepthAnythingV2.infer_image() (dpt.py lines 186-194)

**What does it do?** End-to-end inference: image to tensor to forward to resize to numpy.
**Decorated with `@torch.no_grad()`** — redundant with `inference_mode()` in the caller, but harmless.
**`image2tensor()`** called first: BGR to RGB, /255, resize+normalize+transpose, to CUDA tensor.
**Output:** bilinear-interpolated to original image resolution, moved to CPU as numpy array.

---

## Constants

| Constant | Value | Impact |
|---|---|---|
| `CAMERA_INDEX` | 4 | Which camera to try first |
| `CAMERA_WIDTH/HEIGHT` | 640/480 | Texture allocation size, override by actual camera |
| `WINDOW_WIDTH/HEIGHT` | 1280/720 | Initial window size |
| `AI_SIZE` | 320 | Neural network input resolution |
| `DEPTH_UPDATE_INTERVAL` | 2 | Frames between AI runs |
| `DEPTH_SMOOTHING` | 0.65 | EWMA blend factor |

---

# PART 7 — RECRUITER QUESTIONS

---

**Q: Why did you build this project?**
I wanted to demonstrate that I can build a system where two fundamentally different GPU workloads — AI inference and real-time rendering — run together in one application. Most projects use PyTorch OR OpenGL. DepthFX uses both simultaneously, showing I understand the full GPU compute stack.

**Q: What did you personally learn?**
How to bridge the gap between AI model output and real-time graphics. I learned that the depth model is the bottleneck, not the shader, and that temporal processing (EWMA smoothing, frame skipping) is essential for visual quality. I also learned how to profile GPU operations at the hardware level using OpenGL timer queries.

**Q: What was the hardest part?**
Getting the depth map to look stable in real time. The raw model output flickers between frames — small variations that are barely visible in still images become jarring in video. I had to implement EWMA temporal smoothing and depth update skipping to get visually smooth results without perceptible lag.

**Q: What was the biggest technical challenge?**
Coordinating CUDA (PyTorch) and OpenGL in the same process. Both use the GPU but through different drivers. Data transfer between them goes CPU to GPU: the depth map comes out of PyTorch as a NumPy array and must be uploaded to an OpenGL texture. Getting the coordinate systems right (OpenCV top-left origin vs OpenGL bottom-left origin) required careful vertical flipping.

**Q: What performance problem did you encounter?**
AI inference (~20ms per frame) was the bottleneck. Even with FP16, running depth estimation on every frame limited FPS. I solved this by updating depth only every 2nd frame and using EWMA smoothing to maintain visual quality between updates.

**Q: How did you optimize it?**
Four optimizations: (1) FP16 autocast for faster inference, (2) 320px input instead of 640x480, (3) depth update every 2 frames instead of every frame, (4) all visual effects in GLSL (~0.6ms) instead of CPU-based NumPy (~50ms+ in my earlier implementation).

**Q: Why is GPU acceleration important here?**
Two reasons: the neural network has 25M parameters that need thousands of parallel matrix operations per frame (CUDA), and the visual effects need to process 307,200 pixels per frame in under 1ms (OpenGL). Neither workload is feasible on the CPU in real time.

**Q: Why did you choose OpenGL?**
OpenGL 3.3 Core Profile is the most widely supported modern GPU API. It provides everything I need (programmable shaders, texture units, timer queries) without the complexity of Vulkan. Since the rendering is a single fullscreen quad, Vulkan's explicit resource management would add complexity without benefit.

**Q: Why did you use CUDA?**
The depth model is a PyTorch neural network. PyTorch's CUDA backend is the fastest and most mature inference path. CUDA also provides `synchronize()` for accurate timing, `autocast` for FP16, and `cudnn.benchmark` for kernel auto-tuning.

**Q: How did you verify performance?**
Three methods: (1) FPS counter updated every second, (2) `time.perf_counter()` with `cuda.synchronize()` for AI inference timing, (3) OpenGL `GL_TIME_ELAPSED` hardware timer queries for shader profiling. All three are displayed in the HUD and window title.

**Q: How did you debug rendering problems?**
Display modes. Pressing `D` switches to raw depth grayscale or heatmap, letting me see the AI output directly. If the effects look wrong, I can see whether the depth map is correct. I also saved screenshots for comparison and used the archived CPU-based effects to validate against.

**Q: What would you improve next?**
Async AI inference — running depth estimation in a background thread so the render loop is not blocked. This would decouple AI latency from FPS and potentially double the frame rate.

**Q: What would you do differently if you rebuilt it?**
I would start with async inference from day one, use a configuration file instead of hardcoded constants, write automated tests, and consider ONNX Runtime for cross-vendor GPU support.

**Q: Is this production-ready?**
No. It is a functional demonstration. Production would need async inference, error recovery, configuration management, automated testing, deployment packaging, and semantic segmentation for robust background separation.

**Q: What is the biggest limitation?**
Background blur is depth-based, not semantic. A book held at the same distance as your face will not be blurred. Production video conferencing uses person segmentation, not just depth, for reliable background separation.

**Q: How would you explain the project to a non-technical person?**
"I built a camera app that understands how far away things are. It uses AI to figure out what's close and what's far in the camera picture, then uses that information to add effects like fog in the distance or blur the background — similar to how your phone's portrait mode works, but running in real time on a computer."

**Q: How would you explain it to a senior graphics engineer?**
"It is a real-time pipeline pairing Depth Anything V2 (ViT-S, CUDA FP16 autocast, 320px inference, ~20ms) with an OpenGL 3.3 core fullscreen-quad fragment shader. The depth map is uploaded to an R32F texture and sampled per-fragment to drive smoothstep fog, multi-tap depth-aware blur, mouse-controlled point lighting, and depth-edge enhancement. Temporal stability uses EWMA blending at alpha=0.65 with a 2-frame inference interval. GPU shader time is measured with GL_TIME_ELAPSED timer queries."

**Q: What part demonstrates your strongest engineering skill?**
The performance optimization decisions: choosing where to trade quality for speed (320px inference, 2-frame interval, EWMA smoothing), and implementing all effects in GLSL to keep shader time under 1ms while AI inference takes 20ms.

**Q: What part demonstrates your AI/ML knowledge?**
Using Depth Anything V2 with the correct configuration, understanding DINOv2's role as a self-supervised backbone, implementing FP16 autocast properly (not naively calling .half()), and understanding why temporal smoothing is needed for real-time depth estimation.

**Q: What part demonstrates your GPU/graphics knowledge?**
The complete OpenGL pipeline: creating textures with appropriate formats (R32F vs RGB8), writing GLSL fragment shaders with multi-tap blur kernels, implementing hardware-level GPU profiling with timer queries, and understanding the vertical-flip coordinate difference between OpenCV and OpenGL.

---

# PART 8 — ANSWER QUALITY

---

For the most important questions, here are natural interview-ready answers with deeper explanations:

---

### "What is DepthFX?"

**Short answer:** It is a real-time application that estimates depth from a webcam using AI and applies GPU-accelerated visual effects like fog, blur, and lighting based on that depth.

**Deeper explanation:** DepthFX runs two GPU workloads simultaneously — PyTorch/CUDA for AI depth inference and OpenGL/GLSL for real-time rendering. The depth model (Depth Anything V2, ViT-S) runs at 320px with FP16 autocast on CUDA, producing per-pixel depth maps that are uploaded to an R32F OpenGL texture. A GLSL fragment shader then uses that depth to drive fog, blur, lighting, and edge effects at sub-millisecond speed.

**Why it matters:** It proves you can integrate AI inference into a graphics pipeline at interactive frame rates — a skill required in AR/VR, video conferencing, computational photography, and game engines.

**Relevant code:** `gpu_depth_fx.py` — the main application, `depth_estimator.py` — the AI inference wrapper.

---

### "Why temporal smoothing?"

**Short answer:** The depth model's predictions jitter slightly between frames. Smoothing with an EWMA (65% old + 35% new) reduces visible flicker without adding perceptible lag.

**Deeper explanation:** Neural networks are not deterministic frame-to-frame — small input variations (noise, exposure, compression) cause depth predictions to fluctuate. At 30+ FPS, even 1% depth variation per frame creates visible shimmering in effects like fog ramps and blur transitions. EWMA smoothing acts as a low-pass filter: `current = 0.65 * previous + 0.35 * new`. This suppresses high-frequency flicker while preserving the low-frequency depth signal. The trade-off is ghosting during fast motion — the depth takes ~3 frames to catch up to a suddenly moved object.

**Why it matters:** Any production system using per-frame AI predictions (depth, segmentation, pose) needs temporal consistency. This is a core challenge in real-time computer vision.

**Relevant code:** `gpu_depth_fx.py` lines 1449-1457

---

### "Why R32F?"

**Short answer:** 32-bit float preserves full depth precision. 8-bit would create visible banding in smooth effects like fog ramps.

**Deeper explanation:** Depth values are continuous floats in [0,1]. R32F stores the exact value with ~7 significant digits. R8 quantises to 256 levels (0.004 per step), creating staircase artifacts in `smoothstep()` operations — visible as bands in fog gradients and blur transitions. The VRAM cost is only 1.2MB at 640x480, negligible on modern GPUs. R16F (half float) would also work and save memory, but R32F matches the source data type and avoids any potential precision issues.

**Why it matters:** Texture format choice directly affects visual quality. Choosing the right format for the data type is a fundamental GPU programming skill.

**Relevant code:** `gpu_depth_fx.py` lines 1091-1114 — `create_depth_texture()`

---

### "How does the blur work?"

**Short answer:** A GLSL fragment shader samples 16 neighbouring pixels with depth-dependent radius. Farther pixels get more blur.

**Deeper explanation:** The shader computes a blur factor from depth: `smoothstep(threshold, 1.0, 1.0 - depth) * strength`. If this factor is near zero (close objects), the pixel is returned as-is (early exit optimisation). Otherwise, the kernel samples 4 iterations x 4 directions (horizontal and vertical, positive and negative), with weights decaying as `1.0 / (1.0 + i * 0.45)`. The radius scales from 1 to 5 texels based on blur factor. The weighted average is mixed with the original colour by the blur factor.

**Why it matters:** Multi-tap kernel blur in a fragment shader is a fundamental GPU technique used in depth of field, ambient occlusion, and bloom effects.

**Relevant code:** `gpu_depth_fx.py` lines 141-182 — `depthAwareBlur()`

---

### "How did you optimize performance?"

**Short answer:** FP16 inference, 320px input, every-other-frame depth, and GLSL effects instead of CPU processing.

**Deeper explanation:** The main bottleneck is AI inference (~20ms). I reduced this by: (1) FP16 autocast — halves memory bandwidth, enables Tensor Cores; (2) 320px input — 4x fewer pixels than 640px with minimal quality loss; (3) DEPTH_UPDATE_INTERVAL=2 — runs inference only every other frame, halving compute cost; (4) EWMA smoothing — maintains visual quality between updates. For rendering: all effects run in GLSL (~0.6ms) instead of CPU-based NumPy (the archived `effects.py` approach, which would take 50ms+). The fragment shader has early-exit optimisations for disabled effects and near-zero blur factors.

**Why it matters:** Performance optimization is the difference between a demo that runs at 5 FPS and one that runs at 30+ FPS. The ability to identify bottlenecks and apply targeted optimisations is a critical engineering skill.

**Relevant code:** `depth_estimator.py` lines 98-110 — FP16 inference, `gpu_depth_fx.py` lines 1412-1414 — interval check

---

# PART 9 — FINAL PROJECT SUMMARY

---

## 30-Second Explanation

"DepthFX is a real-time camera application I built that uses AI to understand depth from a single webcam. It figures out what is close and what is far in the camera picture, then uses the GPU to add effects like fog in the distance, blur the background, and create interactive lighting — all running in real time. It combines AI inference with GPU graphics rendering in a single application."

## 1-Minute Explanation

"DepthFX is a real-time pipeline that pairs AI depth estimation with GPU rendering. It captures live webcam video, runs Depth Anything V2 — a transformer-based depth estimation model — on CUDA with FP16 mixed precision at 320px input. The depth map is temporally smoothed using EWMA and uploaded to an OpenGL R32F texture. A GLSL fragment shader then uses that depth to drive visual effects: depth-aware fog, multi-tap depth-driven blur, background blur, mouse-controlled virtual lighting, and depth edge enhancement. The AI inference takes about 20ms per frame on an RTX 4070, while the shader takes under 1ms. I optimize throughput by running depth estimation every other frame and smoothing between updates. The application has three display modes, three effect presets, screenshot capture, and GPU hardware profiling."

## 2-Minute Technical Explanation

"DepthFX demonstrates concurrent GPU workloads in a real-time application. The pipeline starts with OpenCV capturing 640x480 BGR frames from a webcam using DirectShow on Windows. Every second frame, the image is forwarded to a Depth Anything V2 Small model — a ViT-S encoder with DINOv2 backbone and DPT decoder head. Inference runs at 320px inside PyTorch's `inference_mode()` with CUDA FP16 autocast, producing a raw depth map in about 20ms on an RTX 4070 Laptop GPU. The output is min-max normalised to [0,1] and temporally blended with the previous depth using EWMA at alpha=0.65 to suppress per-frame flicker.

Both the RGB frame and the float32 depth map are vertically flipped for OpenGL's bottom-left origin convention and uploaded via `glTexSubImage2D` — RGB to a GL_RGB8 texture on unit 0, depth to a GL_R32F texture on unit 1. A GLSL 3.30 fragment shader processes a fullscreen quad, sampling both textures per pixel to apply five effects in sequence: depth-aware blur (16-tap kernel with depth-dependent radius), background blur (wider depth band for background separation), mouse-controlled virtual point lighting with depth modulation, depth-edge enhancement from neighbour differences, and smoothstep fog. Three effect presets configure six shader uniforms. Three display modes show normal effects, grayscale depth, or a blue-green-red heatmap.

GPU shader execution time is measured with OpenGL `GL_TIME_ELAPSED` timer queries — about 0.6-0.9ms. A professional HUD overlay rendered via PIL to RGBA textures shows real-time performance metrics, effect states, and keyboard controls. Screenshots are captured via `glReadPixels` to timestamped PNGs. The architecture deliberately separates AI inference (PyTorch/CUDA) from rendering (OpenGL/GLSL), allowing each workload to be profiled and optimised independently."

---

## Strongest Project Points

1. **Concurrent GPU workloads** — CUDA AI inference + OpenGL rendering in one process
2. **Performance-conscious design** — FP16, 320px input, 2-frame interval, sub-1ms shader
3. **Complete end-to-end pipeline** — camera, AI, GPU effects, display
4. **Proper temporal processing** — EWMA smoothing solves real-time depth flicker
5. **Hardware-level profiling** — GL_TIME_ELAPSED timer queries, not just wall-clock timing
6. **Professional HUD** — dark-theme overlay with throttled PIL rendering
7. **Clean resource management** — thorough try/finally cleanup
8. **Visible engineering iteration** — archive shows CPU to GPU migration path
9. **Accurate documentation** — README matches the actual codebase

## Weakest Points

1. **No automated tests** — `tests/` is empty; all testing was manual
2. **Single-threaded AI** — inference blocks the render loop; no async pipeline
3. **Windows-only** — DirectShow backend, Windows font paths
4. **No configuration file** — all constants hardcoded in Python source
5. **Depth-based background separation** — not semantic; fails at same-depth ambiguity
6. **No error recovery** — camera disconnect, GPU errors cause degradation without recovery
7. **Lighting has no toggle key** — always enabled, inconsistent with other toggles

## Best Next Steps (Priority Order)

1. **Add automated tests** — `pytest` tests for `normalize_depth()`, shader uniforms, effect toggles, screenshot naming. The `tests/` directory already exists.
2. **Add async AI inference** — run `DepthEstimator.estimate()` in a background thread with a shared depth buffer. This would decouple AI latency from FPS.
3. **Add lighting toggle** — simple: add `L` key to `key_callback()` for consistency.
4. **Externalise configuration** — YAML/JSON config file for camera index, AI size, smoothing factor, etc.
5. **Record demo videos** — populate `assets/videos/` for portfolio presentation (GIF/MP4 showing effects in action).
6. **Remove double normalisation** — eliminate the redundant second min-max normalisation in the main loop.
7. **Add TensorRT export** — convert model to TensorRT for ~2-4x inference speedup.
8. **Add license** — MIT or Apache 2.0 for open-source sharing.
10. **Cache transform pipeline** — move `Compose(...)` from `image2tensor()` to `__init__()`.

---

# PART 10 — STREAMLIT DASHBOARD & REAL-TIME WEB ARCHITECTURE (NEW)

---

## 1. Overview of the Streamlit Web Application

In addition to the standalone desktop OpenGL/GLFW application (`src/gpu_depth_fx.py`), DepthFX includes a modern, high-performance browser-based dashboard implemented in **`streamlit_app.py`**.

The Streamlit interface enables instant web browser access, demonstration capability without requiring a native desktop window manager, and interactive real-time control over depth estimation and post-processing effects.

- **Primary Entry Point:** `.\run_app.bat` or `streamlit run streamlit_app.py`
- **Port:** Defaults to `http://localhost:8501`
- **Execution Model:** Always-on live camera stream with per-frame neural network inference and CPU post-processing effects
- **Rendering Model:** Browser-based responsive rendering with JPEG-compressed frame transfer and dynamic CSS DOM injection

---

## 2. Architectural Comparison: OpenGL/GLFW vs. Streamlit Web Dashboard

| Feature / Dimension | Desktop App (`gpu_depth_fx.py`) | Web Dashboard (`streamlit_app.py`) |
|---|---|---|
| **Interface / Window** | Native GLFW desktop window (1280×720) | Web browser (responsive full-width, zero margins) |
| **Rendering Backend** | OpenGL 3.3 Core Profile (GLSL Shaders) | Streamlit Image component + OpenCV / NumPy |
| **Effect Execution** | GPU Fragment Shader (GLSL) | Vectorized CPU (NumPy, OpenCV GaussianBlur, Blending) |
| **View Layout** | Single fullscreen quad (mode switch via `D` key) | Simultaneous triple-view side-by-side cards |
| **Interactive Lighting** | Real-time mouse-controlled point light | Not applicable in browser stream |
| **Telemetry & HUD** | In-engine PIL texture overlay (GL quads) | Responsive 4-card HTML/CSS telemetry grid |
| **Controls** | Keyboard hotkeys (`F`, `B`, `P`, `1`-`3`, `S`, `Q`) | Integrated interactive toolbar (Selectbox, Toggles, Radio) |
| **Startup Behavior** | Starts window immediately | Instant auto-stream on page load |
| **Snapshot Mechanism** | GPU readback (`glReadPixels`) | CPU array stack (`np.hstack`) saved to PNG |
| **Launcher** | `python src/gpu_depth_fx.py` | `.\run_app.bat` or `streamlit run streamlit_app.py` |

---

## 3. Key Components of `streamlit_app.py`

### 3.1 Torch Module Watcher Compatibility Fix
Streamlit’s internal module watcher scans loaded Python modules for changes. PyTorch C++ extensions export pseudo-modules (such as `torch.classes`) that lack standard `__path__` attributes or raise dynamic `AttributeError` exceptions when inspected by file watchers.
To ensure robust startup, `streamlit_app.py` executes a monkey-patch at startup:
```python
try:
    import torch
    if hasattr(torch, "_classes") and hasattr(torch._classes, "__getattr__"):
        try:
            torch.classes.__path__ = []
        except Exception:
            pass
except Exception:
    pass
```
This guarantees that the Streamlit development server does not crash during hot-reload or file inspection.

### 3.2 Hardware Discovery & Dynamic System Telemetry
The dashboard includes an `@st.cache_data` discovery routine (`get_system_info()`) that probes:
- PyTorch version (`torch.__version__`)
- CUDA runtime status (`torch.cuda.is_available()`)
- GPU device identifier (`torch.cuda.get_device_name(0)`)
- CUDA version (`torch.version.cuda`)
- Hardware Compute Capability (`torch.cuda.get_device_capability(0)`)

GPU name prefixes such as `"NVIDIA GeForce "` or `"AMD Radeon "` are cleaned for compact presentation in dashboard cards.

### 3.3 Model Caching & Lifecycle
The AI model is wrapped with `@st.cache_resource`:
```python
@st.cache_resource
def load_model(input_size=320):
    from depth_estimator import DepthEstimator
    return DepthEstimator(input_size=input_size)
```
This guarantees that `depth_anything_v2_vits.pth` is loaded into GPU VRAM exactly once upon application launch and preserved across user UI interactions and script reruns.

### 3.4 Integrated Control Toolbar (Header Layout)
Rather than cluttering the screen with a collapsible sidebar, all controls are positioned in a streamlined, horizontal toolbar directly below the title header:
1. **Heatmap Palette:** Selectbox containing `Inferno`, `Turbo`, `Jet`, `Magma`, and `Plasma` colormaps.
2. **Atmospheric Fog:** Toggle switch enabling depth-driven exponential fog.
3. **Depth Blur:** Toggle switch enabling depth-aware background softening.
4. **Background Blur:** Toggle switch for aggressive background isolation.
5. **Preset:** Radio selector (`LIGHT`, `MEDIUM`, `STRONG`) configuring blur sigma, fog ramp start/end, and density.
6. **Camera Device:** Selectbox supporting camera device indices `0` through `10`.

### 3.5 Triple-View Synchronization Layout
The video stream layout displays three live visual feeds simultaneously:
1. **NORMAL + EFFECTS (`640×480`):** Live camera feed combined with active visual effects (fog, depth blur, background blur).
2. **DEPTH MAP (`320×320`):** Normalized grayscale depth map (near pixels bright, distant pixels dark).
3. **DEPTH HEATMAP (Colormap name):** High-contrast false-color thermal depth visualization using OpenCV color lookup tables.

Each card header features:
- Left: Uppercase feed title (`NORMAL + EFFECTS`, `DEPTH MAP`, `DEPTH HEATMAP`)
- Right: Monospace technical tag (`640×480`, `320×320`, colormap identifier)

### 3.6 Responsive Video Scaling & 4:3 Aspect Ratio Locking
Standard Streamlit image components frequently introduce unwanted margins or distort 4:3 camera feeds when fitting into columns. The custom CSS in `streamlit_app.py` enforces a locked 4:3 container aspect ratio matching the camera's native dimensions:
```css
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
}
```
This guarantees:
- The video frame spans the exact width of the card header with zero letterbox padding.
- Height dynamically tracks `width × 0.75` across any browser window size.
- 4:3 camera geometry is preserved without stretching or distortion.

### 3.7 Fast Browser Streaming via JPEG Encoding
Rendering raw uncompressed RGB arrays through WebSockets incurs significant serialization latency. The helper:
```python
def render_frame_image(ph, img_rgb):
    ph.image(img_rgb, channels="RGB", output_format="JPEG", width="stretch")
```
encodes frames as lightweight JPEGs in memory before transmission, maximizing browser framerates.

### 3.8 Real-Time 4-Card Telemetry Grid
Positioned below the video cards is a four-card telemetry display updating at ~2 Hz:
1. **Performance:** Live FPS, AI Latency (ms), Inference Rate (1:1 per-frame), AI Resolution (320×320), Motion Delay.
2. **System Status:** Camera connection state, AI Model status, CUDA Engine, Pipeline status, Active Renderer.
3. **Hardware:** Active GPU name, CUDA runtime version, PyTorch version, GPU Compute Capability, Host OS.
4. **AI Model:** Model architecture (`Depth Anything V2`), Backbone (`DINOv2 ViT-S`), Decoder Head (`DPT`), Precision (`FP16 Autocast`), Checkpoint (`vits.pth`).

### 3.9 Bottom Action Bar
Centered at the bottom of the dashboard are two dedicated action buttons:
- **Snapshot (`action-col`):** Captures the current frame across all three feeds, constructs a horizontal composite (`np.hstack([output_normal, output_depth, output_heat])`), writes it to `outputs/depthfx_snapshot_YYYYMMDD_HHMMSS.png`, and displays a non-blocking toast notification.
- **Reset (`reset-col`):** Clears session state and triggers `st.rerun()`, restoring default parameter values.

### 3.10 Vectorized CPU Visual Effects & Mathematical Formulation
Because standard web browsers cannot directly bind to local desktop OpenGL 3.3 Core contexts, `streamlit_app.py` implements CPU-side counterparts for the visual effects using vectorized NumPy and OpenCV operations:

1. **Atmospheric Fog (`apply_fog`):**
   - Inverts normalized depth to obtain optical distance:
     $$\text{dist} = 1.0 - \text{depth}$$
   - Evaluates a linear distance ramp between configurable `start` and `end` thresholds, clamped to $[0.0, 1.0]$ and scaled by `strength`:
     $$t = \text{clip}\left(\frac{\text{dist} - \text{start}}{\max(\text{end} - \text{start}, 10^{-6})}, 0.0, 1.0\right) \times \text{strength}$$
   - Blends scene color with atmospheric fog color `[184, 196, 214]` (RGB, matching GLSL $\text{vec3}(0.72, 0.77, 0.84)$):
     $$\text{Output} = \text{frame} \times (1.0 - t) + \text{fog\_color} \times t$$

2. **Depth-Aware Blur (`apply_blur`):**
   - Applies an initial Gaussian blur across the entire frame with $\sigma_x = 10$:
     $$\text{blurred} = \text{GaussianBlur}(\text{frame}, \sigma=10)$$
   - Computes distance factor $\text{dist} = 1.0 - \text{depth}$ and evaluates an activation ramp starting at $\text{dist} = 0.50$:
     $$\text{mask} = \text{clip}\left(\frac{\text{dist} - 0.50}{0.50}, 0.0, 1.0\right) \times \text{strength}$$
   - **Feathered Mask Boundary Softening:** To prevent jagged silhouette artifacts around foreground boundaries, the mask itself is filtered with a secondary Gaussian blur ($\sigma_x = 3$):
     $$\text{mask}_{\text{soft}} = \text{GaussianBlur}(\text{mask}, \sigma=3)$$
   - Linearly interpolates between original and blurred pixels:
     $$\text{Output} = \text{frame} \times (1.0 - \text{mask}_{\text{soft}}) + \text{blurred} \times \text{mask}_{\text{soft}}$$

3. **Background Blur (`apply_bg_blur`):**
   - Uses an aggressive blur kernel ($\sigma_x = 14$) with an expanded transition band covering $\text{dist} \in [0.25, 0.75]$:
     $$\text{mask} = \text{clip}\left(\frac{\text{dist} - 0.25}{0.50}, 0.0, 1.0\right) \times \text{strength}$$
   - Blends original frame with blurred background for portrait-mode depth isolation.

4. **Preset Calibration Matrix:**
   | Preset | Fog Strength | Blur Strength | Fog Start (`fs`) | Fog End (`fe`) |
   |---|---|---|---|---|
   | `LIGHT` | 0.30 | 0.25 | 0.55 | 0.95 |
   | `MEDIUM` | 0.55 | 0.50 | 0.35 | 0.90 |
   | `STRONG` | 0.85 | 0.85 | 0.20 | 0.80 |

### 3.11 Heatmap Generation & Colormap Color Space Pipeline
Depth estimation produces continuous normalized float32 values in $[0.0, 1.0]$. To render these across diverse thermal visualization profiles:
- Float values are scaled and quantized to 8-bit unsigned integers:
  $$\text{depth}_{u8} = (\text{clip}(\text{depth}, 0.0, 1.0) \times 255.0).\text{astype}(\text{uint8})$$
- Color mapping is applied via OpenCV lookup tables: `cv2.applyColorMap(depth_u8, colormap_id)`.
- Available colormap profiles include:
  - `Inferno`: Deep violet $\to$ warm orange $\to$ bright yellow (standard high-contrast thermal)
  - `Turbo`: Perceptually uniform rainbow (smooth gradient, low color banding)
  - `Jet`: Classic blue $\to$ cyan $\to$ yellow $\to$ red
  - `Magma`: Dark purple $\to$ pink $\to$ white
  - `Plasma`: Deep purple $\to$ magenta $\to$ orange-yellow
- **Color Space Conversion:** Because OpenCV's `applyColorMap` outputs in BGR format, the result is immediately transformed to RGB via `cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)` to prevent blue/red inversion in web browsers.
- **Grayscale Normalization:** `depth_to_grayscale` converts the single-channel depth map to 3-channel RGB (`cv2.COLOR_GRAY2RGB`) to ensure uniform JPEG compression and display handling across all three stream columns.

### 3.12 Latency Management & Direct Hardware Synchronization
1. **Camera Buffer Clamping (`CAP_PROP_BUFFERSIZE = 1`):**
   Standard webcam drivers allocate multi-frame internal buffers. If AI inference and rendering take 25 ms while camera capture runs at 30 FPS (33.3 ms), buffered frames accumulate, causing progressive visual lag. Clamping buffer size to 1 forces OpenCV to immediately read the freshest incoming sensor frame.

2. **Accurate CUDA Timing via `torch.cuda.synchronize()`:**
   PyTorch executes CUDA operations asynchronously. If `time.perf_counter()` is called immediately around `model.estimate()`, it only measures CPU queue dispatch time (~0.1 ms). Calling `torch.cuda.synchronize()` forces the CPU to wait until the GPU finishes all tensor kernels, ensuring the telemetry displays genuine hardware inference latency (~10–15 ms on RTX 4070).

3. **Per-Pixel Dimension Alignment:**
   Depth Anything V2 produces depth predictions at $320\times 320$. To maintain 1:1 pixel alignment with the 640×480 camera frame:
   ```python
   if raw_depth.shape[:2] != frame.shape[:2]:
       raw_depth = cv2.resize(raw_depth, (frame.shape[1], frame.shape[0]))
   ```

### 3.13 Synchronized Triple-Stream Snapshot Generator
When the user clicks the "Snapshot" button:
1. `st.session_state["snapshot_pending"] = True` is flagged.
2. At the conclusion of the current frame iteration, the three synchronized feeds (`output_normal`, `output_depth`, and `output_heat`) are horizontally concatenated into a single panoramic array:
   ```python
   combined = np.hstack([output_normal, output_depth, output_heat])
   ```
3. The resulting $1920\times 480$ RGB image is converted to BGR and written to `outputs/depthfx_snapshot_YYYYMMDD_HHMMSS.png`.
4. A non-blocking toast alert (`st.toast`) confirms the saved file path without pausing or resetting the live video stream.

### 3.14 Dual Throttling & Cooperative Concurrency
- **Cooperative Thread Yielding (`time.sleep(0.004)`):**
  In an infinite `while True` loop, Python can monopolize the Global Interpreter Lock (GIL), starving Streamlit's Tornado/Uvicorn WebSocket communication loop. Inserting a 4 ms cooperative sleep allows the event loop to process user input (slider clicks, toggles, snapshot button) while maintaining high framerates.
- **Dual Throttling Architecture:**
  - **FPS Computation:** Calculated over an 0.8-second rolling window (`elapsed >= 0.8`), filtering out single-frame fluctuations.
  - **Telemetry Grid Updates:** Throttled to 2 Hz (`now - perf_time >= 0.5`), preventing unnecessary DOM reflows and browser re-renders while video feeds update continuously.

### 3.15 Session State & Reset Architecture
Clicking the "Reset" button executes:
```python
for k in list(st.session_state.keys()):
    if k not in ("snap_btn", "reset_btn"):
        del st.session_state[k]
st.rerun()
```
Preserving the widget button keys prevents Streamlit from throwing key-registration errors during rerun while resetting all internal states to clean defaults.

### 3.16 CSS Design System: Split Branding, Flush Alignment & Custom Buttons
The dashboard features an intentionally engineered dark theme inspired by technical developer consoles:
- **Title Branding:** The brand is rendered with split coloration: `"Depth"` in vivid terminal green (`#3fb950`), `"FX"` in pure white (`#f0f6fc`), and `"Dashboard"` centered on the next line in muted silver (`#c9d1d9`).
- **Header-to-Video Zero Margin:** The CSS rule:
  ```css
  [data-testid="stElementContainer"]:has([data-testid="stImage"]) {
      padding: 0 !important;
      margin-top: -6px !important;
  }
  ```
  completely closes the vertical gap between the `.view-header` bar and the video canvas, producing a single seamless visual card.
- **Custom Button Styling:**
  - `action-col button` (Snapshot): Ghost-button styling with 2px solid `#58a6ff` border, subtle ambient glow (`box-shadow: 0 2px 8px rgba(88,166,255,0.15)`), and inverted fill on hover with `#0d1117` text.
  - `reset-col button` (Reset): Industrial dark border (`#30363d`), muted text (`#8b949e`), and smooth contrast inversion on hover.

---

## 4. Technical Questions & Answers on the Streamlit Dashboard

**Q: Why was a Streamlit interface added when an OpenGL desktop app already existed?**
**A:** Streamlit makes the project instantly shareable, demonstrable over local networks or cloud deployments, and usable without needing an X11/Win32 desktop window system. It allows portfolio reviewers and recruiters to inspect live depth inference and interact with effects directly through a web browser.

**Q: Why does the Streamlit version use CPU OpenCV effects instead of OpenGL fragment shaders?**
**A:** Browsers cannot directly execute local desktop OpenGL 3.3 Core Profile contexts created by GLFW. To deliver live video through WebSockets into standard web browsers without WebGL client-side overhead, effects are processed on the host machine using vectorized OpenCV and NumPy before streaming.

**Q: Why was temporal smoothing (EWMA) disabled in the Streamlit loop?**
**A:** The Streamlit pipeline achieves real-time inference on modern GPUs (~10-15 ms per frame). Direct 1:1 inference produces zero motion ghosting and instantaneous response to fast hand gestures. Without EWMA lag, moving foreground objects have razor-sharp depth boundaries.

**Q: Why is `torch.cuda.synchronize()` mandatory for AI latency measurement?**
**A:** PyTorch launches CUDA operations asynchronously on GPU streams. Calling `time.perf_counter()` without `torch.cuda.synchronize()` measures only the time required for the CPU to queue the instructions (typically under 0.2 ms), not the actual execution time. Synchronization halts CPU execution until the GPU has completed the forward pass, providing accurate latency metrics.

**Q: Why did you choose JPEG output format over PNG or raw numpy arrays in Streamlit's `image` component?**
**A:** Uncompressed RGB arrays transferred over local WebSockets require significant bandwidth (~900 KB per 640×480 frame × 3 streams = 2.7 MB/frame). At 30 FPS, this would consume over 80 MB/s of WebSocket throughput, resulting in network congestion, dropped frames, and browser UI freezes. In-memory JPEG compression reduces payload size by ~85–90% with zero perceptible visual degradation.

**Q: How does `aspect-ratio: 4 / 3` in CSS solve the video alignment problem in Streamlit?**
**A:** Native camera frames are 640×480 (a 4:3 aspect ratio). By enforcing `aspect-ratio: 4 / 3` and `width: 100%` on the video container, the height automatically tracks width across any screen or column size ($H = W \times 0.75$). Paired with `object-fit: contain`, the camera image fills the container edge-to-edge with zero side letterboxing or vertical misalignment.

**Q: Why is `cv2.CAP_PROP_BUFFERSIZE = 1` critical for interactive computer vision?**
**A:** Camera drivers typically buffer 3–5 frames. If a computer vision pipeline experiences brief latency variations, the buffer fills, causing the displayed stream to lag several frames behind real-world physical events. Clamping the buffer to 1 guarantees that every frame processed is the most recent capture from the camera sensor.

**Q: How does cooperative `time.sleep(0.004)` prevent thread starvation in Streamlit?**
**A:** Python's Global Interpreter Lock (GIL) can prevent background networking threads from running during compute-heavy continuous loops. Yielding for 4 ms (~1/250th of a second) allows Streamlit's underlying Tornado/Uvicorn WebSocket server to handle incoming browser events (button presses, slider changes) smoothly without measurable impact on frame rate.

**Q: How does the launcher script `run_app.bat` work?**
**A:** It activates the virtual environment at `.venv\Scripts\activate.bat`, verifies the Python environment, and launches Streamlit with `streamlit run streamlit_app.py`, giving Windows users a single-click startup experience.
