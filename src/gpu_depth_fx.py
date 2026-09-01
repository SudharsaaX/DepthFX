import os
import sys
import time
import ctypes
from datetime import datetime
from pathlib import Path

import cv2
import glfw
import numpy as np
import torch

from OpenGL import GL
from OpenGL.GL import shaders
from PIL import Image, ImageDraw, ImageFont

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from depth_estimator import DepthEstimator

CAMERA_INDEX = 0

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

AI_SIZE = 320

DEPTH_UPDATE_INTERVAL = 2

DEPTH_SMOOTHING = 0.65

DISPLAY_MODES = {
    0: "NORMAL",
    1: "DEPTH",
    2: "HEATMAP",
}

EFFECTS = {
    1: {
        "fog_strength": 0.30,
        "blur_strength": 0.25,
        "light_strength": 0.35,
        "ambient": 0.80,
        "fog_start": 0.55,
        "fog_end": 0.95,
    },
    2: {
        "fog_strength": 0.55,
        "blur_strength": 0.50,
        "light_strength": 0.65,
        "ambient": 0.65,
        "fog_start": 0.35,
        "fog_end": 0.90,
    },
    3: {
        "fog_strength": 0.85,
        "blur_strength": 0.85,
        "light_strength": 0.95,
        "ambient": 0.50,
        "fog_start": 0.20,
        "fog_end": 0.80,
    },
}

VERTEX_SHADER = """
#version 330 core

layout(location = 0) in vec2 a_position;
layout(location = 1) in vec2 a_texcoord;

out vec2 v_texcoord;

void main()
{
    v_texcoord = a_texcoord;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330 core

in vec2 v_texcoord;

out vec4 FragColor;

uniform sampler2D u_color;
uniform sampler2D u_depth;

uniform float u_fog_strength;
uniform float u_fog_start;
uniform float u_fog_end;

uniform float u_blur_strength;
uniform float u_depth_threshold;

uniform float u_light_strength;
uniform float u_ambient_strength;

uniform vec2 u_light_position;
uniform vec2 u_texel_size;

uniform int u_fog_enabled;
uniform int u_blur_enabled;
uniform int u_lighting_enabled;
uniform int u_display_mode;
uniform int u_bg_blur_enabled;

uniform float u_time;

float getDepth(vec2 uv)
{
    return texture(u_depth, uv).r;
}

vec3 getColor(vec2 uv)
{
    return texture(u_color, uv).rgb;
}

vec3 depthHeatmap(float depth)
{
    float d = clamp(depth, 0.0, 1.0);
    vec3 color;
    if (d < 0.5)
    {
        color = mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 0.0), d * 2.0);
    }
    else
    {
        color = mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), (d - 0.5) * 2.0);
    }
    return color;
}

vec3 depthAwareBlur(vec2 uv, vec3 original, float depth)
{
    if (u_blur_enabled == 0)
    {
        return original;
    }

    float distanceValue = 1.0 - depth;
    float blurFactor = smoothstep(u_depth_threshold, 1.0, distanceValue);
    blurFactor *= u_blur_strength;
    blurFactor = clamp(blurFactor, 0.0, 1.0);

    if (blurFactor <= 0.001)
    {
        return original;
    }

    float radius = 1.0 + blurFactor * 4.0;
    vec3 sum = original * 0.30;
    float weight = 0.30;

    for (int i = 1; i <= 4; i++)
    {
        float fi = float(i);
        float offset = fi * radius;

        vec2 horizontal = vec2(offset * u_texel_size.x, 0.0);
        vec2 vertical = vec2(0.0, offset * u_texel_size.y);

        float sampleWeight = 1.0 / (1.0 + fi * 0.45);

        sum += getColor(uv + horizontal) * sampleWeight;
        sum += getColor(uv - horizontal) * sampleWeight;
        sum += getColor(uv + vertical) * sampleWeight;
        sum += getColor(uv - vertical) * sampleWeight;

        weight += sampleWeight * 4.0;
    }

    vec3 blurred = sum / weight;
    return mix(original, blurred, blurFactor);
}

vec3 applyBackgroundBlur(vec2 uv, vec3 color, float depth)
{
    if (u_bg_blur_enabled == 0)
    {
        return color;
    }

    float distanceValue = 1.0 - depth;
    float bgFactor = smoothstep(0.25, 0.75, distanceValue);
    bgFactor *= u_blur_strength;
    bgFactor = clamp(bgFactor, 0.0, 1.0);

    if (bgFactor <= 0.001)
    {
        return color;
    }

    float radius = 1.0 + bgFactor * 4.0;
    vec3 sum = color * 0.30;
    float weight = 0.30;

    for (int i = 1; i <= 4; i++)
    {
        float fi = float(i);
        float offset = fi * radius;

        vec2 horizontal = vec2(offset * u_texel_size.x, 0.0);
        vec2 vertical = vec2(0.0, offset * u_texel_size.y);

        float sampleWeight = 1.0 / (1.0 + fi * 0.45);

        sum += getColor(uv + horizontal) * sampleWeight;
        sum += getColor(uv - horizontal) * sampleWeight;
        sum += getColor(uv + vertical) * sampleWeight;
        sum += getColor(uv - vertical) * sampleWeight;

        weight += sampleWeight * 4.0;
    }

    vec3 blurred = sum / weight;
    return mix(color, blurred, bgFactor);
}

float depthEdge(vec2 uv)
{
    float center = getDepth(uv);
    float left = getDepth(uv - vec2(u_texel_size.x, 0.0));
    float right = getDepth(uv + vec2(u_texel_size.x, 0.0));
    float up = getDepth(uv + vec2(0.0, u_texel_size.y));
    float down = getDepth(uv - vec2(0.0, u_texel_size.y));

    float edge = abs(center - left) + abs(center - right) + abs(center - up) + abs(center - down);
    return clamp(edge * 5.0, 0.0, 1.0);
}

vec3 applyLighting(vec3 color, float depth, vec2 uv)
{
    if (u_lighting_enabled == 0)
    {
        return color;
    }

    vec2 delta = uv - u_light_position;
    float distanceToLight = length(delta);
    float radius = 0.70;

    float light = 1.0 - smoothstep(0.0, radius, distanceToLight);
    float depthFactor = mix(0.45, 1.0, depth);

    light *= depthFactor;
    light *= u_light_strength;

    return color * (u_ambient_strength + light);
}

vec3 applyFog(vec3 color, float depth)
{
    if (u_fog_enabled == 0)
    {
        return color;
    }

    float distanceValue = 1.0 - depth;
    float fog = smoothstep(u_fog_start, u_fog_end, distanceValue);
    fog *= u_fog_strength;
    fog = clamp(fog, 0.0, 1.0);

    vec3 fogColor = vec3(0.72, 0.77, 0.84);
    return mix(color, fogColor, fog);
}

void main()
{
    vec2 uv = v_texcoord;
    float depth = getDepth(uv);
    depth = clamp(depth, 0.0, 1.0);

    if (u_display_mode == 1)
    {
        FragColor = vec4(vec3(depth), 1.0);
        return;
    }
    else if (u_display_mode == 2)
    {
        FragColor = vec4(depthHeatmap(depth), 1.0);
        return;
    }

    vec3 color = getColor(uv);
    color = depthAwareBlur(uv, color, depth);
    color = applyBackgroundBlur(uv, color, depth);
    color = applyLighting(color, depth, uv);
    float edge = depthEdge(uv);
    color += edge * 0.035;
    color = applyFog(color, depth);
    color = clamp(color, 0.0, 1.0);

    FragColor = vec4(color, 1.0);
}
"""


# ─── HUD Shader Programs ──────────────────────────────────────────────────────

HUD_VERT = """
#version 330 core
layout(location = 0) in vec2 a_position;
layout(location = 1) in vec2 a_texcoord;
out vec2 v_texcoord;
void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_texcoord = a_texcoord;
}
"""

HUD_FRAG = """
#version 330 core
in vec2 v_texcoord;
out vec4 FragColor;
uniform sampler2D u_tex;
void main() {
    FragColor = texture(u_tex, v_texcoord);
}
"""


# ─── HUD Renderer ─────────────────────────────────────────────────────────────

class HUDRenderer:
    """
    Renders a professional dark-theme overlay HUD on top of the GL scene.

    Architecture
    ------------
    - PIL/Pillow draws text and shapes into RGBA images (off-screen, CPU).
    - Each panel image is uploaded to a dedicated RGBA8 OpenGL texture.
    - A minimal HUD shader blits each texture quad with alpha blending.
    - PIL generation is throttled: stats at ~4 Hz, controls on state change.
    """

    # ── Colour palette ────────────────────────────────────────────
    C_BG          = ( 10,  12,  18, 195)
    C_BORDER      = (  0, 170, 140, 130)
    C_ACCENT      = (  0, 210, 170, 255)
    C_DIM         = (100, 110, 128, 215)
    C_TEXT        = (215, 220, 232, 255)
    C_VALUE       = (  0, 210, 170, 255)
    C_BTN_ON      = (  0,  48,  40, 218)
    C_BTN_OFF     = ( 20,  23,  34, 215)
    C_BTN_BDR_ON  = (  0, 200, 160, 200)
    C_BTN_BDR_OFF = ( 46,  50,  64, 170)
    C_LED_ON      = (  0, 210, 170, 255)
    C_LED_OFF     = ( 60,  65,  82, 200)
    C_SEP         = ( 36,  40,  53, 180)

    PANEL_W = 242
    STATS_W = 207
    STATS_H = 168

    def __init__(self, window_w: int, window_h: int,
                 device_name: str = "", cuda_version: str = ""):
        self.win_w        = window_w
        self.win_h        = window_h
        self.device_name  = device_name
        self.cuda_version = cuda_version

        self._load_fonts()
        self._compile_shader()
        self._init_panels()

        self._last_key         = None
        self._last_legend_mode = -1
        self._last_stats_t     = 0.0

    # ── Font loading ───────────────────────────────────────────────

    def _load_fonts(self):
        candidates = [
            "C:/Windows/Fonts/consola.ttf",
            "C:/Windows/Fonts/cour.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
        ]
        base = None
        for p in candidates:
            if Path(p).exists():
                base = p
                break

        def _ttf(path, size):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                try:
                    return ImageFont.load_default(size=size)
                except Exception:
                    return ImageFont.load_default()

        def _def(size):
            try:
                return ImageFont.load_default(size=size)
            except Exception:
                return ImageFont.load_default()

        if base:
            self.f_title = _ttf(base, 15)
            self.f_label = _ttf(base, 11)
            self.f_val   = _ttf(base, 13)
            self.f_small = _ttf(base, 10)
        else:
            self.f_title = _def(15)
            self.f_label = _def(11)
            self.f_val   = _def(13)
            self.f_small = _def(10)

    # ── Shader compilation ─────────────────────────────────────────

    def _compile_shader(self):
        v = shaders.compileShader(HUD_VERT, GL.GL_VERTEX_SHADER)
        f = shaders.compileShader(HUD_FRAG, GL.GL_FRAGMENT_SHADER)
        self.prog = shaders.compileProgram(v, f)
        GL.glDeleteShader(v)
        GL.glDeleteShader(f)
        GL.glUseProgram(self.prog)
        loc = GL.glGetUniformLocation(self.prog, "u_tex")
        if loc >= 0:
            GL.glUniform1i(loc, 0)
        GL.glUseProgram(0)

    # ── OpenGL helpers ─────────────────────────────────────────────

    def _make_quad(self, x: int, y: int, w: int, h: int):
        """Build a VAO/VBO for a pixel-space rect; UV (0,0)=GL bottom-left."""
        x0 =  (x       / self.win_w) * 2.0 - 1.0
        x1 =  ((x + w) / self.win_w) * 2.0 - 1.0
        y0 =  1.0 - ((y + h) / self.win_h) * 2.0  # NDC bottom of panel
        y1 =  1.0 - ( y      / self.win_h) * 2.0  # NDC top of panel
        # UV rows: (0,0) = GL bottom = PIL row after flipud row-0 = PIL bottom
        # UV (1,1) = GL top = PIL top after flipud
        verts = np.array([
            x0, y0,  0.0, 0.0,
            x1, y0,  1.0, 0.0,
            x1, y1,  1.0, 1.0,
            x0, y0,  0.0, 0.0,
            x1, y1,  1.0, 1.0,
            x0, y1,  0.0, 1.0,
        ], dtype=np.float32)
        vao = GL.glGenVertexArrays(1)
        vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, verts.nbytes, verts, GL.GL_STATIC_DRAW)
        stride = 4 * verts.itemsize
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(8))
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindVertexArray(0)
        return vao, vbo

    def _make_tex(self, w: int, h: int) -> int:
        tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, w, h, 0,
            GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None,
        )
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        return tex

    def _upload(self, tex: int, img):
        """Upload a PIL RGBA image to an OpenGL texture (flip for GL origin)."""
        arr = np.flipud(np.array(img, dtype=np.uint8))
        arr = np.ascontiguousarray(arr)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glTexSubImage2D(
            GL.GL_TEXTURE_2D, 0, 0, 0,
            img.width, img.height,
            GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, arr,
        )
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    # ── Panel setup ────────────────────────────────────────────────

    def _init_panels(self):
        lw, lh = 390, 38
        self.panels = {
            "left":   {"x": 0,                          "y": 0,
                       "w": self.PANEL_W,                "h": self.win_h},
            "stats":  {"x": self.win_w - self.STATS_W,  "y": 0,
                       "w": self.STATS_W,                "h": self.STATS_H},
            "legend": {"x": (self.win_w - lw) // 2,
                       "y": self.win_h - lh - 8,        "w": lw, "h": lh},
        }
        for p in self.panels.values():
            p["vao"], p["vbo"] = self._make_quad(p["x"], p["y"], p["w"], p["h"])
            p["tex"]           = self._make_tex(p["w"], p["h"])

    # ── PIL drawing helpers ────────────────────────────────────────

    def _tw(self, text: str, font) -> int:
        """Text width in pixels (robust across Pillow versions)."""
        try:
            bb = font.getbbox(text)
            return bb[2] - bb[0]
        except AttributeError:
            try:
                return font.getsize(text)[0]
            except Exception:
                return len(text) * 7
        except Exception:
            return len(text) * 7

    def _sep(self, d, y: int, W: int, pad: int = 14):
        d.line([(pad, y), (W - pad, y)], fill=self.C_SEP, width=1)

    def _btn(self, d, x: int, y: int, label: str, active: bool, font,
             btn_h: int = 22) -> int:
        """Draw a state button; returns its pixel width."""
        bw  = self._tw(label, font) + 18
        bg  = self.C_BTN_ON      if active else self.C_BTN_OFF
        bdr = self.C_BTN_BDR_ON  if active else self.C_BTN_BDR_OFF
        tc  = self.C_ACCENT      if active else (145, 150, 168, 220)
        d.rectangle([x, y, x + bw - 1, y + btn_h - 1], fill=bg, outline=bdr)
        d.text((x + 9, y + (btn_h - 12) // 2), label, font=font, fill=tc)
        return bw

    # ── Panel: left sidebar ────────────────────────────────────────

    def _draw_left(self, state: dict):
        p = self.panels["left"]
        W, H = p["w"], p["h"]
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d   = ImageDraw.Draw(img)

        d.rectangle([0, 0, W - 1, H - 1], fill=self.C_BG)
        d.line([(W - 1, 0), (W - 1, H - 1)], fill=self.C_BORDER, width=1)

        PAD = 14
        y   = 14

        # ── Title ────────────────────────────────────────────────
        tw_depth = self._tw("DEPTH", self.f_title)
        d.text((PAD, y),               "DEPTH", font=self.f_title, fill=self.C_ACCENT)
        d.text((PAD + tw_depth + 5, y), "FX",   font=self.f_title,
               fill=(255, 255, 255, 255))
        ul_end = PAD + tw_depth + 5 + self._tw("FX", self.f_title)
        d.line([(PAD, y + 18), (ul_end, y + 18)], fill=(0, 165, 135, 140), width=1)
        y += 27

        # ── AI MODEL ─────────────────────────────────────────────
        d.text((PAD, y), "AI  MODEL", font=self.f_small, fill=self.C_DIM)
        y += 15
        d.text((PAD, y), "Depth Anything V2  \u00b7  ViT-S",
               font=self.f_label, fill=self.C_TEXT)
        y += 15
        d.text((PAD, y), f"CUDA {self.cuda_version}  \u00b7  FP16 Autocast",
               font=self.f_small, fill=(0, 198, 160, 210))
        y += 14
        d.text((PAD, y),
               f"\u0394/{DEPTH_UPDATE_INTERVAL}  \u00b7  \u03b1={DEPTH_SMOOTHING}"
               f"  \u00b7  {AI_SIZE}px input",
               font=self.f_small, fill=(98, 108, 126, 200))
        y += 20

        self._sep(d, y, W)
        y += 10

        # ── DISPLAY MODE ─────────────────────────────────────────
        d.text((PAD, y), "DISPLAY  MODE", font=self.f_small, fill=self.C_DIM)
        y += 15
        cur_mode = DISPLAY_MODES[state["display_mode"]]
        bx = PAD
        for m in ["NORMAL", "DEPTH", "HEATMAP"]:
            bx += self._btn(d, bx, y, m, m == cur_mode, self.f_small) + 5
        y += 32

        self._sep(d, y, W)
        y += 10

        # ── EFFECT PRESET ─────────────────────────────────────────
        d.text((PAD, y), "EFFECT  PRESET", font=self.f_small, fill=self.C_DIM)
        y += 15
        cur_level = state["effect_level"]
        bx = PAD
        for lbl, lvl in [("LIGHT", 1), ("MEDIUM", 2), ("STRONG", 3)]:
            bx += self._btn(d, bx, y, lbl, lvl == cur_level, self.f_small) + 5
        y += 32

        self._sep(d, y, W)
        y += 10

        # ── EFFECTS TOGGLES ───────────────────────────────────────
        d.text((PAD, y), "EFFECTS", font=self.f_small, fill=self.C_DIM)
        y += 15
        for lbl, en, key in [
            ("FOG",     state["fog_enabled"],     "F"),
            ("BLUR",    state["blur_enabled"],    "B"),
            ("BG BLUR", state["bg_blur_enabled"], "P"),
        ]:
            led = self.C_LED_ON if en else self.C_LED_OFF
            d.ellipse([PAD, y + 4, PAD + 8, y + 12], fill=led)
            d.text((PAD + 14, y + 1), lbl, font=self.f_label, fill=self.C_TEXT)
            st_txt = "ON" if en else "OFF"
            st_col = self.C_ACCENT if en else (88, 92, 110, 210)
            sw = self._tw(st_txt, self.f_small)
            d.text((W - PAD - sw, y + 2), st_txt, font=self.f_small, fill=st_col)
            kw = self._tw(key, self.f_small)
            kx = W - PAD - sw - kw - 16
            d.rectangle([kx, y + 1, kx + kw + 8, y + 14],
                        fill=(18, 21, 32, 210), outline=(42, 46, 62, 155))
            d.text((kx + 4, y + 2), key, font=self.f_small,
                   fill=(0, 178, 143, 215))
            y += 20

        y += 4
        self._sep(d, y, W)
        y += 10

        # ── ACTIONS ───────────────────────────────────────────────
        d.text((PAD, y), "ACTIONS", font=self.f_small, fill=self.C_DIM)
        y += 15
        bx = PAD
        for lbl, _ in [("SCREENSHOT", "S"), ("RESET", "R")]:
            bw = self._tw(lbl, self.f_small) + 20
            d.rectangle([bx, y, bx + bw - 1, y + 22],
                        fill=self.C_BTN_OFF, outline=self.C_BTN_BDR_OFF)
            d.text((bx + 10, y + 6), lbl, font=self.f_small,
                   fill=(168, 174, 190, 220))
            bx += bw + 6
        y += 32

        self._sep(d, y, W)
        y += 10

        # ── KEYBOARD ──────────────────────────────────────────────
        d.text((PAD, y), "KEYBOARD", font=self.f_small, fill=self.C_DIM)
        y += 15
        for key, desc in [
            ("D",     "Cycle display mode"),
            ("F/B/P", "Toggle fog / blur / bg"),
            ("1/2/3", "Effect preset"),
            ("S",     "Screenshot"),
            ("R",     "Reset"),
            ("Q",     "Quit"),
            ("Mouse", "Virtual light"),
        ]:
            kw = self._tw(key, self.f_small)
            d.rectangle([PAD, y, PAD + kw + 8, y + 13],
                        fill=(16, 19, 30, 210), outline=(42, 46, 62, 155))
            d.text((PAD + 4, y + 1), key, font=self.f_small,
                   fill=(0, 185, 150, 218))
            d.text((PAD + kw + 14, y + 1), desc, font=self.f_small,
                   fill=(108, 116, 134, 200))
            y += 16

        self._upload(p["tex"], img)

    # ── Panel: top-right stats ─────────────────────────────────────

    def _draw_stats(self, state: dict):
        p = self.panels["stats"]
        W, H = p["w"], p["h"]
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d   = ImageDraw.Draw(img)

        d.rectangle([0, 0, W - 1, H - 1], fill=self.C_BG)
        d.line([(0, 0), (0, H - 1)],         fill=self.C_BORDER, width=1)
        d.line([(0, H - 1), (W - 1, H - 1)], fill=(0, 165, 130, 75), width=1)

        PAD = 10
        y   = 10

        d.text((PAD, y), "PERFORMANCE", font=self.f_small, fill=self.C_DIM)
        y += 17

        def _metric(label: str, value: str, val_col=None):
            nonlocal y
            col = val_col if val_col else self.C_VALUE
            d.text((PAD, y), label, font=self.f_small, fill=(132, 138, 156, 210))
            vw = self._tw(value, self.f_val)
            d.text((W - PAD - vw, y), value, font=self.f_val, fill=col)
            y += 18

        fps = state.get("fps", 0.0)
        _metric("FPS",
                f"{fps:.1f}",
                self.C_ACCENT if fps >= 25 else (210, 150, 55, 255))
        _metric("AI INFERENCE", f"{state.get('ai_ms', 0.0):.1f} ms")
        _metric("GPU SHADER",   f"{state.get('gpu_ms', 0.0):.3f} ms")

        self._sep(d, y + 2, W, pad=PAD)
        y += 12

        dev = self.device_name
        for prefix in ("NVIDIA GeForce ", "NVIDIA "):
            if dev.startswith(prefix):
                dev = dev[len(prefix):]
                break
        if len(dev) > 22:
            dev = dev[:22]

        d.text((PAD, y), dev, font=self.f_small, fill=(0, 198, 160, 190))
        y += 14
        d.text((PAD, y),
               f"CUDA {self.cuda_version}  \u00b7  OpenGL 3.3",
               font=self.f_small, fill=(88, 98, 116, 190))
        y += 14
        d.text((PAD, y), "FP16 Autocast  \u00b7  EWMA 0.65",
               font=self.f_small, fill=(72, 80, 98, 175))

        self._upload(p["tex"], img)

    # ── Panel: heatmap legend ──────────────────────────────────────

    def _draw_legend(self):
        p = self.panels["legend"]
        W, H = p["w"], p["h"]
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d   = ImageDraw.Draw(img)

        d.rectangle([0, 0, W - 1, H - 1], fill=(10, 12, 18, 200))
        d.rectangle([0, 0, W - 1, H - 1], outline=(0, 165, 130, 78), width=1)

        PAD    = 8
        far_w  = self._tw("FAR",  self.f_small)
        near_w = self._tw("NEAR", self.f_small)
        bar_x  = PAD + far_w + 6
        bar_y  = 7
        bar_w  = W - bar_x - PAD - near_w - 6
        bar_h  = 12

        for i in range(bar_w):
            t = i / max(bar_w - 1, 1)
            if t < 0.5:
                r, g, b = 0, int(t * 2 * 255), int(255 - t * 2 * 255)
            else:
                r = int((t - 0.5) * 2 * 255)
                g = int(255 - (t - 0.5) * 2 * 255)
                b = 0
            d.line([(bar_x + i, bar_y), (bar_x + i, bar_y + bar_h)],
                   fill=(r, g, b, 220))

        d.rectangle([bar_x - 1, bar_y - 1, bar_x + bar_w, bar_y + bar_h + 1],
                    outline=(52, 58, 76, 175))
        d.text((PAD, bar_y + 2),                "FAR",
               font=self.f_small, fill=( 75, 105, 215, 220))
        d.text((bar_x + bar_w + 5, bar_y + 2), "NEAR",
               font=self.f_small, fill=(210,  65,  50, 220))

        bl_y = bar_y + bar_h + 3
        d.text((bar_x + 2, bl_y), "BLUE",
               font=self.f_small, fill=( 65,  85, 195, 175))
        gw = self._tw("GREEN", self.f_small)
        d.text((bar_x + bar_w // 2 - gw // 2, bl_y), "GREEN",
               font=self.f_small, fill=( 65, 170,  65, 175))
        rw = self._tw("RED", self.f_small)
        d.text((bar_x + bar_w - rw - 2, bl_y), "RED",
               font=self.f_small, fill=(195,  60,  50, 175))

        self._upload(p["tex"], img)

    # ── Public API ─────────────────────────────────────────────────

    def update(self, state: dict):
        """Regenerate panel textures as needed (throttled / on state change)."""
        now = time.perf_counter()

        # Stats panel: throttled to ~4 Hz
        if now - self._last_stats_t >= 0.25:
            self._draw_stats(state)
            self._last_stats_t = now

        # Controls panel: only on state change
        key = (
            state["display_mode"], state["effect_level"],
            state["fog_enabled"],  state["blur_enabled"],
            state["bg_blur_enabled"],
        )
        if key != self._last_key:
            self._draw_left(state)
            self._last_key = key

        # Legend: only when entering HEATMAP mode
        if state["display_mode"] == 2 and self._last_legend_mode != 2:
            self._draw_legend()
        self._last_legend_mode = state["display_mode"]

    def render(self, state: dict):
        """Blit active panels with GL_BLEND alpha compositing."""
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glUseProgram(self.prog)
        GL.glActiveTexture(GL.GL_TEXTURE0)

        names = ["left", "stats"]
        if state["display_mode"] == 2:
            names.append("legend")

        for name in names:
            p = self.panels[name]
            GL.glBindTexture(GL.GL_TEXTURE_2D, p["tex"])
            GL.glBindVertexArray(p["vao"])
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
            GL.glBindVertexArray(0)

        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glUseProgram(0)
        GL.glDisable(GL.GL_BLEND)

    def resize(self, w: int, h: int):
        """
        Reposition HUD panels for a new framebuffer size.

        Only the quad geometry (VAO/VBO) is rebuilt -- fonts are NEVER reloaded
        so this is fast enough to call from the GLFW callback without impacting FPS.
        The left panel's texture is recreated because its height equals win_h.
        Stats and legend panels keep their existing textures (fixed pixel size).
        """
        if w <= 0 or h <= 0:
            return
        if w == self.win_w and h == self.win_h:
            return  # Nothing to do

        # ─ Delete old quad geometry for all panels ──────────────────────
        for p in self.panels.values():
            try:
                GL.glDeleteBuffers(1, [p["vbo"]])
                GL.glDeleteVertexArrays(1, [p["vao"]])
            except Exception:
                pass

        # Left panel height == win_h; its texture must be recreated.
        try:
            GL.glDeleteTextures([self.panels["left"]["tex"]])
        except Exception:
            pass

        self.win_w = w
        self.win_h = h

        # ─ Recompute pixel positions (panel widths/heights stay FIXED) ─────
        lw, lh = 390, 38
        self.panels["left" ]["h"]  = h
        self.panels["stats" ]["x"]  = w - self.STATS_W
        self.panels["legend"]["x"]  = (w - lw) // 2
        self.panels["legend"]["y"]  = h - lh - 8

        # ─ Rebuild all quad geometry ──────────────────────────────────
        for p in self.panels.values():
            p["vao"], p["vbo"] = self._make_quad(p["x"], p["y"], p["w"], p["h"])

        # ─ Recreate left panel texture (new height) ─────────────────────
        lp = self.panels["left"]
        lp["tex"] = self._make_tex(lp["w"], lp["h"])

        # Force left panel PIL redraw on next update(); stats/legend reuse existing textures.
        self._last_key = None

    def cleanup(self):
        """Delete all OpenGL resources owned by the HUD."""
        for p in self.panels.values():
            try:
                GL.glDeleteTextures([p["tex"]])
                GL.glDeleteBuffers(1, [p["vbo"]])
                GL.glDeleteVertexArrays(1, [p["vao"]])
            except Exception:
                pass
        try:
            GL.glDeleteProgram(self.prog)
        except Exception:
            pass


class GPUTimer:
    def __init__(self):
        self.enabled = False
        self.query = None
        self.last_ms = 0.0
        self.samples = []
        self.error_reported = False

        try:
            raw_query = GL.glGenQueries(1)
            values = np.asarray(raw_query).reshape(-1)
            if values.size == 0:
                return

            self.query = int(values[0])
            if self.query <= 0:
                return

            self.enabled = True
            print("GPU timer query: enabled")
            print("GPU timing mode: standard OpenGL query")

        except Exception as error:
            print("GPU timer query unavailable:")
            print(error)

    def begin(self):
        if not self.enabled:
            return False

        try:
            GL.glBeginQuery(GL.GL_TIME_ELAPSED, int(self.query))
            return True
        except Exception as error:
            self.enabled = False
            if not self.error_reported:
                print("GPU timer start failed:")
                print(error)
                self.error_reported = True
            return False

    def end(self):
        if not self.enabled:
            return

        try:
            GL.glEndQuery(GL.GL_TIME_ELAPSED)
        except Exception as error:
            self.enabled = False
            if not self.error_reported:
                print("GPU timer end failed:")
                print(error)
                self.error_reported = True

    def read(self):
        if not self.enabled:
            return self.last_ms

        try:
            available = GL.glGetQueryObjectuiv(
                int(self.query), GL.GL_QUERY_RESULT_AVAILABLE
            )
            available_array = np.asarray(available).reshape(-1)
            if available_array.size == 0 or int(available_array[0]) == 0:
                return self.last_ms

            result = GL.glGetQueryObjectuiv(
                int(self.query), GL.GL_QUERY_RESULT
            )
            result_array = np.asarray(result).reshape(-1)
            if result_array.size == 0:
                return self.last_ms

            gpu_time_ns = int(result_array[0])
            gpu_time_ms = float(gpu_time_ns) / 1_000_000.0

            self.last_ms = gpu_time_ms
            self.samples.append(gpu_time_ms)
            if len(self.samples) > 300:
                self.samples.pop(0)

            return gpu_time_ms

        except Exception as error:
            self.enabled = False
            if not self.error_reported:
                print("GPU timer read disabled:")
                print(error)
                self.error_reported = True
            return self.last_ms

    def cleanup(self):
        if self.query is None:
            return

        try:
            GL.glDeleteQueries(
                1, np.array([int(self.query)], dtype=np.uint32)
            )
        except Exception:
            pass


def save_screenshot(window):
    try:
        width, height = glfw.get_framebuffer_size(window)
        if width <= 0 or height <= 0:
            return

        gl_pixels = GL.glReadPixels(
            0, 0, width, height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE
        )
        image = np.frombuffer(gl_pixels, dtype=np.uint8).reshape((height, width, 3))
        image = np.flipud(image)
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        outputs_dir = Path("outputs")
        outputs_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now()
        base_name = now.strftime("depthfx_%Y%m%d_%H%M%S")
        filepath = outputs_dir / f"{base_name}.png"
        counter = 1
        while filepath.exists():
            filepath = outputs_dir / f"{base_name}_{counter}.png"
            counter += 1

        success = cv2.imwrite(str(filepath), bgr_image)
        if success:
            print(f"Screenshot saved: {filepath.as_posix()}")
        else:
            print(f"Screenshot failed: cv2.imwrite returned False")
    except Exception as error:
        print(f"Screenshot failed: {error}")


def create_shader_program():
    vertex_shader = shaders.compileShader(VERTEX_SHADER, GL.GL_VERTEX_SHADER)
    fragment_shader = shaders.compileShader(FRAGMENT_SHADER, GL.GL_FRAGMENT_SHADER)
    program = shaders.compileProgram(vertex_shader, fragment_shader)
    GL.glDeleteShader(vertex_shader)
    GL.glDeleteShader(fragment_shader)
    return program


def create_fullscreen_quad():
    vertices = np.array(
        [
            -1.0, -1.0,  0.0, 0.0,
             1.0, -1.0,  1.0, 0.0,
             1.0,  1.0,  1.0, 1.0,

            -1.0, -1.0,  0.0, 0.0,
             1.0,  1.0,  1.0, 1.0,
            -1.0,  1.0,  0.0, 1.0,
        ],
        dtype=np.float32,
    )

    vao = GL.glGenVertexArrays(1)
    vbo = GL.glGenBuffers(1)

    GL.glBindVertexArray(vao)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_STATIC_DRAW)

    stride = 4 * vertices.itemsize
    GL.glEnableVertexAttribArray(0)
    GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(0))

    GL.glEnableVertexAttribArray(1)
    GL.glVertexAttribPointer(
        1, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(2 * vertices.itemsize)
    )

    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
    GL.glBindVertexArray(0)

    return vao, vbo


def create_rgb_texture():
    texture = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture)

    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)

    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D,
        0,
        GL.GL_RGB8,
        CAMERA_WIDTH,
        CAMERA_HEIGHT,
        0,
        GL.GL_RGB,
        GL.GL_UNSIGNED_BYTE,
        None,
    )
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    return texture


def create_depth_texture():
    texture = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture)

    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)

    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 4)
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D,
        0,
        GL.GL_R32F,
        CAMERA_WIDTH,
        CAMERA_HEIGHT,
        0,
        GL.GL_RED,
        GL.GL_FLOAT,
        None,
    )
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    return texture


def main():
    global CAMERA_WIDTH, CAMERA_HEIGHT

    print("=" * 60)
    print("DepthFX - GPU Depth Effects + Optimized AI + Mouse Lighting + Performance Profiler")
    print("=" * 60)

    print()
    print("Loading Depth Anything V2...")
    estimator = DepthEstimator()

    print()
    print("Initializing OpenGL...")
    if not glfw.init():
        raise RuntimeError("GLFW initialization failed.")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(WINDOW_WIDTH, WINDOW_HEIGHT, "DepthFX", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Could not create OpenGL window.")

    glfw.make_context_current(window)
    glfw.swap_interval(0)

    vendor = GL.glGetString(GL.GL_VENDOR)
    renderer = GL.glGetString(GL.GL_RENDERER)
    version = GL.glGetString(GL.GL_VERSION)

    print("OpenGL Vendor:", vendor.decode() if vendor else "Unknown")
    print("OpenGL Renderer:", renderer.decode() if renderer else "Unknown")
    print("OpenGL Version:", version.decode() if version else "Unknown")

    print()
    print("Loading combined GPU shader...")
    program = create_shader_program()
    print("Combined GPU shader created successfully.")

    vao, vbo = create_fullscreen_quad()

    print()
    print("Opening webcam...")

    # Try the configured camera index first, then auto-detect.
    def _open_camera(index):
        cam = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cam.isOpened():
            cam.release()
            cam = cv2.VideoCapture(index)
        if not cam.isOpened():
            cam.release()
            return None
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # Discard stale buffered frames before reading the test frame
        for _ in range(5):
            cam.grab()
        ret, frame = cam.read()
        if not ret or frame is None or frame.size == 0:
            cam.release()
            return None
        # Frame dimensions must be sane (at least 160x120)
        if frame.shape[0] < 120 or frame.shape[1] < 160:
            cam.release()
            return None
        # Verify frame contains actual image content (not dummy/black/noise stream)
        if float(frame.std()) < 5.0 or int(frame.max()) < 10:
            cam.release()
            return None
        return cam

    camera = _open_camera(CAMERA_INDEX)
    if camera is None:
        print(f"Camera index {CAMERA_INDEX} failed, auto-detecting...")
        for _try_idx in range(6):
            if _try_idx == CAMERA_INDEX:
                continue
            camera = _open_camera(_try_idx)
            if camera is not None:
                print(f"Using camera index {_try_idx}")
                break

    if camera is None:
        raise RuntimeError("Could not open any webcam. Please check camera connections.")

    # Read actual resolution reported by the camera driver
    _cam_w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    _cam_h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if _cam_w > 0 and _cam_h > 0:
        CAMERA_WIDTH  = _cam_w
        CAMERA_HEIGHT = _cam_h

    print(f"Camera resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")

    rgb_texture = create_rgb_texture()
    depth_texture = create_depth_texture()

    print("RGB texture created.")
    print("R32F depth texture created.")

    gpu_timer = GPUTimer()

    # ── HUD initialisation ─────────────────────────────────────────
    _device_name  = ""
    _cuda_version = ""
    try:
        if torch.cuda.is_available():
            _device_name  = torch.cuda.get_device_name(0)
            _cuda_version = str(torch.version.cuda) if torch.version.cuda else "N/A"
    except Exception:
        pass

    hud = HUDRenderer(
        WINDOW_WIDTH, WINDOW_HEIGHT,
        device_name=_device_name,
        cuda_version=_cuda_version,
    )
    print("HUD renderer initialised.")

    GL.glUseProgram(program)
    u_color = GL.glGetUniformLocation(program, "u_color")
    u_depth = GL.glGetUniformLocation(program, "u_depth")
    u_texel_size = GL.glGetUniformLocation(program, "u_texel_size")
    u_fog_enabled = GL.glGetUniformLocation(program, "u_fog_enabled")
    u_fog_strength = GL.glGetUniformLocation(program, "u_fog_strength")
    u_fog_start = GL.glGetUniformLocation(program, "u_fog_start")
    u_fog_end = GL.glGetUniformLocation(program, "u_fog_end")
    u_blur_enabled = GL.glGetUniformLocation(program, "u_blur_enabled")
    u_blur_strength = GL.glGetUniformLocation(program, "u_blur_strength")
    u_depth_threshold = GL.glGetUniformLocation(program, "u_depth_threshold")
    u_lighting_enabled = GL.glGetUniformLocation(program, "u_lighting_enabled")
    u_light_position = GL.glGetUniformLocation(program, "u_light_position")
    u_light_strength = GL.glGetUniformLocation(program, "u_light_strength")
    u_ambient_strength = GL.glGetUniformLocation(program, "u_ambient_strength")
    u_display_mode = GL.glGetUniformLocation(program, "u_display_mode")
    u_bg_blur_enabled = GL.glGetUniformLocation(program, "u_bg_blur_enabled")
    u_time = GL.glGetUniformLocation(program, "u_time")

    if u_color >= 0:
        GL.glUniform1i(u_color, 0)
    if u_depth >= 0:
        GL.glUniform1i(u_depth, 1)
    if u_texel_size >= 0:
        GL.glUniform2f(u_texel_size, 1.0 / CAMERA_WIDTH, 1.0 / CAMERA_HEIGHT)

    GL.glUseProgram(0)

    fog_enabled = True
    blur_enabled = True
    lighting_enabled = True
    effect_level = 2
    display_mode = 0
    bg_blur_enabled = False
    take_screenshot = False
    mouse_x = 0.5
    mouse_y = 0.5

    def key_callback(window_handle, key, scancode, action, mods):
        nonlocal fog_enabled, blur_enabled, effect_level, display_mode, bg_blur_enabled, take_screenshot

        if action != glfw.PRESS:
            return

        if key == glfw.KEY_Q:
            glfw.set_window_should_close(window_handle, True)
        elif key == glfw.KEY_S:
            take_screenshot = True
        elif key == glfw.KEY_P:
            bg_blur_enabled = not bg_blur_enabled
            print(f"Background blur: {'ON' if bg_blur_enabled else 'OFF'}")
        elif key == glfw.KEY_D:
            display_mode = (display_mode + 1) % 3
            print(f"Display: {DISPLAY_MODES[display_mode]}")
        elif key == glfw.KEY_F:
            fog_enabled = not fog_enabled
            print("Fog:", "ON" if fog_enabled else "OFF")
        elif key == glfw.KEY_B:
            blur_enabled = not blur_enabled
            print("Blur:", "ON" if blur_enabled else "OFF")
        elif key == glfw.KEY_1:
            effect_level = 1
            print("Effects: LIGHT")
        elif key == glfw.KEY_2:
            effect_level = 2
            print("Effects: MEDIUM")
        elif key == glfw.KEY_3:
            effect_level = 3
            print("Effects: STRONG")
        elif key == glfw.KEY_R:
            effect_level = 2
            fog_enabled = True
            blur_enabled = True
            display_mode = 0
            bg_blur_enabled = False
            print("DepthFX settings reset.")

    def cursor_callback(window_handle, xpos, ypos):
        nonlocal mouse_x, mouse_y

        width, height = glfw.get_window_size(window_handle)
        if width <= 0 or height <= 0:
            return

        mouse_x = float(xpos) / float(width)
        mouse_y = 1.0 - float(ypos) / float(height)

        mouse_x = float(np.clip(mouse_x, 0.0, 1.0))
        mouse_y = float(np.clip(mouse_y, 0.0, 1.0))

    glfw.set_key_callback(window, key_callback)
    glfw.set_cursor_pos_callback(window, cursor_callback)

    # ── Framebuffer-size tracking (resize / maximise support) ──────
    _fb = list(glfw.get_framebuffer_size(window))
    if _fb[0] <= 0 or _fb[1] <= 0:
        _fb = [WINDOW_WIDTH, WINDOW_HEIGHT]
    fb_w, fb_h = _fb

    # If the initial framebuffer already differs (HiDPI) resize HUD now
    if fb_w != WINDOW_WIDTH or fb_h != WINDOW_HEIGHT:
        try:
            hud.resize(fb_w, fb_h)
        except Exception:
            pass

    def framebuffer_size_callback(win, w, h):
        nonlocal fb_w, fb_h
        # Guard: only act if the size actually changed
        if w <= 0 or h <= 0 or (w == fb_w and h == fb_h):
            return
        fb_w, fb_h = w, h
        try:
            hud.resize(w, h)
        except Exception as _e:
            print(f"HUD resize: {_e}")

    glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)

    current_depth = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH), dtype=np.float32)
    previous_depth = None

    total_frames = 0
    depth_updates = 0
    fps_counter = 0
    fps = 0.0
    fps_start = time.perf_counter()
    title_start = time.perf_counter()
    displayed_ai_ms = 0.0

    print()
    print("Starting optimized DepthFX...")
    print()
    print("Controls:")
    print("  D = cycle display mode (NORMAL / DEPTH / HEATMAP)")
    print("  P = toggle background blur")
    print("  S = capture screenshot")
    print("  F = toggle fog")
    print("  B = toggle blur")
    print("  1 = light effects")
    print("  2 = medium effects")
    print("  3 = strong effects")
    print("  R = reset")
    print("  Q = quit")
    print()
    print("Mouse = move virtual light")
    print()
    print("Performance profiling enabled.")
    print("GPU timer query:", "enabled" if gpu_timer.enabled else "disabled")
    print("AI inference size:", AI_SIZE)
    print("Depth update interval:", DEPTH_UPDATE_INTERVAL, "frames")
    print("Depth smoothing:", DEPTH_SMOOTHING)
    print("RGB + AI Depth -> Temporal Depth -> GLSL Fog + Blur + Lighting -> RTX 4070")
    print()

    try:
        while not glfw.window_should_close(window):
            glfw.poll_events()

            success, frame = camera.read()
            if not success:
                print("Warning: camera frame could not be read.")
                continue

            if frame.shape[1] != CAMERA_WIDTH or frame.shape[0] != CAMERA_HEIGHT:
                frame = cv2.resize(
                    frame, (CAMERA_WIDTH, CAMERA_HEIGHT), interpolation=cv2.INTER_AREA
                )

            total_frames += 1
            fps_counter += 1

            should_update_depth = (
                total_frames == 1 or (total_frames % DEPTH_UPDATE_INTERVAL == 0)
            )

            if should_update_depth:
                ai_start = time.perf_counter()

                new_depth = estimator.estimate(frame)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                displayed_ai_ms = (time.perf_counter() - ai_start) * 1000.0
                depth_updates += 1

                new_depth = np.asarray(new_depth, dtype=np.float32)

                depth_min = float(np.min(new_depth))
                depth_max = float(np.max(new_depth))

                if depth_max - depth_min > 1e-8:
                    new_depth = (new_depth - depth_min) / (depth_max - depth_min)
                else:
                    new_depth = np.zeros_like(new_depth, dtype=np.float32)

                if (
                    new_depth.shape[0] != CAMERA_HEIGHT
                    or new_depth.shape[1] != CAMERA_WIDTH
                ):
                    new_depth = cv2.resize(
                        new_depth,
                        (CAMERA_WIDTH, CAMERA_HEIGHT),
                        interpolation=cv2.INTER_LINEAR,
                    )

                new_depth = np.ascontiguousarray(new_depth, dtype=np.float32)

                if previous_depth is None:
                    current_depth = new_depth.copy()
                else:
                    current_depth = (
                        DEPTH_SMOOTHING * previous_depth
                        + (1.0 - DEPTH_SMOOTHING) * new_depth
                    )

                previous_depth = current_depth.copy()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = np.flipud(rgb).copy()
            rgb = np.ascontiguousarray(rgb, dtype=np.uint8)

            depth_upload = np.flipud(current_depth).copy()
            depth_upload = np.ascontiguousarray(depth_upload, dtype=np.float32)

            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, rgb_texture)
            GL.glTexSubImage2D(
                GL.GL_TEXTURE_2D,
                0,
                0,
                0,
                CAMERA_WIDTH,
                CAMERA_HEIGHT,
                GL.GL_RGB,
                GL.GL_UNSIGNED_BYTE,
                rgb,
            )

            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 4)
            GL.glActiveTexture(GL.GL_TEXTURE1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, depth_texture)
            GL.glTexSubImage2D(
                GL.GL_TEXTURE_2D,
                0,
                0,
                0,
                CAMERA_WIDTH,
                CAMERA_HEIGHT,
                GL.GL_RED,
                GL.GL_FLOAT,
                depth_upload,
            )

            GL.glViewport(0, 0, fb_w, fb_h)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            GL.glUseProgram(program)

            settings = EFFECTS[effect_level]

            if u_fog_enabled >= 0:
                GL.glUniform1i(u_fog_enabled, int(fog_enabled))
            if u_fog_strength >= 0:
                GL.glUniform1f(u_fog_strength, settings["fog_strength"])
            if u_fog_start >= 0:
                GL.glUniform1f(u_fog_start, settings["fog_start"])
            if u_fog_end >= 0:
                GL.glUniform1f(u_fog_end, settings["fog_end"])

            if u_blur_enabled >= 0:
                GL.glUniform1i(u_blur_enabled, int(blur_enabled))
            if u_blur_strength >= 0:
                GL.glUniform1f(u_blur_strength, settings["blur_strength"])
            if u_depth_threshold >= 0:
                GL.glUniform1f(u_depth_threshold, 0.50)

            if u_lighting_enabled >= 0:
                GL.glUniform1i(u_lighting_enabled, int(lighting_enabled))
            if u_light_position >= 0:
                GL.glUniform2f(u_light_position, mouse_x, mouse_y)
            if u_light_strength >= 0:
                GL.glUniform1f(u_light_strength, settings["light_strength"])
            if u_ambient_strength >= 0:
                GL.glUniform1f(u_ambient_strength, settings["ambient"])

            if u_display_mode >= 0:
                GL.glUniform1i(u_display_mode, display_mode)

            if u_bg_blur_enabled >= 0:
                GL.glUniform1i(u_bg_blur_enabled, int(bg_blur_enabled))

            if u_color >= 0:
                GL.glUniform1i(u_color, 0)
            if u_depth >= 0:
                GL.glUniform1i(u_depth, 1)
            if u_texel_size >= 0:
                GL.glUniform2f(u_texel_size, 1.0 / CAMERA_WIDTH, 1.0 / CAMERA_HEIGHT)

            if u_time >= 0:
                GL.glUniform1f(u_time, float(time.perf_counter()))

            timer_started = gpu_timer.begin()

            GL.glBindVertexArray(vao)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
            GL.glBindVertexArray(0)

            if timer_started:
                gpu_timer.end()

            GL.glUseProgram(0)

            # ── HUD overlay ──────────────────────────────────────────
            _hud_state = {
                "display_mode":    display_mode,
                "effect_level":    effect_level,
                "fog_enabled":     fog_enabled,
                "blur_enabled":    blur_enabled,
                "bg_blur_enabled": bg_blur_enabled,
                "fps":             fps,
                "ai_ms":           displayed_ai_ms,
                "gpu_ms":          gpu_timer.last_ms,
            }
            hud.update(_hud_state)
            hud.render(_hud_state)

            if take_screenshot:
                save_screenshot(window)
                take_screenshot = False

            glfw.swap_buffers(window)

            gpu_ms = gpu_timer.read()

            now = time.perf_counter()
            elapsed = now - fps_start

            if elapsed >= 1.0:
                fps = fps_counter / elapsed
                fps_counter = 0
                fps_start = now

            if now - title_start >= 0.25:
                title = (
                    f"DepthFX | "
                    f"{fps:.1f} FPS | "
                    f"AI {displayed_ai_ms:.1f} ms | "
                    f"GPU {gpu_ms:.3f} ms | "
                    f"{DISPLAY_MODES[display_mode]}"
                )
                glfw.set_window_title(window, title)
                title_start = now

    except KeyboardInterrupt:
        print()
        print("Keyboard interrupt.")

    finally:
        print()
        print("Stopping DepthFX...")

        try:
            camera.release()
        except Exception:
            pass

        try:
            hud.cleanup()
        except Exception:
            pass

        try:
            gpu_timer.cleanup()
        except Exception:
            pass

        try:
            GL.glDeleteTextures([rgb_texture, depth_texture])
        except Exception:
            pass

        try:
            GL.glDeleteBuffers(1, [vbo])
        except Exception:
            pass

        try:
            GL.glDeleteVertexArrays(1, [vao])
        except Exception:
            pass

        try:
            GL.glDeleteProgram(program)
        except Exception:
            pass

        try:
            glfw.destroy_window(window)
        except Exception:
            pass

        glfw.terminate()

        print(f"Total frames processed: {total_frames}")
        print(f"AI depth updates: {depth_updates}")

        update_ratio = (
            depth_updates / total_frames * 100.0 if total_frames > 0 else 0.0
        )
        print(f"AI update ratio: {update_ratio:.1f}%")

        if gpu_timer.samples:
            samples = np.asarray(gpu_timer.samples, dtype=np.float32)
            print(f"GPU shader average: {samples.mean():.3f} ms")
            print(f"GPU shader minimum: {samples.min():.3f} ms")
            print(f"GPU shader maximum: {samples.max():.3f} ms")

        print("DepthFX GPU pipeline stopped.")


if __name__ == "__main__":
    main()