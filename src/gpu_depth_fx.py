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
    color = applyLighting(color, depth, uv);
    float edge = depthEdge(uv);
    color += edge * 0.035;
    color = applyFog(color, depth);
    color = clamp(color, 0.0, 1.0);

    FragColor = vec4(color, 1.0);
}
"""


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
    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print(f"Camera resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")

    rgb_texture = create_rgb_texture()
    depth_texture = create_depth_texture()

    print("RGB texture created.")
    print("R32F depth texture created.")

    gpu_timer = GPUTimer()

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
    take_screenshot = False
    mouse_x = 0.5
    mouse_y = 0.5

    def key_callback(window_handle, key, scancode, action, mods):
        nonlocal fog_enabled, blur_enabled, effect_level, display_mode, take_screenshot

        if action != glfw.PRESS:
            return

        if key == glfw.KEY_Q:
            glfw.set_window_should_close(window_handle, True)
        elif key == glfw.KEY_S:
            take_screenshot = True
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
    print("RGB + AI Depth → Temporal Depth → GLSL Fog + Blur + Lighting → RTX 4070")
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

            GL.glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
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

            if u_time >= 0:
                GL.glUniform1f(u_time, float(time.perf_counter()))

            timer_started = gpu_timer.begin()

            GL.glBindVertexArray(vao)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
            GL.glBindVertexArray(0)

            if timer_started:
                gpu_timer.end()

            GL.glUseProgram(0)

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