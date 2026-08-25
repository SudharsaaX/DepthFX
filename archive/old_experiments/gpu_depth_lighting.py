import ctypes
import os
import sys
import time

import cv2
import glfw
import numpy as np
import torch
from OpenGL import GL


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SRC_DIR = os.path.join(
    ROOT_DIR,
    "src",
)

CHECKPOINT_PATH = os.path.join(
    ROOT_DIR,
    "checkpoints",
    "depth_anything_v2_vits.pth",
)

SHADER_DIR = os.path.join(
    SRC_DIR,
    "shaders",
)

VERTEX_SHADER_PATH = os.path.join(
    SHADER_DIR,
    "fullscreen.vert",
)

FRAGMENT_SHADER_PATH = os.path.join(
    SHADER_DIR,
    "depth_lighting.frag",
)

sys.path.insert(
    0,
    SRC_DIR,
)


# ============================================================
# DEPTH ANYTHING V2
# ============================================================

from depth_anything_v2.dpt import DepthAnythingV2


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_INDEX = 0

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

WINDOW_WIDTH = 960
WINDOW_HEIGHT = 720

INFERENCE_SIZE = 320


# ============================================================
# SHADER UTILITIES
# ============================================================

def read_shader(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Shader not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return file.read()


def compile_shader(
    source,
    shader_type,
):
    shader = GL.glCreateShader(
        shader_type
    )

    GL.glShaderSource(
        shader,
        source,
    )

    GL.glCompileShader(
        shader
    )

    success = GL.glGetShaderiv(
        shader,
        GL.GL_COMPILE_STATUS,
    )

    if not success:
        log = GL.glGetShaderInfoLog(
            shader
        ).decode(
            "utf-8",
            errors="replace",
        )

        GL.glDeleteShader(
            shader
        )

        raise RuntimeError(
            "GLSL shader compilation failed:\n"
            + log
        )

    return shader


def create_shader_program(
    vertex_path,
    fragment_path,
):
    vertex_source = read_shader(
        vertex_path
    )

    fragment_source = read_shader(
        fragment_path
    )

    vertex_shader = compile_shader(
        vertex_source,
        GL.GL_VERTEX_SHADER,
    )

    fragment_shader = compile_shader(
        fragment_source,
        GL.GL_FRAGMENT_SHADER,
    )

    program = GL.glCreateProgram()

    GL.glAttachShader(
        program,
        vertex_shader,
    )

    GL.glAttachShader(
        program,
        fragment_shader,
    )

    GL.glLinkProgram(
        program
    )

    success = GL.glGetProgramiv(
        program,
        GL.GL_LINK_STATUS,
    )

    if not success:
        log = GL.glGetProgramInfoLog(
            program
        ).decode(
            "utf-8",
            errors="replace",
        )

        GL.glDeleteProgram(
            program
        )

        GL.glDeleteShader(
            vertex_shader
        )

        GL.glDeleteShader(
            fragment_shader
        )

        raise RuntimeError(
            "GLSL program linking failed:\n"
            + log
        )

    GL.glDetachShader(
        program,
        vertex_shader,
    )

    GL.glDetachShader(
        program,
        fragment_shader,
    )

    GL.glDeleteShader(
        vertex_shader
    )

    GL.glDeleteShader(
        fragment_shader
    )

    return program


def get_uniform(
    program,
    name,
):
    location = GL.glGetUniformLocation(
        program,
        name,
    )

    if location == -1:
        print(
            f"Warning: uniform '{name}' "
            f"not found in shader."
        )

    return location


# ============================================================
# FULLSCREEN QUAD
# ============================================================

def create_fullscreen_quad():

    vertices = np.array(
        [
            # position      # texcoord
            -1.0, -1.0,       0.0, 0.0,
             1.0, -1.0,       1.0, 0.0,
             1.0,  1.0,       1.0, 1.0,

            -1.0, -1.0,       0.0, 0.0,
             1.0,  1.0,       1.0, 1.0,
            -1.0,  1.0,       0.0, 1.0,
        ],
        dtype=np.float32,
    )

    vao = GL.glGenVertexArrays(
        1
    )

    vbo = GL.glGenBuffers(
        1
    )

    GL.glBindVertexArray(
        vao
    )

    GL.glBindBuffer(
        GL.GL_ARRAY_BUFFER,
        vbo,
    )

    GL.glBufferData(
        GL.GL_ARRAY_BUFFER,
        vertices.nbytes,
        vertices,
        GL.GL_STATIC_DRAW,
    )

    stride = (
        4 * vertices.itemsize
    )

    # Position
    GL.glEnableVertexAttribArray(
        0
    )

    GL.glVertexAttribPointer(
        0,
        2,
        GL.GL_FLOAT,
        GL.GL_FALSE,
        stride,
        ctypes.c_void_p(0),
    )

    # Texture coordinates
    GL.glEnableVertexAttribArray(
        1
    )

    GL.glVertexAttribPointer(
        1,
        2,
        GL.GL_FLOAT,
        GL.GL_FALSE,
        stride,
        ctypes.c_void_p(
            2 * vertices.itemsize
        ),
    )

    GL.glBindBuffer(
        GL.GL_ARRAY_BUFFER,
        0,
    )

    GL.glBindVertexArray(
        0
    )

    return vao, vbo


# ============================================================
# TEXTURE CREATION
# ============================================================

def create_rgb_texture(
    width,
    height,
):
    texture = GL.glGenTextures(
        1
    )

    GL.glBindTexture(
        GL.GL_TEXTURE_2D,
        texture,
    )

    GL.glTexParameteri(
        GL.GL_TEXTURE_2D,
        GL.GL_TEXTURE_MIN_FILTER,
        GL.GL_LINEAR,
    )

    GL.glTexParameteri(
        GL.GL_TEXTURE_2D,
        GL.GL_TEXTURE_MAG_FILTER,
        GL.GL_LINEAR,
    )

    GL.glTexParameteri(
        GL.GL_TEXTURE_2D,
        GL.GL_TEXTURE_WRAP_S,
        GL.GL_CLAMP_TO_EDGE,
    )

    GL.glTexParameteri(
        GL.GL_TEXTURE_2D,
        GL.GL_TEXTURE_WRAP_T,
        GL.GL_CLAMP_TO_EDGE,
    )

    GL.glTexImage2D(
        GL.GL_TEXTURE_2D,
        0,
        GL.GL_RGB8,
        width,
        height,
        0,
        GL.GL_RGB,
        GL.GL_UNSIGNED_BYTE,
        None,
    )

    GL.glBindTexture(
        GL.GL_TEXTURE_2D,
        0,
    )

    return texture


def create_depth_texture(
    width,
    height,
):
    texture = GL.glGenTextures(
        1
    )

    GL.glBindTexture(
        GL.GL_TEXTURE_2D,
        texture,
    )

    GL.glTexParameteri(
        GL.GL_TEXTURE_2D,
        GL.GL_TEXTURE_MIN_FILTER,
        GL.GL_LINEAR,
    )

    GL.glTexParameteri(
        GL.GL_TEXTURE_2D,
        GL.GL_TEXTURE_MAG_FILTER,
        GL.GL_LINEAR,
    )

    GL.glTexParameteri(
        GL.GL_TEXTURE_2D,
        GL.GL_TEXTURE_WRAP_S,
        GL.GL_CLAMP_TO_EDGE,
    )

    GL.glTexParameteri(
        GL.GL_TEXTURE_2D,
        GL.GL_TEXTURE_WRAP_T,
        GL.GL_CLAMP_TO_EDGE,
    )

    GL.glTexImage2D(
        GL.GL_TEXTURE_2D,
        0,
        GL.GL_R32F,
        width,
        height,
        0,
        GL.GL_RED,
        GL.GL_FLOAT,
        None,
    )

    GL.glBindTexture(
        GL.GL_TEXTURE_2D,
        0,
    )

    return texture


# ============================================================
# DEPTH ESTIMATOR
# ============================================================

class DepthEstimator:

    def __init__(self):

        print(
            "Loading Depth Anything V2 Small..."
        )

        if not os.path.exists(
            CHECKPOINT_PATH
        ):
            raise FileNotFoundError(
                "Depth checkpoint not found:\n"
                + CHECKPOINT_PATH
            )

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"Device: {self.device}"
        )

        if self.device == "cuda":

            print(
                "GPU:",
                torch.cuda.get_device_name(0),
            )

            print(
                "CUDA:",
                torch.version.cuda,
            )

            print(
                "Compute capability:",
                torch.cuda.get_device_capability(0),
            )

        print(
            f"AI inference size: "
            f"{INFERENCE_SIZE}"
        )

        self.model = DepthAnythingV2(
            encoder="vits",
            features=64,
            out_channels=[
                48,
                96,
                192,
                384,
            ],
        )

        print(
            "Loading checkpoint..."
        )

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location="cpu",
        )

        if isinstance(
            checkpoint,
            dict
        ) and "state_dict" in checkpoint:

            checkpoint = checkpoint[
                "state_dict"
            ]

        self.model.load_state_dict(
            checkpoint,
            strict=True,
        )

        self.model = self.model.to(
            self.device
        )

        self.model.eval()

        if self.device == "cuda":
            print(
                "Using FP16 autocast inference."
            )
        else:
            print(
                "Using FP32 CPU inference."
            )

        print(
            "Depth model loaded successfully."
        )

        self.warmup()


    def warmup(self):

        print(
            "Warming up GPU..."
        )

        dummy = np.zeros(
            (
                CAMERA_HEIGHT,
                CAMERA_WIDTH,
                3,
            ),
            dtype=np.uint8,
        )

        for _ in range(3):

            with torch.inference_mode():

                if self.device == "cuda":

                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                    ):
                        self.model.infer_image(
                            dummy,
                            INFERENCE_SIZE,
                        )

                else:

                    self.model.infer_image(
                        dummy,
                        INFERENCE_SIZE,
                    )

        if self.device == "cuda":
            torch.cuda.synchronize()

        print(
            "GPU warm-up complete."
        )


    def estimate(
        self,
        frame,
    ):

        if (
            frame is None
            or frame.size == 0
        ):
            raise ValueError(
                "Invalid camera frame."
            )

        # Depth Anything V2 expects BGR
        # in this implementation.
        input_frame = frame

        with torch.inference_mode():

            if self.device == "cuda":

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):

                    raw_depth = (
                        self.model.infer_image(
                            input_frame,
                            INFERENCE_SIZE,
                        )
                    )

            else:

                raw_depth = (
                    self.model.infer_image(
                        input_frame,
                        INFERENCE_SIZE,
                    )
                )

        depth = np.asarray(
            raw_depth,
            dtype=np.float32,
        )

        depth_min = float(
            depth.min()
        )

        depth_max = float(
            depth.max()
        )

        depth = (
            depth - depth_min
        )

        if depth_max - depth_min > 1e-8:

            depth /= (
                depth_max
                - depth_min
            )

        depth = np.clip(
            depth,
            0.0,
            1.0,
        )

        depth = cv2.resize(
            depth,
            (
                CAMERA_WIDTH,
                CAMERA_HEIGHT,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        return np.ascontiguousarray(
            depth,
            dtype=np.float32,
        )


# ============================================================
# GLFW INITIALIZATION
# ============================================================

def create_window():

    if not glfw.init():
        raise RuntimeError(
            "Failed to initialize GLFW."
        )

    glfw.window_hint(
        glfw.CONTEXT_VERSION_MAJOR,
        3,
    )

    glfw.window_hint(
        glfw.CONTEXT_VERSION_MINOR,
        3,
    )

    glfw.window_hint(
        glfw.OPENGL_PROFILE,
        glfw.OPENGL_CORE_PROFILE,
    )

    window = glfw.create_window(
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
        "DepthFX - GPU Depth Lighting",
        None,
        None,
    )

    if not window:

        glfw.terminate()

        raise RuntimeError(
            "Failed to create OpenGL window."
        )

    glfw.make_context_current(
        window
    )

    # Disable VSync so we can measure
    # actual pipeline performance.
    glfw.swap_interval(0)

    return window


# ============================================================
# GPU INFORMATION
# ============================================================

def print_gpu_information():

    vendor = GL.glGetString(
        GL.GL_VENDOR
    )

    renderer = GL.glGetString(
        GL.GL_RENDERER
    )

    version = GL.glGetString(
        GL.GL_VERSION
    )

    print(
        "OpenGL Vendor:",
        vendor.decode()
        if vendor
        else "Unknown",
    )

    print(
        "OpenGL Renderer:",
        renderer.decode()
        if renderer
        else "Unknown",
    )

    print(
        "OpenGL Version:",
        version.decode()
        if version
        else "Unknown",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "DepthFX - GPU Depth-Aware Lighting"
    )

    print("=" * 60)

    print()


    window = None
    camera = None

    program = None

    vao = None
    vbo = None

    rgb_texture = None
    depth_texture = None


    try:

        # ====================================================
        # DEPTH MODEL
        # ====================================================

        estimator = DepthEstimator()


        # ====================================================
        # OPENGL
        # ====================================================

        print()
        print(
            "Initializing OpenGL..."
        )

        window = create_window()

        print_gpu_information()


        # ====================================================
        # SHADER
        # ====================================================

        print()
        print(
            "Loading GPU lighting shader..."
        )

        program = create_shader_program(
            VERTEX_SHADER_PATH,
            FRAGMENT_SHADER_PATH,
        )

        print(
            "GPU lighting shader created successfully."
        )


        # ====================================================
        # QUAD
        # ====================================================

        vao, vbo = (
            create_fullscreen_quad()
        )


        # ====================================================
        # CAMERA
        # ====================================================

        print()
        print(
            "Opening webcam..."
        )

        camera = cv2.VideoCapture(
            CAMERA_INDEX
        )

        camera.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            CAMERA_WIDTH,
        )

        camera.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            CAMERA_HEIGHT,
        )

        if not camera.isOpened():

            raise RuntimeError(
                "Could not open webcam."
            )

        actual_width = int(
            camera.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        actual_height = int(
            camera.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        print(
            f"Camera resolution: "
            f"{actual_width}x"
            f"{actual_height}"
        )


        # ====================================================
        # TEXTURES
        # ====================================================

        rgb_texture = create_rgb_texture(
            actual_width,
            actual_height,
        )

        depth_texture = create_depth_texture(
            actual_width,
            actual_height,
        )

        print(
            "RGB texture created."
        )

        print(
            "R32F depth texture created."
        )


        # ====================================================
        # SHADER UNIFORMS
        # ====================================================

        GL.glUseProgram(
            program
        )

        color_location = get_uniform(
            program,
            "u_color",
        )

        depth_location = get_uniform(
            program,
            "u_depth",
        )

        texel_size_location = get_uniform(
            program,
            "u_texel_size",
        )

        light_position_location = get_uniform(
            program,
            "u_light_position",
        )

        light_strength_location = get_uniform(
            program,
            "u_light_strength",
        )

        ambient_strength_location = get_uniform(
            program,
            "u_ambient_strength",
        )


        # ====================================================
        # INITIAL UNIFORMS
        # ====================================================

        # Texture 0 = RGB
        if color_location >= 0:

            GL.glUniform1i(
                color_location,
                0,
            )


        # Texture 1 = depth
        if depth_location >= 0:

            GL.glUniform1i(
                depth_location,
                1,
            )


        if texel_size_location >= 0:

            GL.glUniform2f(
                texel_size_location,
                1.0 / actual_width,
                1.0 / actual_height,
            )


        # Light begins at upper-left.
        if light_position_location >= 0:

            GL.glUniform2f(
                light_position_location,
                0.25,
                0.25,
            )


        light_strength = 1.5

        ambient_strength = 0.45


        if light_strength_location >= 0:

            GL.glUniform1f(
                light_strength_location,
                light_strength,
            )


        if ambient_strength_location >= 0:

            GL.glUniform1f(
                ambient_strength_location,
                ambient_strength,
            )


        # IMPORTANT:
        # We finish all uniform configuration
        # while the shader program is active.

        GL.glUseProgram(
            0
        )


        # ====================================================
        # OPENGL STATE
        # ====================================================

        GL.glDisable(
            GL.GL_DEPTH_TEST
        )

        GL.glClearColor(
            0.0,
            0.0,
            0.0,
            1.0,
        )


        # ====================================================
        # CONTROLS
        # ====================================================

        print()
        print(
            "Starting GPU depth-aware lighting..."
        )

        print()

        print(
            "Controls:"
        )

        print(
            "  1 = light lighting"
        )

        print(
            "  2 = medium lighting"
        )

        print(
            "  3 = strong lighting"
        )

        print(
            "  R = reset lighting"
        )

        print(
            "  Q = quit"
        )

        print()

        print(
            "RGB + AI Depth → GLSL Lighting → RTX 4070"
        )

        print()


        # ====================================================
        # PERFORMANCE
        # ====================================================

        total_frames = 0

        fps_counter = 0

        fps = 0.0

        fps_timer = time.perf_counter()

        last_title_time = (
            time.perf_counter()
        )


        # ====================================================
        # KEY STATE
        # ====================================================

        previous_keys = {
            glfw.KEY_1: False,
            glfw.KEY_2: False,
            glfw.KEY_3: False,
            glfw.KEY_R: False,
            glfw.KEY_Q: False,
        }


        def key_pressed(key):

            current = (
                glfw.get_key(
                    window,
                    key,
                )
                == glfw.PRESS
            )

            previous = (
                previous_keys[key]
            )

            previous_keys[key] = current

            return (
                current
                and not previous
            )


        # ====================================================
        # MAIN LOOP
        # ====================================================

        while not glfw.window_should_close(
            window
        ):

            glfw.poll_events()


            # =================================================
            # QUIT
            # =================================================

            if key_pressed(
                glfw.KEY_Q
            ):
                break


            # =================================================
            # LIGHT LEVEL
            # =================================================

            if key_pressed(
                glfw.KEY_1
            ):

                light_strength = 0.8
                ambient_strength = 0.60


            if key_pressed(
                glfw.KEY_2
            ):

                light_strength = 1.5
                ambient_strength = 0.45


            if key_pressed(
                glfw.KEY_3
            ):

                light_strength = 2.5
                ambient_strength = 0.30


            if key_pressed(
                glfw.KEY_R
            ):

                light_strength = 1.5
                ambient_strength = 0.45


            # =================================================
            # CAMERA
            # =================================================

            success, frame = (
                camera.read()
            )

            if not success:
                continue


            frame = cv2.resize(
                frame,
                (
                    actual_width,
                    actual_height,
                ),
                interpolation=cv2.INTER_LINEAR,
            )


            # =================================================
            # AI DEPTH
            # =================================================

            ai_start = time.perf_counter()

            depth = estimator.estimate(
                frame
            )

            if (
                estimator.device
                == "cuda"
            ):

                torch.cuda.synchronize()

            ai_ms = (
                time.perf_counter()
                - ai_start
            ) * 1000.0


            # =================================================
            # RGB DATA
            # =================================================

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            rgb = np.ascontiguousarray(
                rgb
            )


            # =================================================
            # GPU UPLOAD
            # =================================================

            upload_start = (
                time.perf_counter()
            )


            # RGB
            GL.glActiveTexture(
                GL.GL_TEXTURE0
            )

            GL.glBindTexture(
                GL.GL_TEXTURE_2D,
                rgb_texture,
            )

            GL.glTexSubImage2D(
                GL.GL_TEXTURE_2D,
                0,
                0,
                0,
                actual_width,
                actual_height,
                GL.GL_RGB,
                GL.GL_UNSIGNED_BYTE,
                rgb,
            )


            # Depth
            GL.glActiveTexture(
                GL.GL_TEXTURE1
            )

            GL.glBindTexture(
                GL.GL_TEXTURE_2D,
                depth_texture,
            )

            GL.glTexSubImage2D(
                GL.GL_TEXTURE_2D,
                0,
                0,
                0,
                actual_width,
                actual_height,
                GL.GL_RED,
                GL.GL_FLOAT,
                depth,
            )


            upload_ms = (
                time.perf_counter()
                - upload_start
            ) * 1000.0


            # =================================================
            # GPU RENDER
            # =================================================

            gpu_start = (
                time.perf_counter()
            )


            GL.glViewport(
                0,
                0,
                WINDOW_WIDTH,
                WINDOW_HEIGHT,
            )


            GL.glClear(
                GL.GL_COLOR_BUFFER_BIT
            )


            GL.glUseProgram(
                program
            )


            # -------------------------------------------------
            # Update lighting uniforms
            # -------------------------------------------------

            if light_strength_location >= 0:

                GL.glUniform1f(
                    light_strength_location,
                    light_strength,
                )


            if ambient_strength_location >= 0:

                GL.glUniform1f(
                    ambient_strength_location,
                    ambient_strength,
                )


            # -------------------------------------------------
            # Bind RGB
            # -------------------------------------------------

            GL.glActiveTexture(
                GL.GL_TEXTURE0
            )

            GL.glBindTexture(
                GL.GL_TEXTURE_2D,
                rgb_texture,
            )


            # -------------------------------------------------
            # Bind depth
            # -------------------------------------------------

            GL.glActiveTexture(
                GL.GL_TEXTURE1
            )

            GL.glBindTexture(
                GL.GL_TEXTURE_2D,
                depth_texture,
            )


            # -------------------------------------------------
            # Draw
            # -------------------------------------------------

            GL.glBindVertexArray(
                vao
            )

            GL.glDrawArrays(
                GL.GL_TRIANGLES,
                0,
                6,
            )

            GL.glBindVertexArray(
                0
            )


            GL.glUseProgram(
                0
            )


            # -------------------------------------------------
            # Present
            # -------------------------------------------------

            glfw.swap_buffers(
                window
            )


            gpu_ms = (
                time.perf_counter()
                - gpu_start
            ) * 1000.0


            # =================================================
            # FPS
            # =================================================

            total_frames += 1
            fps_counter += 1

            now = time.perf_counter()

            fps_elapsed = (
                now - fps_timer
            )

            if fps_elapsed >= 1.0:

                fps = (
                    fps_counter
                    / fps_elapsed
                )

                fps_counter = 0

                fps_timer = now


            # =================================================
            # WINDOW TITLE
            # =================================================

            if (
                now - last_title_time
                >= 0.20
            ):

                title = (
                    f"DepthFX | "
                    f"{fps:.1f} FPS | "
                    f"AI {ai_ms:.1f}ms | "
                    f"Upload {upload_ms:.1f}ms | "
                    f"GPU {gpu_ms:.1f}ms | "
                    f"DEPTH LIGHTING"
                )

                glfw.set_window_title(
                    window,
                    title,
                )

                last_title_time = now


    except KeyboardInterrupt:

        print(
            "\nKeyboard interrupt."
        )


    finally:

        # ====================================================
        # CLEANUP
        # ====================================================

        print()
        print(
            "Stopping DepthFX lighting..."
        )

        if camera is not None:
            camera.release()


        if window is not None:

            if rgb_texture is not None:

                GL.glDeleteTextures(
                    [
                        rgb_texture
                    ]
                )


            if depth_texture is not None:

                GL.glDeleteTextures(
                    [
                        depth_texture
                    ]
                )


            if vbo is not None:

                GL.glDeleteBuffers(
                    1,
                    [
                        vbo
                    ],
                )


            if vao is not None:

                GL.glDeleteVertexArrays(
                    1,
                    [
                        vao
                    ],
                )


            if program is not None:

                GL.glDeleteProgram(
                    program
                )


            glfw.destroy_window(
                window
            )

            glfw.terminate()


        print(
            f"Total frames processed: "
            f"{total_frames}"
        )

        print(
            "GPU depth-aware lighting stopped."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()