import ctypes
import time
from pathlib import Path

import cv2
import glfw
import numpy as np
from OpenGL import GL

from depth_estimator import DepthEstimator


BASE_DIR = Path(__file__).resolve().parent.parent

VERTEX_SHADER_PATH = (
    BASE_DIR
    / "src"
    / "shaders"
    / "fullscreen.vert"
)

FRAGMENT_SHADER_PATH = (
    BASE_DIR
    / "src"
    / "shaders"
    / "depth_blur.frag"
)

WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480


def read_shader(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"Shader not found: {path}"
        )

    return path.read_text(
        encoding="utf-8"
    )


def compile_shader(source, shader_type):
    shader = GL.glCreateShader(shader_type)

    GL.glShaderSource(
        shader,
        source
    )

    GL.glCompileShader(shader)

    success = GL.glGetShaderiv(
        shader,
        GL.GL_COMPILE_STATUS
    )

    if not success:
        log = GL.glGetShaderInfoLog(
            shader
        ).decode(
            "utf-8",
            errors="replace"
        )

        GL.glDeleteShader(shader)

        raise RuntimeError(
            "Shader compilation failed:\n"
            + log
        )

    return shader


def create_shader_program():
    vertex_source = read_shader(
        VERTEX_SHADER_PATH
    )

    fragment_source = read_shader(
        FRAGMENT_SHADER_PATH
    )

    vertex_shader = compile_shader(
        vertex_source,
        GL.GL_VERTEX_SHADER
    )

    fragment_shader = compile_shader(
        fragment_source,
        GL.GL_FRAGMENT_SHADER
    )

    program = GL.glCreateProgram()

    GL.glAttachShader(
        program,
        vertex_shader
    )

    GL.glAttachShader(
        program,
        fragment_shader
    )

    GL.glLinkProgram(program)

    success = GL.glGetProgramiv(
        program,
        GL.GL_LINK_STATUS
    )

    if not success:
        log = GL.glGetProgramInfoLog(
            program
        ).decode(
            "utf-8",
            errors="replace"
        )

        GL.glDeleteProgram(program)

        raise RuntimeError(
            "Shader linking failed:\n"
            + log
        )

    GL.glDeleteShader(vertex_shader)
    GL.glDeleteShader(fragment_shader)

    return program


def create_fullscreen_quad():
    vertices = np.array(
        [
            -1.0, -1.0, 0.0, 1.0,
             1.0, -1.0, 1.0, 1.0,
             1.0,  1.0, 1.0, 0.0,

            -1.0, -1.0, 0.0, 1.0,
             1.0,  1.0, 1.0, 0.0,
            -1.0,  1.0, 0.0, 0.0,
        ],
        dtype=np.float32
    )

    vao = GL.glGenVertexArrays(1)
    vbo = GL.glGenBuffers(1)

    GL.glBindVertexArray(vao)

    GL.glBindBuffer(
        GL.GL_ARRAY_BUFFER,
        vbo
    )

    GL.glBufferData(
        GL.GL_ARRAY_BUFFER,
        vertices.nbytes,
        vertices,
        GL.GL_STATIC_DRAW
    )

    stride = 4 * vertices.itemsize

    # Position
    GL.glVertexAttribPointer(
        0,
        2,
        GL.GL_FLOAT,
        GL.GL_FALSE,
        stride,
        ctypes.c_void_p(0)
    )

    GL.glEnableVertexAttribArray(0)

    # UV
    GL.glVertexAttribPointer(
        1,
        2,
        GL.GL_FLOAT,
        GL.GL_FALSE,
        stride,
        ctypes.c_void_p(
            2 * vertices.itemsize
        )
    )

    GL.glEnableVertexAttribArray(1)

    GL.glBindBuffer(
        GL.GL_ARRAY_BUFFER,
        0
    )

    GL.glBindVertexArray(0)

    return vao, vbo


def create_color_texture(width, height):
    texture = GL.glGenTextures(1)

    GL.glBindTexture(
        GL.GL_TEXTURE_2D,
        texture
    )

    GL.glTexParameteri(
        GL.GL_TEXTURE_2D,
        GL.GL_TEXTURE_MIN_FILTER,
        GL.GL_LINEAR
    )

    GL.glTexParameteri(
        GL.GL_TEXTURE_2D,
        GL.GL_TEXTURE_MAG_FILTER,
        GL.GL_LINEAR
    )

    GL.glTexParameteri(
        GL.GL_TEXTURE_2D,
        GL.GL_TEXTURE_WRAP_S,
        GL.GL_CLAMP_TO_EDGE
    )

    GL.glTexParameteri(
        GL.GL_TEXTURE_2D,
        GL.GL_TEXTURE_WRAP_T,
        GL.GL_CLAMP_TO_EDGE
    )

    GL.glTexImage2D(
        GL.GL_TEXTURE_2D,
        0,
        GL.GL_RGB8,
        width,
        height,
        0,
        GL.GL_BGR,
        GL.GL_UNSIGNED_BYTE,
        None
    )

    GL.glBindTexture(
        GL.GL_TEXTURE_2D,
        0
    )

    return texture


def upload_color_texture(texture, frame):
    frame = np.ascontiguousarray(frame)

    height, width = frame.shape[:2]

    GL.glBindTexture(
        GL.GL_TEXTURE_2D,
        texture
    )

    GL.glTexSubImage2D(
        GL.GL_TEXTURE_2D,
        0,
        0,
        0,
        width,
        height,
        GL.GL_BGR,
        GL.GL_UNSIGNED_BYTE,
        frame
    )

    GL.glBindTexture(
        GL.GL_TEXTURE_2D,
        0
    )


def create_depth_texture(width, height):
    texture = GL.glGenTextures(1)

    GL.glBindTexture(
        GL.GL_TEXTURE_2D,
        texture
    )

    GL.glTexParameteri(
        GL.GL_TEXTURE_2D,
        GL.GL_TEXTURE_MIN_FILTER,
        GL.GL_LINEAR
    )

    GL.glTexParameteri(
        GL.GL_TEXTURE_2D,
        GL.GL_TEXTURE_MAG_FILTER,
        GL.GL_LINEAR
    )

    GL.glTexParameteri(
        GL.GL_TEXTURE_2D,
        GL.GL_TEXTURE_WRAP_S,
        GL.GL_CLAMP_TO_EDGE
    )

    GL.glTexParameteri(
        GL.GL_TEXTURE_2D,
        GL.GL_TEXTURE_WRAP_T,
        GL.GL_CLAMP_TO_EDGE
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
        None
    )

    GL.glBindTexture(
        GL.GL_TEXTURE_2D,
        0
    )

    return texture


def upload_depth_texture(texture, depth):
    depth = np.asarray(
        depth,
        dtype=np.float32
    )

    depth = np.clip(
        depth,
        0.0,
        1.0
    )

    depth = np.ascontiguousarray(depth)

    height, width = depth.shape

    GL.glBindTexture(
        GL.GL_TEXTURE_2D,
        texture
    )

    GL.glTexSubImage2D(
        GL.GL_TEXTURE_2D,
        0,
        0,
        0,
        width,
        height,
        GL.GL_RED,
        GL.GL_FLOAT,
        depth
    )

    GL.glBindTexture(
        GL.GL_TEXTURE_2D,
        0
    )


def get_uniform(program, name):
    location = GL.glGetUniformLocation(
        program,
        name
    )

    if location == -1:
        raise RuntimeError(
            f"Uniform not found: {name}"
        )

    return location


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
        if vendor else "Unknown"
    )

    print(
        "OpenGL Renderer:",
        renderer.decode()
        if renderer else "Unknown"
    )

    print(
        "OpenGL Version:",
        version.decode()
        if version else "Unknown"
    )


def main():

    print("=" * 60)
    print(
        "DepthFX - GPU Depth-Aware Blur"
    )
    print("=" * 60)

    # --------------------------------------------------
    # Depth model
    # --------------------------------------------------

    print()
    print(
        "Loading Depth Anything V2..."
    )

    estimator = DepthEstimator()

    # --------------------------------------------------
    # OpenGL
    # --------------------------------------------------

    print()
    print(
        "Initializing OpenGL..."
    )

    if not glfw.init():
        raise RuntimeError(
            "Failed to initialize GLFW."
        )

    glfw.window_hint(
        glfw.CONTEXT_VERSION_MAJOR,
        3
    )

    glfw.window_hint(
        glfw.CONTEXT_VERSION_MINOR,
        3
    )

    glfw.window_hint(
        glfw.OPENGL_PROFILE,
        glfw.OPENGL_CORE_PROFILE
    )

    window = glfw.create_window(
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
        "DepthFX - GPU Blur",
        None,
        None
    )

    if not window:
        glfw.terminate()

        raise RuntimeError(
            "Failed to create OpenGL window."
        )

    glfw.make_context_current(window)

    # Disable VSync.
    glfw.swap_interval(0)

    print_gpu_information()

    # --------------------------------------------------
    # Shader
    # --------------------------------------------------

    print()
    print(
        "Loading GPU blur shader..."
    )

    program = create_shader_program()

    print(
        "GPU blur shader created successfully."
    )

    # --------------------------------------------------
    # Geometry
    # --------------------------------------------------

    vao, vbo = create_fullscreen_quad()

    # --------------------------------------------------
    # Webcam
    # --------------------------------------------------

    print()
    print(
        "Opening webcam..."
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam."
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        WINDOW_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        WINDOW_HEIGHT
    )

    success, frame = cap.read()

    if not success:
        cap.release()

        raise RuntimeError(
            "Could not read webcam frame."
        )

    height, width = frame.shape[:2]

    print(
        f"Camera resolution: "
        f"{width}x{height}"
    )

    # --------------------------------------------------
    # Textures
    # --------------------------------------------------

    color_texture = create_color_texture(
        width,
        height
    )

    depth_texture = create_depth_texture(
        width,
        height
    )

    print(
        "RGB texture created."
    )

    print(
        "R32F depth texture created."
    )

    # --------------------------------------------------
    # Uniforms
    # --------------------------------------------------

    color_location = get_uniform(
        program,
        "u_color"
    )

    depth_location = get_uniform(
        program,
        "u_depth"
    )

    blur_strength_location = get_uniform(
        program,
        "u_blur_strength"
    )

    depth_threshold_location = get_uniform(
        program,
        "u_depth_threshold"
    )

    texel_size_location = get_uniform(
        program,
        "u_texel_size"
    )

    # --------------------------------------------------
    # Settings
    # --------------------------------------------------

    blur_strength = 1.0

    depth_threshold = 0.45

    blur_enabled = True

    # --------------------------------------------------
    # FPS
    # --------------------------------------------------

    frame_count = 0

    fps_frames = 0

    fps = 0.0

    fps_start = time.perf_counter()

    print()
    print(
        "Starting GPU depth-aware blur..."
    )

    print(
        "RGB + Depth → GLSL → RTX 4070"
    )

    print()
    print(
        "Controls:"
    )

    print(
        "  F = toggle blur"
    )

    print(
        "  1 = light blur"
    )

    print(
        "  2 = medium blur"
    )

    print(
        "  3 = strong blur"
    )

    print(
        "  R = reset"
    )

    print(
        "  Q = quit"
    )

    print()

    # --------------------------------------------------
    # Main loop
    # --------------------------------------------------

    while not glfw.window_should_close(
        window
    ):

        glfw.poll_events()

        # --------------------------------------------------
        # Keyboard
        # --------------------------------------------------

        if glfw.get_key(
            window,
            glfw.KEY_Q
        ) == glfw.PRESS:
            break

        if glfw.get_key(
            window,
            glfw.KEY_F
        ) == glfw.PRESS:

            blur_enabled = not blur_enabled

            time.sleep(0.15)

        if glfw.get_key(
            window,
            glfw.KEY_1
        ) == glfw.PRESS:

            blur_strength = 0.35

        if glfw.get_key(
            window,
            glfw.KEY_2
        ) == glfw.PRESS:

            blur_strength = 0.70

        if glfw.get_key(
            window,
            glfw.KEY_3
        ) == glfw.PRESS:

            blur_strength = 1.0

        if glfw.get_key(
            window,
            glfw.KEY_R
        ) == glfw.PRESS:

            blur_strength = 1.0
            depth_threshold = 0.45
            blur_enabled = True

            time.sleep(0.15)

        # --------------------------------------------------
        # Capture
        # --------------------------------------------------

        success, frame = cap.read()

        if not success:
            print(
                "ERROR: Failed to read webcam."
            )
            break

        frame_count += 1

        # --------------------------------------------------
        # AI depth
        # --------------------------------------------------

        depth = estimator.estimate(
            frame
        )

        if depth.shape != (
            height,
            width
        ):

            depth = cv2.resize(
                depth,
                (
                    width,
                    height
                ),
                interpolation=cv2.INTER_LINEAR
            )

        depth = np.asarray(
            depth,
            dtype=np.float32
        )

        depth = np.clip(
            depth,
            0.0,
            1.0
        )

        # --------------------------------------------------
        # Upload
        # --------------------------------------------------

        upload_color_texture(
            color_texture,
            frame
        )

        upload_depth_texture(
            depth_texture,
            depth
        )

        # --------------------------------------------------
        # Render
        # --------------------------------------------------

        GL.glClearColor(
            0.0,
            0.0,
            0.0,
            1.0
        )

        GL.glClear(
            GL.GL_COLOR_BUFFER_BIT
        )

        GL.glUseProgram(program)

        # RGB → unit 0
        GL.glActiveTexture(
            GL.GL_TEXTURE0
        )

        GL.glBindTexture(
            GL.GL_TEXTURE_2D,
            color_texture
        )

        GL.glUniform1i(
            color_location,
            0
        )

        # Depth → unit 1
        GL.glActiveTexture(
            GL.GL_TEXTURE1
        )

        GL.glBindTexture(
            GL.GL_TEXTURE_2D,
            depth_texture
        )

        GL.glUniform1i(
            depth_location,
            1
        )

        # Blur strength.
        if blur_enabled:
            strength = blur_strength
        else:
            strength = 0.0

        GL.glUniform1f(
            blur_strength_location,
            strength
        )

        # Depth threshold.
        GL.glUniform1f(
            depth_threshold_location,
            depth_threshold
        )

        # One texel in normalized UV coordinates.
        GL.glUniform2f(
            texel_size_location,
            1.0 / width,
            1.0 / height
        )

        # Draw.
        GL.glBindVertexArray(vao)

        GL.glDrawArrays(
            GL.GL_TRIANGLES,
            0,
            6
        )

        GL.glBindVertexArray(0)

        # --------------------------------------------------
        # FPS
        # --------------------------------------------------

        fps_frames += 1

        current_time = time.perf_counter()

        elapsed = (
            current_time
            - fps_start
        )

        if elapsed >= 0.5:

            fps = (
                fps_frames
                / elapsed
            )

            fps_frames = 0
            fps_start = current_time

            state = (
                "ON"
                if blur_enabled
                else "OFF"
            )

            glfw.set_window_title(
                window,
                f"DepthFX - GPU Blur | "
                f"{fps:.1f} FPS | "
                f"Blur {state} | "
                f"Strength {blur_strength:.2f}"
            )

        glfw.swap_buffers(window)

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------

    cap.release()

    GL.glBindTexture(
        GL.GL_TEXTURE_2D,
        0
    )

    GL.glDeleteTextures(
        [
            color_texture,
            depth_texture
        ]
    )

    GL.glDeleteBuffers(
        1,
        [vbo]
    )

    GL.glDeleteVertexArrays(
        1,
        [vao]
    )

    GL.glDeleteProgram(program)

    glfw.destroy_window(window)

    glfw.terminate()

    print()
    print(
        f"Total frames processed: "
        f"{frame_count}"
    )

    print(
        "GPU depth-aware blur stopped."
    )


if __name__ == "__main__":
    main()