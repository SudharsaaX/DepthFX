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
    / "depth_view.frag"
)


def read_shader(path: Path) -> str:
    """Read a GLSL shader file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Shader not found: {path}"
        )

    return path.read_text(
        encoding="utf-8"
    )


def compile_shader(
    source: str,
    shader_type: int,
) -> int:
    """Compile a GLSL shader."""

    shader = GL.glCreateShader(
        shader_type
    )

    GL.glShaderSource(
        shader,
        source,
    )

    GL.glCompileShader(shader)

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

        GL.glDeleteShader(shader)

        raise RuntimeError(
            "Shader compilation failed:\n"
            + log
        )

    return shader


def create_shader_program() -> int:
    """Create and link the depth visualization shader."""

    vertex_source = read_shader(
        VERTEX_SHADER_PATH
    )

    fragment_source = read_shader(
        FRAGMENT_SHADER_PATH
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

    GL.glLinkProgram(program)

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

        GL.glDeleteProgram(program)
        GL.glDeleteShader(vertex_shader)
        GL.glDeleteShader(fragment_shader)

        raise RuntimeError(
            "Shader linking failed:\n"
            + log
        )

    GL.glDeleteShader(vertex_shader)
    GL.glDeleteShader(fragment_shader)

    return program


def create_fullscreen_quad():
    """Create a fullscreen rectangle."""

    vertices = np.array(
        [
            # Position       Texture coordinates
            -1.0, -1.0,       0.0, 1.0,
             1.0, -1.0,       1.0, 1.0,
             1.0,  1.0,       1.0, 0.0,

            -1.0, -1.0,       0.0, 1.0,
             1.0,  1.0,       1.0, 0.0,
            -1.0,  1.0,       0.0, 0.0,
        ],
        dtype=np.float32,
    )

    vao = GL.glGenVertexArrays(1)
    vbo = GL.glGenBuffers(1)

    GL.glBindVertexArray(vao)

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

    stride = 4 * vertices.itemsize

    # Position
    GL.glVertexAttribPointer(
        0,
        2,
        GL.GL_FLOAT,
        GL.GL_FALSE,
        stride,
        ctypes.c_void_p(0),
    )

    GL.glEnableVertexAttribArray(0)

    # Texture coordinates
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

    GL.glEnableVertexAttribArray(1)

    GL.glBindBuffer(
        GL.GL_ARRAY_BUFFER,
        0,
    )

    GL.glBindVertexArray(0)

    return vao, vbo


def create_depth_texture(
    width: int,
    height: int,
) -> int:
    """Create a floating-point R32F depth texture."""

    texture = GL.glGenTextures(1)

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


def upload_depth_texture(
    texture: int,
    depth: np.ndarray,
):
    """Upload a normalized float32 depth map to the GPU."""

    if depth.ndim != 2:
        raise ValueError(
            f"Expected 2D depth map, got shape {depth.shape}"
        )

    depth = np.asarray(
        depth,
        dtype=np.float32,
    )

    depth = np.clip(
        depth,
        0.0,
        1.0,
    )

    depth = np.ascontiguousarray(
        depth
    )

    height, width = depth.shape

    GL.glBindTexture(
        GL.GL_TEXTURE_2D,
        texture,
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
        depth,
    )

    GL.glBindTexture(
        GL.GL_TEXTURE_2D,
        0,
    )


def print_gpu_information():
    """Print the active OpenGL GPU."""

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
        vendor.decode("utf-8")
        if vendor
        else "Unknown",
    )

    print(
        "OpenGL Renderer:",
        renderer.decode("utf-8")
        if renderer
        else "Unknown",
    )

    print(
        "OpenGL Version:",
        version.decode("utf-8")
        if version
        else "Unknown",
    )


def main():

    print("=" * 60)
    print(
        "DepthFX - Real Depth → GPU Texture"
    )
    print("=" * 60)

    # --------------------------------------------------
    # Load Depth Anything V2
    # --------------------------------------------------

    print()
    print(
        "Loading Depth Anything V2..."
    )

    estimator = DepthEstimator()

    # --------------------------------------------------
    # Initialize GLFW
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
        640,
        480,
        "DepthFX - Real GPU Depth",
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

    # Disable VSync so we can measure performance.
    glfw.swap_interval(0)

    print_gpu_information()

    # --------------------------------------------------
    # Create shader
    # --------------------------------------------------

    print()
    print(
        "Loading GLSL depth shader..."
    )

    program = create_shader_program()

    print(
        "Depth shader created successfully."
    )

    # --------------------------------------------------
    # Create fullscreen quad
    # --------------------------------------------------

    vao, vbo = create_fullscreen_quad()

    # --------------------------------------------------
    # Open webcam
    # --------------------------------------------------

    print()
    print(
        "Opening webcam..."
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        GL.glDeleteProgram(program)
        GL.glDeleteBuffers(1, [vbo])
        GL.glDeleteVertexArrays(1, [vao])

        glfw.destroy_window(window)
        glfw.terminate()

        raise RuntimeError(
            "Could not open webcam."
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480,
    )

    # --------------------------------------------------
    # Capture first frame
    # --------------------------------------------------

    success, frame = cap.read()

    if not success:

        cap.release()

        GL.glDeleteProgram(program)
        GL.glDeleteBuffers(1, [vbo])
        GL.glDeleteVertexArrays(1, [vao])

        glfw.destroy_window(window)
        glfw.terminate()

        raise RuntimeError(
            "Could not read webcam frame."
        )

    height, width = frame.shape[:2]

    print(
        f"Camera resolution: "
        f"{width}x{height}"
    )

    # --------------------------------------------------
    # Create depth texture
    # --------------------------------------------------

    depth_texture = create_depth_texture(
        width,
        height,
    )

    print(
        "GPU R32F depth texture created."
    )

    # --------------------------------------------------
    # Find shader uniform
    # --------------------------------------------------

    depth_location = (
        GL.glGetUniformLocation(
            program,
            "u_depth",
        )
    )

    if depth_location == -1:

        cap.release()

        GL.glDeleteTextures(
            [depth_texture]
        )

        GL.glDeleteProgram(program)
        GL.glDeleteBuffers(1, [vbo])
        GL.glDeleteVertexArrays(1, [vao])

        glfw.destroy_window(window)
        glfw.terminate()

        raise RuntimeError(
            "Could not find u_depth uniform."
        )

    # --------------------------------------------------
    # Main loop
    # --------------------------------------------------

    frame_count = 0

    fps_frames = 0
    fps = 0.0

    fps_start = time.perf_counter()

    print()
    print(
        "Starting real-time depth pipeline..."
    )
    print(
        "Depth Anything V2 → R32F → GLSL"
    )
    print(
        "Press Q or close the window to quit."
    )
    print()

    while not glfw.window_should_close(
        window
    ):

        glfw.poll_events()

        # --------------------------------------------------
        # Capture frame
        # --------------------------------------------------

        success, frame = cap.read()

        if not success:
            print(
                "ERROR: Failed to read webcam."
            )
            break

        frame_count += 1

        # --------------------------------------------------
        # Run Depth Anything V2
        # --------------------------------------------------

        depth = estimator.estimate(
            frame
        )

        # --------------------------------------------------
        # Validate depth
        # --------------------------------------------------

        if depth.shape != (
            height,
            width,
        ):
            depth = cv2.resize(
                depth,
                (
                    width,
                    height,
                ),
                interpolation=cv2.INTER_LINEAR,
            )

        depth = np.asarray(
            depth,
            dtype=np.float32,
        )

        depth = np.clip(
            depth,
            0.0,
            1.0,
        )

        # --------------------------------------------------
        # Upload REAL depth to GPU
        # --------------------------------------------------

        upload_depth_texture(
            depth_texture,
            depth,
        )

        # --------------------------------------------------
        # Render
        # --------------------------------------------------

        GL.glClearColor(
            0.0,
            0.0,
            0.0,
            1.0,
        )

        GL.glClear(
            GL.GL_COLOR_BUFFER_BIT
        )

        GL.glUseProgram(
            program
        )

        # Texture unit 0.
        GL.glActiveTexture(
            GL.GL_TEXTURE0
        )

        GL.glBindTexture(
            GL.GL_TEXTURE_2D,
            depth_texture,
        )

        GL.glUniform1i(
            depth_location,
            0,
        )

        # Fullscreen quad.
        GL.glBindVertexArray(
            vao
        )

        GL.glDrawArrays(
            GL.GL_TRIANGLES,
            0,
            6,
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

            glfw.set_window_title(
                window,
                f"DepthFX - Real GPU Depth "
                f"| {fps:.1f} FPS",
            )

        glfw.swap_buffers(
            window
        )

        # --------------------------------------------------
        # Q key
        # --------------------------------------------------

        if glfw.get_key(
            window,
            glfw.KEY_Q,
        ) == glfw.PRESS:
            break

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------

    cap.release()

    GL.glBindTexture(
        GL.GL_TEXTURE_2D,
        0,
    )

    GL.glDeleteTextures(
        [depth_texture]
    )

    GL.glDeleteBuffers(
        1,
        [vbo],
    )

    GL.glDeleteVertexArrays(
        1,
        [vao],
    )

    GL.glDeleteProgram(
        program
    )

    glfw.destroy_window(
        window
    )

    glfw.terminate()

    print()
    print(
        f"Total frames processed: "
        f"{frame_count}"
    )

    print(
        "Real GPU depth pipeline stopped."
    )


if __name__ == "__main__":
    main()