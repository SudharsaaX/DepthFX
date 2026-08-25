import ctypes
import time
from pathlib import Path

import cv2
import glfw
import numpy as np
from OpenGL import GL


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
    / "camera.frag"
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
    """Compile one OpenGL shader."""

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
        ).decode("utf-8", errors="replace")

        GL.glDeleteShader(shader)

        raise RuntimeError(
            "Shader compilation failed:\n"
            + log
        )

    return shader


def create_shader_program() -> int:
    """Create and link the GLSL shader program."""

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
        ).decode("utf-8", errors="replace")

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
    """
    Create a fullscreen quad.

    Vertex format:

        X, Y, U, V
    """

    vertices = np.array(
        [
            # Position       # Texture coordinates
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

    # --------------------------------------------------
    # Position attribute
    # --------------------------------------------------

    GL.glVertexAttribPointer(
        0,
        2,
        GL.GL_FLOAT,
        GL.GL_FALSE,
        stride,
        ctypes.c_void_p(0),
    )

    GL.glEnableVertexAttribArray(0)

    # --------------------------------------------------
    # Texture coordinate attribute
    # --------------------------------------------------

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


def create_camera_texture(
    width: int,
    height: int,
) -> int:
    """Create an OpenGL texture for the webcam."""

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


def update_camera_texture(
    texture: int,
    frame: np.ndarray,
):
    """Upload an OpenCV BGR frame to the OpenGL texture."""

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    # Make sure the array is contiguous.
    rgb = np.ascontiguousarray(rgb)

    GL.glBindTexture(
        GL.GL_TEXTURE_2D,
        texture,
    )

    GL.glTexSubImage2D(
        GL.GL_TEXTURE_2D,
        0,
        0,
        0,
        rgb.shape[1],
        rgb.shape[0],
        GL.GL_RGB,
        GL.GL_UNSIGNED_BYTE,
        rgb,
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
    print("DepthFX - GPU Camera Renderer")
    print("=" * 60)

    # --------------------------------------------------
    # Initialize GLFW
    # --------------------------------------------------

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
        "DepthFX - GPU Camera",
        None,
        None,
    )

    if not window:
        glfw.terminate()

        raise RuntimeError(
            "Failed to create OpenGL window."
        )

    glfw.make_context_current(window)

    # Enable VSync off for performance testing.
    glfw.swap_interval(0)

    # --------------------------------------------------
    # GPU information
    # --------------------------------------------------

    print_gpu_information()

    # --------------------------------------------------
    # Create shader program
    # --------------------------------------------------

    print("Loading GLSL shaders...")

    program = create_shader_program()

    print(
        "GLSL shader program created successfully."
    )

    # --------------------------------------------------
    # Create fullscreen quad
    # --------------------------------------------------

    vao, vbo = create_fullscreen_quad()

    # --------------------------------------------------
    # Open webcam
    # --------------------------------------------------

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
    # Get first frame
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
        f"Camera resolution: {width}x{height}"
    )

    # --------------------------------------------------
    # Create camera texture
    # --------------------------------------------------

    camera_texture = create_camera_texture(
        width,
        height,
    )

    # --------------------------------------------------
    # Get shader uniform
    # --------------------------------------------------

    camera_location = (
        GL.glGetUniformLocation(
            program,
            "u_camera",
        )
    )

    if camera_location == -1:

        cap.release()

        GL.glDeleteTextures(
            [camera_texture]
        )

        GL.glDeleteProgram(program)
        GL.glDeleteBuffers(1, [vbo])
        GL.glDeleteVertexArrays(1, [vao])

        glfw.destroy_window(window)
        glfw.terminate()

        raise RuntimeError(
            "Could not find u_camera uniform."
        )

    # --------------------------------------------------
    # Main rendering loop
    # --------------------------------------------------

    frame_count = 0

    fps_time = time.perf_counter()
    fps_frames = 0
    fps = 0.0

    print()
    print(
        "Starting GPU camera renderer..."
    )
    print(
        "Press Q or close the window to quit."
    )

    while not glfw.window_should_close(
        window
    ):

        glfw.poll_events()

        # --------------------------------------------------
        # Read webcam
        # --------------------------------------------------

        success, frame = cap.read()

        if not success:

            print(
                "ERROR: Failed to read webcam frame."
            )

            break

        frame_count += 1

        # --------------------------------------------------
        # Upload camera frame to GPU
        # --------------------------------------------------

        update_camera_texture(
            camera_texture,
            frame,
        )

        # --------------------------------------------------
        # Clear framebuffer
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

        # --------------------------------------------------
        # Activate shader
        # --------------------------------------------------

        GL.glUseProgram(program)

        # --------------------------------------------------
        # Bind camera texture
        # --------------------------------------------------

        GL.glActiveTexture(
            GL.GL_TEXTURE0
        )

        GL.glBindTexture(
            GL.GL_TEXTURE_2D,
            camera_texture,
        )

        GL.glUniform1i(
            camera_location,
            0,
        )

        # --------------------------------------------------
        # Draw fullscreen quad
        # --------------------------------------------------

        GL.glBindVertexArray(vao)

        GL.glDrawArrays(
            GL.GL_TRIANGLES,
            0,
            6,
        )

        GL.glBindVertexArray(0)

        # --------------------------------------------------
        # FPS calculation
        # --------------------------------------------------

        fps_frames += 1

        current_time = time.perf_counter()

        fps_elapsed = (
            current_time - fps_time
        )

        if fps_elapsed >= 0.5:

            fps = (
                fps_frames / fps_elapsed
            )

            fps_frames = 0
            fps_time = current_time

            glfw.set_window_title(
                window,
                f"DepthFX - GPU Camera | "
                f"{fps:.1f} FPS",
            )

        # --------------------------------------------------
        # Display
        # --------------------------------------------------

        glfw.swap_buffers(window)

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
        [camera_texture]
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

    glfw.destroy_window(window)
    glfw.terminate()

    print()
    print(
        f"Total frames rendered: "
        f"{frame_count}"
    )

    print(
        "GPU camera renderer stopped."
    )


if __name__ == "__main__":
    main()