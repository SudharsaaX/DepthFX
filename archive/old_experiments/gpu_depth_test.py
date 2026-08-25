import ctypes
from pathlib import Path

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
    / "depth_view.frag"
)


def read_shader(path: Path) -> str:
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

    vertices = np.array(
        [
            # Position      # Texture coordinates
            -1.0, -1.0,      0.0, 1.0,
             1.0, -1.0,      1.0, 1.0,
             1.0,  1.0,      1.0, 0.0,

            -1.0, -1.0,      0.0, 1.0,
             1.0,  1.0,      1.0, 0.0,
            -1.0,  1.0,      0.0, 0.0,
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

    GL.glVertexAttribPointer(
        0,
        2,
        GL.GL_FLOAT,
        GL.GL_FALSE,
        stride,
        ctypes.c_void_p(0),
    )

    GL.glEnableVertexAttribArray(0)

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

    # Floating-point single-channel texture.
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


def create_test_depth(
    width: int,
    height: int,
) -> np.ndarray:
    """
    Create a simple horizontal depth gradient.

    Left side  = 0.0
    Right side = 1.0
    """

    depth = np.linspace(
        0.0,
        1.0,
        width,
        dtype=np.float32,
    )

    depth = np.tile(
        depth,
        (height, 1),
    )

    return np.ascontiguousarray(
        depth
    )


def upload_depth_texture(
    texture: int,
    depth: np.ndarray,
):

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


def main():

    print("=" * 60)
    print(
        "DepthFX - GPU Depth Texture Test"
    )
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
        "DepthFX - GPU Depth Test",
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

    glfw.swap_interval(0)

    # --------------------------------------------------
    # GPU information
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Create shader program
    # --------------------------------------------------

    print(
        "Loading depth visualization shader..."
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
    # Create test depth data
    # --------------------------------------------------

    width = 640
    height = 480

    depth = create_test_depth(
        width,
        height,
    )

    print(
        f"Depth texture size: "
        f"{width}x{height}"
    )

    print(
        f"Depth minimum: "
        f"{depth.min():.3f}"
    )

    print(
        f"Depth maximum: "
        f"{depth.max():.3f}"
    )

    # --------------------------------------------------
    # Create GPU depth texture
    # --------------------------------------------------

    depth_texture = create_depth_texture(
        width,
        height,
    )

    upload_depth_texture(
        depth_texture,
        depth,
    )

    print(
        "Depth texture uploaded to GPU."
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
        raise RuntimeError(
            "Could not find u_depth uniform."
        )

    # --------------------------------------------------
    # Main loop
    # --------------------------------------------------

    print()
    print(
        "Displaying GPU depth texture."
    )
    print(
        "You should see a smooth black-to-white gradient."
    )
    print(
        "Press Q or close the window to quit."
    )

    while not glfw.window_should_close(
        window
    ):

        glfw.poll_events()

        # Clear screen.
        GL.glClearColor(
            0.0,
            0.0,
            0.0,
            1.0,
        )

        GL.glClear(
            GL.GL_COLOR_BUFFER_BIT
        )

        # Use shader.
        GL.glUseProgram(
            program
        )

        # Bind depth texture to texture unit 0.
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

        # Draw fullscreen quad.
        GL.glBindVertexArray(
            vao
        )

        GL.glDrawArrays(
            GL.GL_TRIANGLES,
            0,
            6,
        )

        GL.glBindVertexArray(0)

        glfw.swap_buffers(
            window
        )

        # Q key.
        if glfw.get_key(
            window,
            glfw.KEY_Q,
        ) == glfw.PRESS:
            break

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------

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
        "GPU depth texture test completed."
    )


if __name__ == "__main__":
    main()