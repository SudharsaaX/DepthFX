import glfw
from OpenGL import GL


def main():
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW.")

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
        800,
        600,
        "DepthFX - GPU Test",
        None,
        None,
    )

    if not window:
        glfw.terminate()
        raise RuntimeError(
            "Failed to create OpenGL window."
        )

    glfw.make_context_current(window)

    vendor = GL.glGetString(
        GL.GL_VENDOR
    )

    renderer = GL.glGetString(
        GL.GL_RENDERER
    )

    version = GL.glGetString(
        GL.GL_VERSION
    )

    print("=" * 60)
    print("DepthFX - OpenGL GPU Test")
    print("=" * 60)

    print(
        "OpenGL Vendor:",
        vendor.decode() if vendor else "Unknown",
    )

    print(
        "OpenGL Renderer:",
        renderer.decode() if renderer else "Unknown",
    )

    print(
        "OpenGL Version:",
        version.decode() if version else "Unknown",
    )

    print()
    print("Close the window to finish the test.")

    while not glfw.window_should_close(window):
        glfw.poll_events()

        GL.glClearColor(
            0.05,
            0.05,
            0.05,
            1.0,
        )

        GL.glClear(
            GL.GL_COLOR_BUFFER_BIT
        )

        glfw.swap_buffers(window)

    glfw.destroy_window(window)
    glfw.terminate()

    print("GPU test completed.")


if __name__ == "__main__":
    main()