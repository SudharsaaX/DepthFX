#version 330 core

in vec2 v_texcoord;

out vec4 FragColor;

uniform sampler2D u_camera;

void main()
{
    vec3 color = texture(
        u_camera,
        v_texcoord
    ).rgb;

    FragColor = vec4(
        color,
        1.0
    );
}