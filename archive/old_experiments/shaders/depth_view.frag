#version 330 core

in vec2 v_texcoord;

out vec4 FragColor;

uniform sampler2D u_depth;

void main()
{
    float depth = texture(
        u_depth,
        v_texcoord
    ).r;

    FragColor = vec4(
        depth,
        depth,
        depth,
        1.0
    );
}