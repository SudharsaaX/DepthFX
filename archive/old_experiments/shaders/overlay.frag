#version 330 core

in vec2 v_texcoord;

out vec4 FragColor;

uniform vec4 u_color;

void main()
{
    FragColor = u_color;
}