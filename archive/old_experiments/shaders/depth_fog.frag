#version 330 core

in vec2 v_texcoord;

out vec4 FragColor;

uniform sampler2D u_color;
uniform sampler2D u_depth;

uniform float u_fog_strength;
uniform float u_fog_start;
uniform float u_fog_end;
uniform int u_effect_enabled;

void main()
{
    vec3 color = texture(
        u_color,
        v_texcoord
    ).rgb;

    float depth = texture(
        u_depth,
        v_texcoord
    ).r;

    /*
        Depth Anything V2 relative depth:

        Higher value → closer
        Lower value  → farther

        Convert it to a far-distance value.
    */
    float distance = 1.0 - depth;

    /*
        Fog starts at u_fog_start and reaches
        maximum strength at u_fog_end.
    */
    float fog_factor = smoothstep(
        u_fog_start,
        u_fog_end,
        distance
    );

    fog_factor *= u_fog_strength;

    fog_factor = clamp(
        fog_factor,
        0.0,
        1.0
    );

    /*
        Neutral atmospheric fog.
    */
    vec3 fog_color = vec3(
        0.72,
        0.76,
        0.82
    );

    vec3 result = mix(
        color,
        fog_color,
        fog_factor
    );

    if (u_effect_enabled == 0)
    {
        result = color;
    }

    FragColor = vec4(
        result,
        1.0
    );
}