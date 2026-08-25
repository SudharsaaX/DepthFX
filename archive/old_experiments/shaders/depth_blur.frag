#version 330 core

in vec2 v_texcoord;

out vec4 FragColor;

uniform sampler2D u_color;
uniform sampler2D u_depth;

uniform float u_blur_strength;
uniform float u_depth_threshold;
uniform vec2 u_texel_size;

void main()
{
    vec3 center_color = texture(
        u_color,
        v_texcoord
    ).rgb;

    float depth = texture(
        u_depth,
        v_texcoord
    ).r;

    /*
        Depth Anything V2:
        higher = closer
        lower  = farther

        Convert to distance:
        higher = farther
    */
    float distance = 1.0 - depth;

    /*
        Only objects beyond the threshold
        receive blur.
    */
    float blur_factor = smoothstep(
        u_depth_threshold,
        1.0,
        distance
    );

    blur_factor *= u_blur_strength;

    /*
        Maximum blur radius.
    */
    float radius = 4.0 * blur_factor;

    vec3 result = center_color;

    if (radius > 0.01)
    {
        vec3 sum = center_color;

        float samples = 1.0;

        /*
            Horizontal samples
        */
        for (int i = 1; i <= 4; i++)
        {
            float offset = float(i) * radius;

            vec2 offset_uv = vec2(
                offset * u_texel_size.x,
                0.0
            );

            sum += texture(
                u_color,
                v_texcoord + offset_uv
            ).rgb;

            sum += texture(
                u_color,
                v_texcoord - offset_uv
            ).rgb;

            samples += 2.0;
        }

        /*
            Vertical samples
        */
        for (int i = 1; i <= 4; i++)
        {
            float offset = float(i) * radius;

            vec2 offset_uv = vec2(
                0.0,
                offset * u_texel_size.y
            );

            sum += texture(
                u_color,
                v_texcoord + offset_uv
            ).rgb;

            sum += texture(
                u_color,
                v_texcoord - offset_uv
            ).rgb;

            samples += 2.0;
        }

        result = sum / samples;

        /*
            Blend between original and blurred
            according to depth.
        */
        result = mix(
            center_color,
            result,
            blur_factor
        );
    }

    FragColor = vec4(
        result,
        1.0
    );
}