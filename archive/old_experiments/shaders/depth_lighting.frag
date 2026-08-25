#version 330 core

in vec2 v_texcoord;

out vec4 FragColor;


// ============================================================
// Textures
// ============================================================

uniform sampler2D u_color;
uniform sampler2D u_depth;


// ============================================================
// Depth texture size
// ============================================================

uniform vec2 u_texel_size;


// ============================================================
// Virtual light
// ============================================================

uniform vec2 u_light_position;

uniform float u_light_strength;

uniform float u_ambient_strength;


// ============================================================
// Main
// ============================================================

void main()
{
    // ========================================================
    // Original RGB
    // ========================================================

    vec3 color = texture(
        u_color,
        v_texcoord
    ).rgb;


    // ========================================================
    // Center depth
    // ========================================================

    float depth_center = texture(
        u_depth,
        v_texcoord
    ).r;


    // ========================================================
    // Neighboring depth samples
    // ========================================================

    float depth_left = texture(
        u_depth,
        v_texcoord + vec2(
            -u_texel_size.x,
            0.0
        )
    ).r;

    float depth_right = texture(
        u_depth,
        v_texcoord + vec2(
            u_texel_size.x,
            0.0
        )
    ).r;

    float depth_up = texture(
        u_depth,
        v_texcoord + vec2(
            0.0,
            -u_texel_size.y
        )
    ).r;

    float depth_down = texture(
        u_depth,
        v_texcoord + vec2(
            0.0,
            u_texel_size.y
        )
    ).r;


    // ========================================================
    // Approximate depth gradients
    // ========================================================

    float dx =
        depth_right - depth_left;

    float dy =
        depth_down - depth_up;


    // ========================================================
    // Approximate surface normal
    // ========================================================

    vec3 normal = normalize(
        vec3(
            -dx * 4.0,
            -dy * 4.0,
            1.0
        )
    );


    // ========================================================
    // Virtual light direction
    // ========================================================

    vec2 light_delta =
        u_light_position - v_texcoord;

    float light_distance =
        length(light_delta);


    vec3 light_direction = normalize(
        vec3(
            light_delta.x,
            -light_delta.y,
            0.8
        )
    );


    // ========================================================
    // Diffuse lighting
    // ========================================================

    float diffuse = max(
        dot(
            normal,
            light_direction
        ),
        0.0
    );


    // ========================================================
    // Distance attenuation
    // ========================================================

    float attenuation =
        1.0 / (
            1.0
            + light_distance * 1.5
        );


    diffuse *= attenuation;


    // ========================================================
    // Final lighting
    // ========================================================

    float lighting =
        u_ambient_strength
        +
        diffuse
        *
        u_light_strength;


    lighting = clamp(
        lighting,
        0.25,
        1.8
    );


    // ========================================================
    // Apply lighting
    // ========================================================

    vec3 result =
        color * lighting;


    // ========================================================
    // Output
    // ========================================================

    FragColor = vec4(
        result,
        1.0
    );
}