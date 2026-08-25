#version 330 core

in vec2 v_texcoord;

out vec4 FragColor;


/* ============================================================
   TEXTURES
   ============================================================ */

uniform sampler2D u_color;
uniform sampler2D u_depth;


/* ============================================================
   FOG
   ============================================================ */

uniform int u_fog_enabled;

uniform float u_fog_strength;
uniform float u_fog_start;
uniform float u_fog_end;


/* ============================================================
   BLUR
   ============================================================ */

uniform int u_blur_enabled;

uniform float u_blur_strength;
uniform float u_depth_threshold;

uniform vec2 u_texel_size;


/* ============================================================
   LIGHTING
   ============================================================ */

uniform int u_lighting_enabled;

uniform vec2 u_light_position;

uniform float u_light_strength;
uniform float u_ambient_strength;


/* ============================================================
   DEPTH
   ============================================================ */

float get_depth(vec2 uv)
{
    return clamp(
        texture(
            u_depth,
            uv
        ).r,
        0.0,
        1.0
    );
}


/* ============================================================
   SURFACE NORMAL
   ============================================================ */

vec3 estimate_normal(vec2 uv)
{
    /*
        Sample neighboring depth pixels.

        Higher depth value = closer.

        The depth image is treated as a
        simple screen-space height field.
    */

    float center = get_depth(uv);

    float left = get_depth(
        uv - vec2(
            u_texel_size.x,
            0.0
        )
    );

    float right = get_depth(
        uv + vec2(
            u_texel_size.x,
            0.0
        )
    );

    float up = get_depth(
        uv + vec2(
            0.0,
            u_texel_size.y
        )
    );

    float down = get_depth(
        uv - vec2(
            0.0,
            u_texel_size.y
        )
    );


    /*
        Screen-space depth gradients.

        The scale controls how strongly
        depth changes influence the normal.
    */

    float depth_scale = 8.0;


    float dx =
        (right - left)
        * depth_scale;

    float dy =
        (up - down)
        * depth_scale;


    /*
        Construct a pseudo 3D surface.

        X = horizontal
        Y = vertical
        Z = depth
    */

    vec3 tangent_x =
        normalize(
            vec3(
                1.0,
                0.0,
                dx
            )
        );

    vec3 tangent_y =
        normalize(
            vec3(
                0.0,
                1.0,
                dy
            )
        );


    vec3 normal =
        normalize(
            cross(
                tangent_x,
                tangent_y
            )
        );


    /*
        Make sure the normal faces
        toward the camera.
    */

    if (normal.z < 0.0)
    {
        normal = -normal;
    }


    /*
        Mix with a camera-facing normal.

        This stabilizes noisy AI depth.
    */

    normal =
        normalize(
            mix(
                vec3(
                    0.0,
                    0.0,
                    1.0
                ),
                normal,
                0.75
            )
        );


    /*
        Prevent unused center warning
        and slightly stabilize the result.
    */

    normal *=
        0.98
        + center * 0.02;


    return normal;
}


/* ============================================================
   MAIN
   ============================================================ */

void main()
{
    /* --------------------------------------------------------
       ORIGINAL IMAGE
       -------------------------------------------------------- */

    vec3 original_color =
        texture(
            u_color,
            v_texcoord
        ).rgb;


    /* --------------------------------------------------------
       DEPTH
       -------------------------------------------------------- */

    float depth =
        get_depth(
            v_texcoord
        );


    /*
        Convert inverse depth to
        distance-like representation.

        0 = close
        1 = far
    */

    float distance =
        clamp(
            1.0 - depth,
            0.0,
            1.0
        );


    vec3 result =
        original_color;


    /* ========================================================
       DEPTH-AWARE BLUR
       ======================================================== */

    if (u_blur_enabled == 1)
    {
        float blur_factor =
            smoothstep(
                u_depth_threshold,
                1.0,
                distance
            );


        blur_factor *=
            u_blur_strength;


        blur_factor =
            clamp(
                blur_factor,
                0.0,
                1.0
            );


        float radius =
            3.0 * blur_factor;


        if (radius > 0.001)
        {
            vec3 sum =
                original_color;

            float samples =
                1.0;


            /*
                Horizontal blur
            */

            for (int i = 1; i <= 3; i++)
            {
                float offset =
                    float(i)
                    * radius;


                vec2 uv_offset =
                    vec2(
                        offset
                        * u_texel_size.x,
                        0.0
                    );


                sum += texture(
                    u_color,
                    v_texcoord
                    + uv_offset
                ).rgb;


                sum += texture(
                    u_color,
                    v_texcoord
                    - uv_offset
                ).rgb;


                samples += 2.0;
            }


            /*
                Vertical blur
            */

            for (int i = 1; i <= 3; i++)
            {
                float offset =
                    float(i)
                    * radius;


                vec2 uv_offset =
                    vec2(
                        0.0,
                        offset
                        * u_texel_size.y
                    );


                sum += texture(
                    u_color,
                    v_texcoord
                    + uv_offset
                ).rgb;


                sum += texture(
                    u_color,
                    v_texcoord
                    - uv_offset
                ).rgb;


                samples += 2.0;
            }


            vec3 blurred_color =
                sum / samples;


            result =
                mix(
                    result,
                    blurred_color,
                    blur_factor
                );
        }
    }


    /* ========================================================
       DEPTH-AWARE NORMAL LIGHTING
       ======================================================== */

    if (u_lighting_enabled == 1)
    {
        /*
            Estimate local surface normal
            directly from the AI depth texture.
        */

        vec3 normal =
            estimate_normal(
                v_texcoord
            );


        /*
            Screen-space light position.
        */

        vec2 light_delta =
            u_light_position
            - v_texcoord;


        float light_distance =
            length(
                light_delta
            );


        /*
            Convert the 2D light direction
            into an approximate 3D direction.

            The Z component keeps the light
            coming somewhat toward the surface.
        */

        vec3 light_direction =
            normalize(
                vec3(
                    light_delta.x,
                    light_delta.y,
                    0.65
                )
            );


        /*
            Lambert diffuse lighting.

            Surfaces facing the light
            receive more illumination.
        */

        float diffuse =
            max(
                dot(
                    normal,
                    light_direction
                ),
                0.0
            );


        /*
            Distance falloff.
        */

        float distance_falloff =
            1.0
            - smoothstep(
                0.0,
                0.85,
                light_distance
            );


        /*
            Depth contribution.

            Nearby geometry gets slightly
            stronger lighting.
        */

        float depth_factor =
            mix(
                0.80,
                1.10,
                depth
            );


        /*
            Final dynamic light.
        */

        float dynamic_light =
            diffuse
            * distance_falloff
            * depth_factor
            * u_light_strength;


        /*
            Ambient component prevents
            completely dark regions.
        */

        float illumination =
            u_ambient_strength
            + dynamic_light;


        illumination =
            clamp(
                illumination,
                0.0,
                1.40
            );


        result *=
            illumination;
    }


    /* ========================================================
       DEPTH-AWARE FOG
       ======================================================== */

    if (u_fog_enabled == 1)
    {
        float fog_factor =
            smoothstep(
                u_fog_start,
                u_fog_end,
                distance
            );


        fog_factor *=
            u_fog_strength;


        fog_factor =
            clamp(
                fog_factor,
                0.0,
                1.0
            );


        vec3 fog_color =
            vec3(
                0.72,
                0.76,
                0.82
            );


        result =
            mix(
                result,
                fog_color,
                fog_factor
            );
    }


    /* ========================================================
       FINAL SAFETY
       ======================================================== */

    result =
        clamp(
            result,
            0.0,
            1.0
        );


    FragColor =
        vec4(
            result,
            1.0
        );
}