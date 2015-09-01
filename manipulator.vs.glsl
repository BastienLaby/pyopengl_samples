#version 440

layout(location = 0) in vec3 position_vs_in;

uniform mat4 u_projection;
uniform mat4 u_view;
uniform mat4 u_model = mat4(1.0);

void main()
{
    gl_Position = u_projection * u_view * u_model * vec4(position_vs_in, 1.0);
}