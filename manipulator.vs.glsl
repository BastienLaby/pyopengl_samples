#version 440

layout(location = 0) in vec3 position_vs_in;
layout(location = 1) in vec3 normal_vs_in;

uniform mat4 u_projection;
uniform mat4 u_view;
uniform mat4 u_model = mat4(1.0);

out data_fs_in
{
    vec3 normal;
}fs_in;

void main()
{
    fs_in.normal = normal_vs_in * 0.5 + 0.5;
    gl_Position = u_projection * u_view * u_model * vec4(position_vs_in, 1.0);
}