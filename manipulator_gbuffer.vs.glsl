#version 440

layout(location = 0) in vec3 position_vs_in;

uniform mat4 u_projection;
uniform mat4 u_view;
uniform mat4 u_model = mat4(1.0);

out data_fs_in
{
    vec4 world_pos;
}fs_in;

void main()
{
    fs_in.world_pos = u_model * vec4(position_vs_in, 1.0);
    gl_Position = u_projection * u_view * fs_in.world_pos;
}