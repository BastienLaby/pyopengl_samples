#version 440

in data_fs_in
{
    vec3 normal;
    vec2 texcoord;
}fs_in;

layout (location = 0) out vec4 Color;

void main()
{
    Color = vec4(fs_in.normal, 1.0);
}