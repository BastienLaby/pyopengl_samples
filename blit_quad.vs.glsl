#version 440

layout(location = 0) in vec2 position_vs_in;

smooth out vec2 texcoord_fs_in;

void main()
{
    texcoord_fs_in = position_vs_in * 0.5 + 0.5;
    gl_Position = vec4(position_vs_in, 0.0, 1.0);
}