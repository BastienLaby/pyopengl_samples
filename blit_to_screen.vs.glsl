#version 440

layout(location = 0) in vec2 position_vs_in;

out fs_in_data
{
    vec2 texcoord;
}fs_in;

void main()
{
    fs_in.texcoord = position_vs_in * 0.5 + 0.5;
    gl_Position = vec4(position_vs_in, 0.0, 1.0);
}