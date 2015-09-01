#version 440

in data_fs_in
{
    vec4 world_pos;
}fs_in;

layout (location = 0) out vec4 Worldpos;

void main()
{
    Worldpos = vec4(fs_in.world_pos.xyz, 1.0);
}