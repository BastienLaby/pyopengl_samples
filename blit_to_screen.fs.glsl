#version 440

in fs_in_data
{
    vec2 texcoord;
}fs_in;

uniform sampler2D u_texture;

layout(location = 0) out vec4 Texture;

void main()
{
    Texture = texture(u_texture, fs_in.texcoord);
}