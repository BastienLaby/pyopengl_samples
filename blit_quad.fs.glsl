#version 440

smooth in vec2 texcoord_fs_in;

uniform sampler2D u_texture;

layout(location = 0) out vec4 Texture;

void main()
{
    Texture = 2 * texture(u_texture, texcoord_fs_in);
}