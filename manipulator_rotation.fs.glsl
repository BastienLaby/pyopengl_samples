#version 440

uniform int u_axis;
uniform int u_hover;
uniform int u_clicked;

layout (location = 0) out vec4 ID;

void main()
{

    vec4 red = vec4(1.0, 0.0, 0.0, 1.0);
    vec4 green = vec4(0.0, 1.0, 0.0, 1.0);
    vec4 blue = vec4(0.0, 0.0, 1.0, 1.0);
    vec4 yellow = vec4(1.0, 1.0, 0.0, 1.0);
    vec4 grey = vec4(0.8, 0.8, 0.8, 1.0);

    if (u_axis == 0)
        ID = red;
    if (u_axis == 1)
        ID = green;
    if (u_axis == 2)
        ID = blue;
    if (u_axis == 3)
        ID = yellow;
    if (u_axis == 4)
        ID = grey;
}