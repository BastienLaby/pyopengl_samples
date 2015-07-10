#version 440

in data_fs_in
{
    vec3 normal;
}fs_in;

const int AXIS_X = 0;
const int AXIS_Y = 1;
const int AXIS_Z = 2;
uniform int u_axis;

uniform int u_hover;
uniform int u_clicked;

layout (location = 0) out vec4 AxisX;
layout (location = 1) out vec4 AxisY;
layout (location = 2) out vec4 AxisZ;

void main()
{

    vec4 red = vec4(1.0, 0.0, 0.0, 1.0);
    vec4 green = vec4(0.0, 1.0, 0.0, 1.0);
    vec4 blue = vec4(0.0, 0.0, 1.0, 1.0);
    vec4 yellow = vec4(1.0, 1.0, 0.0, 1.0);

    if (u_axis == AXIS_X)
    {
        AxisX = bool(u_clicked) ? yellow : red;
    }
    else if (u_axis == AXIS_Y)
    {
        AxisY = bool(u_clicked) ? yellow : green;
    }
    else if (u_axis == AXIS_Z)
    {
        AxisZ = bool(u_clicked) ? yellow : blue;
    }
}