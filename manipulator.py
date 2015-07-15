import sys
from math import cos, sin, pi
import ctypes

from PySide import QtCore, QtGui, QtOpenGL
from OpenGL.GL import *
from numpy import array, int32, float32, cross, dot, tan, zeros, eye
from numpy.linalg import norm, inv
import PIL.Image


def createColorTexture(width, height):
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_INT, ctypes.c_void_p(0))
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glBindTexture(GL_TEXTURE_2D, 0)
    return tex

def createDepthTexture(width, height):
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, ctypes.c_void_p(0))
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glBindTexture(GL_TEXTURE_2D, 0)
    return tex

def loadTextureIntoGL(filename):
    ''' Load an image file into GL as a 2D texture (using PIL) '''
    image = PIL.Image.open(filename)
    try:
        width, height, data = image.size[0], image.size[1], image.tobytes("raw", "RGBA", 0, -1)
    except SystemError:
        width, height, data = image.size[0], image.size[1], image.tobytes("raw", "RGBX", 0, -1) # ?
    glID = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, glID)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    return (width, height, glID)

def normalize(vec):
    n = norm(vec)
    return array([i / float(n) for i in vec])

def frustum( left, right, bottom, top, znear, zfar ):
    assert(right  != left)
    assert(bottom != top )
    assert(znear  != zfar)
    M = zeros((4,4), dtype=float32)
    M[0,0] = 2.0 * znear / (right - left)
    M[2,0] = (right + left) / (right - left)
    M[1,1] = 2.0 * znear / (top - bottom)
    M[3,1] = (top + bottom) / (top - bottom)
    M[2,2] = - (zfar + znear) / (zfar - znear)
    M[3,2] = - 2.0 * znear * zfar / (zfar - znear)
    M[2,3] = -1.0
    return M

def perspective(fovy, aspect, znear, zfar):
    assert( znear != zfar )
    h = tan(fovy / 360.0 * pi) * znear
    w = h * aspect
    return frustum( -w, w, -h, h, znear, zfar )

def screenSpaceToViewSpace(screenPos, inverseProjMatrix):
    pos = numpy.dot(screenPos, inverseProjMatrix)
    return [i / pos[3] for i in pos]

def viewSpaceToWorldSpace(viewPos, inverseViewMatrix):

    return dot(viewPos, inverseViewMatrix)

def worldSpaceToViewSpace(worldPos, viewMatrix):

    return dot(worldPos, viewMatrix)

def viewSpaceToScreenSpace(viewPos, projMatrix):
    pos = dot(viewPos, projMatrix)
    return [i / pos[3] for i in pos]

def screenSpaceToWorldSpace(screenPos, inverseProjMatrix, inverseViewMatrix):
    viewPos = screenSpaceToViewSpace(screenPos, inverseProjMatrix)
    return viewSpaceToWorldSpace(viewPos, inverseViewMatrix)

def worldSpaceToScreenSpace(worldPos, projMatrix, viewMatrix):
    viewPos = worldSpaceToViewSpace(worldPos, viewMatrix)
    return viewSpaceToScreenSpace(viewPos, projMatrix)


MOUSE_ZOOM_SPEED = 0.08
MOUSE_PAN_SPEED = 0.001
MOUSE_TURN_SPEED = 0.005


class RenderObject:
    def __init__(self):
        self.vao = -1
        self.vertexCount = 0
        self.drawMode = GL_TRIANGLES
        self.polygonMode = GL_FILL


    def draw(self):
        if glIsVertexArray(self.vao):
            glBindVertexArray(self.vao)
            if self.polygonMode != GL_FILL:
                glPolygonMode(GL_FRONT_AND_BACK, self.polygonMode)
            glDrawElements(self.drawMode, self.vertexCount, GL_UNSIGNED_INT, ctypes.c_void_p(0))
            if self.polygonMode != GL_FILL:
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glBindVertexArray(0)


    def setPolygonMode(self, polygonMode):
        self.polygonMode = polygonMode


    def freeRessources(self):
        if glIsVertexArray(self.vao):
            glDeleteVertexArrays(1, (GLuint)(self.vao))

class Quad2D(RenderObject):
    def __init__(self):
        RenderObject.__init__(self)

        self.vao = glGenVertexArrays(1)
        vbos = glGenBuffers(2)
        glBindVertexArray(self.vao)
        
        indices = array([0, 1, 2, 2, 0, 3], dtype=int32)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbos[0])
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        self.vertexCount = len(indices)
        
        vertices = array([-1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0], dtype=float32)
        glBindBuffer(GL_ARRAY_BUFFER, vbos[1])
        glEnableVertexAttribArray(0) # shader layout location
        glVertexAttribPointer(0, 2, GL_FLOAT, False, 0, ctypes.c_void_p(0))
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        glBindVertexArray(0)
        glDisableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glDeleteBuffers(2, vbos)

class Cube3D(RenderObject):
    def __init__(self):
        RenderObject.__init__(self)

        self.vao = glGenVertexArrays(1)
        vbos = glGenBuffers(4)
        glBindVertexArray(self.vao)

        indices = [0, 1, 2, 2, 1, 3, 4, 5, 6, 6, 5, 7, 8, 9, 10, 10, 9, 11, 12, 13, 14, 14, 13, 15, 16, 17, 18, 19, 17, 20, 21, 22, 23, 24, 25, 26]
        indices = array(indices, dtype=int32)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbos[0])
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        self.vertexCount = len(indices)

        vertices = [-0.5, -0.5, 0.5, 0.5, -0.5, 0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5]
        vertices = array(vertices, dtype=float32)
        glBindBuffer(GL_ARRAY_BUFFER, vbos[1])
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, False, 0, ctypes.c_void_p(0))
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        normals = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0]
        normals = array(normals, dtype=float32)
        glBindBuffer(GL_ARRAY_BUFFER, vbos[2])
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, False, 0, ctypes.c_void_p(0))
        glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)

        uvs = [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]
        uvs = array(uvs, dtype=float32)
        glBindBuffer(GL_ARRAY_BUFFER, vbos[3])
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 2, GL_FLOAT, False, 0, ctypes.c_void_p(0))
        glBufferData(GL_ARRAY_BUFFER, uvs.nbytes, uvs, GL_STATIC_DRAW)

        glBindVertexArray(0)
        glDisableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glDeleteBuffers(4, vbos)

class Grid3D(RenderObject):
    def __init__(self, xres, yres):
        ''' Create a 3D grid which has xres column and yres lines. The grid corners are (-1, 1), (1, 1), (1, -1) and (-1, -1)'''
        RenderObject.__init__(self)

        self.vao = glGenVertexArrays(1)
        vbos = glGenBuffers(2)
        
        glBindVertexArray(self.vao)
        
        indices = []
        for j in range(0, yres):
            offset = j * (xres + 1)
            for i in range(0, xres):
                indices.extend((offset + i,
                                offset + i + 1,
                                offset + (xres + 1) + i + 1,
                                offset + i,
                                offset + (xres + 1) + i + 1,
                                offset + (xres + 1) + i))
        indices = array(indices, dtype=int32)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbos[0])
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        self.vertexCount = len(indices)
        
        vertices = []
        for j in range(0, yres + 1):
            y = (0.5 - j / float(yres)) * 2.0
            for i in range(0, xres + 1):
                vertices.extend((((i / float(xres)) - 0.5) * 2.0, y, 0.0))
        vertices = array(vertices, dtype=float32)
        glBindBuffer(GL_ARRAY_BUFFER, vbos[1])
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, False, 0, ctypes.c_void_p(0))
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        glBindVertexArray(0)
        glDisableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glDeleteBuffers(2, vbos)
        # self.polygonMode = GL_LINE

class TranslationAxis3DSimple(RenderObject):
    def __init__(self, axis="x"):
        RenderObject.__init__(self)

        self.vao = glGenVertexArrays(1)
        vbos = glGenBuffers(3)
        glBindVertexArray(self.vao)
        
        indices = array([0, 1, 2], dtype=int32)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbos[0])
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        self.vertexCount = len(indices)
    
        size = 0.01

        if axis == "x":
            vertices = [0.0, -size, 0.0, 1.0, 0.0, 0.0, 0.0, +size, 0.0]
            normals = 3 * [0.0, 0.0, 1.0]
        elif axis == "y":
            vertices = [-size, 0.0, 0.0, 0.0, 1.0, 0.0, +size, 0.0, 0.0]
            normals = 3 * [0.0, 0.0, 1.0]
        elif axis == "z":
            vertices = [0.0, -size, 0.0, 0.0, +size, 0.0, 0.0, 0.0, 1.0]
            normals = 3 * [1.0, 0.0, 0.0]

        vertices = array(vertices, dtype=float32)
        glBindBuffer(GL_ARRAY_BUFFER, vbos[1])
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, False, 0, ctypes.c_void_p(0))
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        normals = array(normals, dtype=float32)
        glBindBuffer(GL_ARRAY_BUFFER, vbos[2])
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, False, 0, ctypes.c_void_p(0))
        glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)

        glBindVertexArray(0)
        glDisableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glDeleteBuffers(3, vbos)


class TranslationAxis3DComplex(RenderObject):
    def __init__(self, axis="x"):
        RenderObject.__init__(self)

        self.vao = glGenVertexArrays(1)
        vbos = glGenBuffers(3)
        glBindVertexArray(self.vao)
        
        # cylinder

        cylinderSubdiv = 10
        cylinderWidth = 0.7
        cylinderRadius = 0.005
        deltaAngle = 2.0 * pi / (1.0 * cylinderSubdiv)

        vertices = []
        indices = []

        for i in range(cylinderSubdiv):

            if axis == "x":
                origin = [0.0, 0.0, 0.0]
                originFar = [cylinderWidth, 0.0, 0.0]
                v0 = [0.0, cos(i * deltaAngle) * cylinderRadius, sin(i * deltaAngle) * cylinderRadius]
                v1 = [cylinderWidth, cos(i * deltaAngle) * cylinderRadius, sin(i * deltaAngle) * cylinderRadius]
                v2 = [0.0, cos((i+1) * deltaAngle) * cylinderRadius, sin((i+1) * deltaAngle) * cylinderRadius]
                v3 = [cylinderWidth, cos((i+1) * deltaAngle) * cylinderRadius, sin((i+1) * deltaAngle) * cylinderRadius]

            elif axis == "y":
                origin = [0.0, 0.0, 0.0]
                originFar = [0.0, cylinderWidth, 0.0]
                v0 = [cos(i * deltaAngle) * cylinderRadius, 0.0, sin(i * deltaAngle) * cylinderRadius]
                v1 = [cos(i * deltaAngle) * cylinderRadius, cylinderWidth, sin(i * deltaAngle) * cylinderRadius]
                v2 = [cos((i+1) * deltaAngle) * cylinderRadius, 0.0, sin((i+1) * deltaAngle) * cylinderRadius]
                v3 = [cos((i+1) * deltaAngle) * cylinderRadius, cylinderWidth, sin((i+1) * deltaAngle) * cylinderRadius]

            elif axis == "z":
                origin = [0.0, 0.0, 0.0]
                originFar = [0.0, 0.0, cylinderWidth]
                v0 = [cos(i * deltaAngle) * cylinderRadius, sin(i * deltaAngle) * cylinderRadius, 0.0]
                v1 = [cos(i * deltaAngle) * cylinderRadius, sin(i * deltaAngle) * cylinderRadius, cylinderWidth]
                v2 = [cos((i+1) * deltaAngle) * cylinderRadius, sin((i+1) * deltaAngle) * cylinderRadius, 0.0]
                v3 = [cos((i+1) * deltaAngle) * cylinderRadius, sin((i+1) * deltaAngle) * cylinderRadius, cylinderWidth]

            # triangle 1
            vertices += origin
            vertices += v0
            vertices += v2

            # triangle 2
            vertices += v0
            vertices += v1
            vertices += v2

            # triangle 3
            vertices += v1
            vertices += v2
            vertices += v3

            # triangle 4
            vertices += originFar
            vertices += v1
            vertices += v3


        indices = array(range(len(vertices) / 3), dtype=int32)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbos[0])
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        self.vertexCount = len(vertices) / 3

        vertices = array(vertices, dtype=float32)
        glBindBuffer(GL_ARRAY_BUFFER, vbos[1])
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, False, 0, ctypes.c_void_p(0))
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        # normals = array(normals, dtype=float32)
        # glBindBuffer(GL_ARRAY_BUFFER, vbos[2])
        # glEnableVertexAttribArray(1)
        # glVertexAttribPointer(1, 3, GL_FLOAT, False, 0, ctypes.c_void_p(0))
        # glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)

        glBindVertexArray(0)
        glDisableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glDeleteBuffers(3, vbos)


class GLProgram():

    def __init__(self, vsFilename, fsFilename):
        self.id = glCreateProgram()
        vsObject = 0
        try:
            f = open(vsFilename, 'r')
            vsObject = self.createAndCompileShader(GL_VERTEX_SHADER, f.read())
            glAttachShader(self.id, vsObject)
            f.close()
        except IOError as e:
            print "Fail to load", vsFilename
            print "I/O error({0}): {1}".format(e.errno, e.strerror)

        fsObject = 0
        try:
            f = open(fsFilename, 'r')
            fsObject = self.createAndCompileShader(GL_FRAGMENT_SHADER, f.read())
            glAttachShader(self.id, fsObject)
            f.close()
        except IOError as e:
            print "Fail to load", fsFilename
            print "I/O error({0}): {1}".format(e.errno, e.strerror)

        glLinkProgram(self.id)
        glDeleteShader(vsObject)
        glDeleteShader(fsObject)

        status = glGetProgramiv(self.id, GL_LINK_STATUS)
        loglength = glGetProgramiv(self.id, GL_INFO_LOG_LENGTH)
        if(loglength > 1):
            print "Error in linking shaders (status = %s) : %s" % (str(status), glGetProgramInfoLog(self.id))
        print self.id

    def createAndCompileShader(self, shaderType, source):
        shader = glCreateShader(shaderType)
        glShaderSource(shader, source)
        glCompileShader(shader)
        status = glGetShaderiv(shader, GL_COMPILE_STATUS)
        loglength = glGetShaderiv(shader, GL_INFO_LOG_LENGTH)
        if(loglength > 1):
            print "Error in compiling %s (Status = %s): %s" % (str(shaderType), str(status), glGetShaderInfoLog(shader))
            return -1
        return shader


    def use(self):
        glUseProgram(self.id)


    def getLocation(self, uniform):
        return glGetUniformLocation(self.id, uniform)


    def sendUniform1i(self, uniform, i):
        glUniform1i(self.getLocation(uniform), i)


    def sendUniform2i(self, uniform, i1, i2):
        glUniform2i(self.getLocation(uniform), i1, i2)


    def sendUniform3i(self, uniform, i1, i2, i3):
        glUniform3i(self.getLocation(uniform), i1, i2, i3)


    def sendUniform4i(self, uniform, i1, i2, i3, i4):
        glUniform4i(self.getLocation(uniform), i1, i2, i3, i4)


    def sendUniform1iv(self, uniform, vec):
        glUniform1iv(self.getLocation(uniform), len(vec), vec)


    def sendUniform1f(self, uniform, f):
        glUniform1f(self.getLocation(uniform), f)


    def sendUniform2f(self, uniform, f1, f2):
        glUniform2f(self.getLocation(uniform), f1, f2)


    def sendUniform3f(self, uniform, f1, f2, f3):
        glUniform3f(self.getLocation(uniform), f1, f2, f3)


    def sendUniform4f(self, uniform, f1, f2, f3, f4):
        glUniform4f(self.getLocation(uniform), f1, f2, f3, f4)


    def sendUniform1fv(self, uniform, vec):
        glUniform1fv(self.getLocation(uniform), len(vec), vec)


    def sendUniform2fv(self, uniform, vec):
        glUniform2fv(self.getLocation(uniform), len(vec), vec)


    def sendUniform3fv(self, uniform, vec):
        glUniform3fv(self.getLocation(uniform), len(vec), vec)


    def sendUniformMatrix4fv(self, uniform, matrix):
        glUniformMatrix4fv(self.getLocation(uniform), 1, 0, matrix)


    def sendTexture2d(self, uniform, textureId, textureUnit):
        if textureUnit == GL_TEXTURE0:
            i = 0
        elif textureUnit == GL_TEXTURE1:
            i = 1
        elif textureUnit == GL_TEXTURE2:
            i = 2
        elif textureUnit == GL_TEXTURE3:
            i = 3
        elif textureUnit == GL_TEXTURE4:
            i = 4
        elif textureUnit == GL_TEXTURE5:
            i = 5
        elif textureUnit == GL_TEXTURE6:
            i = 6
        elif textureUnit == GL_TEXTURE7:
            i = 7
        else:
            print "bad texture unit"
        self.sendUniform1i(uniform, i)
        glActiveTexture(textureUnit)
        glBindTexture(GL_TEXTURE_2D, textureId)


    def freeRessources(self):
        glDeleteProgram(self.id)

class GLFramebuffer():
    def __init__(self):
        self.id = glGenFramebuffers(1)


    def bind(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.id)


    def unbind(self):
        glBindFramebuffer(GL_FRAMEBUFFER, 0)


    def check(self):
        self.bind()
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (status == GL_FRAMEBUFFER_COMPLETE):
            print "GL_FRAMEBUFFER_COMPLETE"
        elif (status == GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT):
            print "!! GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT"
        elif (status == GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT):
            print "!! GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT"
        elif (status == GL_FRAMEBUFFER_UNSUPPORTED):
            print "!! GL_FRAMEBUFFER_UNSUPPORTED"
        else:
            print "glCheckFramebufferStatus : unsupported result"


    def attachTex2D(self, attachement, tex):
        self.bind()
        glFramebufferTexture2D(GL_FRAMEBUFFER, attachement, GL_TEXTURE_2D, tex, 0)


    def freeRessources(self):
        glDeleteFramebuffers(GLuint(self.id))

class CameraControls:
    def __init__(self):
        self.altPressed = False
        self.turnLock = False
        self.zoomLock = False
        self.panLock = False
        self.lockPositionX = 0.0
        self.lockPositionY = 0.0
        self.leftButton = False
        self.rightButton = False
        self.middleButton = False

class Camera():
    def __init__(self):
        self.reset()

    def compute(self):
        self.eye[0] = self.target[0] + cos(self.theta) * sin(self.phi) * self.radius
        self.eye[1] = self.target[1] + cos(self.phi) * self.radius
        self.eye[2] = self.target[2] + sin(self.theta) * sin(self.phi) * self.radius
        self.up = array([0.0, 1.0 if self.phi < pi else -1.0, 0.0])

    def reset(self, eye=array([0.0, 0.0, -5.0], dtype=float32), target=array([0.0, 0.0, 0.0], dtype=float32)):
        self.target = target
        self.eye = eye
        self.phi = pi / 2.0
        self.theta = pi / 2.0
        self.radius = 5.0
        self.compute()

    def zoom(self, zoomFactor):
        self.radius += self.radius * zoomFactor
        if self.radius < 0.1:
            self.radius = 10.0
            EO = [x - y for x, y in zip(self.target, self.eye)]
            EO = normalize(EO)
            self.target = array([x + y * self.radius for x, y in zip(self.eye, EO)])
        self.compute()

    def turn(self, phi, theta):
        self.theta += theta
        self.phi -= phi
        if self.phi >= 2.0 * pi - 0.1:
            self.phi = 0.00001
        elif self.phi <= 0.0:
            self.phi = 2.0 * pi - 0.1
        self.compute()

    def pan(self, x, y):
        up = array([0.0, 1.0 if self.phi < pi else -1.0, 0.0])
        fwd = normalize(array([i - j for i, j in zip(self.target, self.eye)]))
        side = normalize(cross(fwd, up))
        self.up = normalize(cross(side, fwd))
        self.target[0] += up[0] * y * self.radius * 2
        self.target[1] += up[1] * y * self.radius * 2
        self.target[2] += up[2] * y * self.radius * 2
        self.target[0] -= side[0] * x * self.radius * 2
        self.target[1] -= side[1] * x * self.radius * 2
        self.target[2] -= side[2] * x * self.radius * 2
        self.compute()
        
    def getViewMatrix(self):
        f = normalize([x - y for x, y in zip(self.target, self.eye)])
        s = normalize(cross(f, normalize(self.up)))
        u = cross(s, f)
        result = eye(4)
        result[0][0] = s[0]
        result[1][0] = s[1]
        result[2][0] = s[2]
        result[0][1] = u[0]
        result[1][1] = u[1]
        result[2][1] = u[2]
        result[0][2] = -f[0]
        result[1][2] = -f[1]
        result[2][2] = -f[2]
        result[3][0] = -dot(s, self.eye)
        result[3][1] = -dot(u, self.eye)
        result[3][2] =  dot(f, self.eye)
        return result

class Axis:
    selected = False
    pressed = False
    hover = False

class Manipulator:
    def __init__(self):
        pass

    def draw(self):
        pass

    def initialCompute(self):
        pass

    def onMousePressEvent(self, event):
        pass

    def onMouseReleaseEvent(self, event):
        pass

    def onMouseMoveEvent(self, event):
        pass

    def onMouseWheelEvent(self, event):
        pass

    def onKeyPressedEvent(self, event):
        pass

    def onResizeEvent(self, framebufferDimensions):
        pass

class TranslationManipulator(Manipulator):

    def __init__(self, viewport):
        Manipulator.__init__(self)

        self.axis = {
            "x": Axis(),
            "y": Axis(),
            "z": Axis()
        }

        self.axisRenderObjects = {
            "x": TranslationAxis3DComplex(axis="x"),
            "y": TranslationAxis3DComplex(axis="y"),
            "z": TranslationAxis3DComplex(axis="z")
        }

        self.axisTypeUniform = {
            "x": 0,
            "y": 1,
            "z": 2
        }

        self.utilityRenderObjects = {
            "quad2d": Quad2D()
        }

        self.textures = {
            "axis_x": createColorTexture(viewport["w"], viewport["h"]),
            "axis_y": createColorTexture(viewport["w"], viewport["h"]),
            "axis_z": createColorTexture(viewport["w"], viewport["h"]),
            "depth": createDepthTexture(viewport["w"], viewport["h"])
        }

        self.programs = {
            "manipulator": GLProgram("manipulator.vs.glsl", "manipulator.fs.glsl"),
            "blit_to_screen": GLProgram("blit_to_screen.vs.glsl", "blit_to_screen.fs.glsl")
        }

        self.framebuffer = GLFramebuffer()
        self.framebuffer.attachTex2D(GL_DEPTH_ATTACHMENT, self.textures["depth"])
        self.framebuffer.attachTex2D(GL_COLOR_ATTACHMENT0, self.textures["axis_x"])
        self.framebuffer.attachTex2D(GL_COLOR_ATTACHMENT1, self.textures["axis_y"])
        self.framebuffer.attachTex2D(GL_COLOR_ATTACHMENT2, self.textures["axis_z"])

        self.model = eye(4, dtype=float32)

        self.mouseClicked = False
        self.altClicked = False
        self.selected = False

        self.screenPos = array([0.0, 0.0, 0.0, 1.0])
        self.zStep = -0.01
        self.lastRealX = 0.0
        self.lastRealY = 0.0


    def draw(self, viewport, projection, view):
        self.framebuffer.bind()
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glDrawBuffers((GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2))
        self.framebuffer.attachTex2D(GL_COLOR_ATTACHMENT0, self.textures["axis_x"])
        self.framebuffer.attachTex2D(GL_COLOR_ATTACHMENT1, self.textures["axis_y"])
        self.framebuffer.attachTex2D(GL_COLOR_ATTACHMENT2, self.textures["axis_z"])
        self.framebuffer.attachTex2D(GL_DEPTH_ATTACHMENT, self.textures["depth"])
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.programs["manipulator"].use()
        self.programs["manipulator"].sendUniformMatrix4fv("u_projection", projection)
        self.programs["manipulator"].sendUniformMatrix4fv("u_view", view)
        self.programs["manipulator"].sendUniformMatrix4fv("u_model", self.model)

        for axis in self.axis:
            self.programs["manipulator"].sendUniform1i("u_axis", self.axisTypeUniform[axis])
            self.programs["manipulator"].sendUniform1i("u_clicked", self.axis[axis].selected)
            self.axisRenderObjects[axis].draw()

        self.framebuffer.unbind()

        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.programs["blit_to_screen"].use()

        self.programs["manipulator"].sendTexture2d("u_texture", self.textures["axis_x"], GL_TEXTURE0)
        self.utilityRenderObjects["quad2d"].draw()
        self.programs["manipulator"].sendTexture2d("u_texture", self.textures["axis_y"], GL_TEXTURE0)
        self.utilityRenderObjects["quad2d"].draw()
        self.programs["manipulator"].sendTexture2d("u_texture", self.textures["axis_z"], GL_TEXTURE0)
        self.utilityRenderObjects["quad2d"].draw()

        self.framebuffer.unbind()

        self.projection = projection
        self.view = view


    def drawDebugTextures(self, viewport):

        self.programs["blit_to_screen"].use()

        glDisable(GL_DEPTH_TEST)

        quadw, quadh = viewport["w"] / 5, viewport["h"] / 5

        # draw x
        glViewport(0, 0, quadw, quadh)
        self.programs["manipulator"].sendTexture2d("u_texture", self.textures["axis_x"], GL_TEXTURE0)
        self.utilityRenderObjects["quad2d"].draw()

        # draw y
        glViewport(quadw, 0, quadw, quadh)
        self.programs["manipulator"].sendTexture2d("u_texture", self.textures["axis_y"], GL_TEXTURE0)
        self.utilityRenderObjects["quad2d"].draw()

        # draw z
        glViewport(2 * quadw, 0, quadw, quadh)
        self.programs["manipulator"].sendTexture2d("u_texture", self.textures["axis_z"], GL_TEXTURE0)
        self.utilityRenderObjects["quad2d"].draw()


    def isOnAxis(self, x, y, axis, maskSize=5):
        texture = self.textures["axis_" + axis]
        self.framebuffer.bind()
        self.framebuffer.attachTex2D(GL_COLOR_ATTACHMENT0, texture)
        glReadBuffer(GL_COLOR_ATTACHMENT0)

        data = (GLuint * 1)(0)
        maskRange = range(-maskSize / 2 + 1, maskSize / 2 + 1)
        for i in maskRange:
            for j in maskRange:
                glReadPixels(x + i, y + j, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, data)
                if bool(data[0]):
                    self.framebuffer.unbind()
                    return True

        self.framebuffer.unbind()
        return False


    def updatePosition(self, mouseX, mouseY):

        if self.axis["x"].pressed or self.axis["y"].pressed or self.axis["z"].pressed:
            
            # The letters named with uppercases correspond to the points in screen space
            # The letters named with lowercase correspond to the points in world space

            origin = self.model[3]
            O = worldSpaceToScreenSpace(origin, self.projection, self.view)
            
            x = [origin[0] + 1.0, origin[1], origin[2], origin[3]]
            X = worldSpaceToScreenSpace(x, self.projection, self.view)
            OX = normalize(array([X[0] - O[0], X[1] - O[1]]))

            y = [origin[0], origin[1] + 1.0, origin[2], origin[3]]
            Y = worldSpaceToScreenSpace(y, self.projection, self.view)
            OY = normalize(array([Y[0] - O[0], Y[1] - O[1]]))

            z = [origin[0], origin[1], origin[2] + 1.0, origin[3]]
            Z = worldSpaceToScreenSpace(z, self.projection, self.view)
            OZ = normalize(array([Z[0] - O[0], Z[1] - O[1]]))

            mouseScreenDirection = array([mouseX - self.lastMousePosition[0], mouseY - self.lastMousePosition[1]])
            mouseScreenDirection[0] *= 0.3
            mouseScreenDirection[1] *= 0.3

            speedFactor = 0.03
            if self.axis["x"].pressed:
                origin[0] += dot(mouseScreenDirection, OX) * speedFactor
            if self.axis["y"].pressed:
                origin[1] += dot(mouseScreenDirection, OY) * speedFactor
            if self.axis["z"].pressed:
                origin[2] += dot(mouseScreenDirection, OZ) * speedFactor


    def mousePressEvent(self, x, y):
        self.mouseClicked = True
        self.lastMousePosition = [x, y]
        
        self.axis["x"].selected = self.isOnAxis(x, y, "x")
        if self.axis["x"].selected:
            self.axis["y"].selected = False
            self.axis["z"].selected = False
            self.axis["x"].pressed = True
            return

        self.axis["y"].selected = self.isOnAxis(x, y, "y")
        if self.axis["y"].selected:
            self.axis["x"].selected = False
            self.axis["z"].selected = False
            self.axis["y"].pressed = True
            return

        self.axis["z"].selected = self.isOnAxis(x, y, "z")
        if self.axis["z"].selected:
            self.axis["x"].selected = False
            self.axis["y"].selected = False
            self.axis["z"].pressed = True
            return


    def mouseReleaseEvent(self, x, y):
        self.mouseClicked = False
        self.axis["x"].pressed = False
        self.axis["y"].pressed = False
        self.axis["z"].pressed = False


    def mouseMoveEvent(self, x, y):
        if self.mouseClicked:
            self.updatePosition(x, y)
            self.lastMousePosition = x, y
            return
        self.axis["x"].hover = self.isOnAxis(x, y, "x")
        self.axis["y"].hover = self.isOnAxis(x, y, "y")
        self.axis["z"].hover = self.isOnAxis(x, y, "z")
        self.lastMousePosition = x, y     


    def mouseWheelEvent(self, delta):
        if not self.selected:
            return
        zMin, zMax = 0.0, 0.9999
        if self.altClicked:
            factor = 0.05
        else:
            factor = 1.0
        if delta() < 0:
            self.screenPos[2] = max(zMin, self.screenPos[2] + self.zStep * factor)
        elif delta() > 0:
            self.screenPos[2] = min(zMax, self.screenPos[2] - self.zStep * factor)


    def resizeEvent(self, width, height):
        for tex in self.textures.values():
            glDeleteTextures(GLuint(tex))
        self.textures = {
            "axis_x": createColorTexture(width, height),
            "axis_y": createColorTexture(width, height),
            "axis_z": createColorTexture(width, height),
            "depth": createDepthTexture(width, height)
        }


    def freeRessources(self):
        for rdo in self.axisRenderObjects.values():
            rdo.freeRessources()
        for rdo in self.utilityRenderObjects.values():
            rdo.freeRessources()
        for programm in self.programs.values():
            program.freeRessources()
        for texture in self.textures.values():
            glDeleteTextures(GLuint(textures))


class Window(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        self.setMinimumWidth(800)
        self.setMinimumHeight(600)
        self.viewport = Viewport(self)
        self.setCentralWidget(self.viewport)

    def closeEvent(self, event):
        self.viewport.freeRessources()
        event.accept()

class Viewport(QtOpenGL.QGLWidget):
    def __init__(self, parent):
        super(Viewport, self).__init__(parent)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setMouseTracking(True)


    def initializeGL(self):
        self.viewport = {
            "x": 0,
            "y": 0,
            "w": self.width(),
            "h": self.height()
        }
        self.programs = {
            "blit_quad": GLProgram("blit_quad.vs.glsl", "blit_quad.fs.glsl"),
            "3d": GLProgram("3d.vs.glsl", "3d.fs.glsl")
        }
        self.rdo = {
            "grid": Grid3D(100, 100),
            "quad": Quad2D(),
            "cube": Cube3D()
        }
        self.textures = {
            "displacement": loadTextureIntoGL("displacemap.png")
        }
        self.manipulators = {
            "translation": TranslationManipulator(self.viewport)
        }
        self.camera = Camera()
        self.ctrl = CameraControls()

        glClearColor(0.0, 0.0, 0.0, 1.0)


    def paintGL(self):
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, self.width(), self.height())
        glEnable(GL_DEPTH_TEST)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        projection = perspective(30, self.width() / float(self.height()), 0.1, 10000.0)
        view = self.camera.getViewMatrix()

        self.manipulators["translation"].draw(self.viewport, projection, view)
        # self.manipulators["translation"].drawDebugTextures(self.viewport)


    def resizeGL(self, width, height):
        self.viewport = {
            "x": 0,
            "y": 0,
            "w": width,
            "h": height
        }
        self.manipulators["translation"].resizeEvent(width, height)


    def mousePressEvent(self, event):

        # Camera Controls

        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.ctrl.leftButton = True
            self.ctrl.turnLock = True
        elif event.button() == QtCore.Qt.MouseButton.MiddleButton:
            self.ctrl.middleButton = True
            self.ctrl.panLock = True
        elif event.button() == QtCore.Qt.MouseButton.RightButton:
            self.ctrl.rightButton = True
            self.ctrl.zoomLock = True
        else:
            return
        if self.ctrl.altPressed:
            return
        self.ctrl.lockPositionX = event.x()
        self.ctrl.lockPositionY = event.y()

        # Manipulators controls

        self.manipulators["translation"].mousePressEvent(event.x(), self.height() - event.y())

        self.updateGL()


    def mouseReleaseEvent(self, event):

        # Camera Controls

        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.ctrl.leftButton = False
            self.ctrl.turnLock = False
        elif event.button() == QtCore.Qt.MouseButton.MiddleButton:
            self.ctrl.middleButton = False
            self.ctrl.panLock = False
        elif event.button() == QtCore.Qt.MouseButton.RightButton:
            self.ctrl.rightButton = False
            self.ctrl.zoomLock = False

        # Manipulators controls

        self.manipulators["translation"].mouseReleaseEvent(event.x(), self.height() - event.y())

        self.updateGL()


    def mouseMoveEvent(self, event):
        
        # Camera Controls

        if self.ctrl.altPressed:
            mouseX = event.x()
            mouseY = event.y()
            diffX = mouseX - self.ctrl.lockPositionX
            diffY = mouseY - self.ctrl.lockPositionY

            if self.ctrl.zoomLock:
                zoomDir = -1.0 if diffX > 0 else 1.0
                self.camera.zoom(zoomDir * MOUSE_ZOOM_SPEED)
            elif self.ctrl.turnLock:
                self.camera.turn(diffY * MOUSE_TURN_SPEED, diffX * MOUSE_TURN_SPEED)
            elif self.ctrl.panLock:
                self.camera.pan(diffX * MOUSE_PAN_SPEED, diffY * MOUSE_PAN_SPEED)
            self.ctrl.lockPositionX = mouseX
            self.ctrl.lockPositionY = mouseY

        # Manipulators controls

        self.manipulators["translation"].mouseMoveEvent(event.x(), self.height() - event.y())

        self.updateGL()


    def wheelEvent(self, event):

        # Manipulators controls

        self.manipulators["translation"].wheelEvent(event.delta)

        self.updateGL()


    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_Alt:
            self.ctrl.altPressed = True
        elif event.key() == QtCore.Qt.Key.Key_P:
            mode = GL_FILL if self.grid.polygonMode == GL_POINT else GL_POINT
            self.grid.setPolygonMode(mode)
        self.updateGL()


    def keyReleaseEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_Alt:
            self.ctrl.altPressed = False
        self.updateGL()


    def freeRessources(self):
        for rdo in self.rdo.values():
            rdo.freeRessources()
        for program in self.programs.values():
            program.freeRessources()
        for texture in self.textures.values():
            try:
                glDeleteTextures(GLuint(texture))
            except TypeError:
                glDeleteTextures(GLuint(texture[2])) # for textures loaded with loadTextureIntoGL()




if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())