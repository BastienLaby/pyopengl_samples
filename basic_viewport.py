import sys
from math import cos, sin, pi

from PySide import QtCore, QtGui, QtOpenGL
from OpenGL.GL import *
from numpy import array, int32, float32, cross, dot, tan, zeros, eye
from numpy.linalg import norm
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
        glFramebufferTexture2D(GL_FRAMEBUFFER, attachement, GL_TEXTURE_2D,tex, 0)


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

class Window(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        self.setMinimumWidth(800)
        self.setMinimumHeight(600)
        self.setCentralWidget(Viewport(self))

class Viewport(QtOpenGL.QGLWidget):
    def __init__(self, parent):
        super(Viewport, self).__init__(parent)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setMouseTracking(True)


    def initializeGL(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)
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
        self.camera = Camera()
        self.ctrl = CameraControls()


    def paintGL(self):
        glViewport(0, 0, self.width(), self.height())
        glEnable(GL_DEPTH_TEST)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.programs["3d"].use()
        projection = perspective(30, self.width() / float(self.height()), 0.1, 10000.0)
        self.programs["3d"].sendUniformMatrix4fv("u_projection", projection.astype(float32))
        self.programs["3d"].sendUniformMatrix4fv("u_view", self.camera.getViewMatrix().astype(float32))

        self.rdo["cube"].draw()


    def mousePressEvent(self, event):
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
        self.updateGL()


    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.ctrl.leftButton = False
            self.ctrl.turnLock = False
        elif event.button() == QtCore.Qt.MouseButton.MiddleButton:
            self.ctrl.middleButton = False
            self.ctrl.panLock = False
        elif event.button() == QtCore.Qt.MouseButton.RightButton:
            self.ctrl.rightButton = False
            self.ctrl.zoomLock = False
        self.updateGL()


    def mouseMoveEvent(self, event):
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
        self.updateGL()


    def wheelEvent(self, event):
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
        pass




if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())