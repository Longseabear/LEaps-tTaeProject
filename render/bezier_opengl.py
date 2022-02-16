import OpenGL
from OpenGL.GL import *
from  OpenGL.GLUT import * 
from OpenGL.GLU import * 
import io
print("Imports successful!")


w, h = 500,500

def draw_shape():
    glBegin(GL_LINES)
    glColor4f(1.0,0.0,0.0,1.0)
    glVertex2f(100.0, 100.0)
    glVertex2f(200.0, 100.0)
    glEnd()
    glFlush()

def loop():
    glViewport(0, 0, 500, 500)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, 500, 500, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glClearColor(1.0,1.0,1.0,1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) # Remove everything from screen (i.e. displays all white)
    glLoadIdentity() # Reset all graphic/shape's position

    draw_shape() # Draw function

    glutSwapBuffers()

#---Section 3---
glutInit()
glutInitDisplayMode(GLUT_RGBA) # Set the display mode to be colored
glutInitWindowSize(500, 500)   # Set the w and h of your window
glutInitWindowPosition(0, 0)   # Set the position at which this windows should appear
wind = glutCreateWindow("OpenGL Coding Practice") # Set a window title


glutDisplayFunc(loop)
glutIdleFunc(loop) # Keeps the window open
glutMainLoop()  # Keeps the above created window displaying/running in a loop*