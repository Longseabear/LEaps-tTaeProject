import math
import numpy as np


# Xiaolin Wu's line algorithm

def round(x):
    return ipart(x+0.5)

def fpart(x):
    return x-math.floor(x)

def rfpart(x):
    return 1 - fpart(x)

def ipart(x):
    return math.floor(x)

# color = np.array([1,0,0])
def drawPixel(canvas, y, x, alpha, color):
    if 0 > y or y >= canvas.height or 0 > x or x >= canvas.width:
        return
    alpha *= color[3]
    c = color[:3] * alpha + (1-alpha) * canvas.raster[y,x,:]
    canvas.raster[y,x,:] = c

def drawLine(canvas, y0, x0, y1, x1, color):
    steep = abs(y1-y0) > abs(x1 - x0)
    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    dx = x1 - x0
    dy = y1 - y0

    if dx == 0:
        gradient = 1.0
    else:
        gradient = dy / dx

    xend = round(x0)
    yend = y0 + gradient * (xend - x0)
    xgap = rfpart(x0 + 0.5)
    xpxl1 = xend
    ypxl1 = ipart(yend)

    if steep:
        drawPixel(canvas, xpxl1, ypxl1, rfpart(yend) * xgap, color)
        drawPixel(canvas, xpxl1, ypxl1+1, fpart(yend) * xgap, color)
    else:
        drawPixel(canvas, ypxl1, xpxl1,  rfpart(yend) * xgap, color)
        drawPixel(canvas, ypxl1+1, xpxl1, fpart(yend) * xgap, color)
    intery = yend + gradient

    xend = round(x1)
    yend = y1 + gradient * (xend-x1)
    xgap = fpart(x1+0.5)
    xpxl2 = xend
    ypxl2 = ipart(yend)

    if steep:
        drawPixel(canvas, xpxl2, ypxl2, rfpart(yend) * xgap, color)
        drawPixel(canvas, xpxl2, ypxl2 + 1, fpart(yend) * xgap, color)
    else:
        drawPixel(canvas, ypxl2, xpxl2, rfpart(yend) * xgap, color)
        drawPixel(canvas, ypxl2 + 1, xpxl2, fpart(yend) * xgap, color)

    if steep:
        for x in range(xpxl1 + 1, xpxl2):
            drawPixel(canvas, x, ipart(intery), rfpart(intery), color)
            drawPixel(canvas, x, ipart(intery)+1, fpart(intery), color)
            intery = intery + gradient
    else:
        for x in range(xpxl1 + 1, xpxl2):
            drawPixel(canvas,  ipart(intery),x, rfpart(intery), color)
            drawPixel(canvas,  ipart(intery)+1,x, fpart(intery), color)
            intery = intery + gradient
    return canvas