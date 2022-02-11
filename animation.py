import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

fig = plt.figure()

class RenderBase():
    def __init__(self):
        pass

    def run(self):
        pass

x = np.linspace(0, 1, 128).reshape(1, -1)
y = np.linspace(0, 1, 128).reshape(1, -1)
xx, yy = np.meshgrid(x, y)
xx = xx[:, :, None]
yy = yy[:, :, None]
xy = np.concatenate([xx, yy], axis=-1)

points = [0,0,0.5,0.5,1,0]
x0,y0,x1,y1,x2,y2 = points

canvas = np.ones([128,128,3])
im = plt.imshow(canvas, animated=True)


t = 0
tx_a = np.ones([128,128]) * (x2 - 2 * x1 + x0)
tx_b = np.ones([128,128]) * 2 * (x1 - x0)
tx_c = np.ones([128,128]) * x0 - xy[:,:,0]

ty_a = np.ones([128,128]) * (y2 - 2 * y1 + y0)
ty_b = np.ones([128,128]) * 2 * (y1 - y0)
ty_c = np.ones([128,128]) * y0 - xy[:,:,1]

t = 0.1
print(x1,x2,x0,t)
for i in range(1013):
    t = i/1012
    y = int((t*t*(y2 - 2 * y1 + y0) + 2 * (y1 - y0) * t + y0) * 127 + 0.5)
    x = int((t*t*(x2 - 2 * x1 + x0) + 2 * (x1 - x0) * t + x0)*127 + 0.5)
    canvas[y, x] = 0
    print(t*t*(x2 - 2 * x1 + x0) + 2 * (x1 - x0) * t + x0)
    print(t*t*(y2 - 2 * y1 + y0) + 2 * (y1 - y0) * t + y0)
plt.imshow(canvas)
plt.show()
input()

def between(val, s, e):
    return (s <= val) & (val <= e)

def check(a, b, c, t):
    disc = b**2 - 4*a*c
    plt.imshow(between((-b+np.sqrt(disc))/(2*a),0,t))
    plt.colorbar()
    plt.show()
    tmp = np.zeros([128, 128])
    tmp = np.where((disc >= 0) & (a != 0) & (between((-b+np.sqrt(disc))/(2*a), 0, t) | between((-b-np.sqrt(disc))/(2*a),0,t)), 1, tmp)
    tmp = np.where((a == 0) & (b != 0) & between(-c / b, 0, t), 1, tmp)
     #    tmp = np.where((a == 0) & (b == 0) & between(-c/b,0,t), 1, tmp)
    return tmp

plt.imshow(check(tx_a-ty_a, tx_b-ty_b, tx_c-ty_c, 0.1))
plt.show()



def f(x, y):
    return np.sin(x) + np.cos(y)

def Update(*args):
    global x, y, count, t

    im.set_array(canvas)

    if t >= 1:
        t = 0
    return im,

ani = animation.FuncAnimation(fig, Update, interval=33,)
plt.show()