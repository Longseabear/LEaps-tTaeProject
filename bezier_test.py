import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from leapsDrawLib import *
from time import *

fig = plt.figure()

class Canvas():
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.raster = np.ones([self.height, self.width, 3])
        self.pre_time = time()


# sample_interval = 샘플링된 고정 길이
# 초당 몇 샘플링 할지는?
# 스피드 v가 주어져, (pixel per second)
# 만약 스피드가 10이면 10초당 1pixel을 지나야 돼
# 근데 포인트 당 간격이 interval(px)이고, 1픽셀에 1/interval  e.g., 5개의 포인트가 있어
#  거
#속  시   거리/속력 = 시간
# 10초당 5개의 포인트를 지나야 돼
# 1초당 0.5개의 포인트를 지나야 돼
# 1frame당 0.5/30개의 포인트를 지나야 돼

class BezierQuad():
    # points -> float pixel position
    # sample interval is pixel unit
    def __init__(self, points, sample_num=100, sample_interval=10):
        self.points = points
        self.x0, self.y0, self.x1, self.y1, self.x2, self.y2 = points

        self.tx_a = self.x2 - 2 * self.x1 + self.x0
        self.tx_b = 2 * (self.x1 - self.x0)
        self.tx_c = self.x0

        self.ty_a = (self.y2 - 2 * self.y1 + self.y0)
        self.ty_b = 2 * (self.y1 - self.y0)
        self.ty_c = self.y0

        self.sample_num = sample_num

        sampled_t = np.linspace(0, 1, self.sample_num)
        sampled_y, sampled_x = self.get_point(sampled_t)
        self.length = self.get_line_length(sampled_y, sampled_x)
        self.length_cumsum = self.get_line_length_cumsum(sampled_y, sampled_x)

        self.pre_point = self.get_point(0)
        self.next_point = self.get_point(0)
        self.total_travel = 0
        self.sample_interval = sample_interval
        self.finish = False

    @staticmethod
    def find_nearest(array, value):
        idx = np.searchsorted(array, value, side="left")
        idx = idx - (np.abs(value - array[idx - 1]) < np.abs(value - array[idx]))
        return idx

        # example
        # self.get_line_length(sampled_y, sampled_x)
        #
        # interval_sample = np.arange(0, length_cumsum[-1]+np.finfo(float).eps, self.sample_interval)
        # self.bezier_t = self.find_nearest(length_cumsum, interval_sample) / (self.sample_num-1)
#        self.get_line_length(*self.get_point(self.bezier_t))

    # length_point is pixel unit
    def set_next_point(self, delta_travel):
        out = [self.pre_point]
        while delta_travel and not self.finish:
            self.total_travel += min(delta_travel, self.sample_interval)
            delta_travel -= min(delta_travel, self.sample_interval)

            if self.total_travel >= self.length:
                self.total_travel = self.length
                self.finish = True
            out.append(self.get_point(self.find_nearest(self.length_cumsum, self.total_travel) / (self.sample_num-1)))
        self.pre_point = out[-1]
        return out

    def get_point(self, t):
        return t*t*self.ty_a + t*self.ty_b + self.ty_c, t*t*self.tx_a + t*self.tx_b + self.tx_c

    @staticmethod
    def get_line_length_cumsum(sampled_y, sampled_x):
        out = np.zeros_like(sampled_x)

        x_delta = sampled_x[:-1] - sampled_x[1:]
        y_delta = sampled_y[:-1] - sampled_y[1:]
        dists = np.sqrt(y_delta ** 2 + x_delta ** 2)
        np.cumsum(dists, out=out[1:])
        return out

    @staticmethod
    def get_line_length(sampled_y, sampled_x):
        x_delta = sampled_x[:-1] - sampled_x[1:]
        y_delta = sampled_y[:-1] - sampled_y[1:]
        dist = sum(np.sqrt(y_delta**2 + x_delta**2))
#        print('get_line_length debug:', np.sqrt(y_delta**2 + x_delta**2), min(np.sqrt(y_delta**2 + x_delta**2)), max(np.sqrt(y_delta**2 + x_delta**2)), dist)
        return dist

    def run(self):
        pass

scene = Canvas(128, 128)
im = plt.imshow(scene.raster, animated=True)
curve = BezierQuad([0,0,64,105,127.0,64.1], 500)
v = 1 # pixel per second

min_term = 0.033

def Update(i, Canvas):
    global v, curve
    if curve.finish:
        return im

    if v < 200:
        v += 5
    cur_time = time()
    delta_time = cur_time - Canvas.pre_time

    out = curve.set_next_point(delta_time * v)

    for i in range(len(out)-1):
        drawLine(Canvas, out[i][0], out[i][1], out[i+1][0], out[i+1][1], np.array([1, 0, 0, 1]))
    Canvas.pre_time = cur_time

    im.set_array(Canvas.raster)
    return im

ani = animation.FuncAnimation(fig, Update, fargs={scene,},interval=1/30)


plt.show()