import numpy as np
import time
class Object:
    def __init__(self, name):
        self.name = name

class Canvas():
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.raster = np.ones([self.height, self.width, 3])
        self.pre_time = time.time()

class BezierQuad():
    # points -> float pixel position
    # sample interval is pixel unit
    def __init__(self, points, sample_num=100, sample_interval=5):
        self.points = points
        self.y0, self.x0, self.y1, self.x1, self.y2, self.x2 = points

        self.sample_num = sample_num

        sampled_t = np.linspace(0, 1, self.sample_num)
        sampled_y, sampled_x = self.get_point(sampled_t)
        self.length = self.get_line_length(sampled_y, sampled_x)
        self.length_cumsum = self.get_line_length_cumsum(sampled_y, sampled_x)

        self.pre_point = self.get_point(0)
        self.total_travel = 0
        self.remain_travel = 0
        self.sample_interval = sample_interval
        self.finish = False

    def isFinish(self):
        return self.finish

    @staticmethod
    def find_nearest(array, value):
        idx = np.searchsorted(array, value, side="left")
        idx = idx - (np.abs(value - array[idx - 1]) < np.abs(value - array[idx]))
        return idx

    def get_all_t_sample(self):
        interval_sample = np.arange(0, self.length_cumsum[-1]-self.sample_interval, self.sample_interval)
        interval_sample = np.append(interval_sample, self.length_cumsum[-1])

        return self.find_nearest(self.length_cumsum, interval_sample) / (self.sample_num-1)

    def get_all_points(self):
        y, x = self.get_point(self.get_all_t_sample())

        return list(zip(y,x))

    # length_point is pixel unit
    def set_next_point(self, delta_travel):
        out = [self.pre_point]

        delta_travel += self.remain_travel
        self.remain_travel = 0
        while delta_travel and not self.finish:
            if delta_travel < self.sample_interval:
                self.remain_travel = self.sample_interval - delta_travel
                break

            self.total_travel += self.sample_interval
            delta_travel -= self.sample_interval

            if self.total_travel >= self.length:
                self.total_travel = self.length
                self.finish = True

            out.append(self.get_point(self.find_nearest(self.length_cumsum, self.total_travel) / (self.sample_num-1)))
        self.pre_point = out[-1]
        return out

    def get_point(self, t):
        y = (1-t) * (1-t) * self.y0 + 2 * t * (1-t) * self.y1 + t * t * self.y2
        x = (1-t) * (1-t) * self.x0 + 2 * t * (1-t) * self.x1 + t * t * self.x2
        return y, x

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

        return dist

    def run(self):
        pass

# numpy(3)
def rotation270(vec):
    return np.array([vec[1], -vec[0]])

def rotation180(vec):
    return np.array([-vec[0], -vec[1]])

def rotation90(vec):
    return np.array([-vec[1], vec[0]])

def toUnitVector(vec):
    r = np.sqrt(vec[0]**2 + vec[1]**2)
    return vec / r

def toHomogeneous(vec):
    return np.concatenate([vec, np.ones(1)], axis=0)

def getIntersectionPoint(p1, d1, p2, d2):
    p1_hg = toHomogeneous(p1)
    d1_hg = toHomogeneous(p1 + d1)
    p2_hg = toHomogeneous(p2)
    d2_hg = toHomogeneous(p2 + d2)
    w = np.cross(np.cross(p1_hg, d1_hg), np.cross(p2_hg, d2_hg))

    if w[-1] == 0:
        return (p1 + p2) / 2.
    return w[:2] / w[2]

class BezierQuadOffset():
    # points -> float pixel position
    # sample interval is pixel unit
    def __init__(self, points, thicknesses, sample_num=100, sample_interval=5, max_thickness=0.25 * 128):
        self.max_thickness = max_thickness
        self.sample_interval = sample_interval
        self.sample_num = sample_num

        self.y0, self.x0, self.y1, self.x1, self.y2, self.x2 = points
        self.p0 = np.array([self.y0, self.x0])
        self.p1 = np.array([self.y1, self.x1])
        self.p2 = np.array([self.y2, self.x2])

        self.thickness0, self.thickness1 = thicknesses
        self.thickness0 *= 1.05
        self.thickness1 *= 1.05

        self.v0 = self.p1 - self.p0
        self.v1 = self.p2 - self.p1

        self.cw_d0 = toUnitVector(rotation90(self.v0)) * max_thickness * self.thickness0
        self.ccw_d0 = toUnitVector(rotation270(self.v0)) * max_thickness * self.thickness0

        self.cw_d2 = toUnitVector(rotation90(self.v1)) * max_thickness * self.thickness1
        self.ccw_d2 = toUnitVector(rotation270(self.v1)) * max_thickness * self.thickness1

        self.cw_p0 = self.p0 + self.cw_d0
        self.ccw_p0 = self.p0 + self.ccw_d0

        print(np.abs(np.sum(toUnitVector(self.v0) * toUnitVector(self.v1))))
        if np.abs(np.sum(toUnitVector(self.v0) * toUnitVector(self.v1))) > 0.98:
            self.cw_p1 = (self.p0 + self.cw_d0 + self.p2 + self.cw_d2)/2.
            self.ccw_p1 = (self.p0 + self.ccw_d0 + self.p2 + self.ccw_d2)/2.
        else:
            self.cw_p1 = getIntersectionPoint(self.p0 + self.cw_d0, self.v0, self.p2 + self.cw_d2, self.v1)
            self.ccw_p1 = getIntersectionPoint(self.p0 + self.ccw_d0, self.v0, self.p2 + self.ccw_d2, self.v1)

        self.cw_p2 = self.p2 + self.cw_d2
        self.ccw_p2 = self.p2 + self.ccw_d2

        self.base_curve = BezierQuad(points, sample_num, sample_interval)
        self.cw_points = [self.cw_p0[0], self.cw_p0[1], self.cw_p1[0], self.cw_p1[1], self.cw_p2[0], self.cw_p2[1]]
        self.ccw_points = [self.ccw_p0[0], self.ccw_p0[1], self.ccw_p1[0], self.ccw_p1[1], self.ccw_p2[0], self.ccw_p2[1]]

        self.cw_curve = BezierQuad(self.cw_points, sample_num, sample_interval)
        self.ccw_curve = BezierQuad(self.ccw_points, sample_num, sample_interval)

    def get_all_points(self):
        t = self.base_curve.get_all_t_sample()

        cw_y, cw_x  = self.cw_curve.get_point(t)
        ccw_y, ccw_x = self.ccw_curve.get_point(t)

        return np.float32([cw_y, cw_x, ccw_y, ccw_x]).transpose(), t

    def isFinish(self):
        return self.base_curve.isFinish()

    # length_point is pixel unit
    def set_next_point(self, delta_travel):
        out = [self.pre_point]

        delta_travel += self.base_curve.remain_travel
        self.base_curve.remain_travel = 0

        while delta_travel and not self.isFinish:
            if delta_travel < self.sample_interval:
                self.base_curve.remain_travel = self.sample_interval - delta_travel
                break

            self.base_curve.total_travel += self.sample_interval
            delta_travel -= self.sample_interval

            if self.total_travel >= self.base_curve.length:
                self.total_travel = self.base_curve.length
                self.finish = True

            cw_y, cw_x = self.cw_curve.get_point(self.base_curve.find_nearest(self.base_curve.length_cumsum, self.base_curve.total_travel) / (self.sample_num-1))
            ccw_y, ccw_x = self.ccw_curve.get_point(self.base_curve.find_nearest(self.base_curve.length_cumsum, self.base_curve.total_travel) / (self.sample_num-1))

            out.append([cw_y, cw_x, ccw_y, ccw_x])
        self.pre_point = out[-1]
        return out

    def run(self):
        pass

class BezierQuadPad():
    # points -> float pixel position
    # sample interval is pixel unit
    def __init__(self, points, thicknesses, sample_num=100, sample_interval=5, max_thickness=0.25 * 128):
        self.max_thickness = max_thickness
        self.sample_interval = sample_interval
        self.sample_num = sample_num

        self.y0, self.x0, self.y1, self.x1, self.y2, self.x2 = points
        self.p0 = np.array([self.y0, self.x0])
        self.p1 = np.array([self.y1, self.x1])
        self.p2 = np.array([self.y2, self.x2])

        self.thickness0, self.thickness1 = thicknesses
        self.thickness0 *= self.max_thickness
        self.thickness1 *= self.max_thickness

        self.base_curve = BezierQuad(points, sample_num, sample_interval)

    def get_all_points(self):
        t = self.base_curve.get_all_t_sample()
        pts = np.stack(self.base_curve.get_point(t)) # y, x
        radius = np.expand_dims(self.thickness0 * (1-t) + self.thickness1 * t, 0)

        vts = np.ones_like(pts)
        vts[:, 1:] = pts[:, 1:] - pts[:,:-1]
        vts[:, 0] = vts[:, 1]

        # rotaiton 90
        vts = vts[::-1, :]
        vts[0,:] *= -1

        #normal
        vts /= np.sqrt(vts[0:1, :]**2 + vts[1:,:]**2)

        cw = pts + vts * radius
        ccw = pts + vts * radius * -1
        out = np.stack([cw[0,:], cw[1,:], ccw[0,:], ccw[1,:]]).transpose()

        return out, t

    def isFinish(self):
        return self.base_curve.isFinish()

    # length_point is pixel unit
    def set_next_point(self, delta_travel):
        out = [self.pre_point]

        delta_travel += self.base_curve.remain_travel
        self.base_curve.remain_travel = 0

        while delta_travel and not self.isFinish:
            if delta_travel < self.sample_interval:
                self.base_curve.remain_travel = self.sample_interval - delta_travel
                break

            self.base_curve.total_travel += self.sample_interval
            delta_travel -= self.sample_interval

            if self.total_travel >= self.base_curve.length:
                self.total_travel = self.base_curve.length
                self.finish = True

            cw_y, cw_x = self.cw_curve.get_point(self.base_curve.find_nearest(self.base_curve.length_cumsum, self.base_curve.total_travel) / (self.sample_num-1))
            ccw_y, ccw_x = self.ccw_curve.get_point(self.base_curve.find_nearest(self.base_curve.length_cumsum, self.base_curve.total_travel) / (self.sample_num-1))

            out.append([cw_y, cw_x, ccw_y, ccw_x])
        self.pre_point = out[-1]
        return out

    def run(self):
        pass
