import numpy as np
import time

from skimage.transform import warp, AffineTransform
from scipy import spatial
from skimage import data
import cv2


class Object:
    def __init__(self, name):
        self.name = name


class PiecewiseAffineTransform():
    def __init__(self):
        self._tesselation = None
        self._inverse_tesselation = None
        self.affines = None
        self.inverse_affines = None

    def estimate(self, src, dst):
        if len(src) <= 2:
            return False
        self._tesselation = spatial.Delaunay(src)
        pts = [[i, i + 1, i + 2] for i in range(len(src) - 2)]
        self._tesselation.vertices = pts
        self._tesselation.simplices = np.array(pts)

        # find affine mapping from source positions to destination
        self.affines = []
        for tri in self._tesselation.vertices:
            affine = AffineTransform()
            affine.estimate(src[tri, :], dst[tri, :])
            self.affines.append(affine)

        return True

    def __call__(self, coords):
        out = np.empty_like(coords, np.double)

        # determine triangle index for each coordinate
        simplex = self._tesselation.find_simplex(coords, bruteforce=True)

        # coordinates outside of mesh
        out[simplex == -1, :] = -1

        for index in range(len(self._tesselation.vertices)):
            # affine transform for triangle
            affine = self.affines[index]
            # all coordinates within triangle
            index_mask = simplex == index

            out[index_mask, :] = affine(coords[index_mask, :])

        return out

    def inverse(self, coords):
        """Apply inverse transformation.

        Coordinates outside of the mesh will be set to `- 1`.

        Parameters
        ----------
        coords : (N, 2) array
            Source coordinates.

        Returns
        -------
        coords : (N, 2) array
            Transformed coordinates.

        """

        out = np.empty_like(coords, np.double)

        # determine triangle index for each coordinate
        simplex = self._inverse_tesselation.find_simplex(coords)

        # coordinates outside of mesh
        out[simplex == -1, :] = -1

        for index in range(len(self._inverse_tesselation.vertices)):
            # affine transform for triangle
            affine = self.inverse_affines[index]
            # all coordinates within triangle
            index_mask = simplex == index

            out[index_mask, :] = affine(coords[index_mask, :])

        return out

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

class BezierQuadOffsetRender():
    # points -> float pixel position
    # sample interval is pixel unit
    def __init__(self, act, sample_num=100, sample_interval=5, max_thickness=0.25 * 128):
        self.max_thickness = max_thickness
        self.sample_interval = sample_interval
        self.sample_num = sample_num
        self.act = act

        self.y0, self.x0, self.y1, self.x1, self.y2, self.x2 = act.points
        self.p0 = np.array([self.y0, self.x0])
        self.p1 = np.array([self.y1, self.x1])
        self.p2 = np.array([self.y2, self.x2])

        self.thickness0, self.thickness1 = act.thickness

        self.v0 = self.p1 - self.p0
        self.v1 = self.p2 - self.p1

        self.cw_d0 = toUnitVector(rotation90(self.v0)) * max_thickness * self.thickness0
        self.ccw_d0 = toUnitVector(rotation270(self.v0)) * max_thickness * self.thickness0

        self.cw_d2 = toUnitVector(rotation90(self.v1)) * max_thickness * self.thickness1
        self.ccw_d2 = toUnitVector(rotation270(self.v1)) * max_thickness * self.thickness1

        self.cw_p0 = self.p0 + self.cw_d0
        self.ccw_p0 = self.p0 + self.ccw_d0

        if np.abs(np.sum(toUnitVector(self.v0) * toUnitVector(self.v1))) > 0.95:
            self.cw_p1 = (self.p0 + self.cw_d0 + self.p2 + self.cw_d2)/2.
            self.ccw_p1 = (self.p0 + self.ccw_d0 + self.p2 + self.ccw_d2)/2.
        else:
            self.cw_p1 = getIntersectionPoint(self.p0 + self.cw_d0, self.v0, self.p2 + self.cw_d2, self.v1)
            self.ccw_p1 = getIntersectionPoint(self.p0 + self.ccw_d0, self.v0, self.p2 + self.ccw_d2, self.v1)

        self.cw_p2 = self.p2 + self.cw_d2
        self.ccw_p2 = self.p2 + self.ccw_d2

        self.base_curve = BezierQuad(act.points, sample_num, sample_interval)
        self.cw_points = [self.cw_p0[0], self.cw_p0[1], self.cw_p1[0], self.cw_p1[1], self.cw_p2[0], self.cw_p2[1]]
        self.ccw_points = [self.ccw_p0[0], self.ccw_p0[1], self.ccw_p1[0], self.ccw_p1[1], self.ccw_p2[0], self.ccw_p2[1]]

        self.cw_curve = BezierQuad(self.cw_points, sample_num, sample_interval)
        self.ccw_curve = BezierQuad(self.ccw_points, sample_num, sample_interval)

#        self.start_round = self.base_curve.get_point(1.02)

    def get_all_points(self):
        t = self.base_curve.get_all_t_sample()

        cw_y, cw_x  = self.cw_curve.get_point(t)
        ccw_y, ccw_x = self.ccw_curve.get_point(t)

        return np.float32([cw_y, cw_x, ccw_y, ccw_x]).transpose(), t

    def isFinish(self):
        return self.base_curve.isFinish()

    def render(self, canvas):
        height, width = canvas.height, canvas.width

        all_points, sampled_t = self.get_all_points()
        src_pts = self.act.brush.sample_p(sampled_t)
        dst_pts = all_points.reshape((-1, 2))[:,::-1]

            # cv2.line(canvas.raster, tuple(pts[0]), tuple(pts[1]), color=(1, 0, 0))
            # cv2.line(canvas.raster, tuple(pts[1]), tuple(pts[2]), color=(0, 1, 0))
            # cv2.line(canvas.raster, tuple(pts[2]), tuple(pts[0]), color=(0, 0, 1))
            # # cv2.circle(canvas.raster, (pts[0][0], pts[0][1]), 2, , thickness=-1)
            # # cv2.circle(canvas.raster, (pts[1][0], pts[1][1]), 2, color=(0, 1, 0), thickness=-1)
            # # cv2.circle(canvas.raster, (pts[2][0], pts[2][1]), 2, color=(0, 0, 1), thickness=-1)
            # cv2.imshow('a',canvas.raster)
            # cv2.waitKey(0)
        # for i in range(len(all_points) - 1):
        #     canvas_ptr = np.float32([all_points[i], all_points[i + 1]])
        #     render(scene.raster, act.brush, act.bgr[::], canvas_ptr.reshape(4, 2), (t[i], t[i + 1]))

        tform = PiecewiseAffineTransform()
        if not tform.estimate(dst_pts, src_pts):
            return canvas

        alpha = self.act.brush.sample(0, 1)
        alpha = warp(alpha, tform, output_shape=(height, width))
        alpha = alpha[:,:,None]

        tmp = np.zeros_like(alpha)
        # cv2.circle(tmp, (int(self.p0[1]), int(self.p0[0])), int(self.thickness0 * self.max_thickness), self.act.brush.alpha[0], -1)
        # cv2.circle(tmp, (int(self.p2[1]), int(self.p2[0])), int(self.thickness1 * self.max_thickness), self.act.brush.alpha[1], -1)
        # alpha += tmp * (alpha == 0)

        canvas.raster = canvas.raster * (1 - alpha) + self.act.brush.color * alpha

        return canvas

class BezierQuadOffsetRenderTest():
    # points -> float pixel position
    # sample interval is pixel unit
    def __init__(self, act, sample_num=100, sample_interval=5, max_thickness=0.25 * 128):
        self.max_thickness = max_thickness
        self.sample_interval = sample_interval
        self.sample_num = sample_num
        self.act = act

        self.y0, self.x0, self.y1, self.x1, self.y2, self.x2 = act.points
        self.p0 = np.array([self.y0, self.x0])
        self.p1 = np.array([self.y1, self.x1])
        self.p2 = np.array([self.y2, self.x2])

        self.thickness0, self.thickness1 = act.thickness

        self.v0 = self.p1 - self.p0
        self.v1 = self.p2 - self.p1

        # TEST
        self.p0 = self.p0 + rotation180(toUnitVector(self.v0)) * self.thickness0 * self.max_thickness
        self.p2 = self.p2 + toUnitVector(self.v1) * self.thickness0 * self.max_thickness

        self.cw_d0 = toUnitVector(rotation90(self.v0)) * max_thickness * self.thickness0
        self.ccw_d0 = toUnitVector(rotation270(self.v0)) * max_thickness * self.thickness0

        self.cw_d2 = toUnitVector(rotation90(self.v1)) * max_thickness * self.thickness1
        self.ccw_d2 = toUnitVector(rotation270(self.v1)) * max_thickness * self.thickness1

        self.cw_p0 = self.p0 + self.cw_d0
        self.ccw_p0 = self.p0 + self.ccw_d0

        if np.abs(np.sum(toUnitVector(self.v0) * toUnitVector(self.v1))) > 0.95:
            self.cw_p1 = (self.p0 + self.cw_d0 + self.p2 + self.cw_d2)/2.
            self.ccw_p1 = (self.p0 + self.ccw_d0 + self.p2 + self.ccw_d2)/2.
        else:
            self.cw_p1 = getIntersectionPoint(self.p0 + self.cw_d0, self.v0, self.p2 + self.cw_d2, self.v1)
            self.ccw_p1 = getIntersectionPoint(self.p0 + self.ccw_d0, self.v0, self.p2 + self.ccw_d2, self.v1)

        self.cw_p2 = self.p2 + self.cw_d2
        self.ccw_p2 = self.p2 + self.ccw_d2

        self.base_curve = BezierQuad([self.p0[0], self.p0[1], self.p1[0], self.p1[1], self.p2[0], self.p2[1]], sample_num, sample_interval)
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

    def render(self, canvas):
        height, width = canvas.height, canvas.width

        all_points, sampled_t = self.get_all_points()
        src_pts = self.act.brush.sample_p(sampled_t)
        dst_pts = all_points.reshape((-1, 2))[:,::-1]

            # cv2.line(canvas.raster, tuple(pts[0]), tuple(pts[1]), color=(1, 0, 0))
            # cv2.line(canvas.raster, tuple(pts[1]), tuple(pts[2]), color=(0, 1, 0))
            # cv2.line(canvas.raster, tuple(pts[2]), tuple(pts[0]), color=(0, 0, 1))
            # # cv2.circle(canvas.raster, (pts[0][0], pts[0][1]), 2, , thickness=-1)
            # # cv2.circle(canvas.raster, (pts[1][0], pts[1][1]), 2, color=(0, 1, 0), thickness=-1)
            # # cv2.circle(canvas.raster, (pts[2][0], pts[2][1]), 2, color=(0, 0, 1), thickness=-1)
            # cv2.imshow('a',canvas.raster)
            # cv2.waitKey(0)
        # for i in range(len(all_points) - 1):
        #     canvas_ptr = np.float32([all_points[i], all_points[i + 1]])
        #     render(scene.raster, act.brush, act.bgr[::], canvas_ptr.reshape(4, 2), (t[i], t[i + 1]))

        tform = PiecewiseAffineTransform()
        if not tform.estimate(dst_pts, src_pts):
            return canvas

        d = int(self.act.division)
        py, px = int(self.act.grid_idx[0]), int(self.act.grid_idx[1])
        dy, dx = self.act.height//d, self.act.width//d
        alpha = self.act.brush.sample(0, 1)

        alpha = warp(alpha, tform, output_shape=(height, width))
        alpha = cv2.resize(alpha, (dy, dx))
        alpha = alpha[:,:,None]

        canvas.raster[py*dy:(py+1)*dy, px*dx:(px+1)*dx] = canvas.raster[py*dy:(py+1)*dy, px*dx:(px+1)*dx] * (1 - alpha) + self.act.brush.color * alpha

        return canvas

class BezierQuadOffsetRenderTestWithAnimation():
    # points -> float pixel position
    # sample interval is pixel unit
    def __init__(self, act, sample_num=100, sample_interval=5, max_thickness=0.25 * 128):
        self.max_thickness = max_thickness
        self.sample_interval = sample_interval
        self.sample_num = sample_num
        self.act = act

        self.y0, self.x0, self.y1, self.x1, self.y2, self.x2 = act.points
        self.p0 = np.array([self.y0, self.x0])
        self.p1 = np.array([self.y1, self.x1])
        self.p2 = np.array([self.y2, self.x2])

        self.thickness0, self.thickness1 = act.thickness

        self.v0 = self.p1 - self.p0
        self.v1 = self.p2 - self.p1

        # TEST
        self.p0 = self.p0 + rotation180(toUnitVector(self.v0)) * self.thickness0 * self.max_thickness
        self.p2 = self.p2 + toUnitVector(self.v1) * self.thickness0 * self.max_thickness

        self.cw_d0 = toUnitVector(rotation90(self.v0)) * max_thickness * self.thickness0
        self.ccw_d0 = toUnitVector(rotation270(self.v0)) * max_thickness * self.thickness0

        self.cw_d2 = toUnitVector(rotation90(self.v1)) * max_thickness * self.thickness1
        self.ccw_d2 = toUnitVector(rotation270(self.v1)) * max_thickness * self.thickness1

        self.cw_p0 = self.p0 + self.cw_d0
        self.ccw_p0 = self.p0 + self.ccw_d0

        if np.abs(np.sum(toUnitVector(self.v0) * toUnitVector(self.v1))) > 0.95:
            self.cw_p1 = (self.p0 + self.cw_d0 + self.p2 + self.cw_d2)/2.
            self.ccw_p1 = (self.p0 + self.ccw_d0 + self.p2 + self.ccw_d2)/2.
        else:
            self.cw_p1 = getIntersectionPoint(self.p0 + self.cw_d0, self.v0, self.p2 + self.cw_d2, self.v1)
            self.ccw_p1 = getIntersectionPoint(self.p0 + self.ccw_d0, self.v0, self.p2 + self.ccw_d2, self.v1)

        self.cw_p2 = self.p2 + self.cw_d2
        self.ccw_p2 = self.p2 + self.ccw_d2

        self.base_curve = BezierQuad([self.p0[0], self.p0[1], self.p1[0], self.p1[1], self.p2[0], self.p2[1]], sample_num, sample_interval)
        self.cw_points = [self.cw_p0[0], self.cw_p0[1], self.cw_p1[0], self.cw_p1[1], self.cw_p2[0], self.cw_p2[1]]
        self.ccw_points = [self.ccw_p0[0], self.ccw_p0[1], self.ccw_p1[0], self.ccw_p1[1], self.ccw_p2[0], self.ccw_p2[1]]

        self.cw_curve = BezierQuad(self.cw_points, sample_num, sample_interval)
        self.ccw_curve = BezierQuad(self.ccw_points, sample_num, sample_interval)

        self.image = None
        self.threshold = None

    def get_all_points(self):
        t = self.base_curve.get_all_t_sample()

        cw_y, cw_x  = self.cw_curve.get_point(t)
        ccw_y, ccw_x = self.ccw_curve.get_point(t)

        return np.float32([cw_y, cw_x, ccw_y, ccw_x]).transpose(), t

    def isFinish(self):
        return self.base_curve.isFinish()

    def make(self, canvas):
        height, width = canvas.height, canvas.width

        all_points, sampled_t = self.get_all_points()
        src_pts = self.act.brush.sample_p(sampled_t)
        dst_pts = all_points.reshape((-1, 2))[:,::-1]

        tform = PiecewiseAffineTransform()
        if not tform.estimate(dst_pts, src_pts):
            self.image = np.zeros([height, width, 1])
            self.threshold = np.ones([height, width, 1])
            return

        d = int(self.act.division)
        py, px = int(self.act.grid_idx[0]), int(self.act.grid_idx[1])
        dy, dx = self.act.height//d, self.act.width//d
        alpha = self.act.brush.sample(0, 1)

        threshold = self.act.brush.sample_alpha((0,1))


        alpha = warp(alpha, tform, output_shape=(height, width))
        threshold = warp(threshold, tform, output_shape=(height, width))
        alpha = cv2.resize(alpha, (dy, dx))
        threshold = cv2.resize(threshold, (dy, dx))

        alpha = alpha[:,:,None]
        threshold = threshold[:,:, None]

        self.image = alpha
        self.threshold = threshold
        self.previous_t = 0

    def reset(self):
        if self.threshold is not None:
            self.threshold = np.zeros_like(self.threshold)
        if self.act is not None:
            self.act.reset()
        self.previous_t = 0

    def render(self, canvas, t):
        d = int(self.act.division)
        py, px = int(self.act.grid_idx[0]), int(self.act.grid_idx[1])
        dy, dx = self.act.height//d, self.act.width//d

        alpha = self.image * ((self.threshold < t) & (self.threshold >= self.previous_t))
        self.previous_t = t
        canvas.raster[py*dy:(py+1)*dy, px*dx:(px+1)*dx] = canvas.raster[py*dy:(py+1)*dy, px*dx:(px+1)*dx] * (1 - alpha) + self.act.brush.color * alpha
        return canvas