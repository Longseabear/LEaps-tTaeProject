import numpy as np
import cv2
import math
import random
import numpy as np

def rn():
    return random.random()

brushes = {}

# load brushes from ./brushes directory
def load_brushes():
    brush_dir = './brushes/'
    import os
    for fn in os.listdir(brush_dir):
        if os.path.isfile(brush_dir + fn):
            brush = cv2.imread(brush_dir + fn,0)
            if not brush is None:
                brushes[fn] = brush

load_brushes()

# ONLY VERTICAL ORIENTED
class Brush():
    def __init__(self, color, alpha=(1, 1), key='random'):
        if key=='random':
            key = random.choice(list(brushes.keys()))
        self.key = key
        self.color = color
        self.img = brushes[key] / 255
        self.h, self.w = self.img.shape

        self.alpha = np.array([alpha[0], alpha[1]])
        self.alpha_map = np.stack([np.linspace(alpha[0], alpha[1], self.h), ] * self.w).transpose()

    def sample_alpha(self, alpha):
        alpha_map = np.stack([np.linspace(alpha[0], alpha[1], self.h), ] * self.w).transpose()
        return alpha_map

    def setAlpha(self, alpha):
        self.alpha = alpha
        self.alpha_map = self.sample_alpha(alpha)

    def sample_p(self, t):
        src_cols = np.linspace(0, self.w, 2)
        src_rows = t * self.h

        src_cols, src_rows = np.meshgrid(src_cols, src_rows)
        src = np.dstack([src_cols.flat, src_rows.flat])[0] # x, y order

        return src

    def sample(self, t0, t1):
        return self.img[int(t0 * self.h) : int(t1* self.h), :] * self.alpha_map[int(t0 * self.h) : int(t1* self.h), :]


def homography():
    pts_src = np.array([[141, 131], [480, 159], [493, 630], [64, 601]])
    pts_dst = np.array([[318, 256],[534, 372],[316, 670],[73, 473]])

    h, status = cv2.findHomography(pts_src, pts_dst)
    print(h.shape, (np.concatenate([pts_src, np.ones((4,1))], axis=1)).shape, status)
    print(np.matmul(h, np.concatenate([pts_src, np.ones((4,1))], axis=1).transpose()))
    out = np.matmul(h, np.concatenate([pts_src, np.ones((4,1))], axis=1).transpose())
    print(out[:2, :] / out[2,:])
    print((out[:2, :] / out[2,:]).transpose())

# 시계방향
def render(canvas, brush, color, canvas_pts, ts):
    height, width, _ = canvas.shape
    brush_pts = brush.sample_p(ts[0],ts[1])

    min_y, min_x, max_y, max_x = np.min(canvas_pts[:,0]), np.min(canvas_pts[:,1]), np.max(canvas_pts[:,0]), np.max(canvas_pts[:,1])

    delta_canvas_pts = canvas_pts - np.array([min_y, min_x])
    min_y = np.clip(min_y, 0, height)
    min_x = np.clip(min_x, 0, width)
    max_y = np.clip(max_y, 0, height)
    max_x = np.clip(max_x, 0, width)

    min_y, min_x = int(min_y), int(min_x)
    pad_height = int(max_y - min_y)
    pad_width = int(max_x - min_x)

    if pad_height == 0 or pad_width == 0:
        return canvas
    # points = [2, N] (y, x)
    h = cv2.getPerspectiveTransform(brush_pts[:,::-1], delta_canvas_pts[:,::-1])
    alpha = cv2.warpPerspective(brush.sample(0, 1), h, (pad_width, pad_height)) #(brush.shape[1], brush.shape[0]))
    alpha = alpha[:,:, None]

    canvas[min_y:min_y+pad_height, min_x:min_x + pad_width,:] = canvas[min_y:min_y+pad_height, min_x:min_x + pad_width,:] * (1-alpha) + color * alpha
    return canvas

def render_all_point(canvas, brush, color, all_pts, t):
    height, width, _ = canvas.shape
    alpha = np.zeros_like(canvas)

    pts_num = len(all_pts)
    for i in range(pts_num-1):
        brush_pts = brush.sample_p(t[i], t[i+1])
        canvas_pts = all_pts[i:i+2, :] # y x y x y x y x
        canvas_pts = canvas_pts.reshape((4, 2))
        min_y, min_x, max_y, max_x = np.min(canvas_pts[:,0]), np.min(canvas_pts[:,1]), np.max(canvas_pts[:,0]), np.max(canvas_pts[:,1])

        delta_canvas_pts = canvas_pts - np.array([min_y, min_x])
        min_y = np.clip(min_y, 0, height)
        min_x = np.clip(min_x, 0, width)
        max_y = np.clip(max_y, 0, height)
        max_x = np.clip(max_x, 0, width)

        min_y, min_x = int(min_y), int(min_x)
        pad_height = int(max_y - min_y)
        pad_width = int(max_x - min_x)

        if pad_height == 0 or pad_width == 0:
            continue
        # points = [2, N] (y, x)
        h = cv2.getPerspectiveTransform(brush_pts[:,::-1], delta_canvas_pts[:,::-1])
        a = cv2.warpPerspective(brush.sample(0, 1), h, (pad_width, pad_height)) #(brush.shape[1], brush.shape[0]))
        a = a[:,:, None]

        alpha[min_y:min_y + pad_height, min_x:min_x + pad_width, :] = a * (alpha[min_y:min_y + pad_height, min_x:min_x + pad_width, :] == 0)
    canvas[:] = canvas * (1-alpha) + color * alpha
    return canvas

if __name__ =="__main__":
    width = height = 256
    brush = Brush()


    cv2.imshow('a', brush.sample(0,1))
    cv2.waitKey(0)
    input()

    color = [rn(), rn(), rn()]
    canvas = cv2.imread('../img/Lenna.png')/255.

    render(canvas, brush, color, np.float32([[0, 0], [0, 128], [128, 0], [128, 128]]))
    render(canvas, brush, color, np.float32([[0, 128], [0, 256], [128, 128], [128, 256]]))

    cv2.imshow('brush', brush.sample(0,1))
    cv2.imshow('img', canvas)
    cv2.waitKey(0)