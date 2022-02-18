from paint.render_object import *
from paint.bursh import *
import numpy as np

def homography():
    pts_src = np.array([[141, 131], [480, 159], [493, 630], [64, 601]])
    pts_dst = np.array([[318, 256],[534, 372],[316, 670],[73, 473]])

    h, status = cv2.findHomography(pts_src, pts_dst)
    print(h.shape, (np.concatenate([pts_src, np.ones((4,1))], axis=1)).shape, status)
    print(np.matmul(h, np.concatenate([pts_src, np.ones((4,1))], axis=1).transpose()))
    out = np.matmul(h, np.concatenate([pts_src, np.ones((4,1))], axis=1).transpose())
    print(out[:2, :] / out[2,:])
    print((out[:2, :] / out[2,:]).transpose())


class Action():
    def __init__(self, data, height, width):
        self.division = data[0]
        self.grid_idx = (data[1], data[2])
        self.points = np.array(data[3:9])

        self.points[2] = self.points[0] + (self.points[4] - self.points[0]) * self.points[2]
        self.points[3] = self.points[1] + (self.points[5] - self.points[1]) * self.points[3]

        self.points[::2] *= height
        self.points[1::2] *= width

        self.width = width
        self.height = height

        self.thickness = (data[9], data[10])
        self.transparency = (data[11], data[12])
        self.bgr = data[13:]

        self.brush = Brush(self.bgr, self.transparency, key="random")

    def reset(self):
        self.brush = Brush(self.bgr, self.transparency, key="random")

    def __repr__(self):
        return "division: {}, gIdx {}, points {}, thickness {}, transparency {}, bgr {}".\
            format(self.division, self.grid_idx, self.points, self.thickness, self.transparency, self.bgr)

if __name__ =="__main__":
    draw_file_path = "../drawfile_output/girl_s50_d4/girl_s50_d4_meta.txt"
    draw_file = open(draw_file_path)

    im_h, im_w = [int(i) for i in draw_file.readline().split(' ')]
    draw_file.readline()
    canvas = Canvas(256, 256)

    actions = [Action([float(i) for i in str[:-1].split(' ')], canvas.height, canvas.width) for str in draw_file.readlines()]
    cv2.imshow('a', canvas.raster)

    for act in actions:
        sample_num = 100

#        act.points = np.float32([0, 0, 128, 64, 0, 256])
#        act.points = np.float32([131.11410688, 136.84358144, 154.35334757, 126.14695484, 175.09512192, 116.64090624])
        curve = BezierQuadOffsetRenderTestWithAnimation(act, sample_num, sample_interval=5, max_thickness=0.25*canvas.width)
        act.brush.setAlpha(act.brush.alpha * 0.25)
        for i in range(4):
            curve.make(canvas)
            a = np.linspace(0, 1, 25)
            a = a * a
            for t in a:
                curve.render(canvas, t)
                cv2.imshow('a', canvas.raster)
                cv2.waitKey(5)
            curve.reset()
        cv2.waitKey(100)
    cv2.waitKey(0)
