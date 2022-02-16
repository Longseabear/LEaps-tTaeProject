from render.objects import *
from render.bursh import *
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

        self.thickness = (data[9], data[10])
        self.transparency = (data[11], data[12])
        self.bgr = data[13:]

        self.brush = Brush(self.transparency, key="test.png")

    def __repr__(self):
        return "division: {}, gIdx {}, points {}, thickness {}, transparency {}, bgr {}".\
            format(self.division, self.grid_idx, self.points, self.thickness, self.transparency, self.bgr)

if __name__ =="__main__":
    draw_file_path = "../drawfile_output/girl_s100_d1/girl_s100_d1_meta.txt"
    draw_file = open(draw_file_path)

    im_h, im_w = [int(i) for i in draw_file.readline().split(' ')]
    draw_file.readline()
    scene = Canvas(512, 512)

    actions = [Action([float(i) for i in str[:-1].split(' ')], scene.height, scene.width) for str in draw_file.readlines()]

    for act in actions:
        sample_num = 1000
#        act.points = np.float32([0, 0, 128, 64, 0, 256])
#        act.points = np.float32([131.11410688, 136.84358144, 154.35334757, 126.14695484, 175.09512192, 116.64090624])
        curve = BezierQuadOffset(act.points, act.thickness, sample_num, max_thickness=0.25*512)

        all_points, t = curve.get_all_points()
        print(act)
        for i in range(len(all_points) - 1):
            canvas_ptr = np.float32([all_points[i], all_points[i + 1]])
            render(scene.raster, act.brush, act.bgr[::], canvas_ptr.reshape(4, 2), (t[i], t[i + 1]))

        # for i in range(len(all_points) - 1):
        #     cv2.circle(scene.raster, (all_points[i][1], all_points[i][0]), 1, color=(i / (len(all_points)), 0, 0),
        #                thickness=-1)
        #     cv2.circle(scene.raster, (all_points[i][3], all_points[i][2]), 1, color=(i / (len(all_points)), 0, 0),
        #                thickness=-1)

        act.points = np.int16(act.points)

        # cv2.circle(scene.raster, (act.points[1], act.points[0]), 3, color=(0, 0, 1))
        # cv2.circle(scene.raster, (act.points[3], act.points[2]), 3, color=(0, 1, 0))
        # cv2.circle(scene.raster, (act.points[5], act.points[4]), 3, color=(1, 0, 0))

#        act.brush = Brush((0.2, 1))
#        print((curve.ccw_p1[1], curve.ccw_p1[0]), act.points)

        # t = np.int16([curve.ccw_p0[1], curve.ccw_p0[0]])
        # cv2.circle(scene.raster, (t[0], t[1]), 5, color=(0, 0, 1))
        # t = np.int16([curve.cw_p0[1], curve.cw_p0[0]])
        # cv2.circle(scene.raster, (t[0], t[1]), 5, color=(0, 0, 1))
        #
        # t = np.int16([curve.ccw_p1[1], curve.ccw_p1[0]])
        # cv2.circle(scene.raster, (t[0], t[1]), 5, color=(0, 1, 0))
        # t = np.int16([curve.cw_p1[1], curve.cw_p1[0]])
        # cv2.circle(scene.raster, (t[0], t[1]), 5, color=(0, 1, 0))
        #
        # t = np.int16([curve.ccw_p2[1], curve.ccw_p2[0]])
        # cv2.circle(scene.raster, (t[0], t[1]), 5, color=(1, 0, 0))
        # t = np.int16([curve.cw_p2[1], curve.cw_p2[0]])
        # cv2.circle(scene.raster, (t[0], t[1]), 5, color=(1, 0, 0))

        cv2.imshow("img", scene.raster)
        cv2.waitKey(10)
        # scene.raster = np.ones_like(scene.raster)
    cv2.waitKey(0)
#     print(len(actions))
#     count = 0
#     for action in actions:
#         print(action)
#         curve = BezierQuadPad(action.points, action.thickness, sample_num, max_thickness=0.25 * scene.height)
# #        print(action)
#
#         all_points, t = curve.get_all_points()
#         # for i in range(len(all_points) - 1):
#         #     canvas_ptr = np.float32([all_points[i], all_points[i+1]])
#         #     render(scene.raster, action.brush, action.bgr, canvas_ptr.reshape(4,2), (t[i], t[i+1]))
#         #     if count==14:
#         #         cv2.imshow("img", scene.raster)
#         #         cv2.waitKey(0)
#         render_all_point(scene.raster,action.brush, action.bgr, np.float32(all_points), t)
#         # print(all_points[-1])
#         # print(curve.cw_p0, curve.ccw_p0, curve.cw_p1, curve.ccw_p1, curve.cw_p2, curve.ccw_p2)
#         cv2.imshow("img", scene.raster)
#
#         cv2.waitKey(0)
#         count += 1
#
#     cv2.waitKey(0)
    # brush = Brush()
    # color = [rn(), rn(), rn()]
    # canvas = cv2.imread('../img/Lenna.png')/255.

    # render(canvas, brush, color, np.float32([[0, 0], [0, 128], [128, 0], [128, 128]]))
    # render(canvas, brush, color, np.float32([[0, 128], [0, 256], [128, 128], [128, 256]]))
    #
    # cv2.imshow('brush', brush.sample(0,1))
    # cv2.imshow('img', canvas)
    # cv2.waitKey(0)