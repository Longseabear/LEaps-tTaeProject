from genericpath import exists
import os
import shutil

class DrawFileFormat:
    def __init__(self, title, width, height, img_path, output_folder="drawfile_output") -> None:
        self.title = title

        self.width = width
        self.height = height
        self.action_list = []
        self.input_img_path = img_path
        self.output_folder = output_folder
        self.final_output_path = None
        

    def save(self):
        
        output_folder = os.path.join(self.output_folder, self.title)
        os.makedirs(output_folder, exist_ok=True)

        output_path = os.path.join(output_folder, self.title + "_meta.txt")
        f = open(output_path, 'w')
        f.write("{} {}\n".format(self.width, self.height)) 

        # division grid_index_x grid_index_y x0, y0, x1, y1, x2, y2, r0, t0, r1, t1, B, G, R
        f.write(str(len(self.action_list)) + "\n")
        for action in self.action_list:
            f.write(' '.join([str(item) for item in action]) + '\n')
        
        shutil.copyfile(self.input_img_path, os.path.join(output_folder, self.title + ".png"))
        if self.final_output_path is not None:
            shutil.copyfile(self.final_output_path, os.path.join(output_folder, self.title + "_output.png"))

if __name__ == "__main__":
    import numpy as np
    f = DrawFileFormat(250, 250, 1)
    for i in range(5):
        item = list(np.random.normal(size=(16)))
        f.action_list.append(item)
    f.save('a.drawfile')

