import os
import cv2
import ast
import numpy


class OutputMaker:
    def __init__(self, path='/home/student/Documents/VOC2012/SegmentationClass'):
        self.class_data = {(0, 0, 0): 0}
        self.path = path

    def create_image_data(self):
        for path in self.get_all_files_from_path():
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            image = (image.reshape((image.shape[0] * image.shape[1], 3))).tolist()
            for pixel in image:
                pixel = tuple(pixel)
                if pixel not in self.class_data.keys():
                    self.class_data[pixel] = len(self.class_data) - 1

    def get_all_files_from_path(self):
        files = []
        for r, d, f in os.walk(self.path):
            for file in f:
                files.append(os.path.join(r, file))
        return files

    def save_image_data(self, path="saved_progress.txt"):
        file = open(path, 'w')
        file.write(str(self.class_data))
        file.close()

    def load_image_data(self, path="saved_progress.txt"):
        file = open(path, 'r')
        contents = file.read()
        self.class_data = ast.literal_eval(contents)
        file.close()

    def create_target(self, image):
        image = numpy.array(image.convert("RGB"))
        width, height, dimensions = image.shape
        new_image = numpy.zeros((21, width, height), dtype="int")
        for row in range(width):
            for column in range(height):
                key = image[row][column]
                key[0], key[2] = key[2], key[0]
                key = tuple(key)

                new_image[self.class_data[key]][row][column] = 1
        return new_image
