from os import listdir
from os.path import isfile, join
import math
import json
import sys

sys.path.insert(0, 'D:/desktop/TLSB/PersonDetector/PersonDetector')
import util


class TxtGenerator:
    JSON_EXTENSION = 'json'
    JPG_EXTENSION = 'jpg'

    def __init__(self, path):
        self.path = path

    def indices(self):
        imgs = self.get_jpg_files_from_directory()
        content_to_txt = ""
        for img in imgs:
            content_to_txt += "bg/" + img + "\n"
        f = open("bg.txt", "w+")
        f.write(content_to_txt)
        f.close()

    def generate(self):
        jsons = self.get_json_files_from_directory()
        content_to_txt = ""
        for json_file in jsons:
            json_object = self.get_json_from_file(json_file)
            json_object = json.loads(json_object)
            content_to_txt += self.extract_info(json_object)
        f = open("brazoSamples.txt", "w+")
        f.write(content_to_txt)
        f.close()

    def extract_info(self, obj):
        img_path = str(obj['imagePath'])
        info = "img/" + img_path + " "
        info += str(len(obj['shapes'])) + " "
        for box in obj['shapes']:
            coordinates = self.coordinates_info(box['points'])
            coordinates = coordinates.replace(".", ",")
            info += coordinates + " "
        info = info[:-1]
        info += "\n"
        return info

    def coordinates_info(self, array):
        coordinates = ""
        origin_coordinates, end_coordinates = self.classify_coordinates(array)
        width = end_coordinates[0] - origin_coordinates[0]
        height = end_coordinates[1] - origin_coordinates[1]
        coordinates += str(origin_coordinates[0]) + " " + str(origin_coordinates[1]) + " " + str(width) + " " + str(height)
        return coordinates

    def classify_coordinates(self, array):
        first_coordinates = array[0]
        second_coordinates = array[1]
        if first_coordinates[0] < second_coordinates[0]:
            origin_coordinates, end_coordinates = first_coordinates, second_coordinates
        else:
            origin_coordinates, end_coordinates = second_coordinates, first_coordinates
        self.swap_y_if_necessary(origin_coordinates, end_coordinates)
        return self.clean_coordinate(origin_coordinates), self.clean_coordinate(end_coordinates)

    def clean_coordinate(self, coordinate):
        coordinate[0] = math.ceil(coordinate[0])
        coordinate[1] = math.ceil(coordinate[1])
        return coordinate

    def swap_y_if_necessary(self, origin, end):
        if origin[1] > end[1]:
            temp1 = origin[1]
            temp2 = end[1]
            origin[1] = temp2
            end[1] = temp1
        return origin, end

    def get_json_from_file(self, file_path):
        json_file = open(self.path + "/" + file_path, "r")
        lines = json_file.readlines()
        json_file.close()

        content = ""
        for line in lines:
            content += line

        return content

    def get_json_files_from_directory(self):
        return self.get_extension_file_from_directory(self.is_given_file_json_file)

    def get_jpg_files_from_directory(self):
        return self.get_extension_file_from_directory(self.is_given_file_jpg_file)

    def get_extension_file_from_directory(self, filter_function):
        files = [f for f in listdir(self.path) if isfile(join(self.path, f))]
        filtered_files = list(filter(lambda x: filter_function(x), files))
        return filtered_files

    def is_given_file_json_file(self, file):
        return self.is_file_extension(file, self.JSON_EXTENSION)

    def is_given_file_jpg_file(self, file):
        return self.is_file_extension(file, self.JPG_EXTENSION)

    @staticmethod
    def is_file_extension(file, extension):
        final_token = util.getLastTokenOfPath(file)
        return final_token[1] == extension
