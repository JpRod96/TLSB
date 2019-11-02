import math
import numpy as np
from os import listdir
from os.path import isfile, join
import os
import json
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, 'D:/desktop/TLSB/PersonDetector/PersonDetector')
import util


class Plotter:
    TXT_EXTENSION = 'txt'
    TEST_SIZE = 5
    global_counter = 0

    def __init__(self, switcher):
        self.switcher = switcher

    def is_given_file_txt_file(self, file):
        final_token = util.getLastTokenOfPath(file)
        return final_token[1] == self.TXT_EXTENSION

    @staticmethod
    def is_given_path_a_dir(path):
        final_token = util.getLastTokenOfPath(path)
        return len(final_token) == 1

    def plot_txts_from_path(self, path):
        if self.is_given_path_a_dir(path):
            txt_files = self.getTxtFilesFromDirectory(path)
            for txtFile in txt_files:
                file_name = util.getLastTokenOfPath(txtFile)[0]
                os.mkdir(file_name)
                self.plot_txt_file(path + "/" + txtFile, file_name + "/" + file_name)
        else:
            file_name = util.getLastTokenOfPath(path)[0]
            os.mkdir(file_name)
            self.plot_txt_file(path, file_name + "/" + file_name)

    def plot_txt_file(self, file_path, name):
        txt_file = open(file_path, "r")
        lines = txt_file.readlines()
        txt_file.close()

        self.generate_neurons_number_comparison(lines, name)

    def generate_neurons_number_comparison(self, lines, file_name):
        test_segments = self.get_file_test_segments(lines)
        self.process_for_neurons(test_segments, file_name)

    def generate_activation_function_comparison(self, lines, file_name):
        test_segments = self.get_file_test_segments(lines)
        self.process_segments(test_segments, file_name, self.process_segment_for_function)

    def get_file_test_segments(self, lines):
        test_segment_counter = 0
        test_segments = []
        test_segment = []
        for index in range(0, len(lines)):
            actual_line = lines[index]
            if "---------------" in actual_line:
                if test_segment_counter < self.TEST_SIZE:
                    test_segment_counter += 1
                    test_segment.append(actual_line)
                else:
                    test_segment_counter = 1
                    test_segments.append(test_segment)
                    test_segment = [actual_line]
            else:
                test_segment.append(actual_line)
        test_segments.append(test_segment)
        print(len(test_segments))
        return test_segments

    def process_segments(self, segments, file_name, getter_function):
        for index in range(0, len(segments)):
            descriptor, history = getter_function(segments[index], index)
            self.plot(descriptor, history, file_name + str(index))

    def process_for_neurons(self, segments, file_name):
        to_plot = self.filter_segments(segments)
        for index in range(0, len(to_plot)):
            function_name = self.get_function_name()
            self.process_segment_for_neurons(to_plot[index], function_name, file_name)

    def filter_segments(self, segments):
        filtered = []
        for x in range(0, len(self.switcher)):
            filtered.append([])
        cont = 0
        for segment in segments:
            cont += 1
            filtered[cont - 1].append(segment)
            if cont is len(self.switcher):
                cont = 0
        return filtered

    def plot(self, title, history, file_name):
        fig = plt.figure(figsize=(5, 5))

        plt.plot(history)
        plt.ylabel('accuracy')
        plt.xlabel('N epochs')
        plt.xticks(np.arange(self.TEST_SIZE), (30, 60, 100, 130, 160))
        plt.ylim(0, 100)
        plt.grid(b=True)
        plt.suptitle(title)
        plt.savefig(file_name + '.png')
        plt.close(fig)

    def process_segment_for_function(self, segment, index):
        arq = self.find_neurons_from_segment(segment)
        function = self.get_function_name()
        descriptor = (arq, function)
        history = self.get_accuracy_history_of(segment)

        return descriptor, history

    def get_function_name(self):
        self.global_counter += 1
        value = self.switcher.get(self.global_counter, 1)
        if self.global_counter >= len(self.switcher):
            self.global_counter = 0
        return value

    def process_segment_for_neurons(self, array, function_name, file_name):
        neurons = self.get_neurons_array(array)
        history = self.get_accuracy_history_of_array(array)
        self.plot_2(neurons, history, function_name, file_name)

    def plot_2(self, neurons, history, title, file_name):
        fig = plt.figure(figsize=(15, 5))

        plt.plot(history)
        plt.ylabel('accuracy')
        plt.xlabel('N neurons')
        plt.xticks(np.arange(len(neurons)), neurons)
        plt.ylim(0, 100)
        plt.grid(b=True)
        plt.suptitle(title)
        plt.savefig(file_name + title + '.png')
        plt.close(fig)

    def get_accuracy_history_of_array(self, array):
        history = []
        for element in array:
            history.append(max(self.get_accuracy_history_of(element)))
        return history

    def get_neurons_array(self, array):
        neurons = []
        for element in array:
            neurons.append(self.find_neurons_from_segment(element))
        return neurons

    @staticmethod
    def get_accuracy_history_of(segment):
        local_max = 0
        history = []
        for line in segment:
            if "Test accuracy" in line:
                tokens = line.split()
                potential_max = math.floor(float(tokens[2]) * 100)
                local_max = potential_max if potential_max > local_max else local_max
            if "--------" in line:
                if local_max > 0:
                    history.append(local_max)
                local_max = 0
        history.append(local_max)
        return history

    def find_neurons_from_segment(self, segment):
        first_instance = 8
        neurons = []
        index = first_instance
        while True:
            line = segment[index]
            if "Dense" in line:
                dirty_neurons_string = line.split()[3]
                neurons.append(self.clean_string_number(dirty_neurons_string))
                index += 2
            else:
                break
        neurons.pop()
        return str(neurons)

    @staticmethod
    def clean_string_number(string_number):
        number = ""
        for character in string_number:
            if '0' <= character <= '9':
                number += character
        return int(number)

    def findTestSegments(self, lines, fileName):
        for index in range(0, len(lines)):
            line = lines[index]
            if "--------" in line:
                testNumber = lines[index + 1].split()[2]
                self.findTestEntries(lines, index + 2, fileName + testNumber)

    def findTestEntries(self, lines, initialIndex, fileName):
        line = lines[initialIndex]
        index = initialIndex
        cont = 1
        while (not ("------------" in line)):
            self.plot_entry(lines, index, fileName + str(cont))
            cont += 1
            index += 3
            if (index < len(lines)):
                line = lines[index]
            else:
                break

    @staticmethod
    def plot_entry(lines, initial_index, name):
        object_line = lines[initial_index + 1]
        object_line = object_line.replace(chr(39), '"')
        history = json.loads(object_line)

        fig = plt.figure(figsize=(10, 5))

        plt.subplot(121)
        plt.plot(history['acc'])
        plt.ylabel('accuracy')
        plt.xlabel('epochs')
        plt.ylim(0, 1)

        plt.subplot(122)
        plt.plot(history['loss'])
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.ylim(0, 3)

        plt.suptitle(lines[initial_index])
        plt.savefig(name + ".png")
        plt.close(fig)

    def getTxtFilesFromDirectory(self, path):
        files = [f for f in listdir(path) if isfile(join(path, f))]
        txtFiles = list(filter(lambda x: self.is_given_file_txt_file(x), files))
        return txtFiles
