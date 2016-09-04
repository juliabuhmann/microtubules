__author__ = 'julia'

import numpy as np


class Chunks(object):
    def __init__(self, start_coord, box_size, chunk_size, overlap, verbose=False):
        self.start_coord = start_coord
        self.box_size = box_size
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunk_list = []
        self.verbose = verbose
        self.initialize_chunk_list()

    def initialize_chunk_list(self):
        modular_of_box = np.remainder(self.box_size, self.chunk_size)
        if not modular_of_box.all():
            num_chunks = (self.box_size / self.chunk_size).astype(int)
        else:
            print "Given chunk size is not multiple of boxe size. Enlarging box size."
            add_to_box = np.ones((1, 3), dtype=int)
            add_to_box[modular_of_box == 0] = 1
            num_chunks = (self.box_size / self.chunk_size).astype(int) + add_to_box[0]
        # TODO(julia): implement a more efficient way to do that
        if self.verbose:
            print "initializing chunks"
        for x in range(num_chunks[0]):
            low_x = self.start_coord[0] + self.chunk_size[0] * x - self.overlap[0]
            upper_x = self.start_coord[0] + self.chunk_size[0] * (x + 1) + self.overlap[0]
            for y in range(num_chunks[1]):
                low_y = self.start_coord[1] + self.chunk_size[1] * y - self.overlap[1]
                upper_y = self.start_coord[1] + self.chunk_size[1] * (y + 1) + self.overlap[1]
                for z in range(num_chunks[2]):
                    low_z = self.start_coord[2] + self.chunk_size[2] * z - self.overlap[2]
                    upper_z = self.start_coord[2] + self.chunk_size[2] * (z + 1) + self.overlap[2]
                    lower_coord = np.array([low_x, low_y, low_z])
                    upper_coord = np.array([upper_x, upper_y, upper_z])
                    chunk = Chunk(lower_coord, upper_coord)
                    self.chunk_list.append(chunk)
        if self.verbose:
            print "Box has %i chunks" % len(self.chunk_list)
            print "Exact dimension of num of chunks", num_chunks

    def data_to_chunks(self, data):
        assert data.shape[1] == 3, "Data has to be provided with dimensions n*3."
        for chunk in self.chunk_list:
            for point_index in range(data.shape[0]):
                point = data[point_index, :]
                # Check whether point is in chunk
                point_inside_chunk1 = (chunk.lower_coord <= point).all()
                point_inside_chunk2 = (point < chunk.upper_coord).all()
                if point_inside_chunk1 & point_inside_chunk2:
                    chunk.indeces.append(point_index)


class Chunk(object):
    def __init__(self, lower_coord, upper_coord):
        self.lower_coord = lower_coord
        self.upper_coord = upper_coord
        self.indeces = []
