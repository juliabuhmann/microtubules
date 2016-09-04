__author__ = 'julia'

from modules import utils
import h5py
import numpy as np
from skeleton import networkx_utils
from skeleton import knossos_utils
import skimage
import argparse

parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument('-preprocessing_threshold', type=float, default=0.7)
parser.add_argument('-gaussian_sigma', type=float, default=0.5)
parser.add_argument('-prob_map_inputfilename', default='default')
parser.add_argument('-knossos_outputdirectory', default='default')
parser.add_argument('-twodimensional', action='store_false')
parser.add_argument('-voxel_size', type=float, default=[1.0, 1.0, 1.0], nargs='+',)

verbose = True

results = parser.parse_args()
# Parameter setting
point_threshold = results.preprocessing_threshold
inputfile = results.prob_map_inputfilename
threshold_string = str(results.preprocessing_threshold).replace('.', '_')
gaussian_sigma = str(results.gaussian_sigma).replace('.', '_')
voxel_size = np.array(results.voxel_size)

if results.twodimensional:
    twodim = "2D"
else:
    twodim = "3D"
outputfilename = results.knossos_outputdirectory + threshold_string + gaussian_sigma + twodim + "points.nml"
print inputfile
f = h5py.File(inputfile)
data = f['exported_data'].value
f.close()
z_additive = 1

if results.twodimensional:
    coord_list = []
    nx_skeletons = []
    count = 0
    for slice in range(data.shape[2]):
        current_section = data[:, :, slice]
        bin_image = current_section.copy()
        bin_image = skimage.filters.gaussian_filter(bin_image, results.gaussian_sigma)
        bin_image = bin_image > point_threshold
        edges = bin_image
        coords = utils.detect_points(bin_image)
        coords = coords[1:]
        count += len(coords)
        for point in coords:
            coord_list.append(np.array([point[0], point[1], slice + z_additive]))
        if verbose:
            print "number of lines per section", len(coords)
    if verbose:
        print "number of datapoints in total", count

else:
    bin_image = skimage.filters.gaussian_filter(data, results.gaussian_sigma)
    bin_image = bin_image > point_threshold
    coord_list = utils.detect_points(bin_image)
    if verbose:
        print "number of datapoints in total", len(coord_list)

coord_array = np.array(coord_list)
n = networkx_utils.NxSkeleton(voxel_size=voxel_size)
n.initialize_from_edgelist(coord_array, [[0, 0]])
knossos_utils.from_nx_graphs_to_knossos([n.nx_graph], outputfilename)
