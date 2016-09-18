__author__ = 'julia'
import numpy as np
import random
import h5py
import math
import glob
from scipy import ndimage
from skimage.feature import peak_local_max


def create_array_with_random_numbers(bb_low, bb_upper, count):
    assert bb_low.shape == bb_upper.shape
    col_list = []
    for col in range(bb_low.shape[0]):
        low = int(bb_low[col])
        upper = int(bb_upper[col])
        column_rand = np.array(random.sample(range(low, upper), count))
        col_list.append(column_rand)
    rand_arr = np.array(col_list).T
    return rand_arr


def get_bb(data_points):
    assert data_points.shape[1] == 3, "Bounding bos is only calculated for 3d space"
    bb_low = np.min(data_points, axis=0)
    bb_upper = np.max(data_points, axis=0)
    return bb_low, bb_upper


def write_data_to_h5(outputfile, data, datasetname, overwrite=False, verbose=False):
    f = h5py.File(outputfile)
    if type(data) == set:
        # H5 does not accept set storing
        data = list(data)
    if datasetname in f:
        if overwrite:
            del f[datasetname]
            f.create_dataset(datasetname, data=data)
            f.close()
        else:
            f.close()
            assert datasetname in f, "Dataset already exists. Use overwrite=True for overwriting existing dataset or rename dataset."
    else:
        f.create_dataset(datasetname, data=data)
        f.close()
    if verbose:
        print "data written to ", outputfile


def write_model_list(model_list, outputfile, dataset_name):
    for (ii, model) in enumerate(model_list):
        inliers = model[2]
        write_data_to_h5(outputfile, inliers, dataset_name + "%04d/inliers" % ii, overwrite=True)
        model_obj = model[0]
        write_data_to_h5(outputfile, model_obj.direction_vector, dataset_name + "%04d/direction_vector" % ii,
                         overwrite=True)
        write_data_to_h5(outputfile, model_obj.support_vector, dataset_name + "%04d/support_vector" % ii,
                         overwrite=True)
        error = model[1]
        write_data_to_h5(outputfile, error, dataset_name + "%04d/error" % ii, overwrite=True)


def get_list_with_hdf5_objects(hdf5_group, cond_type='group'):
    if cond_type == 'group':
        hdf5_list = [group for group in hdf5_group.values() if isinstance(group, h5py.Group)]
    if cond_type == "all":
        hdf5_list = [group for group in hdf5_group.values()]
    return hdf5_list


def append_id_col_to_datapoints(data_points, id):
    label_col = np.empty((data_points.shape[0], 1))
    label_col.fill(id)
    data_points = np.hstack([data_points, label_col])
    return data_points


def validate_gt(inliers, noise_id=False):
    # This validation method assumes labels at the forth column of the inliers matrix (first three are x, y, z coords)
    # The majority of the ids is assumed to be the "wanted" micro to be fitted
    id_list = list(inliers[:, 3])
    if noise_id and 14 in id_list:
        id_list = id_list.remove(14)

    id_list = np.array(id_list)
    hist_res = np.histogram(id_list)[0]
    total_points = len(id_list)
    maj_count = np.max(hist_res)
    perc = maj_count / float(total_points)
    return perc


def angle_between_vecs(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    assert len(v1) == 3
    assert len(v2) == 3
    if (v1 == v2).all():
        # print "zero"
        return 0
    elif ((v1 * (-1)) == v2).all():
        # print "zero or 180 degree"
        return 0
    else:
        try:
            angle = math.acos(np.dot(v1, v2) / ((np.linalg.norm(v1)) * np.linalg.norm(v2)))
            return angle
        except ValueError:
            print "value close to zero"
            #     print v1, v2
            angle = 0
        return angle


def calculate_spanning_angle(middle_pos, pos1, pos2):
    dir1 = middle_pos - pos1
    dir2 = middle_pos - pos2
    # Special case if incident vectors are 180 degree or 0 degree. Then you can not differentiate them
    # based on the direction vectors alone
    angle = angle_between_vecs(dir1, dir2)
    if ((dir1 * (-1)) == dir2).all():
        dist = np.linalg.norm(pos2 - pos1)
        dist_middle = np.linalg.norm(dir2)
        if dist < dist_middle:
            angle = 0
        else:
            angle = np.pi
    return angle


def from_singleh5_to_array(inputdirectory):
    inputfiles = glob.glob(inputdirectory + '*.h5')
    inputfiles.sort()
    for ii, inputfile in enumerate(inputfiles):
        f = h5py.File(inputfile)
        data = f['exported_data'].value
        f.close()
        if ii == 0:
            new_matrix = np.zeros((data.shape[0], data.shape[1], len(inputfiles)), dtype=data.dtype)
        new_matrix[:, :, ii] = data[:, :, 0]
    print "New Matrix has shape and dtype", new_matrix.shape, new_matrix.dtype
    return new_matrix


def from_h5_to_h5stack(inputdirectory, outputfile):
    ar = from_singleh5_to_array(inputdirectory)
    f = h5py.File(outputfile, 'w')
    f.create_dataset('exported_data', data=ar)
    f.close()
    print "stack written to %s" % outputfile


def from_linepoints_to_array(linepoints, z_coord):
    coord = {}
    edgelist = []
    for ii, linepair in enumerate(linepoints):
        p0, p1 = linepair
        coord[ii * 2] = np.array([p0[1], p0[0], z_coord])
        coord[ii * 2 + 1] = np.array([p1[1], p1[0], z_coord])
        edgelist.append([ii * 2, ii * 2 + 1])

    return coord, edgelist


def detect_points(binary_image):
    cc, count = ndimage.label(np.array(binary_image, dtype=np.uint32))
    coords = ndimage.measurements.center_of_mass(binary_image, cc, range(count))
    return coords


def detect_points_in_prob_map(image):
    coordinates = peak_local_max(image, min_distance=20)


def from_lists_to_precision_recall(gt_list, test_list):
    total_number_of_gt_objects = len(gt_list)
    total_number_of_test_objects = len(test_list)
    gt_set = set(gt_list)
    test_set = set(test_list)

    true_positives = gt_set.intersection(test_set)
    false_positives = test_set - gt_set
    false_negatives = gt_set - test_set
    # true_negatives_number = total_number_of_objects - len(false_positives) - len(true_positives) - len(false_negatives)
    if float(len(true_positives) + len(false_positives)) == 0:
        print "no true positives and false positives"
        return False
    else:
        precision = len(true_positives) / float(len(true_positives) + len(false_positives))
        recall = len(true_positives) / float(len(true_positives) + len(false_negatives))
        # false_positive_rate = len(false_positives)/float(len(false_positives)+true_negatives_number)
        fscore = 2 * len(true_positives) / float(
            (2 * len(true_positives) + len(false_negatives) + len(false_positives)))
    error_dic = {}
    error_dic['precision'] = precision
    error_dic['recall'] = recall
    error_dic['gt_set'] = gt_set
    error_dic['test_set'] = test_set
    error_dic['true_positives'] = true_positives
    error_dic['false_positives'] = false_positives
    error_dic['false_negatives'] = false_negatives
    error_dic['total_number_of_gt_objects'] = total_number_of_gt_objects
    error_dic['total_number_of_test_objects'] = total_number_of_test_objects
    # error_dic['false_positive_rate'] = false_positive_rate
    # error_dic['total_number_of_objects'] = total_number_of_objects
    error_dic['f_sc0re'] = fscore
    return error_dic


def print_dic(dictionary):
    for key, value in dictionary.iteritems():
        print key
        print value
        print "-----"


def calculate_intersection_plane(p0, p1, normal_vec, support_vec):
    pass


# intersection function
def isect_line_plane_v3(p0, p1, p_co, p_no, epsilon=1e-6):
    """
    p0, p1: define the line
    p_co, p_no: define the plane:
        p_co is a point on the plane (plane coordinate).
        p_no is a normal vector defining the plane direction; does not need to be normalized.

    return a Vector or None (when the intersection can't be found).
    """

    u = sub_v3v3(p1, p0)
    dot = dot_v3v3(p_no, u)
    print dot
    if abs(dot) > epsilon:
        # the factor of the point between p0 -> p1 (0 - 1)
        # if 'fac' is between (0 - 1) the point intersects with the segment.
        # otherwise:
        #  < 0.0: behind p0.
        #  > 1.0: infront of p1.
        w = sub_v3v3(p0, p_co)
        fac = -dot_v3v3(p_no, w) / dot
        u = mul_v3_fl(u, fac)
        return add_v3v3(p0, u)
    else:
        # The segment is parallel to plane
        return None


# ----------------------
# generic math functions

def add_v3v3(v0, v1):
    return (
        v0[0] + v1[0],
        v0[1] + v1[1],
        v0[2] + v1[2],
    )


def sub_v3v3(v0, v1):
    return (
        v0[0] - v1[0],
        v0[1] - v1[1],
        v0[2] - v1[2],
    )


def dot_v3v3(v0, v1):
    return (
        (v0[0] * v1[0]) +
        (v0[1] * v1[1]) +
        (v0[2] * v1[2])
    )


def len_squared_v3(v0):
    return dot_v3v3(v0, v0)


def mul_v3_fl(v0, f):
    return (
        v0[0] * f,
        v0[1] * f,
        v0[2] * f,
    )
