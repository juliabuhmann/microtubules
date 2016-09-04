__author__ = 'julia'
# !/usr/bin/python

from modules import PGM
from gurobipy import *
from gurobipy import Model
import numpy as np
from scipy.spatial import KDTree
from python_utilities.skeleton import networkx_utils
import networkx as nx
from skeleton import knossos_utils
from modules import utils

import argparse
import os
from modules import chunks
import h5py


def add_ground_truth_direction_vector(g, ground_truth_skeleton):
    for node, node_attr in g.nx_graph.nodes_iter(data=True):
        # Get ground truth direction vectors
        pos = node_attr['position']
        # if tuple(pos) in ground_truth_skeleton.coords_to_id:
        node_id = ground_truth_skeleton.coords_to_id[tuple(pos)]
        if 'dir_vector' in ground_truth_skeleton.nx_graph.node[node_id]:
            dir_vec_of_node = ground_truth_skeleton.nx_graph.node[node_id]['dir_vector']
        else:
            dir_vec_of_node = [0, 0, 1]
        node_attr['dir_vector'] = dir_vec_of_node
    return g


def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

    parser.add_argument('-config_filename', default='@example.cfg')
    parser.add_argument('-real_data', '--real_data',
                        help='If true, the provided data is ground truth data', action='store_false')
    parser.add_argument('-data_inputfilename', default='default')
    parser.add_argument('-gt_inputfilename', default='default')
    parser.add_argument('--quadratic', action='store_true')
    parser.add_argument('-voxel_size', type=float, default=[1.0, 1.0, 1.0], nargs='+', )
    # Preprocessing parameters
    parser.add_argument('-preprocessing_threshold', type=float, default=0.7)
    # Chunk parameters
    parser.add_argument('-chunk_size', nargs='+', default=False, type=int)
    parser.add_argument('-chunk_overlap', nargs='+', default=[100, 100, 100], type=int)
    parser.add_argument('-chunkbox_start', nargs='+', default=False, type=int)
    parser.add_argument('-chunkbox_end', nargs='+', default=False, type=int)
    parser.add_argument('--from_gt_bb', action='store_true')
    # ILP Parameters
    parser.add_argument('-dummy_edge_cost', type=float, default=100.0)
    parser.add_argument('-distance_cost', type=float, default=1.0)
    parser.add_argument('-comb_angle_cost', type=float, default=5.0)
    parser.add_argument('-perc_of_noise', type=float, default=0.0)
    parser.add_argument('-angle_cost_factor', type=float, default=0.0)
    parser.add_argument('-tree_query_distance', type=float, default=150.0)
    parser.add_argument('-selection_cost', type=float, default=-50.0)
    parser.add_argument('-pairwise', type=bool, default=True)
    parser.add_argument('--exclusive_distance', action='store_true')
    parser.add_argument('-exclusive_distance_threshold', type=float, default=25.0)
    parser.add_argument('-timelimit', type=float, default=1000.)
    # parser.add_argument('-simulation_parameter', nargs='+',
    #                     help='If ground truth data is used, this is the percentage of noise added and '
    #                          'the number of microtubules.', default=[0.1, 14], type=float)
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('-outputfolder', default='')
    parser.add_argument('-outputdirectory', default='')
    parser.add_argument('--horizontal', action='store_true')
    parser.add_argument('--merge_connected_nodes', action='store_true')
    # Postprocessing parameters
    parser.add_argument('-mask_seg_inputfilename', default='default')

    results = parser.parse_args()
    voxel_size = np.array(results.voxel_size)
    correction_factor = 1. / voxel_size
    # correction_factor = np.array([1/4.6, 1/4.6, 1/50.])
    additive = np.array([0, 0, 0])
    if results.verbose:
        print "config settings"
        print results

    # Parameters
    visualize = results.visualize
    params = PGM.ILPParameters(
        dummy_edge_cost=results.dummy_edge_cost,
        distance_cost=results.distance_cost,
        comb_angle_cost=results.comb_angle_cost,
        angle_cost_factor=results.angle_cost_factor,
        selection_cost=results.selection_cost,
        pairwise=results.pairwise,
        exclusive_distance=results.exclusive_distance,
        exclusive_distance_threshold=results.exclusive_distance_threshold
    )
    # Chunk parameters
    chunk_size = np.array(results.chunk_size)
    chunk_overlap = np.array(results.chunk_overlap)

    # Distance to build up the graphs
    tree_query_distance = results.tree_query_distance

    if results.gt_inputfilename != 'default':
        filename = results.gt_inputfilename
        skeletons = knossos_utils.from_nml_to_nx_skeletons(filename, scaling=1 / correction_factor)
        skeletons = networkx_utils.make_nx_graphs_unique(skeletons)
        all_skeletons = nx.compose_all(skeletons)
        all_skeletons = networkx_utils.NxSkeleton(all_skeletons)
        all_skeletons.add_geom_features_to_edges()
        all_skeletons.add_geom_features_to_nodes()
        all_skeletons.add_coords_to_node_id_dic()
        data = networkx_utils.from_nx_skeleton_to_datapoints(skeletons)
        gt_data = data.copy()


    filename = results.data_inputfilename
    print "preparing input data"
    extracted_skeletons = knossos_utils.from_nml_to_nx_skeletons(filename, scaling=1 / correction_factor)
    extracted_skeletons = networkx_utils.stitch_nx_graphs_together_based_on_same_coord(extracted_skeletons)

    data = networkx_utils.from_nx_skeleton_to_datapoints([extracted_skeletons.nx_graph])
    extracted_skeletons.add_coords_to_node_id_dic()
    print "input data prepared"

    if not results.chunkbox_start:
        bb_low = np.min(data[:, 0:3], axis=0)
    else:
        bb_low = np.array(results.chunkbox_start)
    if not results.chunkbox_end:
        bb_upper = np.max(data[:, 0:3], axis=0)
    else:
        bb_upper = np.array(results.chunkbox_end)

    if results.from_gt_bb:
        try:
            bb_low = np.min(gt_data[:, 0:3], axis=0)
            bb_upper = np.max(gt_data[:, 0:3], axis=0)
        except UnboundLocalError:
            print "please provide a ground truth filename or select other bounding box option"

    if not results.chunk_size:
        chunk_size = bb_upper - bb_low

    if results.verbose:
        print "Bounding box of data", bb_low, bb_upper
    # Set the chunk information
    windows = chunks.Chunks(bb_low, bb_upper - bb_low, chunk_size, chunk_overlap, verbose=results.verbose)
    windows.data_to_chunks(data[:, 0:3])
    # outputbase = os.path.join(results.outputdirectory, 'ILP', results.outputfolder,
    #                           eo.get_daystamp_folder_name(), eo.get_timestamp_folder_name())
    outputbase = os.path.join(results.outputdirectory, 'ILP_results', results.outputfolder)
    if not os.path.exists(outputbase):
        os.makedirs(outputbase)
    outputdirectory = os.path.join(outputbase, 'knossos', '')
    if not os.path.exists(outputdirectory):
        os.makedirs(outputdirectory)
    whole_nx_graph_list = []
    for (window_count, window) in enumerate(windows.chunk_list):
        if results.verbose:
            print "processing chunk %i from %i" % (window_count, len(windows.chunk_list))
        data_in_window = data[window.indeces]
        # Test whether there is enough points (~4) in window
        if data_in_window.shape[0] > 4:
            print "%i datapoints in current chunk" % data_in_window.shape[0]
            tree = KDTree(zip(data_in_window[:, 0].ravel(), data_in_window[:, 1].ravel(), data_in_window[:, 2].ravel()))
            id_to_pos = {}

            for ii in range(tree.data.shape[0]):
                id_to_pos[ii] = tree.data[ii, :]

            # Generate edgelist with the help of kdtree (only those edges are linked)
            distance_graphs = tree.query_pairs(tree_query_distance)

            # Not all data points might be connected (and the ILP can not see this) so split the
            # graphs into connected components and let them be solved separately
            # List of nx_skeletons
            nx_skeletons_all = networkx_utils.from_dtps_to_nx_skeletons(tree.data, distance_graphs)

            print "len of edgelist", len(distance_graphs)
            print "There are %i of independent clusters in the dataset" % len(nx_skeletons_all)
            nx_skeletons = []
            count = 0
            for nx_skeleton in nx_skeletons_all:
                num_of_nodes = nx_skeleton.nx_graph.number_of_nodes()
                if num_of_nodes > 3:
                    nx_skeletons.append(nx_skeleton)
                else:
                    count += 0
            print "%i clusters have been filtered out" % count

            new_skeletons = []
            model_status_list = []
            for ii in range(len(nx_skeletons)):
                # for ii in range(3):
                g = nx_skeletons[ii]
                g.add_geom_features_to_edges()
                g.add_geom_features_to_nodes()
                g.print_statistics()
                # Add Ground truth direction vectors

                # g = add_ground_truth_direction_vector(g, extracted_skeletons)
                new_graph, model = PGM.calculate_ILP_linear(g, params, timelimit=results.timelimit)
                new_skeletons.append(new_graph)
                model_status_list.append((model.status, model.MIPGap))

            skeleton_list = []
            for new_graph in new_skeletons:
                graph_for_evaluation = new_graph.nx_graph.copy()
                whole_nx_graph_list.append(graph_for_evaluation)
                new_graph.scale_positions(correction_factor, additive)
                skeleton_list.append(new_graph.nx_graph)

            print "length of skeletonlist", len(skeleton_list)

            # outputfilename = outputdirectory + "/fitted_models_chunk%i.nml" % window_count
            # knossos_utils.from_nx_graphs_to_knossos(skeleton_list, outputfilename)
        else:
            print "not enough datapoints in window %i" % window_count
            print data_in_window.shape

        data_in_window = data_in_window * correction_factor + np.array([0, 0, 0])
        # knossos_utils.datapoints_to_knossos(data_in_window,
        #                                     outputdirectory + "/orig_datapoints_chunk%i.nml" % window_count)

    unstitched_nx_skeleton = networkx_utils.stitch_nx_graphs_together(whole_nx_graph_list)
    if len(windows.chunk_list) > 1:
        stitched_nx_skeleton = networkx_utils.stitch_nx_graphs_together_based_on_same_coord(whole_nx_graph_list)
    else:
        stitched_nx_skeleton = networkx_utils.stitch_nx_graphs_together(whole_nx_graph_list)
    if visualize:
        stitched_nx_skeleton.visualize()

    if not results.mask_seg_inputfilename == 'default':
        # Mask results
        f = h5py.File(results.mask_seg_inputfilename, 'r')
        data_mask = f['seg'].value
        f.close()
        stitched_nx_skeleton.scale_positions(correction_factor, additive)
        networkx_utils.crop_nx_graph_with_mask(stitched_nx_skeleton.nx_graph, data_mask)
    else:

        stitched_nx_skeleton.scale_positions(correction_factor, additive)
    outputfilename = os.path.join(outputdirectory, 'fitted_models_all_chunks.nml')
    # knossos_utils.from_nx_graphs_to_knossos([stitched_nx_skeleton.nx_graph], outputfilename)

    unstitched_nx_skeleton.scale_positions(correction_factor, additive)
    resultfilename = os.path.join(outputdirectory, 'fitted_models_all_chunks_not_stitched.nml')
    # knossos_utils.from_nx_graphs_to_knossos([unstitched_nx_skeleton.nx_graph], resultfilename)

    input_graph = networkx_utils.stitch_nx_graphs_together([nx_skeleton.nx_graph for nx_skeleton in nx_skeletons_all])

    input_graph.scale_positions(correction_factor, additive)
    inputgraphfilename = os.path.join(outputdirectory, 'input_graph.nml')
    knossos_utils.from_nx_graphs_to_knossos([input_graph.nx_graph], inputgraphfilename)

    for min_num_nodes in range(3):
        splitted_nx_graphs = networkx_utils.split_nx_graphs([stitched_nx_skeleton.nx_graph],
                                                            min_number_of_nodes=min_num_nodes)
        outputfilename = os.path.join(outputdirectory, 'fitted_models_all_chunks_splitted_min%i.nml' % min_num_nodes)

        knossos_utils.from_nx_graphs_to_knossos(splitted_nx_graphs, outputfilename)

    outputfilename = os.path.join(outputdirectory, "input_datapoints.nml")
    extracted_skeletons.scale_positions(correction_factor, additive=additive)
    knossos_utils.from_nx_graphs_to_knossos([extracted_skeletons.nx_graph], outputfilename)
    if not results.gt_inputfilename == 'default':
        all_skeletons.scale_positions(correction_factor, additive)
        outputfilename = outputdirectory + "/GT.nml"
        knossos_utils.from_nx_graphs_to_knossos([all_skeletons.nx_graph], outputfilename)
    # Write results to file
    arguments_outputfile = outputbase + '/arguments.h5'
    termincal_commands = vars(results)
    for key, value in termincal_commands.iteritems():
        utils.write_data_to_h5(arguments_outputfile, value, key, overwrite=True, verbose=False)

    results_outputfile = outputbase + '/results.h5'
    utils.write_data_to_h5(results_outputfile, resultfilename, 'resultfilename', overwrite=True, verbose=False)
    utils.write_data_to_h5(results_outputfile,
                           stitched_nx_skeleton.get_number_of_cc(), 'number_of_skeletons_result',
                           overwrite=True, verbose=False)

    utils.write_data_to_h5(results_outputfile,
                           model_status_list, 'status_report',
                           overwrite=True, verbose=False)
    if results.verbose:
        print 'final knossosskeleton is stored in ', outputdirectory + 'fitted_models_all_chunks_splitted_min2.nml'


if __name__ == '__main__':
    main()
