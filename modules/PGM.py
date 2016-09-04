__author__ = 'julia'
#!/usr/bin/python

try:
    from mayavi import mlab
except ImportError:
    print 'mayavi not installed, visualization methods will not be available'
import matplotlib.pyplot as plt
import sys
import math
from gurobipy import *
from gurobipy import Model
import numpy as np
from scipy.spatial import KDTree
from skeleton import networkx_utils
import networkx as nx
from skeleton import knossos_utils
import utils


from numpy import genfromtxt
import math




class ILPParameters(object):
    # As proposed in python Cookbook. (usage a bit like a c struct)
    def __init__(self, dummy_edge_cost=100, distance_cost=1, comb_angle_cost=5, angle_cost_factor=0,
                 selection_cost=-50, pairwise=True, exclusive_distance=False, exclusive_distance_threshold=25, **kwds):
        #First set default values
        default_dic = {}
        default_dic['dummy_edge_cost'] = dummy_edge_cost
        default_dic['distance_cost'] = distance_cost
        default_dic['comb_angle_cost'] = comb_angle_cost
        default_dic['angle_cost_factor'] = angle_cost_factor
        default_dic['selection_cost'] = selection_cost
        default_dic['pairwise'] = pairwise
        default_dic['exclusive_distance'] = exclusive_distance
        default_dic['exclusive_distance_threshold'] = exclusive_distance_threshold


        self.__dict__.update(default_dic)
        self.__dict__.update(kwds)




def get_angle(g, i, cur_dir_vec, noise_sigma=False):
    #Get ground truth direction vectors
    # #Get ground truth direction vectors
    # pos = g.nx_graph.node[node_id]['position']
    # if tuple(pos) in all_skeletons.coords_to_id:
    #     node_id = all_skeletons.coords_to_id[tuple(pos)]
    # # if 'dir_vector' in all_skeletons.nx_graph.node[node_id]:
    #     dir_vec_of_node = all_skeletons.nx_graph.node[node_id]['dir_vector']
    # else:
    #     dir_vec_of_node = np.random.random((1, 3))
    if not 'dir_vector' in g.nx_graph.node[i]:
        dir_vec_of_node = [0, 0, 1]
    else:
        dir_vec_of_node = g.nx_graph.node[i]['dir_vector']


    # dir_vec_of_node = np.array([0, 0, 1])
    angle = utils.angle_between_vecs(dir_vec_of_node, cur_dir_vec)
    if noise_sigma:
        mu = 0
        sigma = noise_sigma
        angle_noise = np.abs(sigma * np.random.randn() + mu)
        # angle_noise = np.random.uniform(0, np.pi/2.)
        angle += angle_noise

    if angle > np.pi:
        angle = angle - np.pi

    if angle > np.pi/2.:
        angle = np.abs(angle - np.pi)

    if math.isnan(angle):
        print "angle is not defined, check why"
        angle = 0
    return angle


def create_graph_from_solution(g_old, solution):
    print " creating new graph"
    nx_skeleton = networkx_utils.NxSkeleton()
    position_dic = g_old.node_feature_to_dic('position')
    nx_skeleton.initialize_from_edgelist(position_dic, solution, only_from_edgelist=True)
    return nx_skeleton

def check_consistency(comb_solution, solution):
    print "checking consistency"
    for comb_edge, sol in comb_solution.iteritems():
        node_id = comb_edge[0]
        neighbour1 = comb_edge[1]
        neighbour2 = comb_edge[2]
        if sol==1.0:
            # Check for those combi edges that are on, that also the corresponding edge variables are on
            assert solution[node_id, neighbour1] == 1
            assert solution[node_id, neighbour2] == 1
        else:
            assert solution[node_id, neighbour1] + solution[node_id, neighbour2] <= 1


def add_pairwaisefactors_to_model(model, g, params, variables):
    # Add additional variables that represent the combination of two edges respectively
    edge_combinations = {}
    for node_id, node_attr in g.nx_graph.nodes_iter(data=True):
        neighbours = g.nx_graph.neighbors(node_id)
        neighbours.sort()
        for ii in range(len(neighbours)-1):
            neighbour_id1 = neighbours[ii]
            for jj in range(len(neighbours)-ii-1):
                neighbour_id2 = neighbours[jj+ii+1]
                spanning_angle = node_attr['spanning_angles'][(neighbour_id1, neighbour_id2)]
                edge_combinations[node_id, neighbour_id1, neighbour_id2] = \
                    model.addVar(obj=(np.abs(spanning_angle-np.pi)*params.comb_angle_cost)**2, vtype=GRB.BINARY,
                                 name='node_%i_edge_%i_%i' %(node_id, neighbour_id1, neighbour_id2))

    model.update()
    # Add corresponding constraints to new introduced variables (specifically,
    # combination of egdes should only be switched on if both corresponding edges are switched on)

    for node_id, node_attr in g.nx_graph.nodes_iter(data=True):
        neighbours = g.nx_graph.neighbors(node_id)
        neighbours.sort()
        for ii in range(len(neighbours)-1):
            neighbour_id1 = neighbours[ii]
            for jj in range(len(neighbours)-ii-1):
                neighbour_id2 = neighbours[jj+ii+1]
                model.addConstr(edge_combinations[node_id, neighbour_id1, neighbour_id2] <= variables[node_id, neighbour_id1])
                model.addConstr(edge_combinations[node_id, neighbour_id1, neighbour_id2] <= variables[node_id, neighbour_id2])
                model.addConstr(edge_combinations[node_id, neighbour_id1, neighbour_id2] +1 >=
                                variables[node_id, neighbour_id2] + variables[node_id, neighbour_id1])
    model.update()
    return model, edge_combinations

def add_proximity_exclusivity(model, g, selection_cost_variables, distance_threshold):
    for node_id, node_attr in g.nx_graph.nodes_iter(data=True):
        neighbours = g.nx_graph.neighbors(node_id)
        neighbours.sort()
        neighbours_in_proximity = []
        for neighbour in neighbours:
            distance = g.nx_graph[node_id][neighbour]['distance']
            if distance < distance_threshold:
                neighbours_in_proximity.append(neighbour)
        if neighbours_in_proximity:
            model.addConstr(selection_cost_variables[node_id]+
                            1./len(neighbours_in_proximity)*quicksum(selection_cost_variables[neighbour] for neighbour in neighbours_in_proximity) <= 1)
    model.update()
    return model

def calculate_ILP_linear(g, params,
                  verbose=True, visualize=False, timelimit=120):
    # This is a reimplementaiton of the original version of setting up the ILP for microtubule tracking. (as described
    # in paper)
    if verbose:
        print "starting constructing ILP"
        for key, value in params.__dict__.iteritems():
            print key, value
    model = Model()
    if timelimit > 0:
        model.setParam(GRB.Param.TimeLimit, timelimit)
    variables = {}
    # Add variables to the model, per edge in the graph one variable
    for edge in g.nx_graph.edges_iter(data=True):
        i = edge[0]
        j = edge[1]
        dist = edge[2]['distance']
        cur_dir_vector = edge[2]['dir_vector']
        angle_cost = 0
        if params.angle_cost_factor != 0:
            angle1 = get_angle(g, i, cur_dir_vector)
            angle2 = get_angle(g, j, cur_dir_vector)
            angle_cost = (angle1 + angle2)*params.angle_cost_factor


        variables[i, j] = model.addVar(obj=(dist*params.distance_cost+angle_cost)**1, vtype=GRB.BINARY, name='e'+str(i)+'_'+str(j))
        variables[j, i] = variables[i, j]


    sink_variables = {}
    selection_cost_variables = {}
    for node_id in g.nx_graph.nodes_iter():
        # Add dummy node (sink), here the costs decide about how expensive it is to open / close a chain
        sink_variables[node_id] = model.addVar(obj=params.dummy_edge_cost, vtype=GRB.BINARY, name='dummy_%s' %str(node_id))
        # Add cost for selecting a candidate (this is uniformly set,
        # but can potentially exchanged with adaptive costs from underlying pipeline)
        selection_cost_variables[node_id] = model.addVar(obj=params.selection_cost, vtype=GRB.BINARY, name='selectioncost_%s' %str(node_id))
    model.update()

    # Add consistency constraints for guaranteeing coherent topology
    # If a candidate is selected, two edge variables should also be selected
    for node_id in g.nx_graph.nodes_iter():
        neighbours = g.nx_graph.neighbors(node_id)
        model.addConstr(2*selection_cost_variables[node_id] -(quicksum(variables[node_id, j] for j in neighbours) +
                        sink_variables[node_id]) <= 0,
                        name='nodeconstraint1_%i' %node_id)
        model.addConstr(2*selection_cost_variables[node_id] -(quicksum(variables[node_id, j] for j in neighbours) +
                        sink_variables[node_id]) >= 0,
                        name='nodeconstraint2_%i' %node_id)

    # model.addConstr(selection_cost_variables[0] == 1,
    #                 name='nodeconstraint3_%i' %node_id)
    model.update()
    # If a link is selected, the two incident candidates on this link also need to be selected
    # The other way round is not true, if a candidate is selected
    for edge in g.nx_graph.edges_iter():
        node_id1 = edge[0]
        node_id2 = edge[1]
        model.addConstr(2*variables[edge]-selection_cost_variables[node_id1] - selection_cost_variables[node_id2] <= 0,
                        name='edgeconstraint_%i_%i' %edge)

    model.update()
    print "Intermediate number of variables before adding combination aspect", model.NumVars

    if params.pairwise:
        # Microtubules are rigid. Add cost for bending
        model, edge_combinations = add_pairwaisefactors_to_model(model, g, params, variables)
    if params.exclusive_distance:
        # Too many candidates for the same microtubules cause parallel occurence
        model = add_proximity_exclusivity(model, g, selection_cost_variables, params.exclusive_distance_threshold)
    if verbose:
        print "starting Optimization"
        print "Number of variables %i" % model.NumVars
        print "Number of linear constraints %i" %model.NumConstrs
    # model.display()
    model.optimize()

    print 'Obj:', model.ObjVal
    # Contains a label for each edge
    solution = model.getAttr("X", variables)
    if params.pairwise:
        comb_solution = model.getAttr('X', edge_combinations)
        check_consistency(comb_solution, solution)
    new_edgelist = []
    for edge, sol in solution.iteritems():
        if sol == 1:
            new_edgelist.append(edge)

    if visualize:
        g.visualize(edgelist=new_edgelist, only_edges_from_edgelist=True)
        mlab.show()

    # Create new graph from solution
    nx_skeleton_solution = create_graph_from_solution(g, new_edgelist)
    nx_skeleton_solution.print_statistics()

    return nx_skeleton_solution, model


def calculate_ILP(g, quadratic=False, dummy_edge_cost = 100, distance_cost=1, comb_angle_cost = 5,
                  verbose=True, visualize=False, linear=False, angle_cost_factor=0):
    print "starting constructing ILP"
    model = Model()
    # model.setParam(GRB.Param.TimeLimit, 7200.0)
    variables = {}
    # Add variables to the model, per edge in the graph one variable
    for edge in g.nx_graph.edges_iter(data=True):
        i = edge[0]
        j = edge[1]
        dist = edge[2]['distance']

        cur_dir_vector = edge[2]['dir_vector']
        angle_cost = 0
        if angle_cost_factor != 0:
            angle1 = get_angle(g, i, cur_dir_vector)
            angle2 = get_angle(g, j, cur_dir_vector)
            angle_cost = (angle1 + angle2)*angle_cost_factor
            # angle = utils.angle_between_vecs(dir_vec_of_node, cur_dir_vec)
        # print dist
        # print angle1, angle2, cur_dir_vector
        variables[i, j] = model.addVar(obj=(dist*distance_cost+angle_cost)**1, vtype=GRB.BINARY, name='e'+str(i)+'_'+str(j))
        variables[j, i] = variables[i, j]
    # This is the number for the dummy node. This guarantees that the id is not already occcupied by a node id
    n = np.max(g.nx_graph.nodes())+1

    g.add_geom_features_to_edges()
    g.add_geom_features_to_nodes()
    for node_id in g.nx_graph.nodes_iter():
        # Add dummy node, here the costs decide about how expensive it is to open / close a chain
        variables[node_id, n] = model.addVar(obj=dummy_edge_cost, vtype=GRB.BINARY, name='dummy1_%s' %str(node_id))
        variables[node_id, n+1] = model.addVar(obj=dummy_edge_cost, vtype=GRB.BINARY, name='dummy2_%s' %str(node_id))
        variables[n, node_id] = variables[node_id, n]
        variables[n+1, node_id] = variables[node_id, n+1]
    model.update()

    for node_id in g.nx_graph.nodes_iter():

        neighbours = g.nx_graph.neighbors(node_id)
        neighbours.extend([n, n+1])
        model.addConstr(quicksum(variables[node_id, j] for j in neighbours) == 2, name='node_%i' %node_id)

    model.update()
    print "Intermediate number of variables before adding combination aspect", model.NumVars
    if quadratic:
        if verbose:
            print "adding quadratic terms"
        quad = QuadExpr()
        for node_id, node_attr in g.nx_graph.nodes_iter(data=True):
            neighbours = g.nx_graph.neighbors(node_id)
            neighbours.sort()

            # Add the quadratic terms for the combination of each edge pair incident to the same node

            for ii in range(len(neighbours)-1):
                neighbour_id1 = neighbours[ii]
                for jj in range(len(neighbours)-ii-1):
                    neighbour_id2 = neighbours[jj+ii+1]
                    spanning_angle = node_attr['spanning_angles'][(neighbour_id1, neighbour_id2)]
                    quad.add(variables[node_id, neighbour_id1]*variables[node_id, neighbour_id2]*(np.abs(spanning_angle-np.pi)*comb_angle_cost)**2)

        # Add linear costs to quadratic expression
        for edge in g.nx_graph.edges_iter(data=True):
            i = edge[0]
            j = edge[1]
            dist = edge[2]['distance']
            cur_dir_vector = edge[2]['dir_vector']
            angle1 = get_angle(g, i, cur_dir_vector)
            angle2 = get_angle(g, j, cur_dir_vector)
            angle_cost = (angle1 + angle2)*angle_cost_factor
            total_cost = angle_cost + dist*distance_cost
            quad.add(variables[i, j]*(total_cost))

        for node_id in g.nx_graph.nodes_iter():
            # Add dummy node, here the costs decide about how expensive it is to open / close a chain
            quad.add(variables[node_id, n]*dummy_edge_cost)
            quad.add(variables[node_id, n+1]*dummy_edge_cost)
        print "model collection costs finished"
        model.setObjective(quad)
        model.update()
        print "Number of variables after adding quadratic component", model.NumVars

    elif linear:
        # Add additional variables that represent the combination of two edges respectively
        edge_combinations = {}
        for node_id, node_attr in g.nx_graph.nodes_iter(data=True):
            neighbours = g.nx_graph.neighbors(node_id)
            neighbours.sort()
            for ii in range(len(neighbours)-1):
                neighbour_id1 = neighbours[ii]
                for jj in range(len(neighbours)-ii-1):
                    neighbour_id2 = neighbours[jj+ii+1]
                    spanning_angle = node_attr['spanning_angles'][(neighbour_id1, neighbour_id2)]
                    edge_combinations[node_id, neighbour_id1, neighbour_id2] = \
                        model.addVar(obj=(np.abs(spanning_angle-np.pi)*comb_angle_cost)**2, vtype=GRB.BINARY,
                                     name='node_%i_edge_%i_%i' %(node_id, neighbour_id1, neighbour_id2))

        model.update()
        # Add corresponding constraints to new introduced variables (specifically,
        # combination of egdes should only be switched on if both corresponding edges are switched on)

        for node_id, node_attr in g.nx_graph.nodes_iter(data=True):
            neighbours = g.nx_graph.neighbors(node_id)
            neighbours.sort()
            for ii in range(len(neighbours)-1):
                neighbour_id1 = neighbours[ii]
                for jj in range(len(neighbours)-ii-1):
                    neighbour_id2 = neighbours[jj+ii+1]
                    model.addConstr(edge_combinations[node_id, neighbour_id1, neighbour_id2] <= variables[node_id, neighbour_id1])
                    model.addConstr(edge_combinations[node_id, neighbour_id1, neighbour_id2] <= variables[node_id, neighbour_id2])
                    model.addConstr(edge_combinations[node_id, neighbour_id1, neighbour_id2] +1 >=
                                    variables[node_id, neighbour_id2] + variables[node_id, neighbour_id1])
        model.update()





    if verbose:
        print "starting Optimization"
        print "Number of variables %i" % model.NumVars
        print "Number of linear constraints %i" %model.NumConstrs
    model.optimize()
    # Contains a label for each edge
    solution = model.getAttr("X", variables)
    if linear:
        comb_solution = model.getAttr("X", edge_combinations)

    new_edgelist = []

    count2 = 0
    for edge, sol in solution.iteritems():
        if n+1 in edge or n in edge:
            if sol == 1:
                count2 += 1
        elif sol == 1:
            if not edge in new_edgelist or not (edge[1], edge[0]) in new_edgelist:
                new_edgelist.append(edge)

    if visualize:
        g.visualize(edgelist=new_edgelist, only_edges_from_edgelist=True)
        mlab.show()

    # Create new graph from solution
    graph_solution = create_graph_from_solution(g, new_edgelist)
    graph_solution.print_statistics()
    if linear:
        check_consistency(comb_solution, solution)

    if linear:
        return graph_solution, [model, solution, comb_solution]
    else:
        return graph_solution, [model, solution]



