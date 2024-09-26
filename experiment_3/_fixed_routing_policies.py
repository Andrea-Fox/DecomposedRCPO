import numpy as np
import math
from _routing_policy import *

def fixed_routing_constraints(env, consider_areas_of_interest = False):

    # this function defines a routing policy that is fixed and depends on the values of the constraints of the server
    # the routing policy will have to consider the fact that systems with smaller values of the constraints will likely 
    # discard an higher quantity if videos

    routing_policy = np.zeros((env.N_servers, env.N_servers))

    # we define the weight for each server, depending on the value of the constraints
    # the weight will be given by the softmax function
    weight = np.zeros(env.N_servers)
    for i in range(env.N_servers):
        weight[i] = env.servers[i].access_capacity
    weight = weight/np.sum(weight)

    # we define the routing policy
    for i in range(env.N_servers):
        for j in range(env.N_servers):
            if consider_areas_of_interest:
                # we must consider that some areas are not of interest for specific servers
                if env.servers[j].areas_of_interest[i]:
                    routing_policy[i, j] = weight[j]
                else:
                    routing_policy[i, j] = 0
            else:
                routing_policy[i, j] = weight[j]

    # we normalize the routing policy
    for i in range(env.N_servers):
        routing_policy[i, :] = routing_policy[i, :]/np.sum(routing_policy[i, :])

    return routing_policy



# def fixed_routing_rewards(env, consider_areas_of_interest = False):


def fixed_routing_uniform_load_balancing(env, consider_areas_of_interest = False):
    routing_policy = np.zeros((env.N_servers, env.N_servers))

    # we define the weight for each server, depending on the value of the constraints
    # the weight will be given by the softmax function
    weight = np.zeros(env.N_servers)
    for i in range(env.N_servers):
        weight[i] = 1/env.N_servers
    weight = weight/np.sum(weight)

    # we define the routing policy
    for i in range(env.N_servers):
        for j in range(env.N_servers):
            if consider_areas_of_interest:
                # we must consider that some areas are not of interest for specific servers
                if env.servers[j].areas_of_interest[i]:
                    routing_policy[i, j] = weight[j]
                else:
                    routing_policy[i, j] = 0
            else:
                routing_policy[i, j] = weight[j]

    # we normalize the routing policy
    for i in range(env.N_servers):
        sum_i = np.sum(routing_policy[i, :])
        for j in range(env.N_servers):
            routing_policy[i, j] = routing_policy[i, j]/sum_i

    return routing_policy


def fixed_routing_area_ranking(env, consider_areas_of_interest = False):
    # in this case the routing policy is fixed and depends on the value of the instantaneous reward of each area
    # it is assumed that the order is maintained

    routing_policy = np.zeros((env.N_servers, env.N_servers))

    # we define the weight for each server, depending on the value of the constraints
    # the weight will be given by the softmax function
    weight = np.zeros((env.N_servers, env.N_servers))
    for i in range(env.N_servers):
        for j in range(env.N_servers):
            temporary_state = { 'server_occupation': np.zeros((env.N_servers, )), 'last_origin_server': i}
            weight[i, j] = env.servers[j].reward_function(temporary_state) 
    weight = weight/np.sum(weight)

    # we define the routing policy
    for i in range(env.N_servers):
        for j in range(env.N_servers):
            if consider_areas_of_interest:
                # we must consider that some areas are not of interest for specific servers
                if env.servers[j].areas_of_interest[i]:
                    routing_policy[i, j] = weight[i, j]
                else:
                    routing_policy[i, j] = 0
            else:
                routing_policy[i, j] = weight[i, j]

    # we normalize the routing policy
    for i in range(env.N_servers):
        sum_i = np.sum(routing_policy[i, :])
        for j in range(env.N_servers):
            routing_policy[i, j] = routing_policy[i, j]/sum_i

    return routing_policy


def fixed_routing_permanence_time(env, consider_areas_of_interest = False):
    # in this case the routing policy is fixed and depends on the value of the parameter mu of each server
    # This value describes the permanence time of the videos in the server

    # main idea: if i a server, video remain less time, we sent there more videos (it is more likely that 
    # the server is empty)

    routing_policy = np.zeros((env.N_servers, env.N_servers))

    # we define the weight for each server, depending on the value of the constraints
    # the weight will be given by the softmax function
    weight = np.zeros((env.N_servers, ))
    for j in range(env.N_servers):
        weight[j] = (1/env.servers[j].server_processing_rate)**.5
    weight = weight/np.sum(weight)

    # we define the routing policy
    for i in range(env.N_servers):
        for j in range(env.N_servers):
            if consider_areas_of_interest:
                # we must consider that some areas are not of interest for specific servers
                if env.servers[j].areas_of_interest[i]:
                    routing_policy[i, j] = weight[j]
                else:
                    routing_policy[i, j] = 0
            else:
                routing_policy[i, j] = weight[j]

    # we normalize the routing policy
    for i in range(env.N_servers):
        sum_i = np.sum(routing_policy[i, :])
        for j in range(env.N_servers):
            routing_policy[i, j] = routing_policy[i, j]/sum_i

    return routing_policy



def fixed_routing_permanence_time_and_constraint(env, consider_areas_of_interest = False):
    # in this case the routing policy is fixed and depends on the value of the parameter mu of each server
    # This value describes the permanence time of the videos in the server

    # main idea: if i a server, video remain less time, we sent there more videos (it is more likely that 
    # the server is empty)

    routing_policy = np.zeros((env.N_servers, env.N_servers))

    # we define the weight for each server, depending on the value of the constraints
    # the weight will be given by the softmax function
    weight = np.zeros((env.N_servers, ))
    for j in range(env.N_servers):
        weight[j] =  (env.servers[j].access_capacity) * (env.servers[j].server_processing_rate)**.75
    weight = weight/np.sum(weight)

    # we define the routing policy
    for i in range(env.N_servers):
        for j in range(env.N_servers):
            if consider_areas_of_interest:
                # we must consider that some areas are not of interest for specific servers
                if env.servers[j].areas_of_interest[i]:
                    routing_policy[i, j] = weight[j]
                else:
                    routing_policy[i, j] = 0
            else:
                routing_policy[i, j] = weight[j]

    # we normalize the routing policy
    for i in range(env.N_servers):
        sum_i = np.sum(routing_policy[i, :])
        for j in range(env.N_servers):
            routing_policy[i, j] = routing_policy[i, j]/sum_i

    return routing_policy


def fixed_routing_occupation(env, consider_areas_of_interest = False, weight_multiplier = 1):
    # in this case we do not look at the occupation of the area considered, but rather at the total occupation of each server
    # the routing policy looks at the current occupation of the servers and defines a probability distribution (based on softmax) 
    # The actual action (i.e. the server to which the video is sent) is chosen according to this probability distribution
    # max_capacity = np.max([env.servers[i].memory_capacity for i in range(env.N_servers)]) + 1
    # # routing_policy[i1, i2, i3, i4, i5] denotes the probability of sending a video from area i4 to server i5 when the \
    # # total occupation of the servers  is, respectively, i1, i2, i3
    # routing_probability = np.zeros(tuple([max_capacity for i in range(env.N_servers)]) + (env.N_servers, env.N_servers))   
    # # routing_probability = np.zeros((max_capacity, max_capacity, max_capacity, env.N_servers, env.N_servers))
# 
    # approximate_routing_policy = np.zeros((env.N_servers, env.N_servers))
# 
    # if consider_areas_of_interest:
    #     for origin_area in range(env.N_servers):
    #         for server_1_occupation_level in range(env.servers[0].memory_capacity+1):
    #             for server_2_occupation_level in range(env.servers[1].memory_capacity+1):
    #                 for server_3_occupation_level in range(env.servers[2].memory_capacity+1):
    #                     if env.servers[0].areas_of_interest[origin_area]:
    #                         routing_probability[server_1_occupation_level, server_2_occupation_level, server_3_occupation_level, origin_area, 0] = math.exp(- weight_multiplier * server_1_occupation_level/env.servers[0].access_capacity)
    #                     if env.servers[1].areas_of_interest[origin_area]:
    #                         routing_probability[server_1_occupation_level, server_2_occupation_level, server_3_occupation_level, origin_area, 1] = math.exp(- weight_multiplier * server_2_occupation_level/env.servers[1].access_capacity)
    #                     if env.servers[2].areas_of_interest[origin_area]:
    #                         routing_probability[server_1_occupation_level, server_2_occupation_level, server_3_occupation_level, origin_area, 2] = math.exp(- weight_multiplier * server_3_occupation_level/env.servers[2].access_capacity)
    #                     # we normalize the routing policy
    #                     sum_i = np.sum(routing_probability[server_1_occupation_level, server_2_occupation_level, server_3_occupation_level, origin_area, :])
    #                     for j in range(env.N_servers):
    #                         routing_probability[server_1_occupation_level, server_2_occupation_level, server_3_occupation_level, origin_area, j] = routing_probability[server_1_occupation_level, server_2_occupation_level, server_3_occupation_level, origin_area, j]/sum_i
    # else:
    #     # we define the weight for each server, depending on the value of the current occupation and on the capacity
    #     for server_1_occupation_level in range(env.servers[0].memory_capacity):
    #         for server_2_occupation_level in range(env.servers[1].memory_capacity):
    #             for server_3_occupation_level in range(env.servers[2].memory_capacity):
    #                 for origin_area in range(env.N_servers):
    #                     routing_probability[server_1_occupation_level, server_2_occupation_level, server_3_occupation_level, origin_area, 0] = math.exp(-server_1_occupation_level/env.servers[0].access_capacity)
    #                     routing_probability[server_1_occupation_level, server_2_occupation_level, server_3_occupation_level, origin_area, 1] = math.exp(-server_2_occupation_level/env.servers[1].access_capacity)
    #                     routing_probability[server_1_occupation_level, server_2_occupation_level, server_3_occupation_level, origin_area, 2] = math.exp(-server_3_occupation_level/env.servers[2].access_capacity)
# 
    #                     # we normalize the routing policy
    #                     sum_i = np.sum(routing_probability[server_1_occupation_level, server_2_occupation_level, server_3_occupation_level, origin_area, :])
    #                     for j in range(env.N_servers):
    #                         routing_probability[server_1_occupation_level, server_2_occupation_level, server_3_occupation_level, origin_area, j] = routing_probabilty[server_1_occupation_level, server_2_occupation_level, server_3_occupation_level, origin_area, j]/sum_i

    # we define the routing policy (which will not depend on the origin area)
    routing_policy =  TotalOccupationRouting(env)

    # we now compute the approximate routing policy (which will be used to define the arrival rates for each server)
    _, approximate_routing_policy = env.compute_approximated_arrival_rates(routing_policy, n_episodes = 100)

    return routing_policy, approximate_routing_policy
                        