# we define a threshold based admission policy, i.e. the package is admitted if and only if the total occupation of the server is below a certain value T
# we then evaluate this policy in the systems usually studied, in order to deifne a comparison

import numpy as np
import pandas as pd
import random
import math
import copy
import time
import os
import pickle as pk
from multiprocessing import Pool

from _mdp_connected_applications import *
from _load_balancing_policy import *

class NaivePolicy():

    def __init__(self, threshold):
        self.threshold = threshold

    def return_action(self, env, state, exploration = False):
        destination_server = state['destination_server']
        destination_server_occupation = np.sum(state['server_' + str(destination_server+1) +'_occupation'])
        return int(destination_server_occupation <= self.threshold), None

    def __call__(self, state):
        return np.sum(state) < self.threshold


def evaluate_naive_policy(max_threshold_value, servers_parameters, arrival_rates, discount_factor, index):
        # print('Reward functions: ' + str(reward_function_types.index(set_reward_functions)+1) + '/' + str(len(reward_function_types)))
    
    # we could make it depend on the index
    N_servers = len(servers_parameters)

    np.random.seed(2 * index) #  + int(10 * ratio))

    # we must sample the parameters for each device
    # we need to understand if the constraint are sufficiently strict
    # we start from the uniform one with areas of interest

    env = LoadBalancingSystem(N_servers, servers_parameters, arrival_rates, discount_factor)
    routing_policy_uniform_aoi = StatelessUniformLoadBalancing(env) 
    env.load_balancing_policy = routing_policy_uniform_aoi
    env.lagrange_multiplier = np.zeros((env.N_servers, ))

    for j in range(N_servers):
        for i in range(N_servers):
            env.servers[j].arrival_rates_server[i] = arrival_rates[i] * routing_policy_uniform_aoi.load_balancing_policy[i, j]
    initial_time = time.time()

    # reward_always_admit = env.evaluate_policy(NaivePolicy(1000), verbose = True)

    optimal_threshold_value = max_threshold_value
    print([servers_parameters[j]['phi'] for j in range(N_servers)])
    reward_naive, cost_naive = env.evaluate_policy(NaivePolicy(max_threshold_value), verbose = True)
    print('Experiment index: {}, Threshold value: {}'.format(index, optimal_threshold_value))
    print([servers_parameters[j]['phi'] for j in range(N_servers)])
    print(cost_naive, reward_naive)
    print(np.all(np.array(cost_naive[j]) < np.array([servers_parameters[j]['phi'] for j in range(N_servers)])))
    print('----------------')
    while not np.all(np.array(cost_naive) < np.array([.servers_parameters[j]['phi'] for j in range(N_servers)])):
        optimal_threshold_value -= 1
        reward_naive, cost_naive = env.evaluate_policy(NaivePolicy(optimal_threshold_value), verbose=True)
        print('Experiment index: {}, Threshold value: {}'.format(index, optimal_threshold_value))
        print([servers_parameters[j]['phi'] for j in range(N_servers)])
        print(cost_naive, reward_naive)
        print(np.all( np.array(cost_naive) < np.array([servers_parameters[j]['phi'] for j in range(N_servers)]) ))
        print('----------------')
    return [reward_naive, cost_naive, optimal_threshold_value, None]
    



# now we first must define the environments for the experiments and then evaluate the policy

N_servers = 10
n_experiment = 20

list_parameters = []

list_area_of_interest = []
evolution_areas_of_interest = []
list_arrival_rates = []
list_discount_factors = []

folder = os.path.dirname(os.path.abspath(__file__))
print('File path:', folder)
results_folder = os.path.join(folder, 'results/experiment_2/')
print('Results folder:', results_folder)
results_folder = '/home/afox/results/experiment_2/'

# first we create all the environments
for experiment_index in range(n_experiment):
    np.random.seed(experiment_index) 
    servers_parameters = [] 
    # at first we must define the areas of interest so that each are of interest is accepted by at least one server
    arrival_rates =  np.ones((N_servers, )) + np.random.random((N_servers, ))
    list_arrival_rates.append(arrival_rates)
    discount_factor = 0.95 + 0.05 * np.random.rand() 
    list_discount_factors.append(discount_factor)
    
    ratio_constraint = 1/(20 + 20 * np.random.random())
    for j in range(N_servers):
        # C = np.random.randint(10, 16)
        C = np.random.randint(20, 30)
        mu = 0 + .5 * np.random.rand()
        reward_function_type = 2
        constraint_value = (1-discount_factor**250)/(20 * (1-discount_factor))    
        if reward_function_type == 1:
            reward_function_multiplier = None
            reward_function_final_value = None
            reward_function_descent_rate = None
        else:
            reward_function_multiplier = 1 + 4 * np.random.random()
            reward_function_final_value = .1 * np.random.random()
            reward_function_descent_rate = 1 + 4 * np.random.random()
        servers_parameters.append({'C': C, 'mu': mu, 'server_id': j, 'reward_function_type': reward_function_type, 'N_servers': N_servers, 'phi': constraint_value, 'reward_function_parameters': [reward_function_multiplier, reward_function_final_value, reward_function_descent_rate]})

    # we create here the evolution of the areas of interest
    areas_of_interest = np.eye((N_servers))
    for j in range(N_servers):
        servers_parameters[j]['areas_of_interest'] = np.copy(areas_of_interest[j, :])
    list_area_of_interest.append(areas_of_interest)
    evolution_areas_of_interest.append(areas_of_interest)
    exclude_sets = [[i] for i in range(N_servers)]
    list_parameters.append(servers_parameters)


    for _ in range(1, N_servers):
        temporary_exclude_sets = copy.deepcopy(exclude_sets)
        new_permutation = []

        # print(temporary_exclude_sets)
        succeded = False
        n_attempts = 0
        while not succeded:
            n_attempts += 1
            np.random.seed(n_attempts)
            if n_attempts > 10:
                raise Exception('Too many attempts')
            for i in range(N_servers):
                available_elements = [element for element in range(0, N_servers) if element not in temporary_exclude_sets[i]]
                i_max = i
                if len(available_elements) == 0:
                    break
                selected_element = np.random.choice(available_elements)
                for j in range(i+1, N_servers):
                    temporary_exclude_sets[j].append(selected_element)
                exclude_sets[i].append(selected_element)
                new_permutation.append(selected_element)
            if len(new_permutation) == N_servers:
                succeded = True
            else:
                for i in range(i_max):
                    exclude_sets[i] = exclude_sets[i][:-1]
                temporary_exclude_sets = copy.deepcopy(exclude_sets)
                new_permutation = []
        for i in range(N_servers):
            areas_of_interest[i, new_permutation[i]] = 1

        new_servers_parameters = copy.deepcopy(servers_parameters)
        for j in range(N_servers):
            new_servers_parameters[j]['areas_of_interest'] = np.copy(areas_of_interest[j, :])

        list_parameters.append(new_servers_parameters)

print('Number of experiments:', len(list_parameters))
print([list_parameters[5][i]['phi'] for i in range(N_servers)])

print(list_parameters[0][0])
print(list_parameters[1][0])
results = []
pool = Pool(processes = 20)
for app_per_server in range(1, 11): #, 11):
    print('index to consider = {}'.format([N_servers * j + (app_per_server-1) for j in range(n_experiment)]))
# in the original experiment we considered different values of the ratio between the server capacity and the constraint value (first input in the following function)
    results = pool.starmap(evaluate_naive_policy, [(3, list_parameters[N_servers * j + (app_per_server-1)], list_arrival_rates[j], list_discount_factors[j], j) for j in range(n_experiment)])
# print(results)
    rewards_naive = [results[j][0] for j in range(len(results))]
    cost_naive = [results[j][1] for j in range(len(results))]
    optimal_threshold_value = [results[j][2] for j in range(len(results))]
    rewards_always_admit = [results[j][3] for j in range(len(results))]


    save_data = True
    if save_data and n_experiment > 1:
        data_to_save = [list_parameters, list_area_of_interest, list_arrival_rates, list_discount_factors, rewards_naive, cost_naive, optimal_threshold_value, rewards_always_admit]
        # folder = '/home/afox/Dropbox/video_admission_control/python/connected_servers/results/experiment_1/' 
        with open(results_folder + '100_server_results_threshold_policy_different_rewards_{}.pk'.format(app_per_server), 'wb') as f:
            pk.dump(data_to_save, f)

