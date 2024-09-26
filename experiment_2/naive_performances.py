# attempts of plotting the results_DRCPO of experiment 2
import pickle as pk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import math
import copy




folder = '/home/afox/Dropbox/video_admission_control/python/connected_servers/results/experiment_2/'

N_servers = 10
area_of_interest = 10
results_DRCPO = []

N_areas_of_interest = np.arange(1, 11)
N_areas_of_interest[1]  = 3
print(N_areas_of_interest)
for area_of_interest in N_areas_of_interest:
    file_DRCPO = folder + '{}_servers_comparison_{}_areas_of_interest_different_rewards.pk'.format(N_servers, area_of_interest)
    with open(file_DRCPO, 'rb') as f:
        data = pk.load(f)
    list_parameters = data[0]
    print(list_parameters[0])
    list_area_of_interest = data[1]
    list_arrival_rates = data[2]
    list_discount_factors = data[3]
    results_DRCPO += data[4]
file_threshold = folder + '10_server_results_threshold_policy_different_rewards_new_env.pk'
with open(file_threshold, 'rb') as f:
    data_threshold = pk.load(f)
parameter_threshold = data_threshold[0]
rewards_naive = np.zeros((20 * len(N_areas_of_interest), ))
cost_naive = np.zeros((20 * len(N_areas_of_interest), N_servers))
# we cannot consider all the experiments, we must consider only the ones that are in the area of interest
area_index = area_of_interest - 1
rewards_naive = data_threshold[4]
cost_naive = data_threshold[5][:]
threshold_values = data_threshold[6]

area_index = 1
experiment_index = 1

# we take the critic of the ezperiment to consider and evaluate it, in order to understand if the results are consistent


#######################
if False:
    for experiment_index in range(20):
        for area_index in range(len(N_areas_of_interest)):
            print('Experiment: {}, area of interest: {}'.format(experiment_index, N_areas_of_interest[area_index]))
            print('Reward DRCPO: ', results_DRCPO[20 * area_index + experiment_index][0][-1])
            print('Reward naive: ', rewards_naive[20 * area_index + experiment_index])
            print('------'*5)
        print('------'*5)


# attempt to understand why unconstrained reward is lower than the naive one
# we focus on experiment 2 and area of interest 1, which is one in which this behavior is observed

# at first we print the immediate reward function
if False:
    parameters_of_interest = list_parameters[1]
    print('Parameters of interest: ', parameters_of_interest)

    random_multiplier = [parameters_of_interest[j]['reward_function_parameters'][0] for j in range(N_servers)]
    random_final_value = [parameters_of_interest[j]['reward_function_parameters'][1] for j in range(N_servers)]
    random_descent_rate = [parameters_of_interest[j]['reward_function_parameters'][2] for j in range(N_servers)]

    reward_function  = (lambda x, j: ((10 * random_multiplier[j-1])/N_servers) * math.exp(-x * (random_descent_rate[j-1])/(N_servers)) +random_final_value[j-1] )

    C = 30
    xpts = np.linspace(0, C, C+1)
    for j in range(0, N_servers):
        plt.scatter(xpts, [reward_function(x, j+1) for x in xpts], label='j = {}'.format(j))
        plt.plot(xpts, [reward_function(x, j+1) for x in xpts])
    plt.xlabel('Occupation level')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()

# comparison naive performances over evolution N applications
if True:
    for experiment_index in range(20):
        print('Experiment: {}'.format(experiment_index))
        for area_index in range(len(N_areas_of_interest)):
            index = 20 * area_index + experiment_index
            print('Applications: {}, Threshold value = {}, Reward = {}'.format(N_areas_of_interest[area_index], threshold_values[index],rewards_naive[index]))
        print('------'*5)



