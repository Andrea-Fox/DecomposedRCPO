# vedere se migliora mettendo tante applicazione sui server e vedere se migliora rispetto al caso con applicazioni separate per server

# vedere come si comporta l'occupazione delle varie aree di origine

# first of all, we need to define the idea of convergence to a feasible solution

# then, we can simply reuse the same ideas in experiment 1 and show the trend observed

# maybe it could be intersting to study the behavior for different quantities of servers


# Experiment 1: comparison of different admission policies for fixed routing policy 
# we consider 4 servers and 

# We consider a system with N servers and N areas of interest (areas of interest are different for each server)
# we consider randomly selected parameters, the following sets:
# C = random value between 8 and 15
# phi = f(C)
# mu = (0, 1]
# arrival rates = (0, 1.5]
# areas of interest: for each server, probability of accepting one area is 0.66. Each area is accepted by at least one server
# we consider both the case with equal reward fucntions (all three of them) and with different reward functions
# discount factor = (0.9, 1)

# The ratio between the capacity of each server and the constraint value goes from 0.1 to 1.5

# The admission policies compared are:
# 1) DRCPO (Decomposed reward constrained policy optimization)
# 2) RCPO (Reward constrained policy optimization, baseline method which follows the same principle of having the Lagrange multiplier)
# 3) CPO (Constrained policy optimization, baseline method)


# could be interesting to find cases with a limited amount of servers (e.g. 3) but make sure that at the beginning all the constraints 
# are not satisfied, i.e. an improvement is required for every server


from _mdp_connected_applications import *
from _admission_policy_learning import *
from _load_balancing_policy import *

from copy import deepcopy
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import pickle as pk
import time
import random


def computation_total_discounted_reward(ratio, areas_of_interest, servers_parameters, arrival_rates, discount_factor, index,  load_balancing_to_emptiest_server = False):
        # print('Reward functions: ' + str(reward_function_types.index(set_reward_functions)+1) + '/' + str(len(reward_function_types)))
    
    # we could make it depend on the index
    N_servers = len(servers_parameters)

    np.random.seed(2 * index) #  + int(10 * ratio))

    # we need to understand if the constraint are sufficiently strict
    
    # raise Exception('stop')
    print([servers_parameters[j]['phi'] for j in range(N_servers)])
    # print(servers_parameters)
    # for each environment, we evaluate different types of fixed routing policies   
    DRCPO = True
    RCPO = False

    # we start from the uniform one with areas of interest
    
    if RCPO:
        env = LoadBalancingSystem(N_servers, servers_parameters, arrival_rates, discount_factor, load_balancing_to_emptiest_server, reward)
        routing_policy_uniform_aoi = StatelessUniformLoadBalancing(env) 
        env.load_balancing_policy = routing_policy_uniform_aoi
        for j in range(N_servers):
            for i in range(N_servers):
                env.servers[j].arrival_rates_server[i] = arrival_rates[i] * routing_policy_uniform_aoi.load_balancing_policy[i, j]
        initial_time = time.time()
        results_RCPO = constrained_policy_learning(env, learning_method= 'RCPO',  total_updates=50, episodes_between_updates=200, lm_learning_rate_exponent=.6, lm_learning_rate = .5)
        elapsed_time_RCPO = time.time() - initial_time
    else:
        results_RCPO = None
        elapsed_time_RCPO = None

    if DRCPO:
        env = LoadBalancingSystem(N_servers, servers_parameters, arrival_rates, discount_factor, load_balancing_to_emptiest_server)
        routing_policy_uniform_aoi = StatelessUniformLoadBalancing(env) 
        env.load_balancing_policy = routing_policy_uniform_aoi
        for j in range(N_servers):
            for i in range(N_servers):
                env.servers[j].arrival_rates_server[i] = arrival_rates[i] * routing_policy_uniform_aoi.load_balancing_policy[i, j]
        initial_time = time.time()
        # areas of interest = 1 
        # results_DRCPO = constrained_policy_learning(env, learning_method= 'DRCPO_optimized', total_updates=100, critic_learning_rate_exponent=0.51, lm_learning_rate_exponent=.6, lm_learning_rate = .5, initial_value_lm = 0, critic_learning_rate = .00005, episodes_between_updates=100)  # 2, 3, 4 servers
        results_DRCPO = constrained_policy_learning(env, learning_method= 'DRCPO_optimized', total_updates=1, critic_learning_rate_exponent=0.51, lm_learning_rate_exponent = .6, lm_learning_rate = .5, initial_value_lm = 0, critic_learning_rate = .00005, episodes_between_updates=100)     # 1 server
        elapsed_time_DRCPO = time.time() - initial_time
    else:
        results_DRCPO = None
        elapsed_time_DRCPO = None
        
    return [results_DRCPO, results_RCPO, servers_parameters, elapsed_time_DRCPO, elapsed_time_RCPO]
    


reward_function_types = [(i, j, k) for i in range(1, 3) for j in range(3) for k in range(3)]
print(len(reward_function_types)) 
N_servers = 10

n_experiment = 20

load_balancing_to_emptiest_server = False


for servers_index in [1]: # 3, 4, 5, 6, 7, 8, 9, 10] : #, 2, 5, N_servers]:

    average_servers_area_of_interest = servers_index

    list_total_discounted_rewards = []
    list_total_discounted_cost = []
    list_parameters = []

    list_area_of_interest = []
    list_arrival_rates = []
    list_discount_factors = []

    for experiment_index in range(n_experiment):
        np.random.seed(experiment_index) 
        servers_parameters = [] 
        # at first we must define the areas of interest so that each are of interest is accepted by at least one server
        
        
        ratio_constraint = 1/(20 + 20 * np.random.random())
        for j in range(N_servers):
            # C = np.random.randint(10, 16)
            C = np.random.randint(20, 30)
            mu = .5 + .25 * np.random.rand()
            reward_function_type = 2
            constraint_value = ratio_constraint * C    
            if reward_function_type == 1:
                reward_function_multiplier = None
                reward_function_final_value = None
                reward_function_descent_rate = None
            else:
                reward_function_multiplier = 1 + 4 * np.random.random()
                reward_function_final_value = .1 * np.random.random()
                reward_function_descent_rate = 1 + 4 * np.random.random()
            servers_parameters.append({'C': C, 'mu': mu, 'server_id': j, 'reward_function_type': reward_function_type, 'N_servers': N_servers, 'phi': constraint_value, 'reward_function_parameters': [reward_function_multiplier, reward_function_final_value, reward_function_descent_rate]})
        
        # area of interest ratio
        if average_servers_area_of_interest == 0:
            areas_of_interest = np.zeros((N_servers, N_servers)) 
            for i in range(N_servers):
                areas_of_interest[i, i] = 1
        elif average_servers_area_of_interest == N_servers:
            areas_of_interest = np.ones((N_servers, N_servers)) 
        elif average_servers_area_of_interest>0 and average_servers_area_of_interest < N_servers:
            areas_of_interest = np.zeros((N_servers, N_servers)) 
            for i in range(N_servers):
                for j in range(average_servers_area_of_interest):
                    areas_of_interest[i][(i + j) % N_servers] = 1
            areas_of_interest = areas_of_interest[:, np.random.permutation(areas_of_interest.shape[1])]  # Shuffle the columns
            areas_of_interest = areas_of_interest[np.random.permutation(areas_of_interest.shape[0]), :]
        else:
            raise 'Impossible to define area if interest matrix'
        areas_of_interest = np.zeros((N_servers, N_servers)) 
        for i in range(N_servers):
            areas_of_interest[i, i] = 1
        for j in range(N_servers):
            servers_parameters[j]['areas_of_interest'] = areas_of_interest[j, :]
        list_area_of_interest.append(areas_of_interest)
        
        list_parameters.append(servers_parameters)

        arrival_rates =  np.ones((N_servers, )) + np.random.random((N_servers, ))
        list_arrival_rates.append(arrival_rates)
        discount_factor = 0.95 + 0.05 * np.random.rand() 
        list_discount_factors.append(discount_factor)

        print(servers_parameters)

    print(list_arrival_rates)

    # in the original experiment we considered different values of the ratio between the server capacity and the constraint value (first input in the following function)
    with Pool(20) as pool:
        results = pool.starmap(computation_total_discounted_reward, [(None, list_area_of_interest[j], list_parameters[j], list_arrival_rates[j], list_discount_factors[j], j, load_balancing_to_emptiest_server) for j in range(n_experiment)])
    print(results)
    results_DRCPO = []
    results_RCPO = []
    updated_server_parameters = []
    elapsed_time_DRCPO = []
    elapsed_time_RCPO = []
    for experiment_index in range(n_experiment):
        results_DRCPO.append(results[experiment_index][0])
        results_RCPO.append(results[experiment_index][1])
        updated_server_parameters.append(results[experiment_index][2])
        elapsed_time_DRCPO.append(results[experiment_index][3])
        elapsed_time_RCPO.append(results[experiment_index][4])
 
    # needs ot be changed the way we save the results
    print(results_DRCPO)
    print(results_RCPO)
    print(updated_server_parameters)
    print(elapsed_time_DRCPO)
    print(elapsed_time_RCPO)


    save_data = True
    if save_data and n_experiment > 1:
        data_to_save = [updated_server_parameters, list_area_of_interest, list_arrival_rates, list_discount_factors, results_DRCPO, results_RCPO, elapsed_time_DRCPO, elapsed_time_RCPO]
        folder = '' # 'results/experiment_2/' 
        file_name = '{}_servers_comparison_{}.pk'.format(N_servers, average_servers_area_of_interest)
        with open(folder + file_name, 'wb') as f:
            pk.dump(data_to_save, f)




