import numpy as np
import math
from copy import deepcopy


from _load_balancing_policy import StatelessLoadBalancing
from _mdp_connected_applications import LoadBalancingSystem



def projection_load_balancing_policy(env, load_balancing_policy, areas_of_interest = False):
    # first we project the routing policy on the simplex
    # we need to normalize it
    for i in range(load_balancing_policy.shape[0]):
        for j in range(load_balancing_policy.shape[1]):
            if areas_of_interest and not env.servers[j].areas_of_interest[i]:
                load_balancing_policy[i, j] = 0
            else:
                load_balancing_policy[i, j] = max(0.0001, load_balancing_policy[i, j])
                load_balancing_policy[i, j] = min(0.9999, load_balancing_policy[i, j])
    # the values have to be so that the sum on each row is equal to 1
    for i in range(load_balancing_policy.shape[0]):
        load_balancing_policy[i, :] = load_balancing_policy[i, :] / np.sum(load_balancing_policy[i, :])
    return load_balancing_policy


def SPSA_fixed_load_balancing_probability(env, load_balancing_probability, load_balancing_probability_positive_perturbation, load_balancing_probability_negative_perturbation, single_core = True, areas_of_interest = False, admission_policy = None):
    # thid function is the one responsible for computing the gradients wrt each component. It is a simple 
    # application of the SPSA algorithm 

    # we compute the positive and negative perturbations
    # compute_server_arrival_probability_given_load_balancing_policy(env_positive_perturbation, load_balancing_policy)
    # compute_server_arrival_probability_given_load_balancing_policy(env_negative_perturbation, load_balancing_policy)

    # we compute the gradient


    # this operation has to be done parallely, to speed up the process
    gradient = np.zeros((env.N_servers, env.N_servers))
    if single_core:
        for area_of_origin in range(env.N_servers):
            env_positive_perturbation = LoadBalancingSystem(env.N_servers, env.servers_parameters, env.arrival_rates, env.discount_factor)
            env_positive_perturbation.load_balancing_policy = StatelessLoadBalancing(env_positive_perturbation, env.load_balancing_policy.load_balancing_policy)
            env_negative_perturbation = LoadBalancingSystem(env.N_servers, env.servers_parameters, env.arrival_rates, env.discount_factor)
            env_negative_perturbation.load_balancing_policy = StatelessLoadBalancing(env_negative_perturbation, env.load_balancing_policy.load_balancing_policy)
            perturbation_value = np.zeros((env.N_servers, ))
            for j in range(env.N_servers):
                # print('gradient component {}, {}'.format(i, j))
                # we define the new environment, with a modified arrival probability in device j
                perturbation_value[j] = load_balancing_probability_positive_perturbation[area_of_origin, j] - load_balancing_probability_negative_perturbation[area_of_origin, j]
                if (areas_of_interest and not env.servers[j].areas_of_interest[area_of_origin]) or (load_balancing_probability_positive_perturbation[area_of_origin, j] - load_balancing_probability_negative_perturbation[area_of_origin, j]) == 0 :
                    # if we consider the areas of interest and this is not one of them, we set the gradient to 0
                    gradient[area_of_origin, j] = 0
                else:
                    # print(gradient.shape, i, j)
                    env_positive_perturbation.load_balancing_policy.load_balancing_policy[area_of_origin, j] = load_balancing_probability_positive_perturbation[area_of_origin, j]
                    env_positive_perturbation.load_balancing_policy.load_balancing_policy = projection_load_balancing_policy(env_positive_perturbation, env_positive_perturbation.load_balancing_policy.load_balancing_policy, areas_of_interest= areas_of_interest)
                    env_negative_perturbation.load_balancing_policy.load_balancing_policy[area_of_origin, j] = load_balancing_probability_negative_perturbation[area_of_origin, j]
                    env_negative_perturbation.load_balancing_policy.load_balancing_policy = projection_load_balancing_policy(env_negative_perturbation, env_negative_perturbation.load_balancing_policy.load_balancing_policy, areas_of_interest= areas_of_interest)
                    
            return_positive_perturbation = env_positive_perturbation.evaluate_policy(admission_policy, n_episodes=10, verbose=False)[0]
            return_negative_perturbation = env_negative_perturbation.evaluate_policy(admission_policy, n_episodes=10, verbose=False)[0]
            
            delta_return = return_positive_perturbation - return_negative_perturbation
            for j in range(env.N_servers):
                if env.servers[j].areas_of_interest[area_of_origin]:
                    gradient[area_of_origin, j] = delta_return / perturbation_value[j]
                    gradient[area_of_origin, j] = max(-10, min(gradient[area_of_origin, j], 10))
    else:
        raise ValueError('Parallelization not implemented yet')
        # optimized code for parallelization (will be useless when doing multiple attempts simultaneously)
        pool = mp.Pool(processes=18)
        results = pool.starmap(parallel_computing_gradient_component, [(env, load_balancing_probability,  load_balancing_probability_positive_perturbation, load_balancing_probability_negative_perturbation, origin_server, desitnation_server) for origin_server in range(env.N_servers) for desitnation_server in range(env.N_servers) ])

        # reformat results to fit the gradient
        for origin_server in range(env.N_servers):
            for destination_server in range(env.N_servers):
                gradient[origin_server, destination_server] = results[env.N_servers * origin_server + destination_server]

    return gradient

def optimize_stateless_load_balancing_policy(env, load_balancing_policy = None, consider_areas_of_interest = False, verbose = False, admission_policy = None, initial_discounted_reward = None):
    # this function receives as input the environment, the current admission policies (saved within the admission
    # mdp) for all the devices and the current routing policy
    # The goal of this function is to optimize the routing policy, given the admission policies

    # The goal is to maximize the return: for each origin area, we compute the derivative wrt the routing 
    # probability of the return. We follow the SPSA approach (already used in the context of energy harvesing)

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    weight_function_perturbation = (lambda x: 1/(x+2)**2)
    weight_function_ascent = (lambda x: 1/((x+2)**3))     

    best_total_reward = -math.inf
    list_load_balancing_policies = []
    list_total_rewards = []

    list_load_balancing_policies.append(load_balancing_policy)

    # print(load_balancing_policy)

    # we compute the total reward for the current routing policy
    total_reward = 0
    # compute_server_arrival_probability_given_load_balancing_policy(env, load_balancing_policy)
    # print('initial total reward: ', total_reward)
    list_total_rewards.append(initial_discounted_reward)

    # we start the optimization procedure
    optimization_completed = False
    SPSA_steps = 1
    steps_without_improvement = 0
    no_improvement = True

    verbose = True

    env_perturbated = LoadBalancingSystem(env.N_servers, env.servers_parameters, env.arrival_rates, env.discount_factor)
    env_perturbated.load_balancing_policy = StatelessLoadBalancing(env_perturbated, env.load_balancing_policy.load_balancing_policy)

    while not optimization_completed:
        # we compute perturbations for each component of the routing policy
        perturbations = np.zeros((env.N_servers, env.N_servers))
        for i in range(env.N_servers):
            for j in range(env.N_servers):
                if env.servers[j].areas_of_interest[i]:
                    perturbations[i, j] = np.random.uniform(-1, 1)
        # print(perturbations)

        # we compute the two perturbed routing policy (positive and negative)
        load_balancing_policy_positive = load_balancing_policy.load_balancing_policy + weight_function_perturbation(SPSA_steps) * perturbations
        load_balancing_policy_positive = projection_load_balancing_policy(env, load_balancing_policy_positive, areas_of_interest = consider_areas_of_interest)
        # print('positive perturbation')
        # print(load_balancing_policy_positive)
        load_balancing_policy_negative = load_balancing_policy.load_balancing_policy - weight_function_perturbation(SPSA_steps) * perturbations
        load_balancing_policy_negative = projection_load_balancing_policy(env, load_balancing_policy_negative, areas_of_interest = consider_areas_of_interest)
        # print('negative perturbation')
        # print(load_balancing_policy_negative)

        # we compute the gradient
        gradient = SPSA_fixed_load_balancing_probability(env, load_balancing_policy.load_balancing_policy, load_balancing_policy_positive, load_balancing_policy_negative, single_core=True, areas_of_interest= consider_areas_of_interest, admission_policy=admission_policy)
        print('Gradient computed')
        # print(gradient)
        # we update the routing policy
        delta_policy = weight_function_ascent(SPSA_steps) * gradient
        new_load_balancing_policy = env_perturbated.load_balancing_policy.load_balancing_policy + weight_function_ascent(SPSA_steps) * gradient
        new_load_balancing_policy = projection_load_balancing_policy(env, new_load_balancing_policy, areas_of_interest=consider_areas_of_interest)

        if verbose:
            print('new routing policy')
            print(new_load_balancing_policy)

        # print(new_load_balancing_policy)
        # we compute the updated return given by the updated routing policy
        new_total_reward = 0
        env_perturbated = LoadBalancingSystem(env.N_servers, env.servers_parameters, env.arrival_rates, env.discount_factor)
        env_perturbated.load_balancing_policy = StatelessLoadBalancing(env_perturbated, new_load_balancing_policy)
        for j in range(env.N_servers):
            for i in range(env.N_servers):
                env_perturbated.servers[j].arrival_rates_server[i] = env.arrival_rates[i] * new_load_balancing_policy[i, j]
        new_total_reward, new_total_cost = env_perturbated.evaluate_policy(admission_policy, n_episodes=100, verbose = False)
        if verbose:
            print(new_total_reward, new_total_cost)
        # for j in range(env.N_servers):
        #     for i in range(env.N_servers)
        #         env_perturbated.servers[j].arrival_rates_server[i] = env.arrival_rates[i] * new_load_balancing_policy[i, j]
        #     new_total_reward += env_perturbated.servers[j].evaluate_policy_single_server(env_perturbated.servers[j].actor)[0]

        # we need to normalize it (the sum of the row must be 1 as it is a stochastic matrix)

        # if the new total reward is better than the previous one, we update the routing policy
        # raise ValueError('Check the following condition')
        if new_total_reward > total_reward and np.all([new_total_cost[i] < 1.1 * env.servers[i].server_access_capacity for i in range(env.N_servers)]):
            print('New evaluation has higher reward')
            total_reward = new_total_reward
            load_balancing_policy = StatelessLoadBalancing(env, new_load_balancing_policy)
            list_load_balancing_policies.append(load_balancing_policy)
            list_total_rewards.append(total_reward)
            SPSA_steps += 1
            steps_without_improvement = 0
            no_improvement = False
            # print('optimization step completed: new total reward: ', new_total_reward, 'old total reward: ', total_reward)
        else:
            steps_without_improvement += 1
            # print('Failed optimization step: new total reward: ', new_total_reward, 'old total reward: ', total_reward)

        if SPSA_steps > 5 or steps_without_improvement >= 2:
            optimization_completed = True

    print('FINAL LOAD BALANCING POLICY')
    print(load_balancing_policy.load_balancing_policy)
    print()
    return load_balancing_policy, list_load_balancing_policies, list_total_rewards, no_improvement

