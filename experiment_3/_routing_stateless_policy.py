# the policy that we seek to learn is one that learns, for each area of origin, the optimal routing policy

# once the admission policies are fixed, which property do we have on the reward of each server? These are the 
# properties that we want to exploit in order to learn the optimal routing policy

# we could assume differentiability. However do we prove it is true?
# once we are given the differentiability, we can use gradient descent to learn the optimal routing policy

# basically we repeat the reasoning of the APPi algorithm
# does this improve anything?

# how do we deal with the non-stationarity of each of the servers when changing the routing probability? 
# Future works? 

# N.B clearly, make sure that the routing probability is 0 for systems where the server is not interested 
# to a certain application (or it could be interesting to understand if it learns by itself that it is a 
# good thing to do)

# def compute_server_arrival_probability_given_routing_policy(env, routing_policy_perturbated, component_to_change):
#     # component to change must be a tuple (i, j) where i is the origin area and j is the component of the routing policy 
# 
#     # as proved, when we consider the component (i, j) of the routing policy, we pnly have to change the arrival
#     # rates in the server j, according to all the arrival probabilities
# 
#     env.list_servers[component_to_change[1]].arrival_rates[component_to_change[0]] = env.arrival_rates[component_to_change[0]] * routing_policy_perturbated[component_to_change[0], component_to_change[1]]
#     return None, routing_policy=

import numpy as np
import math
import multiprocessing as mp

from copy import deepcopy
from _routing_policy import StatelessRouting


def projection_routing_policy(env, routing_policy, areas_of_interest = False):
    # first we project the routing policy on the simplex
    # we need to normalize it
    for i in range(routing_policy.shape[0]):
        for j in range(routing_policy.shape[1]):
            if areas_of_interest and not env.servers[j].areas_of_interest[i]:
                routing_policy[i, j] = 0
            else:
                routing_policy[i, j] = max(0.0001, routing_policy[i, j])
                routing_policy[i, j] = min(0.9999, routing_policy[i, j])
    # the values have to be so that the sum on each row is equal to 1
    for i in range(routing_policy.shape[0]):
        routing_policy[i, :] = routing_policy[i, :] / np.sum(routing_policy[i, :])
    return routing_policy


def parallel_computing_gradient_component(env, routing_probability, routing_probability_positive_perturbation, routing_probability_negative_perturbation, i, j ):
    # origin and destination server are, respectively, i and j

    # we compute, for each single component, the derivative of the return function
    temporary_routing_probability_positive = deepcopy(routing_probability)
    temporary_routing_probability_positive[i, j] = routing_probability_positive_perturbation[i, j]
    temporary_routing_probability_positive = projection_routing_policy(env, temporary_routing_probability_positive)
    positive_perturbation_routing_policy = StatelessRouting(env, temporary_routing_probability_positive)
    return_positive_perturbation = env.evaluate_routing_policy(positive_perturbation_routing_policy)[0]

    # env_negative_perturbation = deepcopy(env)
    # temporary_routing_probability_negative = deepcopy(routing_probability)
    # temporary_routing_probability_negative[i, j] = routing_probability_negative_perturbation[i, j]
    # temporary_routing_probability_negative = projection_routing_policy(temporary_routing_probability_negative)
    # negative_perturbation_routing_policy = StatelessRouting(env_negative_perturbation, temporary_routing_probability_negative)
    # return_negative_perturbation = env_negative_perturbation.evaluate_routing_policy(negative_perturbation_routing_policy)[0]
    delta_return = return_positive_perturbation - return_negative_perturbation
    
    gradient_component = delta_return / (routing_probability_positive_perturbation[i, j] - routing_probability_negative_perturbation[i, j])
    
    return gradient_component


def SPSA_fixed_routing_probability(env, routing_probability, routing_probability_positive_perturbation, routing_probability_negative_perturbation, single_core = True, areas_of_interest = False):
    # thid function is the one responsible for computing the gradients wrt each component. It is a simple 
    # application of the SPSA algorithm 

    # we compute the positive and negative perturbations
    # compute_server_arrival_probability_given_routing_policy(env_positive_perturbation, routing_policy)
    # compute_server_arrival_probability_given_routing_policy(env_negative_perturbation, routing_policy)

    # we compute the gradient


    # this operation has to be done parallely, to speed up the process
    gradient = np.zeros((env.N_servers, env.N_servers))
    if single_core:
        for i in range(env.N_servers):
            for j in range(env.N_servers):
                # we define the new environment, with a modified arrival probability in device j

                if areas_of_interest and not env.servers[j].areas_of_interest[i] or (routing_probability_positive_perturbation[i, j] - routing_probability_negative_perturbation[i, j]) == 0 :
                    # if we consider the areas of interest and this is not one of them, we set the gradient to 0
                    gradient[i, j] = 0
                else:
                    # print(gradient.shape, i, j)
                    env_positive_perturbation = deepcopy(env)
                    temporary_routing_probability_positive = deepcopy(routing_probability)
                    temporary_routing_probability_positive[i, j] = routing_probability_positive_perturbation[i, j]
                    temporary_routing_probability_positive = projection_routing_policy(env, temporary_routing_probability_positive, areas_of_interest= areas_of_interest)
                    positive_perturbation_routing_policy = StatelessRouting(env_positive_perturbation, temporary_routing_probability_positive)
                    # env_positive_perturbation.servers[j].arrival_rates_server[i] = env.arrival_rates[i] * routing_policy_positive_perturbation[i, j]

                    return_positive_perturbation = env_positive_perturbation.evaluate_routing_policy(positive_perturbation_routing_policy)[0]

                    env_negative_perturbation = deepcopy(env)
                    temporary_routing_probability_negative = deepcopy(routing_probability)
                    temporary_routing_probability_negative[i, j] = routing_probability_negative_perturbation[i, j]
                    temporary_routing_probability_negative = projection_routing_policy(env, temporary_routing_probability_negative, areas_of_interest= areas_of_interest)
                    negative_perturbation_routing_policy = StatelessRouting(env_negative_perturbation, temporary_routing_probability_negative)
                    return_negative_perturbation = env_negative_perturbation.evaluate_routing_policy(negative_perturbation_routing_policy, n_episodes=50)[0]
                    delta_return = return_positive_perturbation - return_negative_perturbation
                    gradient[i, j] = delta_return / (routing_probability_positive_perturbation[i, j] - routing_probability_negative_perturbation[i, j])
                    gradient[i, j] = max(-10, min(gradient[i, j], 10))
    else:
        raise ValueError('Parallelization not implemented yet')
        # optimized code for parallelization (will be useless when doing multiple attempts simultaneously)
        pool = mp.Pool(processes=18)
        results = pool.starmap(parallel_computing_gradient_component, [(env, routing_probability,  routing_probability_positive_perturbation, routing_probability_negative_perturbation, origin_server, desitnation_server) for origin_server in range(env.N_servers) for desitnation_server in range(env.N_servers) ])

        # reformat results to fit the gradient
        for origin_server in range(env.N_servers):
            for destination_server in range(env.N_servers):
                gradient[origin_server, destination_server] = results[env.N_servers * origin_server + destination_server]

    return gradient


def optimize_stateless_routing_policy(env, routing_policy = None, consider_areas_of_interest = False, verbose = False):
    # this function receives as input the environment, the current admission policies (saved within the admission
    # mdp) for all the devices and the current routing policy
    # The goal of this function is to optimize the routing policy, given the admission policies

    # The goal is to maximize the return: for each origin area, we compute the derivative wrt the routing 
    # probability of the return. We follow the SPSA approach (already used in the context of energy harvesing)
    weight_function_perturbation = (lambda x: 1/(x+5)**2)
    weight_function_ascent = (lambda x: 1/((x+5)**2))     

    best_total_reward = -math.inf
    list_routing_policies = []
    list_total_rewards = []

    if routing_policy is None:
        if areas_of_interest:
            # if no routing policy is given, we initialize it to a random one
            routing_policy = np.random.uniform(size = (env.N_servers, env.N_servers))
            # we need to normalize it
            routing_policy = projection_routing_policy(env, routing_policy, areas_of_interest= consider_areas_of_interest)
            # we need to set to 0 the routing probabilities of the areas of interest
            for i in range(env.N_servers):
                for j in range(env.N_servers):
                    if not env.servers[j].areas_of_interest[i]:
                        routing_policy[i, j] = 0
        else:
            # if no routing policy is given, we initialize it to a random one
            routing_policy = np.random.uniform(size = (env.N_servers, env.N_servers))
            # we need to normalize it
            routing_policy = projection_routing_policy(env, routing_policy, areas_of_interest= consider_areas_of_interest)

        routing_policy = routing_policy/np.sum(routing_policy, axis = 1)

    list_routing_policies.append(routing_policy)

    # print(routing_policy)

    # we compute the total reward for the current routing policy
    total_reward = 0
    # compute_server_arrival_probability_given_routing_policy(env, routing_policy)
    total_reward = env.evaluate_routing_policy(routing_policy)[0] # StatelessRouting(env, routing_policy))[0]
    # print('initial total reward: ', total_reward)
    list_total_rewards.append(total_reward)

    # we start the optimization procedure
    optimization_completed = False
    SPSA_steps = 1
    steps_without_improvement = 0
    no_improvement = True

    while not optimization_completed:
        # we compute perturbations for each component of the routing policy
        perturbations = np.random.uniform(size = (env.N_servers, env.N_servers))
        # perturbations = perturbations/np.sum(perturbations, axis = 0)

        # we compute the two perturbed routing policy (positive and negative)
        routing_policy_positive = routing_policy.routing_policy + weight_function_perturbation(SPSA_steps) * perturbations
        routing_policy_positive = projection_routing_policy(env, routing_policy_positive, areas_of_interest= consider_areas_of_interest)
        routing_policy_negative = routing_policy.routing_policy - weight_function_perturbation(SPSA_steps) * perturbations
        routing_policy_negative = projection_routing_policy(env, routing_policy_negative, areas_of_interest= consider_areas_of_interest)

        # we compute the gradient
        gradient = SPSA_fixed_routing_probability(env, routing_policy.routing_policy, routing_policy_positive, routing_policy_negative, single_core=True, areas_of_interest= consider_areas_of_interest)
        # print('gradient')
        # print(gradient)
        # we update the routing policy
        delta_policy = weight_function_ascent(SPSA_steps) * gradient
        new_routing_policy = routing_policy.routing_policy + weight_function_ascent(SPSA_steps) * gradient
        new_routing_policy = projection_routing_policy(env, new_routing_policy, areas_of_interest= consider_areas_of_interest)

        if verbose:
            print('new routing policy')
            print(new_routing_policy)

        new_routing_policy = new_routing_policy/np.sum(new_routing_policy, axis = 1)
        # print(new_routing_policy)
        # we compute the updated return given by the updated routing policy
        new_total_reward = 0
        env_perturbated = deepcopy(env)
        new_total_reward = env_perturbated.evaluate_routing_policy(StatelessRouting(env_perturbated, new_routing_policy), n_episodes=50)[0]
        if verbose:
            print(new_total_reward)
        # for j in range(env.N_servers):
        #     for i in range(env.N_servers):
        #         env_perturbated.servers[j].arrival_rates_server[i] = env.arrival_rates[i] * new_routing_policy[i, j]
        #     new_total_reward += env_perturbated.servers[j].evaluate_policy_single_server(env_perturbated.servers[j].actor)[0]

        # we need to normalize it (the sum of the row must be 1 as it is a stochastic matrix)

        # if the new total reward is better than the previous one, we update the routing policy
        if new_total_reward > total_reward:
            total_reward = new_total_reward
            routing_policy = StatelessRouting(env, new_routing_policy)
            list_routing_policies.append(routing_policy)
            list_total_rewards.append(total_reward)
            SPSA_steps += 1
            steps_without_improvement = 0
            no_improvement = False
            # print('optimization step completed: new total reward: ', new_total_reward, 'old total reward: ', total_reward)
        else:
            steps_without_improvement += 1
            # print('Failed optimization step: new total reward: ', new_total_reward, 'old total reward: ', total_reward)

        if SPSA_steps > 10 or steps_without_improvement >= 3:
            optimization_completed = True
        
    return routing_policy, list_routing_policies, list_total_rewards, no_improvement



def softmax_routing_policy(env, q_values, temperature, origin_area = None):
    action_probabilities = np.zeros((env.N_servers, ))
    if origin_area is None:
        origin_area = env.current_state['last_origin_server']
    for j in range(env.N_servers):
        action_probabilities[j] = math.exp(q_values[origin_area, j]/temperature)
    action_probabilities = action_probabilities/np.sum(action_probabilities)
    return np.random.choice(range(env.N_servers), p = action_probabilities)


def stateless_q_learning(env, routing_policy = None):

    if routing_policy is None:
        # if no routing policy is given, we initialize it to a random one
        routing_policy = np.random.uniform(size = (env.N_servers, env.N_servers))
        # we need to normalize it
        routing_policy = projection_routing_policy(env, routing_policy)
        
        routing_policy = routing_policy/np.sum(routing_policy, axis = 1)

    # first we define the environment according to the initial routing policy
    for j in range(env.N_servers):
        for i in range(env.N_servers):
                env.servers[j].arrival_rates_server[i] = env.arrival_rates[i] * routing_policy[i, j]

    # then we define and update the q values for each state-action pair
    q_values = np.zeros((env.N_servers, env.N_servers))
    learning_rate = 0.1
    discount_factor = 0.99
    state_counts = np.ones((env.N_servers, env.N_servers))
    n_episodes = 250
    for i in range(n_episodes):
        # we reset the environment
        state, _ = env.reset()
        for t in range(env.max_length_episode):
            # we choose the action according to the q-values and boltzmann exploration
            action = softmax_routing_policy(env, q_values, temperature = 1)
            state_counts[state['last_origin_server'], action] += 1
            # action = np.random.choice(env.N_servers, p = routing_policy[env.current_state['last_origin_server'], :])
            # we compute the reward
            next_state, reward, _, _ = env.step(action)
            # print(state['last_origin_server'], action, reward)
            # we update the q values
            lr = learning_rate / state_counts[state['last_origin_server'], action]
            q_values[state['last_origin_server'], action] = (1 - lr) * q_values[state['last_origin_server'], action] + lr * (reward + discount_factor * np.max(q_values[next_state['last_origin_server'], :]))
            state = next_state

    # print(state_counts)

    # finally we compute the routing policy according to the q values and evaluate it
    # print(q_values)
    updated_routing_policy = np.zeros((env.N_servers, env.N_servers))
    for i in range(env.N_servers):
        for j in range(env.N_servers):
            updated_routing_policy[i, j] = math.exp(q_values[i, j])
    # updated_routing_policy = updated_routing_policy/np.sum(updated_routing_policy, axis = 1)

    # updated_routing_policy = projection_routing_policy(updated_routing_policy)
    for i in range(updated_routing_policy.shape[0]):
        updated_routing_policy[i, :] = q_values[i, :] / np.sum(q_values[i, :])
    # we evaluate the routing policy
    total_reward = 0
    for j in range(env.N_servers):
        for i in range(env.N_servers):
            env.servers[j].arrival_rates_server[i] = env.arrival_rates[i] * updated_routing_policy[i, j]
        total_reward += env.servers[j].evaluate_policy_single_server(env.servers[j].actor)[0]

    return updated_routing_policy, total_reward







