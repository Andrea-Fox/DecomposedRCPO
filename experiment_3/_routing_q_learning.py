

# we compute the Q-values for each state-action pair
# the states will be represented through a vector of size N_servers + 1, where the last component is the last origin server
# for each server we will only consider the occupation of the server corresponding to the last origin server

# for the moment, we consider systems sufficiently small that allow us to compute the Q-values for each state-action pair
# in the future we will employ a neural network to approximate the Q-values

# we will use a greedy policy to choose the action
# for convenience reasons, we consider the 1-step version of Q-learning
import numpy as np
import math
from _routing_policy import StatelessRouting, ComponentOccupationRouting


def epsilon_greedy_exploration(env, q_values, state, exploration_rate = 0.05, areas_of_interest = False):
    if np.random.uniform(0, 1) < exploration_rate:
        if areas_of_interest:
            origin_area = state['last_origin_server']
            # we only allow it to send to servers that have interest in this video
            area_interested = []    
            for i in range(env.N_servers):
                if env.servers[i].areas_of_interest[origin_area]:
                    area_interested.append(i)
            return np.random.choice(area_interested)
        else:
            return np.random.choice(env.N_servers)
    else:
        last_origin_server = state['last_origin_server']
        server_1_occupation = state['server_1_occupation'][last_origin_server]
        server_2_occupation = state['server_2_occupation'][last_origin_server]
        server_3_occupation = state['server_3_occupation'][last_origin_server]
        if areas_of_interest:
            origin_area = state['last_origin_server']
            # we only allow it to send to servers that have interest in this video
            area_interested = []    
            for i in range(env.N_servers):
                if env.servers[i].areas_of_interest[origin_area]:
                    area_interested.append(i)
            max_action = np.argmax([q_values[ server_1_occupation, server_2_occupation, server_3_occupation, last_origin_server, action] for action in area_interested])
            # print(max_action)
            return area_interested[max_action]
        else:
            return np.argmax([q_values[ server_1_occupation, server_2_occupation, server_3_occupation, last_origin_server, action] for action in range(env.action_space.n)])

def q_learning_routing(env, ordered_q_learning = False, learning_rate = 0.05, learning_rate_exponent = 0.1, num_episodes = 100, consider_areas_of_interest = False):

    max_capacity = np.max([env.servers[i].memory_capacity for i in range(env.N_servers)])

    # we initialize the state counts table to 0, counting only the instances of videos from area j
    state_counts_table = np.ones((max_capacity+1, max_capacity+1, max_capacity+1, env.N_servers))

    # when routing, we have to look, for each server, at the occupation level of the corresponding area of interest
    # the state will take into consideration: the area of origin[2], the possible area of destination[1] and 
    # the occupancy level of the are of destination[0]
    # q_values[i1, i2, a] denotes the estimate of the discounted return when we receive a video from area i2 and we 
    # sent it to area a, where the occupation level of videos from that area is i1
    q_values = np.zeros((max_capacity+1, max_capacity+1, max_capacity+1, env.N_servers, env.action_space.n))

    # if consider_areas_of_interest:
    #     # in order to avoid sending videos to areas where the server has no interest, we set to -infinite the corresponding q-values  
    #     for i in range(env.N_servers):
    #         for j in range(env.N_servers):
    #             if not env.servers[j].areas_of_interest[i]:
    #                 q_values[:, :, :, i, j] = -math.inf
    total_reward_evolution = []

    ordered_q_learning = False

    for episode_index in range(num_episodes):
        # the following quantities need to be resetted as soon as we change the lagrangian multiplier
        future_states = []
        future_rewards = []
        future_actions = []
        
        t = 0
        terminal_time = math.inf

        state, _ = env.reset()
        # we add S_0 to the list of states
        future_states.append(state)   

        # we add R_0 = 0 to the list of rewards, as the rewarda have to be considered only from 1 onwards
        future_rewards.append(0)
        # future_actions.append(0)
        done = False
        reward_sum = 0

        while t < env.max_length_episode:
            # find the optimal action according to the current policy.
            # which one is a better method? Boltzmann exploration or epsilon-greedy?
            # action = boltzmann_exploration(state=state, q_values=self.table, temperature = 1)
            action = epsilon_greedy_exploration(env, q_values, state, exploration_rate = 0.1, areas_of_interest=consider_areas_of_interest)

            future_actions.append(action)

            # take action A_t
            next_state, _, done, additional_info = env.step(action)

            # observe and store the next reward as R_{t+1} and the next state as S_{t+1}
            # indeed thay are going to be element t+1 in the respective lists
            # print(next_state)
            reward = additional_info['immediate_reward']
            future_states.append(next_state)
            reward_sum += (env.discount_factor**t) * reward
            future_rewards.append(reward)    

            # If S_{t+1} is terminal, then update terminal_time
            if done or t > env.max_length_episode:
                terminal_time = t + 1
            
            server_1_occupation_level = state['server_1_occupation'][state['last_origin_server']]
            server_2_occupation_level = state['server_2_occupation'][state['last_origin_server']]
            server_3_occupation_level = state['server_3_occupation'][state['last_origin_server']]

            next_server_1_occupation_level = next_state['server_1_occupation'][next_state['last_origin_server']]
            next_server_2_occupation_level = next_state['server_2_occupation'][next_state['last_origin_server']]
            next_server_3_occupation_level = next_state['server_3_occupation'][next_state['last_origin_server']]
            state_counts_table[server_1_occupation_level, server_2_occupation_level, server_3_occupation_level, state['last_origin_server']] += 1
            lr = learning_rate / (state_counts_table[server_1_occupation_level, server_2_occupation_level, server_3_occupation_level, state['last_origin_server']]**learning_rate_exponent)

            q_values[ server_1_occupation_level, server_2_occupation_level , server_3_occupation_level, state['last_origin_server'], action] += lr * (reward + env.discount_factor * np.max([q_values[ next_server_1_occupation_level, next_server_2_occupation_level, next_server_3_occupation_level, next_state['last_origin_server'], next_action] for next_action in range(env.action_space.n)]) - q_values[ server_1_occupation_level, server_2_occupation_level , server_3_occupation_level, state['last_origin_server'], action])

            if ordered_q_learning:
                # we assume that:
                # 1) Q(x1, x2, x3, i, j, a) > Q(x1 +1 , x2, x3, i, j, a) for every i, j, a
                # 2) Q(x1, x2, x3, i, j, a) > Q(x1, x2 + 1, x3, i, j, a) for every i, j, a
                # 3) Q(x1, x2, x3, i, j, a) > Q(x1, x2, x3 + 1, i, j, a) for every i, j, a

                # we check if the ordering is satisfied for x1
                for higher_occupation_level in range(server_1_occupation_level, env.servers[0].memory_capacity):
                    q_values[higher_occupation_level, server_2_occupation_level, server_3_occupation_level, state['last_origin_server'], action] = min(q_values[server_1_occupation_level, server_2_occupation_level, server_3_occupation_level, state['last_origin_server'], action], q_values[higher_occupation_level, server_2_occupation_level, server_3_occupation_level, state['last_origin_server'], action])
                for smaller_occupation_level in range(server_1_occupation_level):
                    q_values[smaller_occupation_level, server_2_occupation_level, server_3_occupation_level, state['last_origin_server'], action] = max(q_values[server_1_occupation_level, server_2_occupation_level, server_3_occupation_level, state['last_origin_server'], action], q_values[smaller_occupation_level, server_2_occupation_level, server_3_occupation_level, state['last_origin_server'], action])
                # we check if the ordering is satisfied for x2
                for higher_occupation_level in range(server_2_occupation_level, env.servers[1].memory_capacity):
                    q_values[server_1_occupation_level, higher_occupation_level, server_3_occupation_level, state['last_origin_server'], action] = min(q_values[server_1_occupation_level, server_2_occupation_level, server_3_occupation_level, state['last_origin_server'], action], q_values[server_1_occupation_level, higher_occupation_level, server_3_occupation_level, state['last_origin_server'], action])
                for smaller_occupation_level in range(server_2_occupation_level):
                    q_values[server_1_occupation_level, smaller_occupation_level, server_3_occupation_level, state['last_origin_server'], action] = max(q_values[server_1_occupation_level, server_2_occupation_level, server_3_occupation_level, state['last_origin_server'], action], q_values[server_1_occupation_level, smaller_occupation_level, server_3_occupation_level, state['last_origin_server'], action])
                # we check if the ordering is satisfied for x3
                for higher_occupation_level in range(server_3_occupation_level, env.servers[2].memory_capacity):
                    q_values[server_1_occupation_level, server_2_occupation_level, higher_occupation_level, state['last_origin_server'], action] = min(q_values[server_1_occupation_level, server_2_occupation_level, server_3_occupation_level, state['last_origin_server'], action], q_values[server_1_occupation_level, server_2_occupation_level, higher_occupation_level, state['last_origin_server'], action])
                for smaller_occupation_level in range(server_3_occupation_level):
                    q_values[server_1_occupation_level, server_2_occupation_level, smaller_occupation_level, state['last_origin_server'], action] = max(q_values[server_1_occupation_level, server_2_occupation_level, server_3_occupation_level, state['last_origin_server'], action], q_values[server_1_occupation_level, server_2_occupation_level, smaller_occupation_level, state['last_origin_server'], action])



            state = next_state
            t += 1

        return_evolution = False
        # every 100 episodes we print the total reward
        if episode_index % 10 == 0:
            # print('episode: ', episode_index)
            if return_evolution:
                ql_routing_probability = compute_routing_probability_from_q_values(env, q_values)
                ql_routing_policy = ComponentOccupationRouting(env, ql_routing_probability)
                ql_discounted_reward, discounted_reward_per_server = env.evaluate_routing_policy(ql_routing_policy)
                total_reward_evolution.append(ql_discounted_reward)

    # print(state_counts_table)
    # print(total_reward_evolution)
    return q_values, total_reward_evolution

def compute_routing_probability_from_q_values(env, q_values, consider_areas_of_interest = False):
    # we compute the routing probability for each server according to the softmax function
    # first we initialize the routing probability to 0 for each possible state
    # note how each server could possibly have a different capacity. We denote as max_capacity the quantity that is the maximum of all the capacities
    max_capacity = np.max([env.servers[i].memory_capacity for i in range(env.N_servers)])
    # print(q_values)
    if env.N_servers != 3:
        raise ValueError('The routing probability is computed only for 3 servers')
    routing_probability = np.zeros((max_capacity+1, max_capacity+1, max_capacity+1, env.N_servers, env.action_space.n))
    # routing_probability[i1, i2 ,i3, i4, i5] is the probability of routing a video from area i4 to area i5, when the occupation level of the servers are respectively i1, i2, i3
    for origin_server in range(env.N_servers):
        for server_1_occupation in range(max_capacity+1):
            for server_2_occupation in range(max_capacity+1):
                for server_3_occupation in range(max_capacity+1):
                    # we must ignore the possible values whihc are equal to -infinite
                    denominator = 0
                    if consider_areas_of_interest:
                        # we must only consider the servers that have interest in this video
                        area_interested = []
                        for destination_server in range(env.N_servers):
                            if env.servers[destination_server].areas_of_interest[origin_server]:
                                area_interested.append(destination_server)
                                denominator += np.exp(q_values[server_1_occupation, server_2_occupation, server_3_occupation, origin_server, destination_server])
                        
                        for destination_server in area_interested:
                            routing_probability[server_1_occupation, server_2_occupation, server_3_occupation, origin_server, destination_server] = math.exp(q_values[server_1_occupation, server_2_occupation, server_3_occupation, origin_server, destination_server])/denominator
                    else:
                        for j in range(env.N_servers):
                            if q_values[server_1_occupation, server_2_occupation, server_3_occupation, origin_server, j] != -math.inf:
                                denominator += np.exp(q_values[server_1_occupation, server_2_occupation, server_3_occupation, origin_server, j]) 
                        for destination_server in range(env.N_servers):
                            routing_probability[server_1_occupation, server_2_occupation, server_3_occupation, origin_server, destination_server] = math.exp(q_values[server_1_occupation, server_2_occupation, server_3_occupation, origin_server, destination_server])/denominator
    
    return routing_probability