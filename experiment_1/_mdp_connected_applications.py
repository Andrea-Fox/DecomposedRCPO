import math 
import scipy.integrate as integrate
import numpy as np
import random
import torch
import copy




# NOTE ON THE APPLICATIONS: we consider only one application for each area of origin. The area of interest describes if the server has that application on it

class LoadBalancingSystem():

    def __init__(self, N_servers, servers_parameters, arrival_rates, discount_factor, load_balancing_to_emptiest_server = False, device = 'cpu'):
        # servers_parameters is a list of dictionaries, each dictionary contains the parameters for a server:
        if len(servers_parameters) != N_servers:
            raise 'servers_parameters must be a list of length N_servers'

        self.N_servers = N_servers
        self.discount_factor = discount_factor
        self.max_memory_capacity = -math.inf

        self.servers_parameters = servers_parameters
        self.arrival_rates = arrival_rates

        # to initialize the servers, we need to define the arrival rates for each server
        # We start assuming an uniform routing probaility 
        self.arrival_rates = arrival_rates 
        initial_routing_probability = np.ones((N_servers, N_servers)) / N_servers
        servers_arrival_rates = np.zeros((N_servers, N_servers))
        self.load_balancing_policy = None

        self.servers_parameters = servers_parameters
        self.lagrange_multiplier = np.zeros((N_servers, ))

        self.load_balancing_to_emptiest_server = load_balancing_to_emptiest_server

        # If the initial arrival rate is the one of the large environment, the arrival rate to a specific server is obtained 
        # dividing by the probability of routing to that server. Looking at the parameters of the exponential distribution, 
        # we see that the rate is the inverse of the mean, so we divide by the probability of routing to that server
        for i in range(N_servers):
            servers_arrival_rates[i, :] = arrival_rates[i] * initial_routing_probability[i, :]

        # the parameter defined as the 'pricessing time' is indeed the average permanence time of video from that area. 
        # Therefore in the transitions, it will not depend on the server but rather on the origin area of the video analyzed
        # We change the value of 'mu' so that it is a vector, equal for each server
        permanence_time = np.zeros((N_servers, ))
        for i in range(N_servers):
            permanence_time[i] = servers_parameters[i]['mu']
        for i in range(N_servers):
            servers_parameters[i]['mu_vector'] = permanence_time

        # we initialize the servers
        self.servers = []
        self.access_capacity = np.zeros((N_servers, ))
        for i in range(N_servers):
            self.servers.append(AdmissionServer(servers_parameters[i], N_servers, servers_arrival_rates[i, :], discount_factor))
            self.max_memory_capacity = max(self.max_memory_capacity, self.servers[i].memory_capacity)
            self.access_capacity[i] = self.servers[i].server_access_capacity
        
        self.max_capacity = int(self.max_memory_capacity * self.N_servers/2)

        # the only possible action is the routing to one of the servers

        self.episode_length = 0
        self.max_length_episode = 250
        self.device = device

        self.current_state = {}

        self.current_state['last_origin_server'] = -math.inf
        self.current_state['destination_server'] = -math.inf
        for i in range(N_servers):
            self.current_state['server_' + str(i+1) +'_occupation'] = np.zeros((self.N_servers, ))


    def step(self, action):
        # the server to which we import the stream is the one corresponding to the action
        # this server will have to deal with a new stream
        # the other servers will just complete the necessary transitions

        # first we compute the time to the next arrival, as well as the origin server of the following stream
        times = np.zeros((self.N_servers, ))  
        for j in range(self.N_servers):
            times[j] = np.random.exponential(1/self.arrival_rates[j])
        time_to_next_arrival = np.min(times)
        next_origin_server = np.argmin(times)
        # print(time_to_next_arrival, next_origin_server)

        # then we update the state, to comunicate that it has received a stream from area self._last_origin_server
        last_origin_server = self.current_state['last_origin_server']
        last_destination_server = self.current_state['destination_server']
        # self.servers[last_destination_server].current_state['last_origin_server'] = self.current_state['last_origin_server']

        # then according to the action given as an input (results of the admission policy) we can write the transition
        _, penalized_reward, _, additional_info = self.servers[last_destination_server].step(action, time_to_next_arrival=time_to_next_arrival, system = self) 
        
        immediate_reward = additional_info['immediate_reward']
        # the only component of the cost function different from 0 is the one of the server that receives the stream 
        immediate_cost_component = additional_info['immediate_cost']
        immediate_cost = np.zeros((self.N_servers, ))
        immediate_cost[last_destination_server] = immediate_cost_component

        # finally, we compute the step also for the other servers (action is clearly 0, as they have not received any stream)
        for i in range(self.N_servers):
            if i != last_destination_server:
                self.servers[i].step(0, time_to_next_arrival= time_to_next_arrival, system = self)

        # the reward is possibly larger than 0 only in the server that receives the stream
        if self.episode_length >= self.max_length_episode:
            done = True
        else:
            self.episode_length += 1
            done = False

        # among those interested, choose the server with the smallest occupation
        if self.load_balancing_to_emptiest_server:
            next_destination_server = []
            next_destination_server_occupation = math.inf
            for j in range(self.N_servers):
                if self.servers[j].areas_of_interest[next_origin_server]:
                    total_occupation = np.sum(self.current_state['server_' + str(j+1) + '_occupation'])
                    if total_occupation < next_destination_server_occupation:
                        next_destination_server = [j]
                        next_destination_server_occupation = total_occupation
                    elif total_occupation == next_destination_server_occupation:
                        next_destination_server.append(j)
            # sample among elements of next_destination_server
            next_destination_server = np.random.choice(next_destination_server)
            
        else:
            next_destination_server = self.load_balancing_policy.select_destination_server(self, origin_area = next_origin_server)


        self.current_state['last_origin_server'] = next_origin_server
        self.current_state['destination_server'] = next_destination_server
        # for i in range(self.N_servers):
        #     self.current_state['server_' + str(i+1) +'_occupation'] = self.servers[i].current_state['server_occupation']

        return self.current_state, penalized_reward, done, {'immediate_reward': immediate_reward, 'immediate_cost': immediate_cost, 'component_considered': last_destination_server}


    def reset(self, evaluation = False):
        self.current_state['last_origin_server'] = np.random.randint(self.N_servers)
        next_destination_server = self.load_balancing_policy.select_destination_server(self, origin_area =  self.current_state['last_origin_server'])
        self.current_state['destination_server'] = next_destination_server
        for i in range(self.N_servers):
            self.current_state['server_' + str(i+1) +'_occupation'] = np.zeros((self.N_servers, ), dtype=int)
        
        # we first sample the initial occupation for each application
        application_occupation = np.zeros((self.N_servers,), dtype=int)
        max_initial_occupation = min(5, np.sum(self.servers[0].areas_of_interest))
        if not evaluation:
            for i in range(self.N_servers):
                application_occupation[i] = int(random.randint(0,  0 * max_initial_occupation))

        
        # we now distribute this occupation among the possible applications interested in the information flows from the area
        for origin_area in range(self.N_servers):
            # we need to distribute application_occupation[origin_area] flows
            probability = np.zeros((self.N_servers))
            for destination_server in range(self.N_servers):
                probability[destination_server] = self.servers[destination_server].areas_of_interest[origin_area]
            probability = probability/np.sum(probability)
            
            # print(application_occupation)
            # print(probability)
            for _ in range(application_occupation[origin_area]):
                destination_server = np.random.choice(self.N_servers, p = probability)
                if self.current_state['server_' + str(destination_server+1) +'_occupation'][origin_area] < self.servers[destination_server].memory_capacity:
                    self.current_state['server_' + str(destination_server+1) +'_occupation'][origin_area] += 1
                else:
                    # we need to sample the destination server again and redefine the probability of routing
                    probability[destination_server] = 0
                    probability = probability/np.sum(probability)
       
        return self.current_state, None

    def compute_occupation_origin_area(self, origin_area, state = None):
        # compute the occupation given a certain origin area. It takes into consideration the area of interest of the servers
        # As a consequence, if destination_server i is not interested in data form area j, then the occupation to consider
        # in destination_server is 0

        if state is None:
            state = self.current_state

        occupation = 0
        for i in range(self.N_servers):
            if self.servers[i].areas_of_interest[origin_area]:
                occupation += state['server_' + str(i+1) +'_occupation'][origin_area]
        return occupation

    
    def evaluate_policy(self, policy, n_episodes = 100, length_episode = 100, random_admission = False, verbose = True):
        # function thate evaluates the current routing policy
        # could be interesting to evaluate the contribution of each server to the total reward

        # the cost is a vector, containing the cost of each server (we analyze multiple cost functions simultaneously)
        discounted_reward_episode = []
        discounted_costs_episode = np.zeros((n_episodes, self.N_servers))

        average_occupation_servers = np.zeros((n_episodes, self.N_servers))
        average_action_server = np.zeros((n_episodes, self.N_servers))
        length_episode = self.max_length_episode
        final_occupation =  np.zeros((n_episodes, self.N_servers))
        if verbose:
            for i in range(n_episodes):
                state, _ = self.reset(evaluation=True)

                discounted_reward = 0
                discounted_cost = np.zeros((self.N_servers, ))
                average_occupation = np.zeros((self.N_servers, ))
                average_action = np.zeros((self.N_servers, ))
                average_destination_server = np.zeros((self.N_servers, ))
                for index in range(self.N_servers):
                    average_occupation[index] += np.sum(state['server_'+str(index+1)+'_occupation'])
            
                for t in range(length_episode):
                    action, _ = policy.return_action(self, state, exploration = False)
                    average_action[state['destination_server']] += action
                    average_destination_server[state['destination_server']] += 1
                    next_state, _, _, additional_info = self.step(action)
                    discounted_reward += self.discount_factor**t * additional_info['immediate_reward']
                    discounted_cost += self.discount_factor**t * additional_info['immediate_cost']
                    state = next_state
                    for server in range(self.N_servers):
                        average_occupation[server] += np.sum(state['server_'+str(server+1)+'_occupation'])

                for server in range(self.N_servers):
                    final_occupation[i, server] = np.sum(state['server_'+str(server+1)+'_occupation'])

                discounted_reward_episode.append(np.copy(discounted_reward))
                discounted_costs_episode[i, :] = np.copy(discounted_cost)
                average_occupation_servers[i, :] = np.copy(average_occupation)/length_episode
                if average_destination_server.any() == 0:
                    average_action_server[i, :] = 0
                else:
                    average_action_server[i, :]      = average_action/average_destination_server
            print(np.mean(average_occupation_servers, axis = 0), np.mean(average_action_server, axis = 0), average_destination_server, np.mean(final_occupation, axis = 0))
            return np.mean(discounted_reward_episode), np.mean(discounted_costs_episode, axis = 0)
        else:
            for i in range(n_episodes):
                state, _ = self.reset(evaluation=True)
                discounted_reward = 0
                discounted_cost = np.zeros((self.N_servers, ))
                for t in range(length_episode):
                    action, _ = policy.return_action(self, state, exploration = False)
                    next_state, _, _, additional_info = self.step(action)
                    discounted_reward += self.discount_factor**t * additional_info['immediate_reward']
                    discounted_cost += self.discount_factor**t * additional_info['immediate_cost']
                    state = next_state
                    
                discounted_reward_episode.append(np.copy(discounted_reward))
                discounted_costs_episode[i, :] = np.copy(discounted_cost)
            return np.mean(discounted_reward_episode), np.mean(discounted_costs_episode, axis = 0)


    def state_to_tensor(self, state):
        # this function takes as an input the state of the system and returns a tensor that can be used as an input for the neural network
        # the tensor is a vector of length 2 * N_servers + 2
        # the first N_servers**2 components are the occupation of the servers

        state_tensor_array = np.zeros((self.N_servers**2 + 2, ))
        for i in range(self.N_servers):
            state_tensor_array[i*self.N_servers:(i+1)*self.N_servers] = state['server_' + str(i+1) +'_occupation']
        state_tensor_array[-2] = state['last_origin_server']
        state_tensor_array[-1] = state['destination_server']
        return torch.tensor(state_tensor_array, dtype=torch.float32, device=self.device).unsqueeze(0)


    def __deepcopy__(self, memodict = {}):
        new_object = LoadBalancingSystem(self.N_servers, self.servers_parameters, self.arrival_rates, self.discount_factor, self.device)
        # new_object.load_balancing_policy = copy.deepcopy(self.load_balancing_policy)

        # If the initial arrival rate is the one of the large environment, the arrival rate to a specific server is obtained 
        # dividing by the probability of routing to that server. Looking at the parameters of the exponential distribution, 
        # we see that the rate is the inverse of the mean, so we divide by the probability of routing to that server
        servers_arrival_rates = np.zeros((new_object.N_servers, new_object.N_servers))
        # for i in range(new_object.N_servers):
        #     servers_arrival_rates[i, :] = self.arrival_rates[i] * new_object.load_balancing_policy.load_balancing_policy[i, :]

        # we initialize the servers
        new_object.servers = []
        new_object.access_capacity = np.zeros((new_object.N_servers, ))
        for i in range(new_object.N_servers):
            new_object.servers.append(AdmissionServer(new_object.servers_parameters[i], new_object.N_servers, servers_arrival_rates[i, :], new_object.discount_factor))
            new_object.max_memory_capacity = max(new_object.max_memory_capacity, new_object.servers[i].memory_capacity)
            new_object.access_capacity[i] = new_object.servers[i].server_access_capacity
            
        new_object.max_capacity = int(new_object.max_memory_capacity * new_object.N_servers/2)
        
        # the only possible action is the routing to one of the servers
        

        new_object.episode_length = 0
        new_object.max_length_episode = 250
        new_object.device = 'cpu'

        new_object.current_state = {}
        new_object.current_state['last_origin_server'] = -math.inf
        new_object.current_state['destination_server'] = -math.inf
        for i in range(new_object.N_servers):
            new_object.current_state['server_' + str(i+1) +'_occupation'] = np.zeros((new_object.N_servers, ))


        return new_object


    def evaluate_routing_policy(self, routing_policy, discount_factor = 0.99, n_episodes = 100):

        list_discounted_reward = []
        list_discounted_reward_per_server = [] 
        
        # we simulate 100 episodes and then take the average discounted reward given by the routing policy and the admission policies
        for i in range(n_episodes):
            discounted_reward = 0
            discounted_reward_per_server = np.zeros((self.N_servers, ))
            state, _ = self.reset()
            t = 0
            while t < self.max_length_episode:
                # routing_policy.print_probabilities(self, state)
                action = routing_policy.select_section(self, state)
                # routing_policy.print_probabilities(self, state)
                # print(action)
                # print('-------------------')
                next_state, _, done, additional_info = self.step(action)
                reward = additional_info['immediate_reward']
                discounted_reward += discount_factor**t * reward
                discounted_reward_per_server[action] += discount_factor**t * reward
                state = next_state
                t += 1
            list_discounted_reward.append(discounted_reward)
            list_discounted_reward_per_server.append(discounted_reward_per_server)
        
        mean_discounted_reward = np.mean(list_discounted_reward)
        mean_discounted_reward_per_server = np.mean(list_discounted_reward_per_server, axis = 0)
        # print(list_discounted_reward)

        return mean_discounted_reward, None


    def compute_approximated_arrival_rates(self, routing_policy, n_episodes = 100):
        routing_probability = np.zeros((self.N_servers, self.N_servers))
        # routing_probability[i, j] is the the probability of routing a stream from area i to area j
        
        # server_1_occupation_level = np.zeros((15,))
        # server_2_occupation_level = np.zeros((15,))
        # server_3_occupation_level = np.zeros((15,))

        for i in range(n_episodes):
            state, _ = self.reset()
            t = 0
            while t < self.max_length_episode:
                # routing_policy.print_probabilities(self, state)
                # server_1_occupation_level[state['server_1_occupation'][state['last_origin_server']]] += 1
                # server_2_occupation_level[state['server_2_occupation'][state['last_origin_server']]] += 1
                # server_3_occupation_level[state['server_3_occupation'][state['last_origin_server']]] += 1
                action, _ = routing_policy.select_section(self, state)
                
                # routing_policy.print_probabilities(self, state)
                # print(action)
                # print('-------------------')
                routing_probability[state['last_origin_server'], action] += 1
                next_state, _, _, _ = self.step(action)
                state = next_state
                t += 1

        # print(server_1_occupation_level, server_2_occupation_level, server_3_occupation_level)
        # print(np.max(routing_probability), np.min(routing_probability))
        # we normalize the routing probability
        for i in range(self.N_servers):
            sum_i = np.sum(routing_probability[i, :])
            for j in range(self.N_servers):
                routing_probability[i, j] = routing_probability[i, j] / sum_i
        # print(routing_probability)
        return routing_policy, StatelessRouting(self, routing_probability)





class AdmissionServer():

    # we consider only one arrival server and several origin servers
    def __init__(self, parameters, N_servers, arrival_rates, discount_factor):
        super().__init__()

        # in parameters, we find a dictionary containing the parameters of the environment that won't change
        self.server_id = parameters['server_id']
        self.N_servers = N_servers
        self.num_states = N_servers * parameters['C']
        self.memory_capacity = parameters['C']
        # self.binding_properties = parameters['chi']
        self.server_access_capacity = parameters['phi']
        # now this is a vector
        self.area_permanence_time = parameters['mu_vector']
        
        self.areas_of_interest = parameters['areas_of_interest']    # it will be a vector of length N_servers, containing either True or False
        # we assume one application for each area of interest
        self.application_number = np.sum(self.areas_of_interest)

        # reward function parameters (if not specified, we assume the decreasing one that changes order)
        try:
            self.reward_function_type = parameters['reward_function_type']
        except:
            self.reward_function_type = 1

        if self.reward_function_type == 2:
            self.reward_multiplier = parameters['reward_function_parameters'][0]
            self.reward_final_value = parameters['reward_function_parameters'][1]
            self.reward_descent_rate = parameters['reward_function_parameters'][2]
        else:
            self.reward_multiplier = None
            self.reward_final_value = None
            self.reward_descent_rate = None

        self.discount_factor = discount_factor

        # This quantity has to be a vector of length M
        if len(arrival_rates) != N_servers:
            raise 'arrival_rates must be a vector of length M'

        # the arrival rates will be reset often, due to the routing policy
        self.arrival_rates_server = arrival_rates
        self.device = 'cpu'


    def reward_function(self, state, system):
        # we use the input state (saved as a dictionary) to compute the reward function
        application_to_consider = state['last_origin_server']
        # then, for all the server, we conside the occupation of the servers which have interest in the area
        application_occupation = system.compute_occupation_origin_area(application_to_consider, state)
        if self.reward_function_type == 0:
            raise 'reward function type not implemented'
            return (server_to_consider + 1)/self.N_servers    
        elif self.reward_function_type == 1:
            return 2 * math.exp(-application_occupation/5) +2 
        elif self.reward_function_type == 2:
            return ((10 * self.reward_multiplier)/system.N_servers) * math.exp(-application_occupation * self.reward_descent_rate/system.N_servers) + self.reward_final_value 
        else:
            raise 'reward function type not recognized'

    def compute_immediate_reward(self, action, system = None):
        if action == 0:
            return 0
        else:
            immediate_reward = self.reward_function(system.current_state, system)
            return immediate_reward

    def compute_immediate_cost(self, action, system = None):
        # for the moment we assume that the cost function corresponds to the total occupation of the server
        if action == 0:
            return 0
        else:
            destination_server = system.current_state['destination_server']
            server_occupation = system.current_state['server_' +str(destination_server+1)+ '_occupation']
            return np.sum(server_occupation)

    def step(self, action, system = None, time_to_next_arrival = None):    
        # if the action is to accept the stream, we need to update the state (by adding the new information flow) and compute the required reward
        if action == 1:
            immediate_reward = self.compute_immediate_reward(action, system)
            immediate_cost = self.compute_immediate_cost(action, system)
            lagrangian_reward = immediate_reward - system.lagrange_multiplier[self.server_id] * immediate_cost
            # print(lagrangian_reward)
            if self.server_id != system.current_state['destination_server']:
                raise 'something went wrong'

            system.current_state['server_' + str(self.server_id +1) +'_occupation'][system.current_state['last_origin_server']] += 1

        # on the other hand, if the action is to reject, the reward is 0
        elif action == 0:
            immediate_reward = 0
            immediate_cost = 0
            lagrangian_reward = 0
        
        else:
            print(action)
            raise 'action not recognized'

        # finally, regardless of the action, we must simulate the transition of the system to the next state for the server considered

        # then we compute, for each arrival server, the amount of streams that have left
        streams_that_have_left = np.zeros((self.N_servers, ))
        for origin_area in range(self.N_servers):
            if system.current_state['server_' + str(self.server_id +1) +'_occupation'][origin_area] > 0:
                for _ in range(system.current_state['server_' + str(self.server_id +1) +'_occupation'][origin_area]):
                    if np.random.exponential(1/self.area_permanence_time[origin_area]) < time_to_next_arrival:
                        streams_that_have_left[origin_area] += 1
        # print(time_to_next_arrival, self.current_state['server_occupation'], np.sum(streams_that_have_left))

        for origin_area in range(self.N_servers):
            system.current_state['server_'  + str(self.server_id +1) +'_occupation'][origin_area] -= streams_that_have_left[origin_area]
        # we then update the state
           
        # return self._state_to_tensor(self.current_state), lagrangian_reward, False, False, None
        additional_info = {"immediate_reward": immediate_reward, "immediate_cost": immediate_cost}
        return None, lagrangian_reward, None, additional_info
