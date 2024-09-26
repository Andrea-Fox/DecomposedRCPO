from _mdp_admission import *
from _routing_policy import *
import copy



# the large environment we consider will contain a list of servers where the admission policy will be computed
# moreover, the state space will be a consequence of the state space defined by each server

# in the __init__ function for this new MDP, we will need to define all the parameters for each video analytics server



class RoutingServer(gym.Env):

    def __init__(self, N_servers, servers_parameters, arrival_rates, discount_factor):

        # servers_parameters is a list of dictionaries, each dictionary contains the parameters for a server:
        if len(servers_parameters) != N_servers:
            raise 'servers_parameters must be a list of length N_servers'

        self.N_servers = N_servers
        self.discount_factor = discount_factor

        # to initialize the servers, we need to define the arrival rates for each server
        # We start assuming an uniform routing probaility 
        self.arrival_rates = arrival_rates 
        initial_routing_probability = np.ones((N_servers, N_servers)) / N_servers
        servers_arrival_rates = np.zeros((N_servers, N_servers))
        self.routing_policy = None

        self.servers_parameters = servers_parameters

        # If the initial arrival rate is the one of the large environment, the arrival rate to a specific server is obtained 
        # dividing by the probability of routing to that server. Looking at the parameters of the exponential distribution, 
        # we see that the rate is the inverse of the mean, so we divide by the probability of routing to that server
        for i in range(N_servers):
            servers_arrival_rates[i, :] = arrival_rates[i] * initial_routing_probability[i, :]

        # we initialize the servers
        self.servers = []
        for i in range(N_servers):
            self.servers.append(AdmissionServer(servers_parameters[i], N_servers, servers_arrival_rates[i, :], discount_factor))
        
        # the only possible action is the routing to one of the servers
        spaces = {"last_origin_server": Discrete(self.N_servers)}
        for i in range(N_servers):
            spaces['server_' + str(i+1) +'_occupation'] = Box(0, self.servers[i].memory_capacity, shape=(self.N_servers,), dtype=int )
        self.observation_space = Dict(spaces)

        self.action_space = Discrete(N_servers)

        self.episode_length = 0
        self.max_length_episode = 1000
        self.device = 'cpu'

        self.current_state = self.observation_space.sample()
        self.current_state['last_origin_server'] = -math.inf
        for i in range(N_servers):
            self.current_state['server_' + str(i+1) +'_occupation'] = self.servers[i].current_state['server_occupation']


    def step(self, action):
        # the server to which we import the stream is the one corresponding to the action
        # this server will have to deal with a new stream
        # the other servers will just complete the necessary transitions


        # first we compute the time to the next arrival, as well as the origin server of the following stream
        times = np.zeros((self.N_servers, 1))  
        for j in range(self.N_servers):
            times[j] = np.random.exponential(1/self.arrival_rates[j])
        time_to_next_arrival = np.min(times)
        next_origin_server = np.argmin(times)
        # print(time_to_next_arrival, next_origin_server)

        # then we update the state, to comunicate that it has received a stream from area self._last_origin_server
        last_origin_server = self.current_state['last_origin_server']
        self.servers[action].current_state['last_origin_server'] = self.current_state['last_origin_server']

        # then we choose the action according to its admission policy
        if np.sum(self.servers[action].current_state['server_occupation']) < self.servers[action].memory_capacity:
            component_occupation_level_destination_server = self.servers[action].current_state['server_occupation'][self.current_state['last_origin_server']]
            admission_action = self.servers[action].actor.admission_policy[component_occupation_level_destination_server, self.current_state['last_origin_server'], np.sum(self.servers[action].current_state['server_occupation'])]
        else:
            admission_action = 0
        # print('admission action: ', admission_action)
        _, _, _, additional_info = self.servers[action].step( admission_action, time_to_next_arrival )
        reward = additional_info['immediate_reward']
        # finally, we compute the step also for the other servers (action is clearly 0, as they have not received any stream)
        for i in range(self.N_servers):
            if i != action:
                self.servers[i].current_state['last_origin_server'] = None
                self.servers[i].step(0,  time_to_next_arrival)

        # the reward is possibly larger than 0 only in the server that receives the stream
        if self.episode_length >= self.max_length_episode:
            done = True
        else:
            self.episode_length += 1
            done = False

        self.current_state['last_origin_server'] = next_origin_server
        for i in range(self.N_servers):
            self.current_state['server_' + str(i+1) +'_occupation'] = self.servers[i].current_state['server_occupation']

        return self.current_state, reward, done, {'immediate_reward': reward}


    def reset(self):
        self.current_state['last_origin_server'] = np.random.randint(self.N_servers)
        for i in range(self.N_servers):
            self.current_state['server_' + str(i+1) +'_occupation'] = np.zeros((self.N_servers, ), dtype=int)
            self.servers[i].reset()
        return self.current_state, None

    
    def evaluate_routing_policy(self, policy):
        # function thate evaluates the current routing policy
        # could be interesting to evaluate the contribution of each server to the total reward
        raise NotImplementedError('This function has not been implemented yet')
        return discounted_reward, discounted_reward_per_server, discounted_cost, discounted_cost_per_server

    def __deepcopy__(self, memodict = {}):
        new_object = RoutingServer(self.N_servers, self.servers_parameters, self.arrival_rates, self.discount_factor)

        new_object.N_servers = copy.deepcopy(self.N_servers)
        new_object.discount_factor = copy.deepcopy(self.discount_factor)

        # to initialize the servers, we need to define the arrival rates for each server
        # We start assuming an uniform routing probaility 
        new_object.arrival_rates = copy.deepcopy(self.arrival_rates) 
        initial_routing_probability = np.ones((new_object.N_servers, new_object.N_servers)) / new_object.N_servers
        servers_arrival_rates = np.zeros((new_object.N_servers, new_object.N_servers))
        new_object.routing_policy = None

        new_object.servers_parameters = copy.deepcopy(self.servers_parameters)


        # If the initial arrival rate is the one of the large environment, the arrival rate to a specific server is obtained 
        # dividing by the probability of routing to that server. Looking at the parameters of the exponential distribution, 
        # we see that the rate is the inverse of the mean, so we divide by the probability of routing to that server
        for i in range(new_object.N_servers):
            servers_arrival_rates[i, :] = new_object.arrival_rates[i] * initial_routing_probability[i, :]

        # we initialize the servers
        new_object.servers = []
        for i in range(new_object.N_servers):
            new_object.servers.append(AdmissionServer(new_object.servers_parameters[i], new_object.N_servers, servers_arrival_rates[i, :], new_object.discount_factor))
            new_object.servers[i].actor = copy.deepcopy(self.servers[i].actor)
            new_object.servers[i].critic = copy.deepcopy(self.servers[i].critic)

        # the only possible action is the routing to one of the servers
        spaces = {"last_origin_server": Discrete(new_object.N_servers)}
        for i in range(new_object.N_servers):
            spaces['server_' + str(i+1) +'_occupation'] = Box(0, new_object.servers[i].memory_capacity, shape=(new_object.N_servers,), dtype=int )
        new_object.observation_space = Dict(spaces)

        new_object.action_space = Discrete(new_object.N_servers)

        new_object.episode_length = 0
        new_object.max_length_episode = 1000
        new_object.device = 'cpu'

        new_object.current_state = new_object.observation_space.sample()
        new_object.current_state['last_origin_server'] = -math.inf
        for i in range(new_object.N_servers):
            new_object.current_state['server_' + str(i+1) +'_occupation'] = new_object.servers[i].current_state['server_occupation']


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
                action = routing_policy.select_section(self, state)
                
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




