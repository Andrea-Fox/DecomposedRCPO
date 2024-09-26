import math
import numpy as np

from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical



class Critic():

    def __init__(self, num_episodes, learning_rate, learning_steps, learning_rate_exponent):
        self.table = None
        self.state_counts_table = None
        self.num_episodes = num_episodes
        self.learning_rate = learning_rate
        self.learning_steps = 1
        # it could be intersting to understand how to deal with more than one learning step
        self.learning_rate_exponent = learning_rate_exponent
        # self.double()

    def return_value(self, env, state, action):
        return None

    def reset_state_counts(self):
        self.state_counts_table = np.ones(self.table.shape[:-1])

    def update_state_counts(self, state):
        return None
    
    def single_state_update():
        return None


class DecomposedQLearning_withOptimizedLearning(Critic):

# this critic takes into consideraiton the global action and not only the local one
    
    def __init__(self, env, num_episodes, learning_rate, learning_steps, learning_rate_exponent, initial_critic = None):
        super().__init__(num_episodes, learning_rate, learning_steps, learning_rate_exponent)
        # we also save an estimate relative to the total sum of the occupation of the servers (third component of the table)
        if initial_critic is None:
            self.table = np.zeros((env.N_servers, int(env.N_servers) * env.max_memory_capacity+1, env.N_servers, env.max_memory_capacity+2, 2))#  env.action_space.n))
        else:
            self.table = initial_critic.table

    def reset_state_counts(self, component = None):
        if component is None:
            self.state_counts_table = np.ones(self.table.shape[:-1])
        else:
            self.state_counts_table[component, :, :, :] = 1

    def reset_value_function_server(self, server_to_reset):
        self.table[server_to_reset, :, :, :, :] = 0

    def return_action(self, env, state, exploration = False):
        # we need to redefine the function in order to consider all the possible actions and the consequences on the other components
        value_function_0 = 0
        value_function_1 = 0

        destination_server = state['destination_server']
        origin_area = state['last_origin_server']

        occupation_destination_server = np.sum(abs(state['server_'+str(destination_server+1)+'_occupation']))
        # print(destination_server, state['server_'+str(destination_server+1)+'_occupation'], occupation_destination_server)
        if occupation_destination_server == 0:
            return 1, None
        elif occupation_destination_server < env.servers[destination_server].memory_capacity:
            for origin_area_component in range(env.N_servers):
                for destination_server_component in range(env.N_servers):
                    if env.servers[destination_server_component].areas_of_interest[origin_area_component]:
                        occupation_origin_area_component = env.compute_occupation_origin_area(origin_area_component, state = state)
                        occupation_destination_server_component = np.sum(state['server_'+str(destination_server_component+1)+'_occupation'])
                        if origin_area_component == origin_area and destination_server_component == destination_server:
                            value_function_0 += self.table[destination_server_component, occupation_origin_area_component, origin_area_component, occupation_destination_server_component, 0]
                            value_function_1 += self.table[destination_server_component, occupation_origin_area_component, origin_area_component, occupation_destination_server_component, 1]
                        elif origin_area_component == origin_area and destination_server_component != destination_server:
                            value_function_0 += self.table[destination_server_component, occupation_origin_area_component, origin_area_component, occupation_destination_server_component, 0]
                            value_function_1 += self.table[destination_server_component, occupation_origin_area_component+1, origin_area_component, occupation_destination_server_component, 0]
                        elif origin_area_component != origin_area and destination_server_component == destination_server:
                            value_function_0 += self.table[destination_server_component, occupation_origin_area_component, origin_area_component, occupation_destination_server_component, 0]
                            value_function_1 += self.table[destination_server_component, occupation_origin_area_component, origin_area_component, occupation_destination_server_component+1, 0]
                        else:
                            value_function_0 += self.table[destination_server_component, occupation_origin_area_component, origin_area_component, occupation_destination_server_component, 0]
                            value_function_1 += self.table[destination_server_component, occupation_origin_area_component, origin_area_component, occupation_destination_server_component, 0]
            return np.argmax([value_function_0, value_function_1]), None
        else:
            # print('max capacity reached server ', destination_server)
            return 0, None


    def parameter_update(self, env, state, action, reward, next_state, n_lagrangian_updates):
        # final return depends on the value of 'last origin server' of next state, as well as the future optimal energy
        # raise ValueError("check learning rule")
        # print(next_state)

        origin_area = state['last_origin_server']
        destination_server = state['destination_server']

        next_state_action = self.return_action(env, next_state)

        # we update every single component of the table, considering the adequate reward and action space, given the state and next state
        for origin_area_component in range(env.N_servers):
            for destination_server_component in range(env.N_servers):
                if env.servers[destination_server_component].areas_of_interest[origin_area_component]:
                    next_occupation_origin_area_component = env.compute_occupation_origin_area(origin_area_component, state = next_state)
                    next_occupation_destination_server_component = np.sum(abs(next_state['server_'+str(destination_server_component+1)+'_occupation']))

                    occupation_origin_area_component = env.compute_occupation_origin_area(origin_area_component, state = state)
                    occupation_destination_server_component = np.sum(abs(state['server_'+str(destination_server_component+1)+'_occupation']))

                    learning_rate_multiplier = .01
                    # first we compute the immediate reward
                    if origin_area_component == origin_area and destination_server_component == destination_server:
                        immediate_reward = reward
                        component_action = action
                        # we increase the counter of the visits to the state only for the component actually visited
                        self.state_counts_table[destination_server_component, occupation_origin_area_component, origin_area_component, occupation_destination_server_component] += 1
                        learning_rate_multiplier = 1
                    elif origin_area_component == origin_area and destination_server_component != destination_server:
                        occupation_origin_area_component += action
                        immediate_reward = 0
                        component_action = 0
                    elif origin_area_component != origin_area and destination_server_component == destination_server:
                        occupation_destination_server_component += action
                        immediate_reward = 0
                        component_action = 0
                    else:
                        immediate_reward = 0
                        component_action = 0

                    # then we compute the contribution of the future state
                    if next_state['last_origin_server'] == origin_area_component and next_state['destination_server'] == destination_server_component:
                        # we must consider the action that maximizes the value function (two possible component actions)
                        final_return = env.discount_factor * self.table[destination_server_component, next_occupation_origin_area_component, origin_area_component, next_occupation_destination_server_component, next_state_action[0]] 
                        learning_rate = learning_rate_multiplier * self.learning_rate/((self.state_counts_table[destination_server_component, occupation_origin_area_component, origin_area_component, occupation_destination_server_component]**1) * ((n_lagrangian_updates[destination_server_component]+1)**self.learning_rate_exponent) )
                    elif next_state['last_origin_server'] == origin_area_component and next_state['destination_server'] != destination_server_component:
                        next_occupation_origin_area_component += next_state_action[0]
                        print('kkkkkkkkkkkkkkkkkkkkkkkkkkkk')
                        final_return = env.discount_factor * self.table[destination_server_component, next_occupation_origin_area_component, origin_area_component, next_occupation_destination_server_component, 0]
                        learning_rate = learning_rate_multiplier * self.learning_rate/((self.state_counts_table[destination_server_component, occupation_origin_area_component, origin_area_component, occupation_destination_server_component]**1) * ((n_lagrangian_updates[destination_server_component]+1)**self.learning_rate_exponent) )
                    elif next_state['last_origin_server'] != origin_area_component and next_state['destination_server'] == destination_server_component:
                        next_occupation_destination_server_component += next_state_action[0]
                        final_return = env.discount_factor * self.table[destination_server_component, next_occupation_origin_area_component, origin_area_component, next_occupation_destination_server_component, 0]
                        learning_rate = learning_rate_multiplier * self.learning_rate/((self.state_counts_table[destination_server_component, occupation_origin_area_component, origin_area_component, occupation_destination_server_component]**1) * ((n_lagrangian_updates[destination_server_component]+1)**self.learning_rate_exponent) )
                    else:
                        final_return = env.discount_factor * self.table[destination_server_component, next_occupation_origin_area_component, origin_area_component, next_occupation_destination_server_component, 0]
                        learning_rate = learning_rate_multiplier * self.learning_rate/((self.state_counts_table[destination_server_component, occupation_origin_area_component, origin_area_component, occupation_destination_server_component]**1) * ((n_lagrangian_updates[destination_server_component]+1)**self.learning_rate_exponent) )
                    
                    # finally, we update the value corresponding to this state in the table
                    self.table[destination_server_component, occupation_origin_area_component, origin_area_component, occupation_destination_server_component, component_action] += learning_rate  * (immediate_reward + final_return - self.table[destination_server_component, occupation_origin_area_component,  origin_area_component, occupation_destination_server_component, component_action] )
        


class DecomposedWithOccupancyQLearning(Critic):

    def __init__(self, env, num_episodes, learning_rate, learning_steps, learning_rate_exponent):
        super().__init__(num_episodes, learning_rate, learning_steps, learning_rate_exponent)
        # we also save an estimate relative to the total sum of the occupation of the servers (third component of the table)
        self.table = np.zeros((env.N_servers, int(env.N_servers) * env.max_memory_capacity+1, env.N_servers, env.max_memory_capacity+1, env.action_space.n))


    def reset_state_counts(self, component = None):
        if component is None:
            self.state_counts_table = np.ones(self.table.shape[:-1])
        else:
            self.state_counts_table[component, :, :, :] = 1

    def return_component_value(self, env, state, action, origin_area = None, destination_server = None):
        # this function returns the value function of a single origin server, given the state and the action
        # print(state['server_occupation'][origin_server_component])
        if origin_area is None:
            origin_area = state['last_origin_server']
        
        if destination_server is None:
            destination_server = state['destination_server']

        origin_area_occupation = env.compute_occupation_origin_area(origin_area)
        destination_server_occupation = np.sum(abs(state['server_'+str(destination_server+1)+'_occupation']))

    
        return self.table[destination_server, origin_area_occupation, origin_area, destination_server_occupation, action]


    def return_value(self, env, state, action, exploration = False):
        # this function returns the sum of all the value functions for a given state and action
        sum_returns = 0
        for origin_area in range(env.N_servers):
            for destination_server in range(env.N_servers):
                if destination_server != state['destination_server'] or origin_area != state['last_origin_server']:
                    sum_returns += self.return_component_value(env, state, 0, origin_area, destination_server)
                else:
                    sum_returns += self.return_component_value(env, state, action, origin_area, destination_server)
        return sum_returns

    def reset_value_function_server(self, server_to_reset):
        self.table[server_to_reset, :, :, :, :] = 0

    def return_action(self, env, state, exploration = False):
        origin_area = state['last_origin_server']
        destination_server = state['destination_server']
        occupation_origin_area = env.compute_occupation_origin_area(origin_area, state = state)
        occupation_destination_server = np.sum(abs(state['server_'+str(destination_server+1)+'_occupation']))
        # print(self.table[destination_server, occupation_origin_area, origin_area, occupation_destination_server, :])
        if occupation_destination_server < env.servers[destination_server].memory_capacity:
            # print(env._server_occupation[env._last_origin_server], env._last_origin_server)
            # it is sufficient to compute the best action for the component to be updated (?)
            # return np.argmax(critic.table[destination_server, occupation_origin_area, origin_area, occupation_destination_server, :]), None
            # return np.argmax([critic.return_value(env, state, action) for action in range(env.action_space.n)]), None
            return np.argmax(self.table[destination_server, occupation_origin_area, origin_area, occupation_destination_server, :]), None
        else:
            return 0, None

    def parameter_update(self, env, state, action, reward, next_state, n_lagrangian_updates):
        # final return depends on the value of 'last origin server' of next state, as well as the future optimal energy
        # raise ValueError("check learning rule")
        # print(next_state)

        # we update every single component of the table, considering the adequate reward and action space, given the state and next state
        for origin_area in range(env.N_servers):
            for destination_server in range(env.N_servers):
                next_occupation_origin_area = env.compute_occupation_origin_area(origin_area, state = next_state)
                next_occupation_destination_server = np.sum(abs(next_state['server_'+str(destination_server+1)+'_occupation']))

                occupation_origin_area = env.compute_occupation_origin_area(origin_area, state = state)
                occupation_destination_server = np.sum(abs(state['server_'+str(destination_server+1)+'_occupation']))


                # first we compute the immediate reward
                if state['last_origin_server'] == origin_area and state['destination_server'] == destination_server:
                    immediate_reward = reward
                    # the action in the definition of the function is clearly a function just for the component chosen
                    component_action = action

                    # we increase the counter of the visits to the state only for the component actually visited
                    self.state_counts_table[destination_server, occupation_origin_area, origin_area, occupation_destination_server] += 1
                    learning_rate =  self.learning_rate/(( self.state_counts_table[destination_server, occupation_origin_area, origin_area, occupation_destination_server]**.5) * ((n_lagrangian_updates[destination_server]+1)**self.learning_rate_exponent) )

                else:
                    immediate_reward = 0
                    component_action = 0

                    learning_rate = 0

                # then we compute the contribution of the future state
                if next_state['last_origin_server'] == origin_area and next_state['destination_server'] == destination_server:
                    # we must consider the action that maximizes the value function (two possible component actions)
                    final_return = env.discount_factor * np.max(self.table[destination_server, next_occupation_origin_area, origin_area, next_occupation_destination_server, :])
                    # learning_rate =  self.learning_rate/(( self.state_counts_table[destination_server, occupation_origin_area, origin_area, occupation_destination_server]**.5) * ((n_lagrangian_updates[destination_server]+1)**self.learning_rate_exponent) )
                else:
                    # the only possible action in the future is 0
                    final_return = env.discount_factor *  self.table[destination_server, next_occupation_origin_area, origin_area, next_occupation_destination_server, 0]
                    # learning_rate = self.learning_rate/(( self.state_counts_table[destination_server, occupation_origin_area, origin_area, occupation_destination_server]**.5) * ((n_lagrangian_updates[destination_server]+1)**self.learning_rate_exponent) )

                # finally, we update the value corresponding to this state in the table

                self.table[destination_server, occupation_origin_area,  origin_area, occupation_destination_server,  component_action] += learning_rate  * (immediate_reward + final_return - self.table[destination_server, occupation_origin_area,  origin_area, occupation_destination_server, component_action] )
    


class NaiveQLearning(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(NaiveQLearning,  self).__init__()

        multiplier = .5
        self.layer1 = nn.Linear(n_observations, int(n_observations * multiplier))
        self.layer2 = nn.Linear(int(n_observations * multiplier), int(n_observations * multiplier))
        # self.layer3 = nn.Linear(n_observations * 5, n_observations * 5)
        self.layer4 = nn.Linear(int(n_observations * multiplier), 1)

        # what if we want to have a neural network with more than one output? Like one per action?
        self.BATCH_SIZE = 250
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.TAU = 0.005
        self.LR_critic = 1e-2
        self.steps_done = 1
        self.optimizer = optim.AdamW(self.parameters(), lr=self.LR_critic, amsgrad=True)
        # self.double()

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x, inplace=False)
        x = self.layer2(x)
        x = F.relu(x, inplace=False)
        # x= self.layer3(x)
        # x= F.relu(x, inplace=False)
        return self.layer4(x)

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return torch.argmax(self(state)).view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], dtype=torch.long)

    def update(self, env, actor = None,  memory = None, batch = None):
        
        if memory is None or batch is None:
            raise ValueError("Either memory or batch must be provided")
        # transitions = memory.sample(BATCH_SIZE)
        # batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        future_state_batch = torch.cat(batch.next_state)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        # state_action_values = self(state_batch) #.gather(1, action_batch)

        state_values = self(state_batch)
        new_reward_batch = reward_batch.unsqueeze(1)
        # reward_batch = torch.reshape(reward_batch, (state_values.shape))
        expected_state_values = new_reward_batch + env.discount_factor * self(future_state_batch)
        expected_state_values = expected_state_values.to(state_values.dtype)

        loss = F.mse_loss(expected_state_values, state_values)

        # print('loss critic = ', str(loss))
        # would it be equivalent to have loss = nn.MSELoss(advantage, 0) ?

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.parameters(), 100)
        self.optimizer.step()
        