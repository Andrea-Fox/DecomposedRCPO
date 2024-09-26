import itertools
import numpy as np
import math
import pandas as pd
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class Actor():

    def __init__(self):
        self.policy = None
        self.occupancy_based = False
        self.exploration_rate = 0.25

    def return_action(self, env, state):
        return None

    def print_policy(self, env = None):
        print(self.policy)

    def define_policy_from_input(self, file):
        # this function receives as an input a file, containing a matlab matrix saved as a csv file
        # in this function the input matrix is tranformed into a suitable actor policy, that later will be used 
        # for computing the values for the disocunted cost/reward for the optimal policy
        df = pd.read_csv(file, header=None)
        self.policy = df.values
        self.print_policy()

class DecomposedwithOccupancy_greedyActor(Actor):

    def __init__(self, env, occupancy_based = False):
        super().__init__()
        self.occupancy_based = occupancy_based
        # we multiply the occupation by 5, as it corresponds to the fact that the occupation involves also the other servers
        # The factor 5 is chosen arbitrariliy and we artificially set that if the occupation is above 5 * env.memory_capacity
        # then we consider the maximum value (it is already quite full, too much to have an actual impact on the reward)
         
        self.policy = np.zeros((env.N_servers, env.N_servers * env.max_memory_capacity+1, env.N_servers, env.max_memory_capacity+1), dtype = int)

    def return_action(self, env, state = None, exploration = True, critic = None):
        if state == None:
            raise ValueError('The state must be specified')
            if np.sum(env._server_occupation) < env.memory_capacity:
                # print(env._server_occupation[env._last_origin_server], env._last_origin_server)
                occupation_origin_area = 0
                for component in range(env.N_servers):
                    if env.servers[component].area_of_interest[env._last_origin_server] == 1:
                        occupation_origin_area += env.servers[component]._server_occupation[component]

                occupation_destination_server = np.sum(abs(env.servers[env._destination_server]._server_occupation))
                return self.policy[env._destination_server, occupation_origin_area, env._last_origin_server, occupation_destination_server], None
            else:
                return 0, None
        else:
            origin_area = state['last_origin_server']
            destination_server = state['destination_server'] 
            occupation_origin_area = env.compute_occupation_origin_area(origin_area, state = state)
            occupation_destination_server = np.sum(abs(state['server_'+str(destination_server+1) +'_occupation']))
            
            if occupation_destination_server < env.servers[destination_server].memory_capacity:
                # print(env._server_occupation[env._last_origin_server], env._last_origin_server)
                # it is sufficient to compute the best action for the component to be updated (?)
                # return np.argmax(critic.table[destination_server, occupation_origin_area, origin_area, occupation_destination_server, :]), None
                # return np.argmax([critic.return_value(env, state, action) for action in range(env.action_space.n)]), None
                return self.policy[destination_server, occupation_origin_area, origin_area, occupation_destination_server], None
            else:
                return 0, None
    

    def parameter_update(self, env, critic):
        for destination_server in range(env.N_servers):
            for origin_area in range(env.N_servers):
                for occupation_origin_area in range(env.N_servers * env.max_memory_capacity):
                    for occupation_destination_server in range(env.max_memory_capacity):
                        self.policy[destination_server, occupation_origin_area, origin_area, occupation_destination_server] = np.argmax(critic.table[destination_server, occupation_origin_area, origin_area, occupation_destination_server, :])






class PolicyNetwork(nn.Module):

    def __init__(self, observation_space, action_space):
        super(PolicyNetwork, self).__init__()
        # input: state, output: probability distribution over actions
        self.input_layer = nn.Linear(observation_space, observation_space )
        self.hidden_layer = nn.Linear(observation_space, observation_space)
        self.output_layer = nn.Linear(observation_space, action_space)
        self.LR_actor = 1e-3
        self.optimizer = optim.AdamW(self.parameters(), lr=self.LR_actor, amsgrad=True)

    def forward(self, x):
        #input states

        x = F.relu(self.input_layer(x), inplace=False)
        #hidden layer
        # x = F.relu(self.hidden_layer(x), inplace=False)
        # x = F.relu(self.hidden_layer(x), inplace=False)
        #actions with softmax for a probability distribution
        action_values = self.output_layer(x)
        action_probs = F.softmax(action_values, dim=1)
        torch.autograd.set_detect_anomaly(True)
        return action_probs

    def return_action(self, env, state, exploration=True):
        # print(state)
        if not type(state) == torch.Tensor:
            state = env.state_to_tensor(state)
        distr = Categorical(self(state))
        vector = state.cpu().detach().numpy()[0]
        destination_server = int(vector[-1])
        interval_min = 10 * (destination_server)
        interval_max = 10 * (destination_server +1) 
        # print(destination_server, interval_min, interval_max)
        occupation_origin_area = np.sum(vector[interval_min:interval_max])
        if exploration:
            action = distr.sample()
            if occupation_origin_area == 0:
                action.value = 1
            return action, distr.log_prob(action)
        else:
            action = torch.argmax(distr.probs).item()
            if occupation_origin_area == 0:
                action = 1
            return action, None
        # the probability distribution will be useful for the update step
        


    def update(self, env, critic, memory, batch):
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        action_log_prob_batch = torch.cat(batch.action_log_prob)

        # compute the advantage
        # print(critic(next_state_batch))
        # print(critic(next_state_batch).clone().detach())
        
        advantage = reward_batch + env.discount_factor * critic(next_state_batch) - critic(state_batch)
        entropy = action_log_prob_batch * advantage
        # the loss is given by the policy gradient theorem
        loss = - torch.mean(entropy )
        
        # print('loss actor = ' + str(loss))
        # Optimize the model
        
        loss.backward(retain_graph=True)
        # In-place gradient clipping (what is the meaning of this?)
        torch.nn.utils.clip_grad_value_(self.parameters(), 100)

        self.optimizer.step()