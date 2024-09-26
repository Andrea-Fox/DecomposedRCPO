

# 1) we consider the whole environment to write the transitions
# 2) we save N_servers different policies, one for each server
# 3) at every step, we update only the policy of the server that has received the new information
# 4) we simply do Q-learning update for the server considered

import math
import numpy as np

import torch
import torch.optim as optim
from collections import namedtuple, deque
import random


from _actor import DecomposedwithOccupancy_greedyActor, PolicyNetwork
from _critic import DecomposedWithOccupancyQLearning, NaiveQLearning, DecomposedQLearning_withOptimizedLearning


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'action_log_prob'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory.clear()

def fill_memory(env, actor, batch_size, n_lagrangian_updates):
    memory = ReplayMemory(10000)
    for i_episode in range(batch_size):
        state, _ = env.reset()
        # discounted reward of the whole episode
        discounted_reward = 0
        # state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in range(env.max_length_episode+1):
            exploration_rate = 0
            if np.random.rand() < exploration_rate:
                destination_server = state['destination_server']
                destination_server_occupation = np.sum(state['server_' + str(destination_server+1) +'_occupation'])
                if destination_server_occupation < env.servers[destination_server].memory_capacity:
                    action, action_log_prob = actor.return_action(env,  env.state_to_tensor(state), exploration = True)
                    action.value = np.random.randint(0, 2)
                else:
                    action, action_log_prob = actor.return_action(env,  env.state_to_tensor(state), exploration = True)
                    action.value = 0
            else:
                action, action_log_prob = actor.return_action(env,  env.state_to_tensor(state), exploration = True)
            next_state, reward, _, _ = env.step(action)
    
            # state are always indicated by tensors
            reward = torch.tensor([reward], device=env.device)
    
            # discounted_reward += reward*env.discount_factor**t
            
            # Store the transition in memory
            memory.push(env.state_to_tensor(state), action, env.state_to_tensor(next_state), reward, action_log_prob)
            
            # Move to the next state
            state = next_state

    return memory

def print_result(mean_reward, mean_cost, update_index, method, episodes_between_updates):
    print(f"{method}: {episodes_between_updates * (update_index+1)} episodes ({update_index+1}-th evaluation): mean reward:{mean_reward:.2f}, mean cost:{mean_cost:.2f}")


def agents_update(env, actor, critic, n_lagrangian_updates, total_updates, constrained_learning = True, learning_method = None, memory = None, batch_size = 100):
    # note how within this function we always refer to state. The representation of the state we 
    # actually use will depend on the critic and the actor (and will be handled in the corresponding classes)
    states_visited = []
    # state_counts = np.ones((env.max_memory_capacity+1, ))   
    
    if learning_method == 'DRCPO' or learning_method == 'DRCPO_optimized':
        critic.reset_state_counts()
        
        for _ in range(critic.num_episodes):

            terminal_time = math.inf
            state, _ = env.reset()
            
            done = False
            reward_sum = 0
            for t in range(env.max_length_episode):
                origin_area = state['last_origin_server']
                destination_server = state['destination_server']

                destination_server_occupation = np.sum(state['server_' + str(destination_server+1) +'_occupation'])
                origin_area_occupation = env.compute_occupation_origin_area(origin_area, state=state)
                # areas of interest = 1
                # exploration_rate = .25/(critic.state_counts_table[destination_server, origin_area_occupation, origin_area, destination_server_occupation]**.5)

                exploration_rate = .25/(critic.state_counts_table[destination_server, origin_area_occupation, origin_area, destination_server_occupation]**.5)
                # we consider a certain exploration rate
                if np.random.rand() < exploration_rate and destination_server_occupation < env.servers[destination_server].memory_capacity:
                    action = 1 #random.randint(0, 1)
                elif destination_server_occupation >= env.servers[destination_server].memory_capacity:
                    action = 0
                else:
                    action, _ = critic.return_action(env, state)

                next_state, penalized_reward, _, _ = env.step(action)

                # if action == 1 and env.lagrange_multiplier[destination_server] == 0:
                #     print(penalized_reward)
                
                # actor and critic update            
                critic.parameter_update(env, state, action, penalized_reward, next_state, n_lagrangian_updates)
                
                # print('actor updated')
                state = next_state

    elif learning_method == 'RCPO':
        actor.optimizer.zero_grad()
        critic.optimizer.zero_grad()

        # we reset the learnign rate of the critic and the actor at each step, as the system has changed due to the cnahge in the lagrangian multiplier
        # add as parameter the learning rate decay (?) and the weight decay (?)
        
        critic.LR_critic *= .975 # (0.965**np.max(n_lagrangian_updates))
        # critic.LR_critic *= (0.95**np.max(n_lagrangian_updates))
        actor.LR_actor *= .975 # (0.975**np.max(n_lagrangian_updates))
        # actor.LR_actor *= (0.95**np.max(n_lagrangian_updates))
        critic.optimizer = optim.AdamW(critic.parameters(), lr=critic.LR_critic, amsgrad=True)
        actor.optimizer = optim.AdamW(actor.parameters(), lr=actor.LR_actor, amsgrad=True)

        
        critic.steps_done = 0
        # weight_decay = 0.95
        memory.clear()
        # create a batch of episodes to be used in the learning phase of the critic and the actor
        memory = fill_memory(env, actor, batch_size, memory)
        transitions = memory.sample(memory.__len__())
        batch = Transition(*zip(*transitions))

        # first we update the critic network
        critic.update(env, actor, n_lagrangian_updates, batch)       
        # we then upgrade the actor: we can either assume a greedy policy or a threshold one (with the consequent continuous update)
        actor.update(env, critic, n_lagrangian_updates, batch)

    else:
        raise ValueError("The learning method is not defined")



# Decomposed reward constrained threshold policy optimization
def constrained_policy_learning(env, learning_method, 
    lm_learning_rate = 0.5, lm_learning_rate_exponent = 0.9, initial_value_lm = 0,
    actor_learning_rate = 0.05, actor_learning_rate_exponent = 0.75, actor_initial_policy_multiplier = 0.5, actor_sigmoid_multiplier = 3,
    critic_learning_rate = 0.05, critic_learning_rate_exponent = 0.51, critic_learning_steps = 1, initial_critic = None,
    episodes_between_updates = 100, total_lagrangian_updates = 100, total_updates = 50, 
    only_final_evaluation = False, verbose = True, index = None
    ):
    # this funciton should be able to learn the "optimal" policy for a single server

    assert 1/2 < lm_learning_rate_exponent <= 1, 'Learning rate exponent for the lagrange multiplier is not in the correct range'
    assert 1/2 < critic_learning_rate_exponent <= 1, 'Learning rate exponent for the critic is not in the correct range'

    assert lm_learning_rate_exponent > critic_learning_rate_exponent, 'Learning rate exponent for the lagrange multiplier is not greater than the one for the critic'
    
    # episodes_between_updates substitutes both critic_learning_num_episodes and actor_learning_num_episodes
    discounted_reward_evolution = []
    discounted_cost_evolution = []

    
    env.lagrange_multiplier = initial_value_lm * np.ones((env.N_servers, ))

    lagrangian_gradient_norm = 0 * np.ones((env.N_servers, ))
    stopping_criteria = 0

    n_lagrangian_updates = np.zeros(env.N_servers, dtype = int)
    n_update_steps = 0


    memory = ReplayMemory(10000)

    # definition of the critic and the actor
    if learning_method == 'DRCPO':
        actor = DecomposedwithOccupancy_greedyActor(env)
        critic = DecomposedWithOccupancyQLearning(env, episodes_between_updates, critic_learning_rate, critic_learning_steps, critic_learning_rate_exponent)
    elif learning_method == 'DRCPO_optimized':
        actor = DecomposedwithOccupancy_greedyActor(env)
        if initial_critic is None:
            critic = DecomposedQLearning_withOptimizedLearning(env, episodes_between_updates, critic_learning_rate, critic_learning_steps, critic_learning_rate_exponent)
        else:
            critic = initial_critic
        # critic = DecomposedQLearning_withOptimizedLearning(env, episodes_between_updates, critic_learning_rate, critic_learning_steps, critic_learning_rate_exponent, initial_critic)
    elif learning_method == 'RCPO':
        observation_space_size = env.N_servers * env.N_servers + 2
        action_space_size = 2
        actor = PolicyNetwork(observation_space_size, action_space_size).to(env.device)
        critic = NaiveQLearning(observation_space_size, action_space_size).to(env.device)

    env.actor = actor
    env.critic = critic
    discounted_reward, discounted_cost = env.evaluate_policy(actor, n_episodes=100, verbose = False)


    cost_above_capacity = np.zeros((env.N_servers, ))
    for i in range(env.N_servers):
        cost_above_capacity[i] = (discounted_cost[i] > env.access_capacity[i])

    # print_result(discounted_reward, discounted_cost, -1, method, episodes_between_updates) 
    discounted_reward_evolution.append(discounted_reward)
    discounted_cost_evolution.append(discounted_cost)

    best_discounted_reward = -math.inf
    best_lagrange_multiplier = None

    if verbose:
        print(discounted_reward, discounted_cost, n_lagrangian_updates, n_update_steps, env.lagrange_multiplier)

    if learning_method == 'DRCPO' or learning_method == 'DRCPO_optimized':
        critic.reset_state_counts()
    while n_update_steps < total_updates and np.max(n_lagrangian_updates) <= total_lagrangian_updates:
        # we first update the actor and the critic        
        agents_update(env, actor, critic, n_lagrangian_updates, total_updates, constrained_learning= (np.min(env.access_capacity) != math.inf), learning_method = learning_method, memory = memory, batch_size = episodes_between_updates)
        # we now update the lagrangian multiplier
        # to do so, we follow the the procedure indicated in RCPO paper: first we compute the discounted cost of the current policy
        print('----------------'*3, index)
        if learning_method == 'RCPO':
            discounted_reward, discounted_cost = env.evaluate_policy(actor, n_episodes=100, verbose = verbose) #, n_simulations=n_simulations)
        else:
            discounted_reward, discounted_cost = env.evaluate_policy(critic, n_episodes=100, verbose = False) #, n_simulations=n_simulations)

        discounted_reward_evolution.append(discounted_reward)
        discounted_cost_evolution.append(discounted_cost)

        # then we update the value of the lagrangian using the gradient ascent technique 

        for i in range(env.N_servers):
            lagrangian_gradient_norm[i] = ((discounted_cost[i] - env.access_capacity[i])/env.access_capacity[i]) * lm_learning_rate/((n_lagrangian_updates[i]+1)**lm_learning_rate_exponent)
        
        ##################################################################
        for i in range(env.N_servers):
            lagrangian_gradient_norm[i] = max(-1, min(1, lagrangian_gradient_norm[i]))
            if cost_above_capacity[i] and lagrangian_gradient_norm[i] > 0:
                env.lagrange_multiplier[i] += 2 * lagrangian_gradient_norm[i]
            else:
                env.lagrange_multiplier[i] += lagrangian_gradient_norm[i]

        for i in range(env.N_servers):
            # ISSUE: stops considering adimssion in certain components once the greedy policy becomes always 0
            if discounted_cost[i] == 0: # and env.lagrange_multiplier[i] <= -120:
                env.lagrange_multiplier[i] -= 1
                # critic.reset_state_counts(component = i)
                # we could also reset its components of the value function (in order to recreate the value function from scratch)
                # critic.reset_value_function_server(i)
                print('reset component ', i)
            env.lagrange_multiplier[i] = max(0, env.lagrange_multiplier[i])
        # print(discounted_reward, discounted_cost, env.lagrange_multiplier, lagrangian_gradient_norm, n_lagrangian_updates)

        # actor.print_policy(env)
        # print(critic.table)
        # print(env.lagrange_multiplier)
        # print_result(discounted_reward, discounted_cost, n_update_steps, method, episodes_between_updates)
        n_update_steps += 1

        feasible_solution = True
        # we update the lagrangian stopping criteria only if the the cost has crossed the threshold
        for i in range(env.N_servers):
            if discounted_cost[i] > env.access_capacity[i]:
                feasible_solution = False
            if cost_above_capacity[i] and discounted_cost[i] <= env.access_capacity[i]:
                n_lagrangian_updates[i] += 1
                cost_above_capacity[i] = False
            elif not cost_above_capacity[i] and discounted_cost[i] > env.access_capacity[i]:
                n_lagrangian_updates[i] += 1
                cost_above_capacity[i] = True
            # else:
            #     critic.reset_state_counts(component = i)
        if verbose:
            if index is not None:
                print(discounted_reward, discounted_cost, n_lagrangian_updates, n_update_steps, env.lagrange_multiplier, feasible_solution, index)
            else:
                print(discounted_reward, discounted_cost, n_lagrangian_updates, n_update_steps, env.lagrange_multiplier, feasible_solution)
        print('----------------'*5, index)
        

        # print(round(discounted_reward, 3), round(discounted_cost, 3), round(env.lagrange_multiplier, 3), round(lagrangian_gradient_norm, 3), round(n_lagrangian_updates, 3))

    env.actor = actor
    env.critic = critic

    return discounted_reward_evolution, discounted_cost_evolution, best_discounted_reward, critic, critic, None
