import math
import numpy as np

import torch
import torch.optim as optim

from _admission_actor import GreedyActor, DecomposedwithOccupancy_greedyActor
from _admission_critic import DecomposedQLearning, DecomposedWithOccupancyQLearning


def print_result(mean_reward, mean_cost, update_index, method, episodes_between_updates):
    print(f"{method}: {episodes_between_updates * (update_index+1)} episodes ({update_index+1}-th evaluation): mean reward:{mean_reward:.2f}, mean cost:{mean_cost:.2f}")


def online_learning(env, actor, critic, n_lagrangian_updates, total_updates, constrained_learning = True):
    # note how within this function we always refer to state. The representation of the state we 
    # actually use will depend on the critic and the actor (and will be handled in the corresponding classes)
    batch_size = critic.num_episodes * env.max_length_episode
    states_visited = []
    state_counts = np.ones((env.memory_capacity+1, ))

    if constrained_learning:
        critic.reset_state_counts()
    exploration_rate = .05/(n_lagrangian_updates+1)
    for _ in range(critic.num_episodes):

        terminal_time = math.inf
        state = env.reset(give_last_origin_server=True)
        
        done = False
        reward_sum = 0
        for t in range(env.max_length_episode):
            # we consider a certain exploration rate
            if np.random.rand() < exploration_rate and np.sum(state['server_occupation']) < env.memory_capacity:
                action = np.random.randint(0, env.action_space.n)
            else:
                action, _ = actor.return_action(env, state)
            next_state, reward, _, _ = env.step(action)
            
            critic.update_state_counts(state)

            # actor and critic update            
            critic.online_learning(env, state, action, reward, next_state, n_lagrangian_updates)
            actor.online_learning(env, state, action, reward, next_state, n_lagrangian_updates, critic)

            state = next_state



# Decomposed reward constrained threshold policy optimization
def constrained_policy_learning(env, method = 'DRCPO', 
    lm_learning_rate = 0.15, lm_learning_rate_exponent = 0.95, initial_value_lm = 0,
    actor_learning_rate = 0.05, actor_learning_rate_exponent = 0.75, actor_initial_policy_multiplier = 0.5, actor_sigmoid_multiplier = 3,
    critic_learning_rate = 0.05, critic_learning_rate_exponent = 0.51, critic_learning_steps = 1, 
    episodes_between_updates = 25, total_lagrangian_updates = 10, total_updates = 50, 
    only_final_evaluation = False
    ):
    # this funciton should be able to learn the "optimal" policy for a single server
    cost_above_capacity = None
    if method == 'RCPO' or method == 'DRCTPO':
        incremental_updates_actor = True
    elif method == 'DRCPO':
        incremental_updates_actor = False
    else:
        incremental_updates_actor = False

    assert 1/2 < lm_learning_rate_exponent <= 1, 'Learning rate exponent for the lagrange multiplier is not in the correct range'
    if incremental_updates_actor:
        assert 1/2 < actor_learning_rate_exponent <= 1, 'Learning rate exponent for the actor is not in the correct range'
    assert 1/2 < critic_learning_rate_exponent <= 1, 'Learning rate exponent for the critic is not in the correct range'

    if incremental_updates_actor:
        assert lm_learning_rate_exponent > actor_learning_rate_exponent, 'Learning rate exponent for the lagrange multiplier is not greater than the one for the actor'
        assert actor_learning_rate_exponent > critic_learning_rate_exponent, 'Learning rate exponent for the actor is not greater than the one for the critic'
    else:
        assert lm_learning_rate_exponent > critic_learning_rate_exponent, 'Learning rate exponent for the lagrange multiplier is not greater than the one for the critic'
    
    # episodes_between_updates substitutes both critic_learning_num_episodes and actor_learning_num_episodes
    discounted_reward_evolution = []
    discounted_cost_evolution = []
    
    env.lagrange_multiplier = initial_value_lm

    lagrangian_gradient_norm = math.inf
    stopping_criteria = 0

    n_lagrangian_updates = 0
    n_update_steps = 0
    batch_learning = False

    # definition of the critic and the actor
    if method == 'OccupancyBased-QLearning':
        learning = 'online'
        actor = GreedyActor(env, occupancy_based = True)
        critic = OccupancyBased_OptimizedQLearning(env, episodes_between_updates, critic_learning_rate, critic_learning_steps, critic_learning_rate_exponent)
    elif method == 'OccupancyBased-OptimizedQLearning':
        learning = 'online'
        actor = GreedyActor(env, occupancy_based=True)
        critic = OccupancyBased_OptimizedQLearning(env, episodes_between_updates, critic_learning_rate, critic_learning_steps, critic_learning_rate_exponent)
    elif method == 'OccupancyBased-RCTPO':
        learning = 'online'
        actor = OccupancyBased_StochasticThresholdPolicyActor(env, actor_learning_rate, actor_learning_rate_exponent, episodes_between_updates, actor_initial_policy_multiplier, actor_sigmoid_multiplier)
        critic = OccupancyBased_OptimizedQLearning(env, episodes_between_updates, critic_learning_rate, critic_learning_steps, critic_learning_rate_exponent)
    elif method == 'DecomposedWithOccupancyQLearning':
        learning = 'online'
        actor = DecomposedwithOccupancy_greedyActor(env)
        critic = DecomposedWithOccupancyQLearning(env, episodes_between_updates, critic_learning_rate, critic_learning_steps, critic_learning_rate_exponent)
    elif method == 'DecomposedWithOccupancy-RTCPO':
        learning = 'online'
        actor = DecomposedWithOccupancy_StochasticThresholdPolicyActor(env, actor_learning_rate, actor_learning_rate_exponent, episodes_between_updates, actor_initial_policy_multiplier, actor_sigmoid_multiplier)
        critic = DecomposedWithOccupancyQLearning(env, episodes_between_updates, critic_learning_rate, critic_learning_steps, critic_learning_rate_exponent)
    elif method == 'RCPO':
        print('RCPO: Reward Constrained Policy Optimization')
        # if available, use the GPU
        env.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        learning = 'batch'
        critic = NaiveQLearning(env.origin_servers + 1, env.action_space.n).to(env.device)
        actor = PolicyNetwork(env.origin_servers + 1, env.action_space.n).to(env.device)
    elif method == 'DRCTPO':
        # print('DRCTPO: Decomposed Reward Constrained Threshold Policy Optimization')
        learning = 'online'
        critic = DecomposedQLearning(env, episodes_between_updates, critic_learning_rate, critic_learning_steps, critic_learning_rate_exponent)
        actor = StochasticThresholdPolicyActor(env, actor_learning_rate, actor_learning_rate_exponent, episodes_between_updates, actor_initial_policy_multiplier, actor_sigmoid_multiplier)
    elif method == 'DRCPO':
        # print('DRCPO: Decomposed Reward Constrained Policy Optimization')
        learning = 'online'
        critic = DecomposedQLearning(env, episodes_between_updates, critic_learning_rate, critic_learning_steps, critic_learning_rate_exponent)
        actor = GreedyActor(env)
    elif method == 'PPO-lagrangian':
        learning = 'batch'
        model = PPO('MultiInputPolicy', env, verbose=1)
    else:
        raise ValueError('Method not recognized')

    
    # first we evaluate the random policy: for PPO-lagrangain the method is slightly different
    try:
        critic.reset_state_counts()
    except:
        pass
    
    # if not only_final_evaluation:

    if only_final_evaluation:
        n_simulations = 50
    else:
        n_simulations = 100
    discounted_reward, discounted_cost = env.evaluate_policy_single_server(actor, n_simulations=n_simulations)
    cost_above_capacity = (discounted_cost > env.access_capacity)

    # print_result(discounted_reward, discounted_cost, -1, method, episodes_between_updates) 
    discounted_reward_evolution.append(discounted_reward)
    discounted_cost_evolution.append(discounted_cost)

    best_discounted_reward = -math.inf
    best_lagrange_multiplier = None

    while n_update_steps < total_updates and n_lagrangian_updates <= total_lagrangian_updates:
        # it could be a good idea to deifne a function which creates a bath of episodes and then use it for the critic and the actor update
        # this could be useful especially in the approximate case 
        if learning == 'batch':
            pass
        elif learning == 'online':
            # if batch is None, we use the online learning approach (we do that when decomposing the problem)
            online_learning(env, actor, critic, n_lagrangian_updates, total_updates, constrained_learning= (env.access_capacity != math.inf))


        # we now update the lagrangian multiplier
        # to do so, we follow the the procedure indicated in RCPO paper: first we compute the discounted cost of the current policy
        # print(actor)
        discounted_reward, discounted_cost = env.evaluate_policy_single_server(actor, n_simulations=n_simulations)
        discounted_reward_evolution.append(discounted_reward)
        discounted_cost_evolution.append(discounted_cost)

        # then we update the value of the lagrangian using the gradient ascent technique 
        lagrangian_gradient_norm = (discounted_cost - env.access_capacity) * lm_learning_rate/((n_lagrangian_updates+1)**lm_learning_rate_exponent)
        lagrangian_gradient_norm = min(1, lagrangian_gradient_norm)
        env.lagrange_multiplier += lagrangian_gradient_norm
        env.lagrange_multiplier = max(0, env.lagrange_multiplier)
        # print(discounted_reward, discounted_cost, env.lagrange_multiplier, lagrangian_gradient_norm, n_lagrangian_updates)

        # actor.print_policy(env)
        # print(critic.table)
        # print(env.lagrange_multiplier)
        # print_result(discounted_reward, discounted_cost, n_update_steps, method, episodes_between_updates)
        n_update_steps += 1

        # we update the lagrangian stopping criteria only if the the cost has crossed the threshold
        if cost_above_capacity and discounted_cost <= env.access_capacity:
            n_lagrangian_updates += 1
            cost_above_capacity = False
        elif not cost_above_capacity and discounted_cost > env.access_capacity:
            n_lagrangian_updates += 1
            cost_above_capacity = True

        # print(round(discounted_reward, 3), round(discounted_cost, 3), round(env.lagrange_multiplier, 3), round(lagrangian_gradient_norm, 3), round(n_lagrangian_updates, 3))

    env.actor = actor
    env.critic = critic

    return discounted_reward_evolution, discounted_cost_evolution, best_discounted_reward, actor, critic, best_lagrange_multiplier