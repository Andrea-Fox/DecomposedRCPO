# in experiment 2, the goal is to show, that choosing a routing policy which only sends videos to servers which 
# are interested is a good idea. This implies that it is optimal when the single servers take into account the
# problem of admission, while leaving to the routing device only the choice of routing.
# We show the evolution of the total reward when considering an increasing ratio between the server capacity and 
# the constraint value. 


# to show this result, we consider a system with 3 servers and 3 areas of interest
# we consider randomly selected parameters, the following sets:
# C = random value between 8 and 15
# phi = f(C)
# mu = (0, 1]
# arrival rates = (0, 1.5]
# areas of interest: for each server, probability of accepting one area is 0.66. Each area is accepted by at least one server
# we consider both the case with equal reward fucntions (all three of them) and with different reward functions
# discount factor = (0.9, 1)

# the ratio goes from 0.1 to 1.5

# for each value of the ratio, we consider 10 different systems and we compute the average for each method

# in this experiment we only consider fixed policies. Note how, when considering the policy that takes an action based on the 
# occupation level of each server, in order to compute the arrival rates for each server we would require the stationary probability 
# and hence we compute an approximate value. We note how, routing only to the areas where the server has interest is a good idea and 
# improves the performances (i.e. discarding videos at routing level seems to be a bad idea). Furthermore, it is likely that the 
# difference in performances increases even more when adding an higher number of servers 

# we also show how employing Q-learning to learn the optimal routing policy is not a good idea. This happens as the change in the admission
# policy is not taken into account when computing the Q-values (the immediate reward changes). This is a problem in the field of RL for 
# non-stationary environments, and it is not trivial to solve

# why does this not happen for the other policies? the do not depend on the admission policy
# in a similar way, if we consider the routing policy which depends on the occupation level of the server, we face the same issue

# On the other hand, the other policies do not depend on the admission policy, and hence they are not affected by this problem and give 
# good results. Clearly, depending on the setting of the problem the optimal routing policy can change



from _mdp_routing import *
from _routing_q_learning import *
from _admission_policy_learning import *
from _routing_stateless_policy import *
from _routing_policy import *
from _fixed_routing_policies import *

from copy import deepcopy
import numpy as np
from multiprocessing import Pool
import pickle as pk


def computation_total_discounted_reward(ratio, areas_of_interest, servers_parameters, arrival_rates, discount_factor, index):
        # print('Reward functions: ' + str(reward_function_types.index(set_reward_functions)+1) + '/' + str(len(reward_function_types)))
    
    # we could make it depend on the index
    N_servers = len(servers_parameters)

    np.random.seed(2 * index) #  + int(10 * ratio))

    total_rewards_uniform_with_area_of_interest = []
    total_rewards_uniform_without_area_of_interest = []
    total_rewards_constraints_with_area_of_interest = []
    total_rewards_constraints_without_area_of_interest = []
    total_rewards_occupation = []
    total_rewards_Qlearning = []
    total_rewards_alternate_learning = []



    # we must sample the parameters for each device
    print('C/phi: ' + str(ratio) +' Experiment: ' +str(index)) 
    for j in range(N_servers):
        servers_parameters[j]['phi'] = ratio * servers_parameters[j]['C']
    
    # print(servers_parameters)
    # for each environment, we evaluate different types of fixed routing policies   
    uniform_and_area_of_interest =  True
    uniform_without_area_of_interest = True
    constraints_and_area_of_interest = True
    constraints_without_area_of_interest = False
    occupation = True
    Qlearning = False
    alternate_learning = True

    # we start from the uniform one without areas of interest
    if uniform_without_area_of_interest:
         # we create the environment
        u_env = RoutingServer(N_servers, servers_parameters, arrival_rates, discount_factor)
        routing_policy = StatelessRouting(u_env, fixed_routing_uniform_load_balancing(u_env, consider_areas_of_interest = False))
        for j in range(u_env.N_servers):
            for i in range(u_env.N_servers):
                u_env.servers[j].arrival_rates_server[i] = arrival_rates[i] * routing_policy.routing_policy[i, j]
        for i in range(u_env.N_servers):
            constrained_policy_learning(u_env.servers[i], method = 'DecomposedWithOccupancyQLearning')
        total_return_no_aoi, _ = u_env.evaluate_routing_policy(routing_policy)
        total_rewards_uniform_without_area_of_interest.append(total_return_no_aoi)  
    
    if uniform_and_area_of_interest:
        # we repeat the same operations for the case with areas of interest
        uAoi_env = RoutingServer(N_servers, servers_parameters, arrival_rates, discount_factor)
        routing_policy_uniform_aoi = StatelessRouting(uAoi_env, fixed_routing_uniform_load_balancing(uAoi_env, consider_areas_of_interest = True))
        for j in range(N_servers):
            for i in range(N_servers):
                uAoi_env.servers[j].arrival_rates_server[i] = arrival_rates[i] * routing_policy_uniform_aoi.routing_policy[i, j]
        for i in range(N_servers):
            constrained_policy_learning(uAoi_env.servers[i], method = 'DecomposedWithOccupancyQLearning')
        total_return_aoi, _ = uAoi_env.evaluate_routing_policy(routing_policy_uniform_aoi)
        total_rewards_uniform_with_area_of_interest.append(total_return_aoi)    
    
    if constraints_without_area_of_interest:
        # we now consider the fixed routing based on the constraints
        c_env = RoutingServer(N_servers, servers_parameters, arrival_rates, discount_factor)
        routing_policy = StatelessRouting(c_env, fixed_routing_constraints(c_env, consider_areas_of_interest = False))
        for j in range(c_env.N_servers):
            for i in range(c_env.N_servers):
                c_env.servers[j].arrival_rates_server[i] = arrival_rates[i] * routing_policy.routing_policy[i, j]
        for i in range(c_env.N_servers):
            constrained_policy_learning(c_env.servers[i], method = 'DecomposedWithOccupancyQLearning')
        total_return_no_aoi, _ = c_env.evaluate_routing_policy(routing_policy)
        total_rewards_constraints_without_area_of_interest.append(total_return_no_aoi)  
    
    if constraints_and_area_of_interest:
        # we repeat the same operations for the case with areas of interest
        cAoi_env = RoutingServer(N_servers, servers_parameters, arrival_rates, discount_factor)
        routing_policy = StatelessRouting(cAoi_env, fixed_routing_constraints(cAoi_env, consider_areas_of_interest = True))
        for j in range(cAoi_env.N_servers):
            for i in range(cAoi_env.N_servers):
                cAoi_env.servers[j].arrival_rates_server[i] = arrival_rates[i] * routing_policy.routing_policy[i, j]
        for i in range(N_servers):
            constrained_policy_learning(cAoi_env.servers[i], method = 'DecomposedWithOccupancyQLearning')
        total_return_aoi, _ = cAoi_env.evaluate_routing_policy(routing_policy)
        total_rewards_constraints_with_area_of_interest.append(total_return_aoi)    
    # we now consider the routing based on the occupation of the servers
    # Note how in this case, at first we have to compute the admission policies for each server as they are required 
    # to approximate the stationary probability of the system and therefore compute the arrival rates for each server

    if occupation:
        # for this algorithm, we directly consider the case where we only consider the areas of interest
        print('begin occupation')
        o_env = RoutingServer(N_servers, servers_parameters, arrival_rates, discount_factor)
        for i in range(o_env.N_servers):
            constrained_policy_learning(o_env.servers[i], method = 'DecomposedWithOccupancyQLearning')
        routing_policy, approximate_routing_policy = fixed_routing_occupation(o_env, consider_areas_of_interest = True)
        for j in range(o_env.N_servers):
            for i in range(o_env.N_servers):
                o_env.servers[j].arrival_rates_server[i] = arrival_rates[i] * approximate_routing_policy.routing_policy[i, j]
        for i in range(o_env.N_servers):
            constrained_policy_learning(o_env.servers[i], method = 'DecomposedWithOccupancyQLearning')
        total_return_occupation, _ = o_env.evaluate_routing_policy(routing_policy)
        total_rewards_occupation.append(total_return_occupation)    
    
    if Qlearning:
        # we conclude with the case where we consider the routing policy given by Q-learning applied to the routing MDP
        ql_env = RoutingServer(N_servers, servers_parameters, arrival_rates, discount_factor)
        # at first we need to compute the admission policies, assuming a uniform routing policy
        for i in range(ql_env.N_servers):
            constrained_policy_learning(ql_env.servers[i], method = 'DecomposedWithOccupancyQLearning')
        q_values, _ = q_learning_routing(ql_env, ordered_q_learning= True, consider_areas_of_interest = True, num_episodes = 1000)
        routing_probability = compute_routing_probability_from_q_values(ql_env, q_values, consider_areas_of_interest = True)
        routing_policy, approximate_routing_policy = ql_env.compute_approximated_arrival_rates(ComponentOccupationRouting(ql_env, routing_probability), n_episodes = 100) 
        # print('iteration '+str(iteration_counter) +': new routing policy computed')
        # approximate_routing_policy is used to chenage the arrival rates of the servers
        for j in range(ql_env.N_servers):
            for i in range(ql_env.N_servers):
                ql_env.servers[j].arrival_rates_server[i] = ql_env.arrival_rates[i] * approximate_routing_policy.routing_policy[i, j]
        for i in range(N_servers):
            constrained_policy_learning(ql_env.servers[i], method = 'DecomposedWithOccupancyQLearning')
        # evaluation of the environment with the fixed routing policy and the corresponding admission policies
        total_return_Qlearning, _ = ql_env.evaluate_routing_policy(routing_policy)
        total_rewards_Qlearning.append(total_return_Qlearning)

    if alternate_learning:
        # alternate learning
        a_env = deepcopy(uAoi_env)
        total_rewards_alternate_learning.append(total_rewards_uniform_with_area_of_interest[-1])
        iteration_counter = 0
        ending_condition_satisfied = False
        routing_policy = routing_policy_uniform_aoi
        while not ending_condition_satisfied:
            temporary_a_env = deepcopy(a_env)
            routing_policy, list_routing_policies, list_total_rewards, no_improvement = optimize_stateless_routing_policy(temporary_a_env , routing_policy= routing_policy, consider_areas_of_interest = True, verbose=False)

            print('new routing policy computed. Total reward = ', list_total_rewards[-1])
            # given the new routing policy, we must update the arrival rates of each server and therefore the optimal admission policies for each server
            for j in range(temporary_a_env.N_servers):
                for i in range(temporary_a_env.N_servers):
                    temporary_a_env.servers[j].arrival_rates_server[i] = temporary_a_env.arrival_rates[i] * routing_policy.routing_policy[i, j]

            for i in range(temporary_a_env.N_servers):
                constrained_policy_learning(temporary_a_env.servers[i], method = 'DecomposedWithOccupancyQLearning', only_final_evaluation = True)

            print('new admission policies computed')
            discounted_reward, _ = temporary_a_env.evaluate_routing_policy(routing_policy)
            total_rewards_alternate_learning.append(discounted_reward)
            print(discounted_reward)
            if discounted_reward > total_rewards_alternate_learning[-2]:
                print('iteration '+str(iteration_counter) +': improvement found. Improved total reward = ' + str(discounted_reward) )
                a_env = temporary_a_env
            else:
                # maybe add a mechanism that retrieves the best routing policy found so far (even at the cost of remaining with the old admission policy)
                print('iteration '+str(iteration_counter) +': no improvement found' )

            print('-------------------------------------')
            iteration_counter += 1
            if iteration_counter > 0 or no_improvement:
                # find some better ending condition (such as the number of consecutive non-improvements)
                ending_condition_satisfied = True


    return np.mean(total_rewards_uniform_without_area_of_interest), np.mean(total_rewards_uniform_with_area_of_interest), np.mean(total_rewards_constraints_without_area_of_interest), np.mean(total_rewards_constraints_with_area_of_interest), np.mean(total_rewards_occupation), total_rewards_alternate_learning

reward_function_types = [(i, j, k) for i in range(1, 3) for j in range(3) for k in range(3)]
print(len(reward_function_types)) 
N_servers = 7

list_total_rewards_uniform_with_area_of_interest = []
list_total_rewards_uniform_without_area_of_interest = []
list_total_rewards_uniform_improvement = []

list_total_rewards_constraints_with_area_of_interest = []
list_total_rewards_constraints_without_area_of_interest = []
list_total_rewards_constraints_improvement = []

list_total_rewards_occupation = []
list_total_rewards_Qlearning = []
list_total_rewards_alternate_learning = []
list_total_rewards_alternate_learning_best_result = []

pool = Pool(processes=mp.cpu_count())

n_experiment = 4

list_parameters = []
list_area_of_interest = []
list_arrival_rates = []
list_discount_factors = []

for j in range(n_experiment):
    np.random.seed(j)
    servers_parameters = [] 
    # at first we must define the areas of interest so that each are of interest is accepted by at least one server
    areas_of_interest = np.zeros((N_servers, N_servers))
    while min(np.sum(areas_of_interest, axis = 0)) == 0 or min(np.sum(areas_of_interest, axis = 1)) == 0:
        areas_of_interest = np.zeros((N_servers, N_servers))   
        for i in range(N_servers):
            for j in range(N_servers):
                areas_of_interest[j, i] = (np.random.rand() < 0.5)
    list_area_of_interest.append(areas_of_interest)

    for j in range(N_servers):
        C = np.random.randint(8, 16)
        mu = np.random.rand()
        reward_function_type = np.random.randint(3)
        servers_parameters.append({'C': C, 'areas_of_interest': areas_of_interest[j, :], 'mu': mu, 'server_id': j, 'reward_function_type': reward_function_type})
    list_parameters.append(servers_parameters)

    arrival_rates = 1.5 * np.random.rand(N_servers) + 0.01
    list_arrival_rates.append(arrival_rates)

    discount_factor = 0.9 + 0.1 * np.random.rand() 
    list_discount_factors.append(discount_factor)

    print(servers_parameters)

results = pool.starmap(computation_total_discounted_reward, [(i/10, list_area_of_interest[j], list_parameters[j], list_arrival_rates[j], list_discount_factors[j], j) for i in range(1, 16, 3) for j in range(n_experiment)])

print(results)
for i in range(len(results)):
    list_total_rewards_uniform_without_area_of_interest.append(results[i][0])
    list_total_rewards_uniform_with_area_of_interest.append(results[i][1])
    list_total_rewards_uniform_improvement.append(results[i][0]/results[i][1])

    list_total_rewards_constraints_without_area_of_interest.append(results[i][2])
    list_total_rewards_constraints_with_area_of_interest.append(results[i][3])
    list_total_rewards_constraints_improvement.append(results[i][2]/results[i][3])

    list_total_rewards_occupation.append(results[i][4])
    list_total_rewards_alternate_learning.append(results[i][5])
    list_total_rewards_alternate_learning_best_result.append(max(results[i][5]))



        # we save the results
        # list_total_rewards_with_area_of_interest.append(total_return_aoi)
        # list_total_rewards_without_area_of_interest.append(total_return_no_aoi)
        # list_total_rewards_improvement.append(total_return_no_aoi/total_return_aoi)

folder = '' 
with open(folder + 'results.pk', 'wb') as f:
    pk.dump(results, f)

print('Uniform routing policy: ')
print('Without areas of interest: ', np.mean(list_total_rewards_uniform_without_area_of_interest))
print('With areas of interest: ', np.mean(list_total_rewards_uniform_with_area_of_interest))
print('Improvement: ', np.mean(list_total_rewards_uniform_improvement))

print('Constraints routing policy: ')
print('Without areas of interest: ', np.mean(list_total_rewards_constraints_without_area_of_interest))
print('With areas of interest: ', np.mean(list_total_rewards_constraints_with_area_of_interest))
print('Improvement: ', np.mean(list_total_rewards_constraints_improvement))

print('Occupation level:')
print('With areas of interest: ', np.mean(list_total_rewards_occupation))

print('Q-learning:')
print('With areas of interest: ', np.mean(list_total_rewards_Qlearning))

print('Alternate learning:')
print('With areas of interest: ', np.mean(list_total_rewards_alternate_learning_best_result))


