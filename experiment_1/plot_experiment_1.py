
# attempts of plotting the results_DRCPO of experiment 2
import pickle as pk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import math
import copy


def feasible_policy(costs, parameters):
    # we look at the constraints of the system and check if the policy satisfies them
    N_servers = len(costs[0])
    constraint = np.zeros((N_servers, ))
    index_of_feasible_policy = math.inf
    for server_index in range(N_servers):
        constraint[server_index] = parameters[server_index]['phi']

    constraint_satisfied = True
    episode_index = len(costs) -1
    while constraint_satisfied and episode_index > 0:
        local_costs = costs[episode_index] 
        constraints_broken = 0
        for server_index in range(N_servers):
            if costs[episode_index][server_index] > constraint[server_index] * 2000000:
                constraints_broken += 1
        if constraints_broken > 1 :
            constraint_satisfied = False
            index_of_feasible_policy = episode_index + 1
        episode_index -= 1
    if index_of_feasible_policy == math.inf:
        return 1
    else:
        return int(index_of_feasible_policy)



def compute_optimal_result(reward, costs, parameters, constraint_surplus = 0):
    # first we understand which subset to consider (i.e. all the future policies are feasible)
    index_of_feasible_policy = feasible_policy(costs, parameters)
    # we consider the maximum value of reward obtained by the system during the learning process, among the considered policies     
    max_feasible_reward = 0
    best_policy = None
    for episode_index in range(index_of_feasible_policy, len(reward)):
        # first we consider if the policy is feasible (really feasible, without tricks...)
        constraint_satisfied = 0
        for server_index in range(N_servers):
            if costs[episode_index][server_index] < parameters[server_index]['phi'] * 1:
                constraint_satisfied += 1
            elif costs[episode_index][server_index] > parameters[server_index]['phi'] * (1 + .1):
                constraint_satisfied = -math.inf
        if reward[episode_index] > max_feasible_reward and constraint_satisfied >= N_servers * .5:
            max_feasible_reward = reward[episode_index]
            best_policy = episode_index
    return max_feasible_reward



################################################
new_parameters = True
plot_RCPO = True
naive_policy = True
plot_discounted_reward = True
plot_discounted_cost = True
single_cost_plot = True
single_plot = True
savefig = True
data_from_exp_2 = True
N_servers = 10
################################################

experiments_to_consider = []
folder = ''

N_areas_of_interest = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
N_areas_of_interest = [10]#, 4, 7, 10]  
results_DRCPO = []
results_RCPO = []
list_parameters = []
for area_of_interest in N_areas_of_interest:
    file_DRCPO = folder + '{}_servers_{}_applications_DRCPO.pk'.format(N_servers, area_of_interest)    
    with open(file_DRCPO, 'rb') as f:
        data = pk.load(f)
    list_parameters += data[0]
    list_area_of_interest = data[1]
    list_arrival_rates = data[2]
    list_discount_factors = data[3]
    if area_of_interest == 1:
        for index in range(len(list_discount_factors)):
            if list_discount_factors[index] < 1:
                for n_applications in N_areas_of_interest:
                    experiments_to_consider.append(index + 20 * (n_applications - 1))
                print('Discount factor =', list_discount_factors[index])
    # print(list_parameters[0][0]['phi'])
    # print(list_discount_factors)
    results_DRCPO += data[4]
    # print(len(results_DRCPO))arXiv.org
    # print(results_DRCPO[0])
    if plot_RCPO:
        file_RCPO = folder + '{}_servers_{}_applications_RCPO.pk'.format(N_servers, area_of_interest)
        with open(file_RCPO, 'rb') as f:
            data_RCPO = pk.load(f)
        RCPO_multiplier = 2
        results_RCPO += data_RCPO[5]
    else:
        results_RCPO += copy.deepcopy(data[4])

    if naive_policy:
        parameter_threshold = []
        discount_factor_naive = []
        rewards_naive = []
        cost_naive = []
        for n_applications in N_areas_of_interest:
            file_threshold = folder + '{}_server_{}_applications_threshold_policy.pk'.format(N_servers, n_applications)
            with open(file_threshold, 'rb') as f:
                data_threshold = pk.load(f)
            parameter_threshold += data_threshold[0]

            discount_factor_naive += data_threshold[3]
            print('----------')
            print(discount_factor_naive)
            # rewards_naive = data_threshold[4]
            # cost_naive = data_threshold[5]
            rewards_naive += data_threshold[4]
            cost_naive += data_threshold[5]
        print(len(rewards_naive))
        print(len(cost_naive))





print(experiments_to_consider)
if naive_policy:
    app_per_server = 7
    experiment_index = 3

    index = 20 * (app_per_server - 1) + experiment_index
    print(index)

    n_experiments_per_group = 20
    new_app_per_server =  index // n_experiments_per_group + 1
    new_exp_index = index - n_experiments_per_group * (new_app_per_server - 1)

    print(new_app_per_server, new_exp_index)
    new_index = N_servers * new_exp_index + new_app_per_server-1
    print(new_index)

    # print(list_parameters[index])
    # print(parameter_threshold[new_index])


# raise Exception('Stop here')
n_evaluations_DRCPO = math.inf
n_experiments = len(results_DRCPO)
print(n_experiments)
if data_from_exp_2:
    n_experiments_per_group = len(list_parameters)
else:
    n_experiments_per_group = n_experiments
print('n experiments =', n_experiments)
for i in range(n_experiments):
    n_evaluations_DRCPO = min(n_evaluations_DRCPO, len(results_DRCPO[i][0]))

n_evaluations_RCPO = math.inf
for i in range(len(results_RCPO)):
    n_evaluations_RCPO = min(n_evaluations_RCPO, len(results_RCPO[i][0]))

print("n evaluations DRCPO =", n_evaluations_DRCPO)

if plot_RCPO:
    print('n evaluations RCPO =', n_evaluations_RCPO)
    # assert (n_evaluations_DRCPO-1) == (2 * (n_evaluations_RCPO-1))


list_opt_res_DRCPO = []
list_opt_res_RCPO = []

colors = {'DRCPO':'#1f77b4', 'RCPO': '#2A2E4B', 'naive': '#CDB54A', 'constraint':'r'}

rewards_naive_normalized = np.zeros((n_experiments, ))



# first we compute the maximum value experiment wise
n_env = 20
max_values = np.zeros((n_env, ))
max_values_app = np.zeros((n_env, ))
for experiment_index in range(n_env):
    max_value_DRCPO = -math.inf
    max_value_RCPO = -math.inf
    for n_apps in range(1):
        max_value_DRCPO = max(max_value_DRCPO, np.max(results_DRCPO[experiment_index + n_apps * n_env][0]))
        max_value_RCPO = max(max_value_RCPO, np.max(results_RCPO[experiment_index + n_apps * n_env][0]))
    max_values[experiment_index] = max(max_value_DRCPO, max_value_RCPO)

print(max_values)
for experiment_index in range(n_experiments):
    # print(results_DRCPO[i][0])
    # max value should be the same for each experiment
    # max_value_DRCPO = np.max(results_DRCPO[experiment_index][0])
    # max_value_RCPO = np.max(results_RCPO[experiment_index][0])
    max_value =  max_values[experiment_index % 20]
    opt_res_DRCPO = compute_optimal_result(results_DRCPO[experiment_index][0][1:], results_DRCPO[experiment_index][1][1:], list_parameters[experiment_index%n_experiments_per_group])
    opt_res_RCPO =  compute_optimal_result(results_RCPO[experiment_index][0][1:], results_RCPO[experiment_index][1][1:], list_parameters[experiment_index%n_experiments_per_group])
    values_to_normalize = min(len(results_DRCPO[experiment_index][0]), len(results_RCPO[experiment_index][0]))
    # DRCPO
    for j in range(n_evaluations_DRCPO):
        results_DRCPO[experiment_index][0][j] = results_DRCPO[experiment_index][0][j]/max_value
    if plot_RCPO:
        for j in range(n_evaluations_RCPO):
            results_RCPO[experiment_index][0][j] = results_RCPO[experiment_index][0][j]/max_value
    if naive_policy:
        # n_experiments_per_group = 20
        # app_per_server =  (experiment_index // n_experiments_per_group) + 1
        # exp_index = experiment_index - n_experiments_per_group * (app_per_server - 1)
        print(max_values[experiment_index % n_env], opt_res_DRCPO, opt_res_RCPO, rewards_naive[experiment_index], experiment_index % n_env)
        # rewards_naive_normalized[experiment_index] = rewards_naive[N_servers * exp_index + app_per_server-1]/max_value
        rewards_naive_normalized[experiment_index] = rewards_naive[experiment_index]/max_value
    else:
        print(max_value, opt_res_DRCPO, opt_res_RCPO, experiment_index % n_env)
    # print(results_DRCPO[experiment_index][0][:values_to_normalize])
    # print(results_RCPO[experiment_index][0][:values_to_normalize])
    list_opt_res_DRCPO.append(opt_res_DRCPO)
    list_opt_res_RCPO.append(opt_res_RCPO)
    print('-----------------'*3)
if naive_policy:
    print(np.mean(list_opt_res_DRCPO), np.mean(list_opt_res_RCPO), np.mean(rewards_naive))
else:
    print(np.mean(list_opt_res_DRCPO), np.mean(list_opt_res_RCPO))

print('-----------------'*3)
print(np.mean(rewards_naive_normalized), len(rewards_naive_normalized))
print(np.mean(rewards_naive_normalized), len(rewards_naive_normalized))
print('-----------------'*3)
# RCPO
# mean value given a fixed number of episodes
normalized_rewards_DRCPO = []
normalized_rewards_RCPO = []
# DRCPO
for i in range(n_evaluations_DRCPO):
    vector_to_average_DRCPO = []
    for j in range(n_experiments):
        vector_to_average_DRCPO.append(results_DRCPO[j][0][i])

    normalized_rewards_DRCPO.append(np.mean(vector_to_average_DRCPO))

# rewards_naive_normalized = rewards_naive_normalized[20:40]

# RCPO
for i in range(n_evaluations_RCPO):
    vector_to_average_RCPO = []
    for j in range(n_experiments):
        vector_to_average_RCPO.append(results_RCPO[j][0][i])
    normalized_rewards_RCPO.append(np.mean(vector_to_average_RCPO))



############# COST NORMALIZATION #############
cost_naive_normalized = np.zeros((n_experiments, N_servers))
for experiment_index in range(n_experiments):
    constraint = np.zeros((N_servers, ))
    for server_index in range(N_servers):
        # print(results_DRCPO[i][1])
        # constraint = list_parameters[experiment_index][server_index]['C'] * .15
        constraint[server_index] = list_parameters[experiment_index%n_experiments_per_group][server_index]['phi']
        # print(constraint[server_index])
        # DRCPO
        for episode_index in range(n_evaluations_DRCPO):
            results_DRCPO[experiment_index][1][episode_index][server_index] = results_DRCPO[experiment_index][1][episode_index][server_index]/constraint[server_index]
        if plot_RCPO:
            for episode_index in range(n_evaluations_RCPO):
                results_RCPO[experiment_index][1][episode_index][server_index] = results_RCPO[experiment_index][1][episode_index][server_index]/constraint[server_index]
        if naive_policy:
            # n_experiments_per_group = 20
            # app_per_server =  experiment_index // n_experiments_per_group + 1
            # exp_index = experiment_index - n_experiments_per_group * (app_per_server - 1)            
            # cost_naive_normalized[experiment_index][server_index] = cost_naive[N_servers * exp_index + app_per_server-1][server_index]/constraint[server_index]
            cost_naive_normalized[experiment_index][server_index] = cost_naive[experiment_index][server_index]/constraint[server_index]
            if cost_naive_normalized[experiment_index][server_index] > 1:
                print('Experiment index = {}, Server index = {}, Value = {}'.format(experiment_index, server_index, cost_naive_normalized[experiment_index][server_index]))
            # print(results_DRCPO[experiment_index][1], constraint[server_index])
    # print(results_RCPO[experiment_index][1], constraint[server_index])
    print('-----------------'*10)
# normalization of the results_DRCPO
normalized_costs_DRCPO = []
normalized_costs_RCPO = []
# DRCPO 
for i in range(n_evaluations_DRCPO):
    normalized_result_single_server_DRCPO = np.zeros((N_servers, ))
    for k in range(N_servers):
        vector_to_average_DRCPO = []
        for j in range(n_experiments):
            vector_to_average_DRCPO.append(results_DRCPO[j][1][i][k])
        normalized_result_single_server_DRCPO[k] = np.mean(vector_to_average_DRCPO)
    normalized_costs_DRCPO.append(normalized_result_single_server_DRCPO)

# RCPO
for i in range(n_evaluations_RCPO):
    normalized_result_single_server_RCPO = np.zeros((N_servers, ))
    for k in range(N_servers):
        vector_to_average_RCPO = []
        for j in range(n_experiments):
            vector_to_average_RCPO.append(results_RCPO[j][1][i][k])
        normalized_result_single_server_RCPO[k] = np.mean(vector_to_average_RCPO)
    normalized_costs_RCPO.append(normalized_result_single_server_RCPO)

if naive_policy:
    normalized_naive_cost = np.mean(cost_naive_normalized, axis = 1)
    print(len(normalized_naive_cost))
    normalized_naive_cost = np.mean(normalized_naive_cost)

evolution_cost_DRCPO = []
evolution_cost_RCPO = []
for i in range(N_servers):
    evolution_cost_DRCPO.append([])
    evolution_cost_RCPO.append([])

for evaluation_index in range(len(normalized_costs_DRCPO)):
    for server_index in range(N_servers):
        evolution_cost_DRCPO[server_index].append(normalized_costs_DRCPO[evaluation_index][server_index])
    

for evaluation_index in range(len(normalized_costs_RCPO)):
    for server_index in range(N_servers):
        evolution_cost_RCPO[server_index].append(normalized_costs_RCPO[evaluation_index][server_index])

labels = ['DRCPO', 'RCPO', 'Naive', 'Constraint']
linestyles = {'DRCPO': '-', 'RCPO': '--', 'naive': '-.', 'constraint': ':'}
lines_list = []
labels_to_plot = []

major_yticks_rew = np.arange(.5, 1.1, .25)
minor_yticks_rew = np.arange(.5, 1.1, .25/5)
max_cost_lim = 1.5
major_yticks_cost = np.arange(0, max_cost_lim, .5)
minor_yticks_cost = np.arange(0, max_cost_lim, .5/5)
major_xticks = np.arange(0, 201, 50)
minor_xticks = np.arange(0, 201, 10)
minor_grid_alpha = .3
major_grid_alpha = .6 

# we must make a single figure with two rectangulars figure inside
# the top one shows the discounted reward
# the bottom one shows the discounted cost

figure, axis = plt.subplots(2, sharex=True)
figure.set_figheight(5)
figure.set_figwidth(5.3)

line_DRCPO,  = axis[0].plot(normalized_rewards_DRCPO, label=labels[0], color = colors['DRCPO'], linestyle = linestyles['DRCPO'])
labels_to_plot.append(labels[0])
lines_list.append(line_DRCPO)
if plot_RCPO:
    line_RCPO,  = axis[0].plot(np.arange(0, RCPO_multiplier * n_evaluations_RCPO, RCPO_multiplier), normalized_rewards_RCPO, label = labels[1], color = colors['RCPO'], linestyle = linestyles['RCPO'])
    labels_to_plot.append(labels[1])
    lines_list.append(line_RCPO)
if naive_policy:
    line_naive = axis[0].hlines(np.mean(rewards_naive_normalized), 0, n_evaluations_DRCPO, color = colors['naive'], label = labels[2], linestyle = linestyles['naive'])
    labels_to_plot.append(labels[2])
    lines_list.append(line_naive)
    
# plt.plot(mean_discounted_rewards_unconstrained, label='Unconstrained Decomposed reward learning')
# plt.hlines(np.mean(max_discounted_rewards_unconstrained), 0, 50, label = 'Optimal reward', colors='r', linestyles='dashed')
# axis[0].legend()
axis[0].set_ylabel('Discounted Reward', fontsize=14)

axis[0].set_ylim([.5, 1.1])
axis[0].set_yticks(major_yticks_rew)
axis[0].set_yticks(minor_yticks_rew, minor=True)
axis[0].set_xticks(major_xticks)
axis[0].set_xticks(minor_xticks, minor=True)

axis[0].grid(which = 'major', alpha = major_grid_alpha)
axis[0].grid(which = 'minor', alpha = minor_grid_alpha)

# computation mean costs
mean_normalized_cost_DRCPO = np.zeros((len(normalized_costs_DRCPO), ))
mean_normalized_cost_RCPO = np.zeros((len(normalized_costs_RCPO), ))
for evaluation_index in range(len(normalized_costs_DRCPO)):
    mean_normalized_cost_DRCPO[evaluation_index] = np.mean([evolution_cost_DRCPO[i][evaluation_index] for i in range(N_servers)])
for evaluation_index in range(len(normalized_costs_RCPO)):
    mean_normalized_cost_RCPO[evaluation_index] = np.mean([evolution_cost_RCPO[i][evaluation_index] for i in range(N_servers)])


axis[1].plot(mean_normalized_cost_DRCPO, label=labels[0], linestyle = linestyles['DRCPO'], color = colors['DRCPO'])
if plot_RCPO:
    axis[1].plot(np.arange(0, RCPO_multiplier * n_evaluations_RCPO, RCPO_multiplier), mean_normalized_cost_RCPO, color=colors['RCPO'], label = labels[1], linestyle = linestyles['RCPO'])
# plt.plot(mean_discounted_costs_unconstrained, label='Unconstrained Decomposed reward learning')
if naive_policy:
    axis[1].hlines(normalized_naive_cost, 0, n_evaluations_DRCPO, label = labels[2], linestyles = linestyles['naive'], color = colors['naive'])
        

line_constraint = axis[1].hlines(1, 0, n_evaluations_DRCPO, label=labels[3], color=colors['constraint'], linestyle = linestyles['constraint'])
labels_to_plot.append(labels[3])
lines_list.append(line_constraint)
labels.append('Constraint')
# axis[1].legend()
axis[1].set_ylabel('Discounted Cost', fontsize=14)
axis[1].set_xlabel('Episodes', fontsize=14)
axis[1].set_ylim([0, max_cost_lim])
# axis[1].set_xticks(xticks)
# axis[1].set_xticklabels(xticks_label)
axis[1].set_yticks(major_yticks_cost)
axis[1].set_yticks(minor_yticks_cost, minor=True)
axis[1].set_xticks(major_xticks)
axis[1].set_xticks(minor_xticks, minor=True)
axis[1].grid(which = 'major', alpha = major_grid_alpha)
axis[1].grid(which = 'minor', alpha = minor_grid_alpha)
plt.figlegend(lines_list, labels_to_plot, loc = 'upper center', ncol=2, labelspacing=0., fontsize = 14, fancybox=True, bbox_to_anchor=(.55, .89))
plt.xticks(np.arange(0, 201, 50), np.arange(0, 20001, 5000))
# plt.xticks_labels = np.arange(0, 10001, 2000)
if savefig:
    plt.savefig(folder + "experiment_1.pdf", format="pdf", bbox_inches="tight")
plt.show()


