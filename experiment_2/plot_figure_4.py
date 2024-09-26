# attempts of plotting the results_single_server of experiment 2

# goal show how for less applications on a server, despite having lower average cost, the cost of some servers is higher than the constraint, 
# in a worse way compared to the other cases. This is explicable, how? 


import pickle as pk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import math


def compute_optimal_result(reward, costs, parameters):
    
    constraint = np.zeros((4, ))
    n_servers = 4
    for i in range(4):
        constraint[i] = parameters[i]['phi']
    max_feasible_reward = -math.inf
    for episode_index in range(len(reward)):
        constraint_satisfied = True
        for server_index in range(n_servers):
            if costs[episode_index][server_index] > constraint[server_index]:
                constraint_satisfied = False
        
        if constraint_satisfied:
            max_feasible_reward = max(max_feasible_reward, reward[episode_index])

    return max_feasible_reward



N_servers = 10
N_areas_of_interest = [1, 4, 7, 10] 
list_results = []
list_elapsed_times = []

longer_episodes = False
single_plot = True
savefig = True

# read the data
folder = '' #'/results/experiment_2/'
for areas_of_interest in N_areas_of_interest:
    file = folder + '{}_servers_{}_applications_DRCPO.pk'.format(N_servers, areas_of_interest)
    with open(file, 'rb') as f:
        data = pk.load(f)
    print(file)
    if areas_of_interest >= N_areas_of_interest[0]:
        list_parameters = data[0]
        # print(list_parameters[0])
        list_area_of_interest = data[1]
        list_arrival_rates = data[2]
        print(list_arrival_rates[0])
        list_discount_factors = data[3]
        print(list_discount_factors[0])
    results = data[4]
    list_results.append(results)
    elapsed_time = data[6]
    list_elapsed_times.append(elapsed_time)


    


n_evaluations = math.inf * np.ones((len(N_areas_of_interest), ), dtype=int)
n_experiments = len(list_results[0])         # we assume thay are the same for every element of the list
print('N experiments=', n_experiments)

for index in range(len(N_areas_of_interest)):
    for experiment_index in range(n_experiments):
        if len(list_results[index][experiment_index][0]) < n_evaluations[index]:
            if len(list_results[index][experiment_index][0]) < 200:
                print('Experiment index = {}. Servers for each application = {}'.format(experiment_index, N_areas_of_interest[index]))
                print('N evaluations = ', len(list_results[index][experiment_index][0]))   
            n_evaluations[index] = min(n_evaluations[index], len(list_results[index][experiment_index][0]))

print('N evaluations = ', n_evaluations)

colors = {'DRCPO':'#1f77b4', 'RCPO': '#ff7f0e'}

# DISCOUNTED REWARD
# normalization wrt max value
for experiment_index in range(n_experiments):
    # print(results_single_server[i][0])
    max_value = np.zeros((len(N_areas_of_interest), ))
    for index in range(len(N_areas_of_interest)):
        max_value[index] = np.max(list_results[index][experiment_index][0])
    # for index in range(1, len(N_areas_of_interest)):
    #     max_value = max(max_value, np.max(list_results[index][experiment_index][0]))

    list_opt_res = []
    for index in range(len(N_areas_of_interest)):
        opt_res = compute_optimal_result(list_results[index][experiment_index][0], list_results[index][experiment_index][1], list_parameters[experiment_index])
        list_opt_res.append(opt_res)
        for evaluation_index in range(int(n_evaluations[index])):
            list_results[index][experiment_index][0][evaluation_index] = list_results[index][experiment_index][0][evaluation_index]/max_value[index]

    print(max_value, list_opt_res)
    print('-----------------'*3)

# mean value given a fixed number of episodes
normalized_rewards = []
for index in range(len(N_areas_of_interest)):
    normalized_rewards.append([])

for index in range(len(N_areas_of_interest)):
    for evaluation_index in range(int(n_evaluations[index])):
        vector_to_average = []
        for experiment_index in range(n_experiments):
            vector_to_average.append(list_results[index][experiment_index][0][evaluation_index])
        normalized_rewards[index].append(np.mean(vector_to_average))

# DISCOUNTED COST
# normalization wrt the constraint of each server
for experiment_index in range(n_experiments):
    constraint = np.zeros((N_servers, ))
    for server_index in range(N_servers):
        constraint[server_index] = list_parameters[experiment_index][server_index]['phi']
        for app_per_server in range(len(N_areas_of_interest)):
            for episode_index in range(int(n_evaluations[app_per_server])):
                list_results[app_per_server][experiment_index][1][episode_index][server_index] = list_results[app_per_server][experiment_index][1][episode_index][server_index]/constraint[server_index]
            
# extract the maximum value of the normalized costs and average wrt all the experiments
normalized_costs_max = []
normalized_costs_mean = []
normalized_costs_min = []
for index in range(len(N_areas_of_interest)):
    normalized_costs_max.append([])
    normalized_costs_mean.append([])
    normalized_costs_min.append([])
for app_per_server in range(len(N_areas_of_interest)):
    for evaluation_index in range(int(n_evaluations[app_per_server])):
        vector_evaluation_max = []
        vector_evaluation_mean = []
        vector_evaluation_min = []
        for experiment_index in range(n_experiments):
            vector_to_average = []
            for server_index in range(N_servers):
                vector_to_average.append(list_results[app_per_server][experiment_index][1][evaluation_index][server_index])
            vector_evaluation_max.append(np.max(vector_to_average))
            vector_evaluation_mean.append(np.mean(vector_to_average))
            vector_evaluation_min.append(np.min(vector_to_average))
        normalized_costs_max[app_per_server].append(np.mean(vector_evaluation_max))
        normalized_costs_mean[app_per_server].append(np.mean(vector_evaluation_mean))
        normalized_costs_min[app_per_server].append(np.mean(vector_evaluation_min))
        # normalized_costs[app_per_server].append(max_normalized_costs)

evolution_cost_max = []
evolution_cost_mean = []
evolution_cost_min = []
for app_per_server in range(len(N_areas_of_interest)):
    evolution_cost_max.append([])
    evolution_cost_mean.append([])
    evolution_cost_min.append([])
for app_per_server in range(len(N_areas_of_interest)):
    for evaluation_index in range(int(n_evaluations[app_per_server])):
        evolution_cost_max[app_per_server].append(normalized_costs_max[app_per_server][evaluation_index])
        evolution_cost_mean[app_per_server].append(normalized_costs_mean[app_per_server][evaluation_index])
        evolution_cost_min[app_per_server].append(normalized_costs_min[app_per_server][evaluation_index])

ratios_vector = [1.5]
for i in range(len(N_areas_of_interest)):
    ratios_vector.append(1)
figure, axis = plt.subplots(len(N_areas_of_interest) + 1, sharex=True, gridspec_kw={'height_ratios': ratios_vector})
figure.set_figheight(8)
figure.set_figwidth(6)
labels = []
linestyles = []
style = ['dotted', 'dashed', 'dashdot', 'solid']
color = ['#EF2D56', '#ED7D3A', '#66C237', '#1f77b4']
for index in range(len(N_areas_of_interest)):
    line, = axis[0].plot(normalized_rewards[index], linestyle = style[index], color = color[index]) 
    linestyles.append(line)
    labels.append('{}'.format(N_areas_of_interest[index]))
    
# plt.plot(mean_discounted_rewards_unconstrained, label='Unconstrained Decomposed reward learning')
# plt.hlines(np.mean(max_discounted_rewards_unconstrained), 0, 50, label = 'Optimal reward', colors='r', linestyles='dashed')
# axis[0].legend()

alpha_lines = .5
alpha_filling = 0.5
width_line = .5

minor_grid_alpha = .3
major_grid_alpha = .6

axis[0].set_ylabel('Discounted\nReward', fontsize=14)
axis[0].set_ylim([.5, 1.1])

major_yticks = np.arange(.5, 1.1, .25)
minor_yticks = np.arange(.5, 1.1, .25/4)

major_xticks = np.arange(0, 201, 50)
minor_xticks = np.arange(0, 201, 10)
axis[0].set_yticks(major_yticks)
axis[0].set_yticks(minor_yticks, minor=True)
axis[0].set_xticks(major_xticks)
axis[0].set_xticks(minor_xticks, minor=True)
axis[0].grid(which='minor', alpha = minor_grid_alpha)
axis[0].grid(which='major', alpha = major_grid_alpha)



for index in range(len(N_areas_of_interest)):
    axis[index+1].plot(np.arange(0, len(evolution_cost_max[index])), evolution_cost_max[index], label = '{}'.format(N_areas_of_interest[index]), color = linestyles[index].get_color(), alpha = alpha_lines, linewidth = width_line)
    axis[index+1].plot(np.arange(0, len(evolution_cost_mean[index])), evolution_cost_mean[index], label = '{}'.format(N_areas_of_interest[index]), color = linestyles[index].get_color(), linestyle = linestyles[index].get_linestyle(), linewidth = 2)
    axis[index+1].plot(np.arange(0, len(evolution_cost_min[index])), evolution_cost_min[index], label = '{}'.format(N_areas_of_interest[index]), color = linestyles[index].get_color(), alpha = alpha_lines, linewidth = width_line)
    axis[index+1].set_ylim([0, 2])
    axis[index+1].set_xlim([-10, 200])
    axis[index+1].fill_between(np.arange(0, len(evolution_cost_max[index])), evolution_cost_max[index], evolution_cost_min[index], alpha=alpha_filling, color = linestyles[index].get_color())
    line_constraint = axis[index+1].hlines(1, -10, 251, color='black', alpha = .3)
    # axis[index+1].set_ylabel('Discounted\nCost', fontsize=14)
    major_yticks = np.arange(0, 2.1, 1)
    minor_yticks = np.arange(0, 2.1, .25)

    major_xticks = np.arange(0, 201, 50)
    minor_xticks = np.arange(0, 201, 10)

    axis[index+1].set_yticks(major_yticks)
    axis[index+1].set_yticks(minor_yticks, minor=True)

    axis[index+1].set_xticks(major_xticks)
    axis[index+1].set_xticks(minor_xticks, minor=True)
    # axis[1].set_xticks(xticks)
    # axis[1].set_xticklabels(xticks_label)
    #  axis[index+1].grid(alpha = .3)
    axis[index+1].grid(which='minor', alpha=minor_grid_alpha)
    axis[index+1].grid(which='major', alpha=major_grid_alpha)

figure.text(0.02, 0.39, 'Discounted cost', va='center', rotation='vertical', fontsize=14)


axis[index+1].set_xlabel('Episodes', fontsize=14)
plt.figlegend(linestyles, labels, loc = 'upper center', ncol=4, labelspacing=0., fontsize = 14, title = r'Applications per server ($d^i)$', title_fontsize = 14, borderaxespad=0.1, bbox_to_anchor=(0.512, .97))
plt.xticks(np.arange(0, 201, 50), np.arange(0, 20001, 5000))
# plt.xticks_labels = np.arange(0, 10001, 2000)
figure.align_ylabels(axis[:])
if savefig:
    plt.savefig(folder + "experiment_2_evolution_cost_intervals.pdf", format="pdf", bbox_inches="tight")
plt.show()
