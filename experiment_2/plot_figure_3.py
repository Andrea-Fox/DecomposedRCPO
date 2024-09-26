# FIGURE 1
# x-axis: number of servers on whicH each application is installed
# y-axis: normalized discounted reward (normalized wrt the value obtained when each application is installed on a single server)

# FIGURE 2
# x-axis: number of servers on whicH each application is installed
# y-axis: normalized discounted cost (normalized wrt the value obtained when each application is installed on a single server)

# VALUE CONSIDERED: we consider the maximum value of reward obtained by the system during the learning process, 
# and restrict our attention to the feasible policies such that all future policies are feasible
# 
# FEASIBLE POLICY: a policy is feasible if it satisfies the constraints on the admission policies  
import numpy as np
import math
import pickle as pk
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def feasible_policy(costs, parameters):
    # we look at the constraints of the system and check if the policy satisfies them
    N_servers = len(costs[0])
    constraint = np.zeros((N_servers, ))
    index_of_feasible_policy = math.inf
    for server_index in range(N_servers):
        constraint[server_index] = parameters[server_index]['phi']

    constraint_satisfied = True
    episode_index = len(costs) -1
    while constraint_satisfied and episode_index > 1:
        local_costs = costs[episode_index] 
        constraints_broken = 0
        for server_index in range(N_servers):
            if costs[episode_index][server_index] > constraint[server_index] * 10:
                constraints_broken += 1
        if constraints_broken > 1 :
            constraint_satisfied = False
            index_of_feasible_policy = episode_index + 1
        episode_index -= 1
    if index_of_feasible_policy == math.inf:
        return 10
    else:
        return int(index_of_feasible_policy)



def value_to_consider(reward, costs, parameters, constraint_surplus = 0):
    # first we understand which subset to consider (i.e. all the future policies are feasible)
    index_of_feasible_policy = feasible_policy(costs, parameters)
    # we consider the maximum value of reward obtained by the system during the learning process, among the considered policies     
    max_feasible_reward = -math.inf
    best_policy = None
    for episode_index in range(index_of_feasible_policy, len(reward)):
        # first we consider if the policy is feasible (really feasible, without tricks...)
        constraint_satisfied = 0
        for server_index in range(N_servers):
            if costs[episode_index][server_index] < parameters[server_index]['phi'] * 1:
                constraint_satisfied += 1
            elif costs[episode_index][server_index] > parameters[server_index]['phi'] * (1 + .05):
                constraint_satisfied = -math.inf

        if reward[episode_index] > max_feasible_reward and constraint_satisfied >= N_servers * .5:
            max_feasible_reward = reward[episode_index]
            best_policy = episode_index
    return max_feasible_reward, best_policy


N_servers = 10
N_areas_of_interest = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
colors = {'DRCPO':'#1f77b4', 'RCPO': '#ff7f0e'}
plot_RCPO = False
savefig = True

# read the data
folder = '' #'/results/experiment_2/'

list_rewards_DRCPO = []
list_rewards_RCPO = []

for area_of_interest in N_areas_of_interest:
    file_DRCPO = folder + '{}_servers_{}_applications_DRCPO.pk'.format(N_servers, area_of_interest)

    with open(file_DRCPO, 'rb') as f:
        data = pk.load(f)
    
    # print(data)

    list_parameters = data[0]
    # print(list_parameters[0])
    list_area_of_interest = data[1]
    list_arrival_rates = data[2]
    list_discount_factors = data[3]
    results_DRCPO = data[4]
    results_RCPO = None
    elapsed_time = data[6]

    n_experiments_DRCP0 = len(results_DRCPO)
    min_length_DRCPO = math.inf
    for i in range(n_experiments_DRCP0):
        if len(results_DRCPO[i][0]) < min_length_DRCPO:
            min_length_DRCPO = len(results_DRCPO[i][0])
    n_experiments = n_experiments_DRCP0
    
    # assert min_length_DRCPO-1 == 2 * (min_length_RCPO-1)
    

    reward_DRCPO = np.zeros((n_experiments, ))
    best_index_DCRPO = np.zeros((n_experiments, ))

    for experiment_index in range(n_experiments):
        reward_DRCPO[experiment_index], best_index_DCRPO[experiment_index] = value_to_consider(results_DRCPO[experiment_index][0], results_DRCPO[experiment_index][1], list_parameters[experiment_index], constraint_surplus=.0)
    list_rewards_DRCPO.append(reward_DRCPO)

    print('----'*10)


all_feasible_policies = 0
for index in range(len(reward_DRCPO)):
    if np.min([list_rewards_DRCPO[i][index] for i in range(len(list_rewards_DRCPO))]) != -math.inf:
        all_feasible_policies += 1

normalized_rewards_DRCPO = np.zeros((all_feasible_policies, len(list_rewards_DRCPO)))
style = ['dotted', 'solid' , 'solid', 'dashed', 'solid', 'solid',  'dashdot', 'solid', 'solid', 'solid']
color = ['#EF2D56', '#808080', '#808080', '#ED7D3A', '#808080', '#808080', '#66C237', '#808080', '#808080', '#1f77b4']
alpha_other = .4
alpha = [1, alpha_other, alpha_other, 1, alpha_other, alpha_other, 1, alpha_other, alpha_other, 1]
# normalization wrt the value of the reward obtained when each application is installed on a single server
index_of_feasible_policy = 0
for index in range(len(reward_DRCPO)):
    if np.min([list_rewards_DRCPO[i][index] for i in range(len(list_rewards_DRCPO))]) != -math.inf:
        for i in range(len(list_rewards_DRCPO)):
            normalized_rewards_DRCPO[index_of_feasible_policy][i] = list_rewards_DRCPO[i][index] / list_rewards_DRCPO[0][index]
        index_of_feasible_policy += 1
print(normalized_rewards_DRCPO)
print(len(normalized_rewards_DRCPO))
for i in range(len(list_rewards_DRCPO)):
    print(np.mean(normalized_rewards_DRCPO[:, i], axis=0), np.median(normalized_rewards_DRCPO[:, i], axis=0))


offset = .3 * plot_RCPO
width = .3 * (1 + plot_RCPO)
# we now plot the results
plt.figure(1, figsize=(5.3, 3.5))
# plt.figheight(5)
# plt.figwidth(5.3)
alpha_value = .6
boxprops_DRCPO = dict(color=colors['DRCPO'], alpha = alpha_value)
whiskerprops_DRCPO = dict(color = colors['DRCPO'], alpha = alpha_value)
capprops_DRCPO = dict(color = colors['DRCPO'], alpha = alpha_value)

boxprops_RCPO = dict(color=colors['RCPO'], alpha = alpha_value)
whiskerprops_RCPO = dict(color = colors['RCPO'], alpha = alpha_value)
capprops_RCPO = dict(color = colors['RCPO'], alpha = alpha_value)

for index in range(len(N_areas_of_interest)):
    bp = plt.boxplot(normalized_rewards_DRCPO[:, index], positions= [N_areas_of_interest[index]-offset], widths=width,showfliers=True) #, boxprops=boxprops_DRCPO, whiskerprops=whiskerprops_DRCPO, capprops=capprops_DRCPO) 
    for _, line_list in bp.items():
        for line in line_list:
            line.set_color(color[index])
            line.set_alpha(alpha[index])
            line.set_linestyle(style[index])
            line.set_linewidth(alpha[index] * 2)
    # plt.setp(bp['medians'], color=color[index])
    
# finding the polynomial that best fits the median of the data for DRCPO
degree = 2  # Specify the degree of the polynomial curve
x_DRCPO = [N_areas_of_interest[i]-offset for i in range(len(N_areas_of_interest))]  # x-values for the curve
y_DRCPO = np.median(normalized_rewards_DRCPO, axis=0)  # Median of the data along the columns

coefficients_DRCPO = np.polyfit(x_DRCPO, y_DRCPO, degree)  # Fit the polynomial curve
curve_DRCPO = np.poly1d(coefficients_DRCPO)  # Create a polynomial function from the coefficients

# Plot the polynomial curve
x_curve_DRCPO = np.linspace(1-offset, N_servers-offset, 100)  # x-values for the curve
y_curve_DRCPO = curve_DRCPO(x_curve_DRCPO)  # Evaluate the polynomial function at x_curve

# plt.hlines(1, 0, N_servers+1, colors='blue', linestyles='dashed', alpha = 0.25)

plt.plot(x_curve_DRCPO, y_curve_DRCPO, color='black', alpha = .75 ) 
plt.grid(alpha = .3)

plt.xticks(N_areas_of_interest, N_areas_of_interest)
plt.xlabel(r'Applications per server ($d^i$)')
plt.ylabel('Normalized discounted reward')
# plt.legend(['DRCPO'])


# plt.ylim([.75, 2.5])
# plt.title('Impact of application installation')

if savefig:
    plt.savefig(folder + "experiment_2.pdf", format="pdf", bbox_inches="tight")
# Show the plot
plt.show()