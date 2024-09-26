import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import os, sys

def plot_colored_boxplot_N(image_index, axis, data, median, index, labels, colors, linestyles_list, markers):
    N_servers_to_consider = [3, 5, 7, 10]
    alpha_value = 0.4
    medianprops = dict(linewidth=0)
    boxprops = dict(color=colors[index], alpha = alpha_value)
    whiskerprops = dict(color = colors[index], alpha = alpha_value)
    capprops = dict(color = colors[index], alpha = alpha_value)
    width = 0.2
    shift = -1.5 + index 
    axis[image_index].boxplot(data, positions=np.array(N_servers_to_consider) + shift * width,  widths = width, showfliers = False, medianprops = medianprops, boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops)
    scatter_plot = axis[image_index].scatter(np.array(N_servers_to_consider) + shift * width, median, color = colors[index], marker=markers[index])
    m_1, b_1 = np.polyfit(N_servers_to_consider, median, 1)
    line_plot,  = axis[image_index].plot(np.array(N_servers_to_consider) + shift * width, m_1 * np.array(N_servers_to_consider) + b_1, label=labels[index], color = colors[index], linestyle = linestyles_list[index], linewidth = 2)

    return scatter_plot, line_plot

def plot_colored_boxplot_ratio(image_index, axis, data, median, index, labels, colors, linestyles_list, markers):
    ratios_to_consider = [.1, .4, .7, 1.0, 1.3]
    alpha_value = 0.4
    medianprops = dict(linewidth=0)
    boxprops = dict(color=colors[index], alpha = alpha_value)
    whiskerprops = dict(color = colors[index], alpha = alpha_value)
    capprops = dict(color = colors[index], alpha = alpha_value)
    width = 0.04
    shift = -1.5 + index 
    axis[image_index].boxplot(data, positions=np.array(ratios_to_consider) + shift * width,  widths = width, showfliers = False, medianprops = medianprops, boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops)
    scatter_plot =axis[image_index].scatter(np.array(ratios_to_consider) + shift * width, median, color = colors[index], marker = markers[index])
    m_1, b_1 = np.polyfit(ratios_to_consider, median, 1)
    line_plot,  = axis[image_index].plot(np.array(ratios_to_consider) + shift * width, m_1 * np.array(ratios_to_consider) + b_1, label=labels[index], color = colors[index], linestyle = linestyles_list[index], linewidth = 2)

    print(scatter_plot, line_plot)
    return scatter_plot, line_plot



# load the data
folder = os.path.dirname(os.path.abspath(sys.argv[0]))
print(folder)
folder  = folder + '/results/experiment_3/'
# folder = '/home/afox/Dropbox/video_admission_control/python/routing/results/experiment_2/'
N_servers_to_consider = [3, 5, 7, 10]
values_of_N_server = len(N_servers_to_consider)
results = []
for N_servers in N_servers_to_consider:
    filename = folder + 'N_' + str(N_servers)+ '.pk' 
    results_temp = pk.load(open(filename, 'rb'))

    results.append(results_temp)


different_ratio_values = 5
n_experiments = 4



list_total_rewards_uniform_with_area_of_interest = np.zeros(shape = (different_ratio_values, n_experiments, values_of_N_server))
list_total_rewards_uniform_without_area_of_interest = np.zeros(shape = (different_ratio_values, n_experiments, values_of_N_server))

list_total_rewards_constraints_with_area_of_interest = np.zeros(shape = (different_ratio_values, n_experiments, values_of_N_server))
list_total_rewards_constraints_without_area_of_interest = np.zeros(shape = (different_ratio_values, n_experiments, values_of_N_server))

list_total_rewards_occupation = np.zeros(shape = (different_ratio_values, n_experiments, values_of_N_server))
list_total_rewards_alternate = np.zeros(shape = (different_ratio_values, n_experiments, values_of_N_server))
# extract the average for each method
print(len(results[0]), len(results[1]))


for k in range(values_of_N_server):
    for i in range(different_ratio_values):
        for j in range(n_experiments):
            index = i*n_experiments + j
            list_total_rewards_uniform_without_area_of_interest[i, j, k] = results[k][index][0]
            list_total_rewards_uniform_with_area_of_interest[i, j, k] = results[k][index][1]
            list_total_rewards_constraints_without_area_of_interest[i, j, k] = results[k][index][2]
            list_total_rewards_constraints_with_area_of_interest[i, j, k] = results[k][index][3]
            list_total_rewards_occupation[i, j, k] = results[k][index][4]
            list_total_rewards_alternate[i, j, k] = np.max(results[k][index][5])


#############################################################################
# evolution wrt the value of N_servers


N_server_average_total_rewards_uniform_with_area_of_interest = np.mean(list_total_rewards_uniform_with_area_of_interest, axis = (0, 1))
N_server_average_total_rewards_uniform_without_area_of_interest = np.mean(list_total_rewards_uniform_without_area_of_interest, axis = (0, 1))
N_server_average_total_rewards_constraints_with_area_of_interest = np.mean(list_total_rewards_constraints_with_area_of_interest, axis = (0, 1))
N_server_average_total_rewards_constraints_without_area_of_interest = np.mean(list_total_rewards_constraints_without_area_of_interest, axis = (0, 1))
N_server_average_total_rewards_occupation = np.mean(list_total_rewards_occupation, axis = (0, 1))
N_server_average_total_rewards_alternate = np.mean(list_total_rewards_alternate, axis = (0, 1))
# print(list_total_rewards_uniform_with_area_of_interest)
print(N_server_average_total_rewards_uniform_with_area_of_interest)
print(N_server_average_total_rewards_uniform_without_area_of_interest)
print(N_server_average_total_rewards_constraints_with_area_of_interest)
print(N_server_average_total_rewards_constraints_without_area_of_interest)
print(N_server_average_total_rewards_occupation)
print(N_server_average_total_rewards_alternate)
# we now format each list into a 15x10 matrix


# data normalization
# 1 is the value obtained with uniform load balancing, even toward the ares without interest
# moreover, we exclude Q learning
N_server_normalized_uniform = list_total_rewards_uniform_with_area_of_interest/list_total_rewards_uniform_without_area_of_interest
N_server_normalized_constraint = list_total_rewards_constraints_with_area_of_interest/list_total_rewards_uniform_without_area_of_interest
N_server_normalized_occupation = list_total_rewards_occupation/list_total_rewards_uniform_without_area_of_interest
N_server_normalized_alternate = list_total_rewards_alternate/list_total_rewards_uniform_without_area_of_interest

# we now compute all the averages, so that we can create a new plot
N_server_mean_normalized_uniform = np.mean(N_server_normalized_uniform, axis = (0, 1))
N_server_median_normalized_uniform = np.median(N_server_normalized_uniform, axis = (0, 1))
N_server_std_normalized_uniform = np.std(N_server_normalized_uniform, axis = (0, 1))

N_server_mean_normalized_constraint = np.mean(N_server_normalized_constraint, axis = (0, 1))
N_server_median_normalized_constraint = np.median(N_server_normalized_constraint, axis = (0, 1))
N_server_std_normalized_constraint = np.std(N_server_normalized_constraint, axis = (0, 1))

N_server_mean_normalized_occupation = np.mean(N_server_normalized_occupation, axis = (0, 1))
N_server_median_normalized_occupation = np.median(N_server_normalized_occupation, axis = (0, 1))
N_server_std_normalized_occupation = np.std(N_server_normalized_occupation, axis = (0, 1))

N_server_mean_normalized_alternate = np.mean(N_server_normalized_alternate, axis = (0, 1))
N_server_median_normalized_alternate = np.median(N_server_normalized_alternate, axis = (0, 1))
N_server_std_normalized_alternate = np.std(N_server_normalized_alternate, axis = (0, 1))


# reproduction of the plot used in experiment 1b (convergence speed)
m_uniform, b_uniform = np.polyfit(N_servers_to_consider, N_server_mean_normalized_uniform, 1)
m_constraint, b_constraint = np.polyfit(N_servers_to_consider, N_server_mean_normalized_constraint, 1)
m_occupation, b_occupation = np.polyfit(N_servers_to_consider, N_server_mean_normalized_occupation, 1)
m_alternate, b_alternate = np.polyfit(N_servers_to_consider, N_server_mean_normalized_alternate, 1)

ratio_average_total_rewards_uniform_with_area_of_interest = np.mean(list_total_rewards_uniform_with_area_of_interest, axis = (1, 2))
ratio_average_total_rewards_uniform_without_area_of_interest = np.mean(list_total_rewards_uniform_without_area_of_interest, axis = (1, 2))
ratio_average_total_rewards_constraints_with_area_of_interest = np.mean(list_total_rewards_constraints_with_area_of_interest, axis = (1, 2))
ratio_average_total_rewards_constraints_without_area_of_interest = np.mean(list_total_rewards_constraints_without_area_of_interest, axis = (1, 2))
ratio_average_total_rewards_occupation = np.mean(list_total_rewards_occupation, axis = (1, 2))
ratio_average_total_rewards_alternate = np.mean(list_total_rewards_alternate, axis = (1, 2))


ratio_normalized_uniform = list_total_rewards_uniform_with_area_of_interest/list_total_rewards_uniform_without_area_of_interest
ratio_normalized_constraint = list_total_rewards_constraints_with_area_of_interest/list_total_rewards_uniform_without_area_of_interest
ratio_normalized_occupation = list_total_rewards_occupation/list_total_rewards_uniform_without_area_of_interest
ratio_normalized_alternate = list_total_rewards_alternate/list_total_rewards_uniform_without_area_of_interest

ratio_mean_normalized_uniform = np.mean(ratio_normalized_uniform, axis = (1, 2))
ratio_median_normalized_uniform = np.median(ratio_normalized_uniform, axis = (1, 2))

ratio_mean_normalized_constraint = np.mean(ratio_normalized_constraint, axis = (1, 2))
ratio_median_normalized_constraint = np.median(ratio_normalized_constraint, axis = (1, 2))

ratio_mean_normalized_occupation = np.mean(ratio_normalized_occupation, axis = (1, 2))
ratio_median_normalized_occupation = np.median(ratio_normalized_occupation, axis = (1, 2))

ratio_mean_normalized_alternate = np.mean(ratio_normalized_alternate, axis = (1, 2))
ratio_median_normalized_alternate = np.median(ratio_normalized_alternate, axis = (1, 2))

colors = ['#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
linestyles_list = ['solid', 'dashdot', 'dashed', 'dotted']
labels = ['Uniform', 'Origin-based', 'Occupation-based', 'Adaptive']
markers = ['o', 's', 'x', '^']
ratios_to_consider = [.1, .4, .7, 1.0, 1.3]
############################################################################################################
# single plot 

fixed_ratio_uniform = []
fixed_ratio_constraint = []
fixed_ratio_occupation = []
fixed_ratio_alternate = []
for i in range(len(ratios_to_consider)):
    fixed_ratio_uniform.append(ratio_normalized_uniform[i, :, :].flatten())
    fixed_ratio_constraint.append(ratio_normalized_constraint[i, :, :].flatten())
    fixed_ratio_occupation.append(ratio_normalized_occupation[i, :, :].flatten())
    fixed_ratio_alternate.append(ratio_normalized_alternate[i, :, :].flatten())

figure, axis = plt.subplots(2, sharey=True)
figure.set_figheight(5.5)
figure.set_figwidth(5.3)


s_uniform, p_uniform = plot_colored_boxplot_ratio(0, axis, fixed_ratio_uniform, ratio_median_normalized_uniform, 0, labels, colors, linestyles_list, markers)
s_constraint, p_constraint = plot_colored_boxplot_ratio(0, axis, fixed_ratio_constraint, ratio_median_normalized_constraint, 1, labels, colors, linestyles_list, markers)
s_occupation, p_occupation = plot_colored_boxplot_ratio(0, axis, fixed_ratio_occupation, ratio_median_normalized_occupation, 2, labels, colors, linestyles_list, markers)
s_alternate, p_alternate = plot_colored_boxplot_ratio(0, axis, fixed_ratio_alternate, ratio_median_normalized_alternate, 3, labels, colors, linestyles_list, markers)
axis[0].set_xlim([-.1, 1.5])
axis[0].set_ylim([1, 2.1])
axis[0].set_xticks(ratios_to_consider)
axis[0].set_xticklabels(ratios_to_consider)
axis[0].set_xlabel(r'$Z/\theta$', fontsize = 13)
axis[0].grid(alpha = 0.5)



fixed_N_uniform = []
fixed_N_constraint = []
fixed_N_occupation = []
fixed_N_alternate = []

mean_N_3 = []
for i in range(len(N_servers_to_consider)):
    fixed_N_uniform.append(N_server_normalized_uniform[:, :, i].flatten())
    fixed_N_constraint.append(N_server_normalized_constraint[:, :, i].flatten())
    fixed_N_occupation.append(N_server_normalized_occupation[:, :, i].flatten())
    fixed_N_alternate.append(N_server_normalized_alternate[:, :, i].flatten())

plot_colored_boxplot_N(1, axis, fixed_N_uniform, N_server_median_normalized_uniform, 0, labels, colors, linestyles_list, markers)
plot_colored_boxplot_N(1, axis, fixed_N_constraint, N_server_median_normalized_constraint, 1, labels, colors, linestyles_list, markers)
plot_colored_boxplot_N(1, axis, fixed_N_occupation, N_server_median_normalized_occupation, 2, labels, colors, linestyles_list, markers)
plot_colored_boxplot_N(1, axis, fixed_N_alternate, N_server_median_normalized_alternate, 3, labels, colors, linestyles_list, markers)

axis[1].set_xlabel('M (number of servers in the system)', fontsize = 13)
axis[1].set_xlim([2, 11])
axis[1].set_xticks(N_servers_to_consider)
axis[1].set_xticklabels(N_servers_to_consider)
axis[1].grid(alpha = 0.5)

plt.figlegend([(s_uniform, p_uniform), (s_constraint, p_constraint), (s_occupation, p_occupation), (s_alternate, p_alternate)], labels, loc='upper center', handlelength=3, ncol=2, handler_map={tuple: HandlerTuple(ndivide=None,pad = .1)}, fontsize = 13)
plt.subplots_adjust(left=0.12, top=0.85, hspace=.4, bottom = 0.1, right = 0.95)
plt.show()


#