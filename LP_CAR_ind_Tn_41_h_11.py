#!/usr/bin/env python
# coding: utf-8

# ### 1. Import libraries and formulate problems
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from LP_RH import *
# from Monte_Carlo import *


# ### 2. Upload file and Construct the Spatial-temporal Network G(N,A)
class Cost:
    def __init__(self):
        self.idle = 0
        self.reposition = 0
        self.vehicle = 0
        self.revenue = 0
        self.penalty = 0
        self.revenue_unit = 0

    def reset_cost(self):
        self.idle = 0
        self.reposition = 0
        self.vehicle = 0
        self.revenue = 0
        self.penalty = 0
        self.revenue_unit = 0


cost = Cost()


# generate random demands for a graph G(N,E)
def generate_trip_size(I, T, S):
    """
        Inputs:
            I number of regions
            T number of time node, thus T-1 time period, 1-indexed
            S number of scenarios
            type_of_dist type of distrbution, 1 is normal, 2 is bimodal
        Output:
            samples_demands: list with **NON-INTEGER** trip sizes
    """
    samples_demands = np.random.normal(6000, 1200, S)
    count, bins, ignored = plt.hist(samples_demands, 20, density=False)
    plt.xlabel('Demands')
    plt.ylabel('Number of Scenarios')
    plt.show()
    plt.savefig('Normal Distribution of 100 scenarios.png')
    return samples_demands


def initialize_demands(I, T, travel_time_mat):
    """
        Inputs:
            I number of regions
            T number of time node, thus T-1 time period, 1-indexed
        Output:
            g: dictionarys with the key as tuple: t_o,t_d,z_o,z_d, and value as 0
    """
    g = {}  # key:(t_o,t_d,z_o,z_d), value = 0

    # valid pairs satisfy the condition: t_d > t_o
    for t_o in range(1, T):
        for z_o in range(1, I + 1):
            # for t_d in range(t_o+1,T+1):
            for z_d in range(1, I + 1):
                if z_d == z_o:
                    continue
                t_d = t_o + int(travel_time_mat[z_o - 1][z_d - 1])
                if t_d > T:
                    continue
                g[str(t_o) + "_" + str(t_d) + "_" + str(z_o) + "_" + str(z_d)] = 0
    return g


def generate_samples(time_mat, I, T, S, samples_demands, dist_type):
    """
         Inputs:
            samples_demands: random trip size generated from function 'generate_trip_size'
        Output:
            g: n scenarios of g
    """
    g = {}
    i = 0
    zone_list = [i + 1 for i in range(I)]
    for demands in samples_demands:
        i += 1
        g_o = initialize_demands(I, T, time_mat)
        demands = int(demands)
        if dist_type == 1:
            start_time_list = np.random.normal((T - 1) / 2, (T - 1) * 0.25, 1000)
            start_time_list = [x for x in start_time_list if x >= 1 and x <= T - 1]
            if i == 1:
                count, bins, ignored = plt.hist(start_time_list, 100, density=True)
                plt.xticks(np.arange(1, T + 5, 4))
                plt.xlim([0, T + 1])
                plt.xlabel('Time Period')
                plt.ylabel('Demands Density')
                plt.show()
        elif dist_type == 2:
            x1 = np.random.gamma(4, 3, 1000)  # shape, scale
            x2 = np.random.gamma((T - 1) / 2, 2, 1000)
            start_time_list = np.hstack(([[x for x in x1 if x >= 1], [x for x in x2 if x <= T - 1]]))
            if i == 1:
                count, bins, ignored = plt.hist(start_time_list, 100, density=True)
                plt.xticks(np.arange(1, T + 5, 4))
                plt.xlim([0, T + 1])
                plt.xlabel('Time Period')
                plt.ylabel('Demands Density')
                plt.show()
        elif dist_type == 3:
            start_time_list = [i + 1 for i in range(T - 1)]
        else:
            print('no distribution type found')
            return
        while demands >= 0:
            start_time_node = int(random.choice(start_time_list))
            if start_time_node <= 0 or start_time_node > T - 1:
                continue
            start_zone = random.choice(zone_list)
            end_zone = random.choice(zone_list)
            if start_zone == end_zone:
                continue
            end_time_node = start_time_node + int(time_mat[start_zone - 1][end_zone - 1])
            if end_time_node > T:
                continue
            pair = str(int(start_time_node)) + '_' + str(int(end_time_node)) + '_' + str(int(start_zone)) + '_' + str(
                int(end_zone))
            #         print(pair)
            if pair in g_o:
                #             print('check valid pair')
                g_o[pair] += 1
                demands -= 1
        g[i] = g_o
    g_RH = {}
    for key in range(1, S * T + 1):
        g_RH[key] = {}
    for s in g:
        for pair, value in g[s].items():
            t_start, t_d, zone_start, zone_end = pair.split("_")
            pair_use = str((s - 1) * T + int(t_start)) + '_' + str((s - 1) * T + int(t_d)) + '_' + str(
                int(zone_start)) + '_' + str(int(zone_end))
            g_RH[int(t_start) + T * (s - 1)][pair_use] = value
    return g, g_RH


##########STAGE TWO
def LP_stage_two(t_o, I_n, horizon, overlap, fulfill_rate, requests, prev_cost, vehicle):
    '''
    inputs: t: time node
            parameters: parameters
            fulfill_rate: the lowest fulfillment rate
            demand_ful: dictionary which stores the demand fulfillment arc
            horizon: number of time period is being considered in one horizon
    outputs:
            cost:stage one cost
            a: supply at each location at time t = 0
            demand_ful: updated demand_ful
    '''
    t_end = min(t_o + horizon, T_n * S)  # horizon number of time period
    # Set up the LP problem
    prob = pulp.LpProblem("Stage_two", pulp.LpMinimize)

    ########!Set up the variables
    # variables include flow at idle arc, at relocation arc and supply at each node
    fi_dict, fr_dict, x_dict, fd_dict, requests_horizon = {}, {}, {}, {}, {}
    fd_obj = []

    for t in range(t_o, t_end + 1):
        if t != t_end:
            for pair in requests[t]:
                requests_horizon[pair] = requests[t][pair]
                t_start, t_d, zone_start, zone_end = pair.split("_")
                name_fd = "fd_" + pair
                fd_dict[name_fd] = pulp.LpVariable(name=name_fd, lowBound=0, upBound=requests_horizon[pair],
                                                   cat="Integer")
                fd_obj.append(fd_dict[name_fd] * (int(t_d) - int(t_start)))

                name_fr = "fr_" + pair
                fr_dict[name_fr] = pulp.LpVariable(name=name_fr, lowBound=0, cat="Integer")
                # fr (reposition flow):
                # name convention: fr_t_t'_i_j: reposition vehicle from region i to j can take 1 or 2 time period
                name_fr = "fr_" + pair
                fr_dict[name_fr] = pulp.LpVariable(name=name_fr, lowBound=0, cat="Integer")

    for t in range(t_o, t_end + 1):
        # if t in requests:
        fi_dict[t], x_dict[t] = {}, {}
        for i in range(1, I_n + 1):
            # fi (idle flow): the flow is always from i->i and t->t+1
            # name convention: fi_t_i
            if t != t_end:
                name_fi = "fi_" + str(t) + "_" + str(i)
                fi_dict[t][name_fi] = pulp.LpVariable(name=name_fi, lowBound=0, cat="Integer")

            # x (supply level):
            # name convention: x_t_i
            name_x = "x_" + str(t) + "_" + str(i)
            x_dict[t][name_x] = pulp.LpVariable(name=name_x, lowBound=0, cat="Integer")
            ############!Constraints
            # C1: Flow balance AT each node n_it
            # outbound:
            if t != t_end:  # truncate at t_end
                prob += (
                            # supply at n_i,t:
                                x_dict[t][name_x]

                                # idle flow from n_i,t to n_i,(t+1):
                                - fi_dict[t][name_fi]

                                # demands flow from n_i,t: fd_t_t'_i_j
                                - (pulp.lpSum(fd_dict[name] for name in fd_dict
                                              if name.split("_")[1] == str(t) and name.split("_")[3] == str(i))
                                   )
                                # reposition flow from n_i,t: fr_t_t'_i_j
                                - (pulp.lpSum(fr_dict[name] for name in fr_dict
                                              if name.split("_")[1] == str(t) and name.split("_")[3] == str(i))
                                   )
                        ) == 0

            # inbound:
            if t != t_o:
                prob += (
                            # supply at n_i,t:
                                x_dict[t][name_x]

                                # idle flow towards n_i,t:
                                - fi_dict[t - 1]["fi_" + str(t - 1) + "_" + str(i)]

                                # demands flow toward n_i,t:
                                - (pulp.lpSum(fd_dict[name] for name in fd_dict
                                              if name.split("_")[2] == str(t) and name.split("_")[4] == str(i))
                                   )

                                # reposition flow toward n_i,t:
                                - (pulp.lpSum(fr_dict[name] for name in fr_dict
                                              if name.split("_")[2] == str(t) and name.split("_")[4] == str(i))
                                   )

                        ) == 0

    # C3: total actual fulfilled demands >= total fulfillment rate*requests
    prob += pulp.lpSum(x_dict[t_o]) - vehicle == 0
    prob += pulp.lpSum(fd_dict) - fulfill_rate * pulp.lpSum(requests_horizon) >= 0
    idle_cost = h_i * pulp.lpSum(fi_dict)
    reposition_cost = h_r * pulp.lpSum(fr_dict)
    revenue = r * pulp.lpSum(fd_obj)
    penalty = p * (pulp.lpSum(requests_horizon) - pulp.lpSum(fd_dict))
    if t_o == 1:
        prob += (
            # stage-one cost: initialization
                c * vehicle
                # expected stage-two cost
                + 1 / S * (idle_cost + reposition_cost + penalty - revenue)
        )
    else:
        prob += (
                prev_cost
                # expected stage-two cost
                + 1 / S * (idle_cost + reposition_cost + penalty - revenue)
        )
    status = prob.solve()
    ############## Problem solved
    fd_unit = 0
    fd_with_time = 0
    fr_unit = 0
    request_unit = 0
    fi_unit = 0
    fr_with_time = 0
    for pair in requests_horizon.keys():
        t_start, t_d, zone_start, zone_end = pair.split("_")
        if t_end != T_n * S:
            if int(t_start) < t_end - overlap:
                fd_value = pulp.value(fd_dict["fd_" + pair])
                fd_unit += fd_value
                fd_with_time += fd_value * (int(t_d) - int(t_start))

                fr_value = pulp.value(fr_dict["fr_" + pair])
                fr_unit += fr_value
                fr_with_time += fr_value * (int(t_d) - int(t_start))
                request_unit += requests_horizon[pair]
        else:
            fd_value = pulp.value(fd_dict["fd_" + pair])
            fd_unit += fd_value
            fd_with_time += fd_value * (int(t_d) - int(t_start))

            fr_value = pulp.value(fr_dict["fr_" + pair])
            fr_unit += fr_value
            fr_with_time += fr_value * (int(t_d) - int(t_start))
            request_unit += requests_horizon[pair]
    for t in range(t_o, t_end - overlap + 1):
        fi_unit += pulp.value(pulp.lpSum(fi_dict[t]))
    cost.idle += fi_unit * h_i
    cost.revenue += fd_with_time * r
    cost.reposition += fr_with_time * h_r
    cost.penalty += p * (request_unit - fd_unit)
    cost.revenue_unit += fd_unit
    current_cost = (c * vehicle
                    + 1 / S *
                    (
                            cost.idle  # idle
                            - cost.revenue  # revenue
                            + cost.penalty  # penalty
                            + cost.reposition  # reposition
                    )
                    )
    return current_cost


##########Drivers
def driver_vehicle(i_n, horizon, overlap, parameters, requests, fulfill_rate):
    prev_vehicle = 0
    for t in range(1, T_n * (S) - horizon + 1, horizon - overlap):
        prev_vehicle = LP_stage_one(t_o=t, parameters=parameters, I_n=i_n, T_n=T_n, horizon=horizon, overlap=overlap,
                                    fulfill_rate=fulfill_rate, requests=requests, prev_vehicle=prev_vehicle)
    return prev_vehicle


def driver_cost(i_n, horizon, overlap, parameters, requests, fulfill_rate, vehicle):
    T = T_n
    cost.reset_cost()
    prev_cost = 0
    for t in range(1, T * S - horizon + 1, horizon - overlap):
        prev_cost = LP_stage_two(t_o=t, I_n=i_n, horizon=horizon, overlap=overlap,
                                 fulfill_rate=fulfill_rate, requests=requests, prev_cost=prev_cost, vehicle=vehicle)
    return prev_cost


def major_driver_RH(parameters, i_n, requests, horizon, overlap, S, range_rate=np.linspace(0.5, 1, num=6)):
    cache_lst = []  # cache list for each fulfillment rate
    for fulfill_rate in range_rate:
        print(fulfill_rate)
        fleet_size = driver_vehicle(i_n, horizon, overlap, parameters, requests, fulfill_rate)
        final_cost = driver_cost(i_n, horizon, overlap, parameters, requests, fulfill_rate, vehicle=fleet_size)
        cache_temp = {}
        cache_temp['actual_rate_list'] = fulfill_rate
        cache_temp['number_of_vehicles'] = fleet_size
        cache_temp["actual_fulfilment_rate"] = 0
        for dic in requests.values():
            cache_temp["actual_fulfilment_rate"] += sum(dic.values())
        cache_temp["actual_fulfilment_rate"] = cost.revenue_unit / cache_temp["actual_fulfilment_rate"]
        cache_temp['avg_cost_idle'] = cost.idle / S
        cache_temp['avg_cost_revenue'] = cost.revenue / S
        cache_temp['avg_penalty'] = cost.penalty / S
        cache_temp["total_cost"] = final_cost
        cache_temp['avg_cost_reposition'] = cost.reposition / S
        cache_lst.append(cache_temp)
    print("Done!")
    return cache_lst


# ### 3. Set up LP Model
# 
# ### 3.1 Set up and Solve the two-stage stochastic Problem 

# ### 3.2 Run the test
# 
# ### 3.2.1 Set up variables
# #### Variables:
#     - depreciation value c per vehicle per day
#     - positive impact r, revenue per time period per car
#     - reposition cost h_r, cost per time period per car
#     - idle cost h_i, cost per time period per car
#     - p penalty if demand is not fulfilled per request
#     - budget B per day,
#     - Zone number I_n
#     - Time period T, time point T_n
#     - Number of scenarios: S
#     
# 

# ### 4. Results
# 
# **Case 1**: Relationship between the **actual fulfillment rate** and the **total number of vehicle**
def plot_results(cache_lst):
    fig, axs = plt.subplots()
    x0 = np.array([i for i in range(0, 6)])
    x = np.array([round(cache["actual_rate_list"], 3) for cache in cache_lst])
    y_overall_rate = np.array([round(cache['actual_fulfilment_rate'], 3) for cache in cache_lst])
    y2 = np.array([cache["number_of_vehicles"] for cache in cache_lst])
    y3 = np.array([round(cache["total_cost"], 0) for cache in cache_lst])
    width = 0.3
    #     axs[0,0].bar(x0,x)
    #     axs[0,0].bar(x0,y1)
    bar1 = axs.bar(x0 - width / 2, x, width=width, label="Actual minimum fulfilment")
    bar2 = axs.bar(x0 + width / 2, y_overall_rate, width=width,
                      label="Actual overall fulfilment rate for all scenarios")
    axs.set_title("Fullfilment Rate at different Minimum rate restraints", fontdict={'fontname': 'Comic Sans MS'})
    axs.set(ylabel="Fulfilment rate")
    axs.set(xlabel="Minimum Fulfilment Constraint")
    axs.set_xticks(x0)
    axs.set_xticklabels(x)
    axs.legend()
    # axs.legend(bbox_to_anchor=(0, 1), loc='best', ncol=1)

    def autolabel_bar(rects):
        for rect in rects:
            height = rect.get_height()
            axs.annotate("{}".format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')

    autolabel_bar(bar1)
    autolabel_bar(bar2)

    def autolabel_plot(plot, x, y, n):
        for i in x0:
            plot.annotate("{}".format(y[i]), xy=(x[i], y[i]), xytext=(0, 3), textcoords="offset points",
                          ha='center', va='bottom')

    fig1, ax1 = plt.subplots()
    ax1.plot(y_overall_rate, y2, 'r^--', label="Number of Vehicles")
    ax1.set_title("Number of Vehicles at different actual average fulfilment rate",
                     fontdict={'fontname': 'Comic Sans MS'})
    ax1.set(xlabel="Actual Overall fulfilment")
    ax1.set(ylabel="Number of Vehicles")
    ax1.legend()
    #     axs[1].legend(bbox_to_anchor=(0, 1), loc='best', ncol=1)
    autolabel_plot(ax1, y_overall_rate, y2, x0)

    fig2, ax2 = plt.subplots()
    ax2.plot(y_overall_rate, y3, 'b^--', label="Total Cost")
    ax2.set_title("Total cost at different actual overall fulfilment rate", fontdict={'fontname': 'Comic Sans MS'})
    ax2.set(xlabel="Actual Overall fulfilment")
    ax2.set(ylabel="Total daily cost")
    ax2.legend()
    autolabel_plot(ax2, y_overall_rate, y3, x0)

    fig3, ax3 = plt.subplots()
    ax3.plot(y2, y3, 'r.--', label="Total cost")
    ax3.set_title("Total cost at different number of initialized vechiles ", fontdict={'fontname': 'Comic Sans MS'})
    ax3.set(xlabel="Total number of vehicles")
    ax3.set(ylabel="Total daily cost")
    ax3.legend()
    autolabel_plot(ax3, y2, y3, x0)
    plt.show()


def plot_results_SAA(cache_lst):
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(24, 20)
    x0 = np.array([i for i in range(0, 8)])
    x = np.array([round(cache["min_fulfilment_rate"], 3) for cache in cache_lst])
    y1 = np.array([round(min(cache["actual_rate_list"]), 3) for cache in cache_lst])  # Actual minimum fulfilment rate
    y_overall_rate = np.array([round(cache['actual_fulfilment_rate'], 3) for cache in cache_lst])
    y2 = np.array([cache["number_of_vehicles"] for cache in cache_lst])
    y3 = np.array([round(cache["total_cost"], 0) for cache in cache_lst])
    width = 0.3
    #     axs[0,0].bar(x0,x)
    #     axs[0,0].bar(x0,y1)
    bar2 = axs[0, 0].bar(x0 - width / 2, y1, width=width, label="Actual minimum fulfilment")
    bar3 = axs[0, 0].bar(x0 + width / 2, y_overall_rate, width=width,
                         label="Actual overall fulfilment rate for all scenarios")
    axs[0, 0].set_title("Fullfilment Rate at different Minimum rate restraints", fontdict={'fontname': 'Comic Sans MS'})
    axs[0, 0].set(ylabel="Fulfilment rate")
    axs[0, 0].set(xlabel="Minimum Fulfilment Constraint")
    axs[0, 0].set_xticks(x0)
    axs[0, 0].set_xticklabels(x)
    axs[0, 0].legend()

    def autolabel_bar(rects):
        for rect in rects:
            height = rect.get_height()
            axs[0, 0].annotate("{}".format(height),
                               xy=(rect.get_x() + rect.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom')

    autolabel_bar(bar2)
    autolabel_bar(bar3)

    def autolabel_plot(plot, x, y, n):
        for i in x0:
            plot.annotate("{}".format(y[i]), xy=(x[i], y[i]), xytext=(0, 3), textcoords="offset points",
                          ha='center', va='bottom')

    axs[0, 1].plot(y_overall_rate, y2, 'r^--', label="Number of Vehicles")
    axs[0, 1].set_title("Number of Vehicles at different actual average fulfilment rate",
                        fontdict={'fontname': 'Comic Sans MS'})
    axs[0, 1].set(xlabel="Actual Overall fulfilment")
    axs[0, 1].set(ylabel="Number of Vehicles")
    axs[0, 1].legend()
    autolabel_plot(axs[0, 1], y_overall_rate, y2, x0)
    axs[1, 0].plot(y_overall_rate, y3, 'b^--', label="Total Cost")
    axs[1, 0].set_title("Total cost at different actual overall fulfilment rate",
                        fontdict={'fontname': 'Comic Sans MS'})
    axs[1, 0].set(xlabel="Actual Overall fulfilment")
    axs[1, 0].set(ylabel="Total daily cost")
    axs[1, 0].legend()
    autolabel_plot(axs[1, 0], y_overall_rate, y3, x0)
    axs[1, 1].plot(y2, y3, 'r.--', label="Total cost")
    axs[1, 1].set_title("Total cost at different number of initialized vechiles ",
                        fontdict={'fontname': 'Comic Sans MS'})
    axs[1, 1].set(xlabel="Total number of vehicles")
    axs[1, 1].set(ylabel="Total daily cost")
    axs[1, 1].legend()
    autolabel_plot(axs[1, 1], y2, y3, x0)
    plt.show()


def plot_results_price(cache_lst):
    fig, axs = plt.subplots()
    fig.set_size_inches(10, 8)
    n = np.array([i for i in range(2, 7)])
    x0 = n
    y3 = np.array([round(cache["total_cost"], 0) for cache in cache_lst])
    mark = np.array([5500 for i in range(2, 7)])

    def autolabel_plot(plot, x, y, n):
        for i in n:
            plot.annotate("{}".format(y[i]), xy=(x[i], y[i]), xytext=(0, 3), textcoords="offset points",
                          ha='center', va='bottom')

    axs.plot(x0, y3, 'b^--', label="Total Cost ($)")
    axs.plot(x0, mark, label='Daily budget ($5500)', color='r')
    axs.set_title("Total cost at different unit positive impact r ($)", fontdict={'fontname': 'Comic Sans MS'})
    axs.set(xlabel="Unit positive impact r ($/Time Period)")
    axs.set(ylabel="Total daily cost ($/day)")
    axs.legend()
    #     autolabel_plot(axs,x0,y3,n)
    plt.show()


def result_table(cache):
    columns = ['Number of Vehicle', 'Min Fulfill Rate', "Actual Fulfil Rate", 'Idle Cost $', 'Reposition Cost $',
               'Revenue generated $', 'Penalty $', 'Total cost $']
    df = pd.DataFrame(columns=columns, index=None)
    for i, cache in enumerate(cache):
        df = df.append({
            columns[1]: round((cache["actual_rate_list"]), 3),
            columns[0]: int(cache['number_of_vehicles']),
            columns[2]: round(cache["actual_fulfilment_rate"], 3),
            columns[3]: round(cache['avg_cost_idle'], 2),
            columns[4]: round(cache['avg_cost_reposition'], 2),
            columns[5]: round(cache['avg_cost_revenue'], 2),
            columns[6]: round(cache['avg_penalty'], 2),
            columns[7]: round(cache["total_cost"], 2)}, ignore_index=True)
    return df


def comparison(cache_list):
    fig, axs = plt.subplots()
    fig.set_size_inches(10, 8)
    labels = ['Uniform', 'Normal', 'Bimodal']
    colors = ['b', 'orange', 'green']
    for i, cache_lst in enumerate(cache_list):
        y_overall_rate = np.array([round(cache['total_cost'], 3) for cache in cache_lst])
        y_num_vehicle = np.array([cache["number_of_vehicles"] for cache in cache_lst])
        #         y_total_coast=np.array([round(cache["total_cost"],0) for cache in cache_lst])
        axs.plot(y_num_vehicle, y_overall_rate, '^--', label=labels[i], color=colors[i])
    #         axs.plot(y_overall_rate,y_total_coast,label = labels[i])
    #     axs.set_title("Total cost at different unit positive impact r ($)",fontdict = {'fontname':'Comic Sans MS'})
    axs.set(xlabel="Fleet size")
    axs.set(ylabel="Total daily cost ($/day)")
    axs.legend()
    #     autolabel_plot(axs,x0,y3,n)
    plt.show()


def main(city, poi, parameters=dict(c=54, r=5, h_r=3, h_i=0.5, B=5500, p=2, S=90, t=15)):
    global T_n, c, r, h_r, h_i, p, B, S
    c = parameters["c"]
    r = parameters["r"]
    h_r = parameters["h_r"]
    h_i = parameters["h_i"]
    p = parameters["p"]
    B = parameters["B"]
    S = parameters["S"]
    t = int(parameters["t"])
    T_n = 600 // t + 1

    travel_time_mat = pd.read_excel('travel_time_mat.xlsx', sheet_name=city, index_col=0)
    travel_time_mat = travel_time_mat.loc[travel_time_mat.Type.isin(poi), travel_time_mat.loc['Type'].isin(poi)]
    travel_time_mat = travel_time_mat.to_numpy() // t + 1
    i_n = len(travel_time_mat)

    samples_demands = generate_trip_size(i_n, T_n, S)
    g_normal = generate_samples(travel_time_mat, i_n, T_n, S, samples_demands, 1)  # g_normal is a list [g,g_RH]
    g_bi = generate_samples(travel_time_mat, i_n, T_n, S, samples_demands, dist_type=2)
    g_uni = generate_samples(travel_time_mat, i_n, T_n, S, samples_demands, dist_type=3)
    # ### Impact of Horizon
    horizon = 11
    overlap = 3
    fulfill_rate = 1

    cache_normal = major_driver_RH(parameters, i_n, g_normal[1], horizon, overlap, S)
    # cache_bi = major_driver_RH(requests=g_bi[1])
    # cache_uni = major_driver_RH(requests=g_uni[1])

    plot_results(cache_normal)
    # plot_results(cache_bi)
    # plot_results(cache_uni)

    # #### Optimal Cost vs Unit deman flow cost 
    # cache_price = driver_price()
    # plot_results_price(cache_price)

    # df_normal_RH=result_table(cache_normal)
    # df_uni_RH=result_table(cache_uni)
    # df_bi_RH=result_table(cache_bi)
    # comparison([cache_uni, cache_normal, cache_bi])


if __name__ == '__main__':
    main('Columbia', [1, 4, 6])
