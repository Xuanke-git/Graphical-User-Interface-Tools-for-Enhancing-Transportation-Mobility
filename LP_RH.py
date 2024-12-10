import pulp


##########STAGE ONE
def LP_stage_one(t_o, parameters, I_n, T_n, horizon,overlap,fulfill_rate,requests,prev_vehicle):
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
    c = parameters["c"]
    r=parameters["r"]
    h_r= parameters["h_r"]
    h_i = parameters["h_i"]
    p = parameters["p"]
    B=parameters["B"]
    S=parameters["S"]
    t_end = min(t_o+horizon,T_n*S) #horizon number of time period
#     print(t_end)
    #Set up the LP problem
    prob = pulp.LpProblem("Stage_One", pulp.LpMinimize)

    ########!Set up the variables
    #variables include flow at idle arc, at relocation arc and supply at each node
    fi_dict, fr_dict, x_dict, fd_dict,requests_horizon = {},{},{},{},{}
    #x_dict = {   name: pulp.variables   }
    #fr_dict = { name:pulp.variables}
    #fd_dict, requests_horizon={s:{pair:integer}}
    fd_obj= []

    #fd (fulfilled demand flow): the flow on the arc should be no greater than the requests
    #name convention: s_fd_t_t'_i_j

    for t in range(t_o,t_end+1):
        if t!=t_end:
            for pair in requests[t]:
                requests_horizon[pair] = requests[t][pair]
                t_start,t_d, zone_start,zone_end = pair.split("_")
                name_fd = "fd_"+pair
                fd_dict[name_fd]= pulp.LpVariable(name=name_fd,lowBound = 0,upBound = requests_horizon[pair],cat="Integer")
                fd_obj.append(fd_dict[name_fd]*(int(t_d)-int(t_start)))
                name_fr = "fr_"+pair
                fr_dict[name_fr]= pulp.LpVariable(name=name_fr,lowBound = 0,cat="Integer")
            #fr (reposition flow):
            #name convention: fr_t_t'_i_j: reposition vehicle from region i to j can take 1 or 2 time period
                name_fr = "fr_"+pair
                fr_dict[name_fr]= pulp.LpVariable(name=name_fr,lowBound = 0,cat="Integer")

    for t in range(t_o,t_end+1):
        # if t in requests:
        fi_dict[t],x_dict[t] = {},{}
        for i in range(1,I_n+1):
            #fi (idle flow): the flow is always from i->i and t->t+1
            #name convention: fi_t_i
            if t!=t_end:
                name_fi="fi_"+str(t)+"_"+str(i)
                fi_dict[t][name_fi]= pulp.LpVariable(name=name_fi,lowBound = 0,cat="Integer")

            #x (supply level):
            #name convention: x_t_i
            name_x="x_"+str(t)+"_"+str(i)
            x_dict[t][name_x]=pulp.LpVariable(name=name_x,lowBound = 0,cat="Integer")

   ############!Constraints

            #C1: Flow balance AT each node n_it
            #outbound:
            if t!=t_end: #truncate at t_end
                prob+=(
                    #supply at n_i,t:
                    x_dict[t][name_x]

                    #idle flow from n_i,t to n_i,(t+1):
                    -fi_dict[t][name_fi]

                    #demands flow from n_i,t: fd_t_t'_i_j
                    -(pulp.lpSum(fd_dict[name] for name in fd_dict
                                 if name.split("_")[1] == str(t) and name.split("_")[3]==str(i))
                     )
                    #reposition flow from n_i,t: s_fr_t_t'_i_j
                    -(pulp.lpSum(fr_dict[name] for name in fr_dict
                                 if name.split("_")[1] == str(t) and name.split("_")[3]==str(i))
                     )
                )==0

            #inbound:
            if t!=t_o:
                prob+= (
                    #supply at n_i,t:
                    x_dict[t][name_x]

                    #idle flow towards n_i,t:
                    -fi_dict[t-1]["fi_"+str(t-1)+"_"+str(i)]

                    #demands flow toward n_i,t:
                    -(pulp.lpSum(fd_dict[name] for name in fd_dict
                                 if name.split("_")[2] == str(t) and name.split("_")[4]==str(i))
                     )

                     #reposition flow toward n_i,t:
                    -(pulp.lpSum(fr_dict[name] for name in fr_dict
                                 if name.split("_")[2] == str(t) and name.split("_")[4]==str(i))
                     )

                )==0

#C3: total actual fulfilled demands >= total fulfillment rate*requests
    prob+=pulp.lpSum(x_dict[t_o])-prev_vehicle>=0
    prob+= pulp.lpSum(fd_dict)-fulfill_rate*pulp.lpSum(requests_horizon)>=0

    Total_vehicle = pulp.lpSum(x_dict[t_o]) #[s][t][name]
    idle_cost = h_i*pulp.lpSum(fi_dict)
    reposition_cost = h_r*pulp.lpSum(fr_dict)
    revenue = r*pulp.lpSum(fd_obj)
    penalty=p*(pulp.lpSum(requests_horizon)-pulp.lpSum(fd_dict))


    prob+= (
        #stage-one cost: initialization
        c*Total_vehicle
        #expected stage-two cost
        +1/S*(idle_cost+reposition_cost+penalty-revenue)
    )

    status=prob.solve()
    ############## Problem solved
    return pulp.value(Total_vehicle)
