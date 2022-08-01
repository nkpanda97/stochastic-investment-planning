# -*- coding: utf-8 -*-
# @Time : 28/07/22 11:29 PM
# @Author : nkpanda
# @FileName: helper_functions.py
# @Git ：https://github.com/nkpanda97

import numpy as np
from pyomo.environ import *
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
# from pysp import *
from tabulate import tabulate
import mpisppy.utils.sputils as sputils
import pandas as pd
from itertools import product
from mpisppy.opt.lshaped import LShapedMethod
from numpy.random import default_rng
from tqdm import tqdm


# https://mpi-sppy.readthedocs.io/en/latest/examples.html


# ------------------------------ Models for Q(a) MOD1 ------------------------------------------------------------------

# Create deeterministic model instance
def create_model_qa(capacity_per_mode=[5, 3, 2], capacity_max=[7, 4, 3]):
    """
    This function creates a diterminestic model for the MOD1 model given in the report and jupyter
    notebook
    """
    model = ConcreteModel()

    # Declaring SETS
    model.n = Set(initialize=np.arange(4))  # Number of technologies
    model.k = Set(initialize=np.arange(3))  # Number of modes

    # Declaring Variables
    model.x = Var(model.n, within=NonNegativeReals)  # Decision variable for capacity of each techology
    model.y = Var(model.n, model.k,
                  within=NonNegativeReals)  # Capacity of technology i={0,1,2,3} effectively used in mode j={0,1,2}

    # Defining Parameters
    model.capacity_max = Param(model.k, within=NonNegativeReals,
                               mutable=True)  # Maximum possible value of \xi_j, j={0,1,2}
    model.capacity_per_mode = Param(model.k, within=NonNegativeReals,
                                    mutable=True)  # Stochastic load demand for model j but value is known as the expected value
    model.tech_cost = Param(model.n, within=NonNegativeReals, mutable=True)  # Cost of each technology per unit capacity
    model.operation_cost = Param(model.n, within=NonNegativeReals,
                                 mutable=True)  # Cost of each technology for production of per unit capacity
    model.max_investment = 120  # Max budget
    model.T = Param(model.k, initialize=[10, 6, 1], mutable=False)

    # Filling parameter
    operating_costs = [4, 4.5, 3.2, 5.5]
    capital_cost = [10, 7, 16, 6]

    for i in model.n:
        model.operation_cost[i] = operating_costs[i]
        model.tech_cost[i] = capital_cost[i]

    for j in model.k:
        model.capacity_per_mode[j] = capacity_per_mode[j]
        model.capacity_max[j] = capacity_max[j]

    def max_demand(model):
        return sum(model.x[i] for i in model.n) >= sum(model.capacity_max[j] for j in model.k)

    def budget_constraint(model):
        return sum(model.tech_cost[i] * model.x[i] for i in model.n) <= model.max_investment

    def capacity_per_mode(model, i):
        return sum(model.y[i, j] for j in model.k) <= model.x[i]

    def demand_bal(model, j):
        return sum(model.y[i, j] for i in model.n) >= model.capacity_per_mode[j]

    model.con1 = Constraint(rule=max_demand)
    model.con2 = Constraint(rule=budget_constraint)
    model.con3 = Constraint(model.n, rule=capacity_per_mode)
    model.con4 = Constraint(model.k, rule=demand_bal)

    def obj_second_stage(model):
        expr = sum(model.operation_cost[i] * sum(model.T[j] * model.y[i, j] for j in model.k) for i in model.n)
        return expr

    def obj_first_stage(model):
        expr = sum(model.tech_cost[i] * model.x[i] for i in model.n)
        return expr

    model.ObjCost_first_stage = Expression(rule=obj_first_stage)
    model.ObjCost_second_stage = Expression(rule=obj_second_stage)

    def total_cost_rule(model):
        return model.ObjCost_first_stage + model.ObjCost_second_stage

    model.Total_Cost_Objective = Objective(rule=total_cost_rule, sense=minimize)
    return model


# Createmodel to calculate EEV
def create_eev_model_qa(x_bar=[1, 2, 3, 4], capacity_per_mode=[5, 3, 2], capacity_max=[7, 4, 3]):
    """
    This function creates a diterminestic model for the MOD1.1 model  to solve for the EEEV value given in the report and
    jupyter notebook
    """
    model = ConcreteModel()

    # Declaring SETS
    model.n = Set(initialize=np.arange(4))  # Number of technologies
    model.k = Set(initialize=np.arange(3))  # Number of modes

    # Declaring Variables
    model.y = Var(model.n, model.k,
                  within=NonNegativeReals)  # Capacity of technology i={0,1,2,3} effectively used in mode j={0,1,2}

    # Defining Parameters
    model.x_bar = Param(model.n, within=NonNegativeReals, mutable=True)  # Expected value solution
    model.capacity_max = Param(model.k, within=NonNegativeReals,
                               mutable=True)  # Maximum possible value of \xi_j, j={0,1,2}
    model.capacity_per_mode = Param(model.k, within=NonNegativeReals,
                                    mutable=True)  # Stochastic load demand for model j but value is known as the expected value
    model.tech_cost = Param(model.n, within=NonNegativeReals, mutable=True)  # Cost of each technology per unit capacity
    model.operation_cost = Param(model.n, within=NonNegativeReals,
                                 mutable=True)  # Cost of each technology for production of per unit capacity
    model.max_investment = 120  # Max budget
    model.T = Param(model.k, initialize=[10, 6, 1], mutable=False)

    # Filling parameter
    operating_costs = [4, 4.5, 3.2, 5.5]

    for i in model.n:
        model.operation_cost[i] = operating_costs[i]
        model.x_bar[i] = x_bar[i]

    for j in model.k:
        model.capacity_per_mode[j] = capacity_per_mode[j]

    def capacity_per_mode(model, i):
        return sum(model.y[i, j] for j in model.k) <= model.x_bar[i]

    def demand_bal(model, j):
        return sum(model.y[i, j] for i in model.n) >= model.capacity_per_mode[j]

    model.con3 = Constraint(model.n, rule=capacity_per_mode)
    model.con4 = Constraint(model.k, rule=demand_bal)

    def obj_second_stage(model):
        expr = sum(model.operation_cost[i] * sum(model.T[j] * model.y[i, j] for j in model.k) for i in model.n)
        return expr

    model.obj = Objective(expr=sum(model.operation_cost[i] * sum(model.T[j] * model.y[i, j] for j in model.k)
                                   for i in model.n), sense=minimize)
    return model


# ------------------------------ Models for Q(b) MOD2------------------------------------------------------------------

# Create deterministic model instance for Q(b)
def create_model_qb(random_parameters=np.array([1, 1, 1, 1, 1, 1, 1])):
    """
    This function creates a diterminestic model for the MOD2 model given in the report and jupyter
    notebook
    """
    # random_parameters=[a1,a2,a3,a4,xi1,xi2,xi3]
    alpha_i = list(random_parameters[0:4]) + [1]
    capacity_per_mode = list(random_parameters[4:7])
    model = ConcreteModel()

    # Declaring SETS------------------------------------------------------------------------------
    model.n = Set(initialize=np.arange(5))  # Number of technologies
    model.k = Set(initialize=np.arange(3))  # Number of modes

    # Declaring Variables------------------------------------------------------------------------------
    model.x = Var(model.n, within=NonNegativeReals)  # Decision variable for capacity of each techology
    model.y = Var(model.n, model.k,
                  within=NonNegativeReals)  # Capacity of technology i={0,1,2,3} effectively used in mode j={0,1,2}

    # Defining Parameters------------------------------------------------------------------------------
    model.capacity_per_mode = Param(model.k, within=NonNegativeReals,
                                    mutable=True)  # Stochastic load demand for model j but value is known as the expected value
    model.tech_cost = Param(model.n, within=NonNegativeReals, mutable=True)  # Cost of each technology per unit capacity
    model.operation_cost = Param(model.n, within=NonNegativeReals,
                                 mutable=True)  # Cost of each technology for production of per unit capacity
    model.max_investment = 120  # Max budget
    model.T = Param(model.k, initialize=[10, 6, 1], mutable=False)
    model.alpha = Param(model.n, mutable=True)  # Random parameter for operational availability

    # Filling parameter-----------------------------------------------------------------------------------
    operating_costs = [4, 4.5, 3.2, 5.5, 10]
    capital_cost = [10, 7, 16, 6, 0]

    for i in model.n:
        model.operation_cost[i] = operating_costs[i]
        model.tech_cost[i] = capital_cost[i]
        model.alpha[i] = alpha_i[i]

    for j in model.k:
        model.capacity_per_mode[j] = capacity_per_mode[j]

    def budget_constraint(model):
        return sum(model.tech_cost[i] * model.x[i] for i in model.n) <= model.max_investment

    def capacity_per_mode(model, i):
        return sum(model.y[i, j] for j in model.k) <= model.alpha[i] * model.x[i]

    def demand_bal(model, j):
        return sum(model.y[i, j] for i in model.n) >= model.capacity_per_mode[j]

    model.con2 = Constraint(rule=budget_constraint)
    model.con3 = Constraint(model.n, rule=capacity_per_mode)
    model.con4 = Constraint(model.k, rule=demand_bal)

    def obj_second_stage(model):
        expr = sum(model.operation_cost[i] * sum(model.T[j] * model.y[i, j] for j in model.k) for i in model.n)
        return expr

    def obj_first_stage(model):
        expr = sum(model.tech_cost[i] * model.x[i] for i in model.n)
        return expr

    model.ObjCost_first_stage = Expression(rule=obj_first_stage)
    model.ObjCost_second_stage = Expression(rule=obj_second_stage)

    def total_cost_rule(model):
        return model.ObjCost_first_stage + model.ObjCost_second_stage

    model.Total_Cost_Objective = Objective(rule=total_cost_rule, sense=minimize)
    return model


# Createmodel to calculate EEV for Qb
def create_eev_model_qb(x_bar=[1, 2, 3, 4, 5], random_parameters=np.array([1, 1, 1, 1, 1, 1, 1])):
    """
    This function creates a diterminestic model for the MOD1.1 model  to solve for the EEEV value given in the report and
    jupyter notebook
    """
    # random_parameters=[a1,a2,a3,a4,xi1,xi2,xi3]
    alpha_i = list(random_parameters[0:4]) + [1]
    capacity_per_mode = list(random_parameters[4:7])
    model = ConcreteModel()

    # Declaring SETS
    model.n = Set(initialize=np.arange(5))  # Number of technologies
    model.k = Set(initialize=np.arange(3))  # Number of modes

    # Declaring Variables
    model.y = Var(model.n, model.k,
                  within=NonNegativeReals)  # Capacity of technology i={0,1,2,3} effectively used in mode j={0,1,2}

    # Defining Parameters
    model.x_bar = Param(model.n, within=NonNegativeReals, mutable=True)  # Expected value solution
    model.capacity_per_mode = Param(model.k, within=NonNegativeReals,
                                    mutable=True)  # Stochastic load demand for model j but value is known as the expected value
    model.tech_cost = Param(model.n, within=NonNegativeReals, mutable=True)  # Cost of each technology per unit capacity
    model.operation_cost = Param(model.n, within=NonNegativeReals,
                                 mutable=True)  # Cost of each technology for production of per unit capacity
    model.max_investment = 120  # Max budget
    model.T = Param(model.k, initialize=[10, 6, 1], mutable=False)
    model.alpha = Param(model.n, mutable=True)  # Random parameter for operational availability

    # Filling parameter
    operating_costs = [4, 4.5, 3.2, 5.5, 10]

    for i in model.n:
        model.operation_cost[i] = operating_costs[i]
        model.x_bar[i] = x_bar[i]
        model.alpha[i] = alpha_i[i]

    for j in model.k:
        model.capacity_per_mode[j] = capacity_per_mode[j]

    def capacity_per_mode(model, i):
        return sum(model.y[i, j] for j in model.k) <=  model.alpha[i]*model.x_bar[i]

    def demand_bal(model, j):
        return sum(model.y[i, j] for i in model.n) >= model.capacity_per_mode[j]

    model.con3 = Constraint(model.n, rule=capacity_per_mode)
    model.con4 = Constraint(model.k, rule=demand_bal)


    model.obj = Objective(expr=sum(model.operation_cost[i] * sum(model.T[j] * model.y[i, j] for j in model.k)
                                   for i in model.n), sense=minimize)
    return model


def create_model_qc(random_parameters=np.array([1, 1, 1, 1, 1, 1, 1])):
    """
    This function creates a diterminestic model for the MOD2 model given in the report and jupyter
    notebook
    """
    # random_parameters=[tau1, tau2, a1,a2,a3,a4,xi1,xi2,xi3]
    tau_i = [10] + list(random_parameters[0:2])
    alpha_i = list(random_parameters[2:6]) + [1]
    capacity_per_mode = list(random_parameters[6:9])
    model = ConcreteModel()

    # Declaring SETS------------------------------------------------------------------------------
    model.n = Set(initialize=np.arange(5))  # Number of technologies
    model.k = Set(initialize=np.arange(3))  # Number of modes

    # Declaring Variables------------------------------------------------------------------------------
    model.x = Var(model.n, within=NonNegativeReals)  # Decision variable for capacity of each techology
    model.y = Var(model.n, model.k,
                  within=NonNegativeReals)  # Capacity of technology i={0,1,2,3} effectively used in mode j={0,1,2}

    # Defining Parameters------------------------------------------------------------------------------
    model.capacity_per_mode = Param(model.k, within=NonNegativeReals,
                                    mutable=True)  # Stochastic load demand for model j but value is known as the expected value
    model.tech_cost = Param(model.n, within=NonNegativeReals, mutable=True)  # Cost of each technology per unit capacity
    model.operation_cost = Param(model.n, within=NonNegativeReals,
                                 mutable=True)  # Cost of each technology for production of per unit capacity
    model.max_investment = 120  # Max budget
    model.T = Param(model.k,mutable=True)
    model.alpha = Param(model.n, mutable=True)  # Random parameter for operational availability


    # Filling parameter-----------------------------------------------------------------------------------
    operating_costs = [4, 4.5, 3.2, 5.5, 10]
    capital_cost = [10, 7, 16, 6, 0]

    for i in model.n:
        model.operation_cost[i] = operating_costs[i]
        model.tech_cost[i] = capital_cost[i]
        model.alpha[i] = alpha_i[i]

    for j in model.k:
        model.capacity_per_mode[j] = capacity_per_mode[j]
        model.T[j] = tau_i[j]

    def budget_constraint(model):
        return sum(model.tech_cost[i] * model.x[i] for i in model.n) <= model.max_investment

    def capacity_per_mode(model, i):
        return sum(model.y[i, j] for j in model.k) <= model.alpha[i] * model.x[i]

    def demand_bal(model, j):
        return sum(model.y[i, j] for i in model.n) >= model.capacity_per_mode[j]

    model.con2 = Constraint(rule=budget_constraint)
    model.con3 = Constraint(model.n, rule=capacity_per_mode)
    model.con4 = Constraint(model.k, rule=demand_bal)

    def obj_second_stage(model):
        expr = sum(model.operation_cost[i] * sum(model.T[j] * model.y[i, j] for j in model.k) for i in model.n)
        return expr

    def obj_first_stage(model):
        expr = sum(model.tech_cost[i] * model.x[i] for i in model.n)
        return expr

    model.ObjCost_first_stage = Expression(rule=obj_first_stage)
    model.ObjCost_second_stage = Expression(rule=obj_second_stage)

    def total_cost_rule(model):
        return model.ObjCost_first_stage + model.ObjCost_second_stage

    model.Total_Cost_Objective = Objective(rule=total_cost_rule, sense=minimize)
    return model


# Createmodel to calculate EEV for Qb
def create_eev_model_qc(x_bar=[1, 2, 3, 4, 5], random_parameters=np.array([1, 1, 1, 1, 1, 1, 1])):
    """
    This function creates a diterminestic model for the MOD1.1 model  to solve for the EEEV value given in the report and
    jupyter notebook
    """
    # random_parameters=[a1,a2,a3,a4,xi1,xi2,xi3]
    alpha_i = list(random_parameters[0:4]) + [1]
    capacity_per_mode = list(random_parameters[4:7])
    model = ConcreteModel()

    # Declaring SETS
    model.n = Set(initialize=np.arange(5))  # Number of technologies
    model.k = Set(initialize=np.arange(3))  # Number of modes

    # Declaring Variables
    model.y = Var(model.n, model.k,
                  within=NonNegativeReals)  # Capacity of technology i={0,1,2,3} effectively used in mode j={0,1,2}

    # Defining Parameters
    model.x_bar = Param(model.n, within=NonNegativeReals, mutable=True)  # Expected value solution
    model.capacity_per_mode = Param(model.k, within=NonNegativeReals,
                                    mutable=True)  # Stochastic load demand for model j but value is known as the expected value
    model.tech_cost = Param(model.n, within=NonNegativeReals, mutable=True)  # Cost of each technology per unit capacity
    model.operation_cost = Param(model.n, within=NonNegativeReals,
                                 mutable=True)  # Cost of each technology for production of per unit capacity
    model.max_investment = 120  # Max budget
    model.T = Param(model.k, initialize=[10, 6, 1], mutable=False)
    model.alpha = Param(model.n, mutable=True)  # Random parameter for operational availability

    # Filling parameter
    operating_costs = [4, 4.5, 3.2, 5.5, 10]

    for i in model.n:
        model.operation_cost[i] = operating_costs[i]
        model.x_bar[i] = x_bar[i]
        model.alpha[i] = alpha_i[i]

    for j in model.k:
        model.capacity_per_mode[j] = capacity_per_mode[j]

    def capacity_per_mode(model, i):
        return sum(model.y[i, j] for j in model.k) <=  model.alpha[i]*model.x_bar[i]

    def demand_bal(model, j):
        return sum(model.y[i, j] for i in model.n) >= model.capacity_per_mode[j]

    model.con3 = Constraint(model.n, rule=capacity_per_mode)
    model.con4 = Constraint(model.k, rule=demand_bal)


    model.obj = Objective(expr=sum(model.operation_cost[i] * sum(model.T[j] * model.y[i, j] for j in model.k)
                                   for i in model.n), sense=minimize)
    return model


# ------------------------------ Additional Utility Functions---------------------------------------------------------

def print_res(model, question='qa'):
    """
        This function is used to print the results of the expected value solution for all the models.
    """
    if question == 'qb':
        res_x = [['Technology-1', model.x[0].value, model.y[0, 0].value, model.y[0, 1].value, model.y[0, 2].value],
                 ['Technology-2', model.x[1].value, model.y[1, 0].value, model.y[1, 1].value, model.y[1, 2].value],
                 ['Technology-3', model.x[2].value, model.y[2, 0].value, model.y[2, 1].value, model.y[2, 2].value],
                 ['Technology-4', model.x[3].value, model.y[3, 0].value, model.y[3, 1].value, model.y[3, 2].value],
                 ['Technology-5', model.x[4].value, model.y[4, 0].value, model.y[4, 1].value, model.y[4, 2].value]
                 ]
        total_investment = value(model.ObjCost_first_stage)
        total_operation_cost = value(model.ObjCost_second_stage)
        print(tabulate(res_x, headers=['Technologies', 'Installed Capacity (x̄)',
                                       'Usage (T1)', 'Usage (T2)', 'Usage (T3)'], tablefmt="pretty"))
        print('Total investment cost     =', total_investment)
        print('Total operation cost =', total_operation_cost)
        print('Total objective cost (EV) =', total_operation_cost + total_investment)

    elif question == 'qa':
        res_x = [['Technology-1', model.x[0].value, model.y[0, 0].value, model.y[0, 1].value, model.y[0, 2].value],
                 ['Technology-2', model.x[1].value, model.y[1, 0].value, model.y[1, 1].value, model.y[1, 2].value],
                 ['Technology-3', model.x[2].value, model.y[2, 0].value, model.y[2, 1].value, model.y[2, 2].value],
                 ['Technology-4', model.x[3].value, model.y[3, 0].value, model.y[3, 1].value, model.y[3, 2].value],
                 ]
        total_investment = value(model.ObjCost_first_stage)
        total_operation_cost = value(model.ObjCost_second_stage)
        print(tabulate(res_x, headers=['Technologies',
                                       'Installed Capacity (x̄)',
                                       'Usage (T1)', 'Usage (T2)',
                                       'Usage (T3)'], tablefmt="pretty"))
        print('Total investment cost     =', total_investment)
        print('Total operation cost =', total_operation_cost)
        print('Total objective cost (EV) =', total_operation_cost + total_investment)


def print_res_ts(model, question='qa'):
    """
          This function is used to print the results of the stochastic solution for all the models.
      """
    if question == 'qa':
        res_x = [['Technology-1', model.x[0].value],
                 ['Technology-2', model.x[1].value],
                 ['Technology-3', model.x[2].value],
                 ['Technology-4', model.x[3].value],
                 ]
        print(tabulate(res_x, headers=['Technologies', 'Installed Capacity (x̄)'], tablefmt="pretty"))
    elif question == 'qb':
        res_x = [['Technology-1', model.x[0].value],
                 ['Technology-2', model.x[1].value],
                 ['Technology-3', model.x[2].value],
                 ['Technology-4', model.x[3].value],
                 ['Technology-5', model.x[4].value]]

        print(tabulate(res_x, headers=['Technologies', 'Installed Capacity (x̄)'], tablefmt="pretty"))


def generate_scenario_qa():

    # This function is used to generate 27 scenarios based on the provoded discrete distributions for \xi_i

    scenario_names = []
    prob_dict = {'xi_1_3': 0.3,
                 'xi_1_5': 0.4,
                 'xi_1_7': 0.3,
                 'xi_2_2': 0.3,
                 'xi_2_3': 0.4,
                 'xi_2_4': 0.3,
                 'xi_3_1': 0.3,
                 'xi_3_2': 0.4,
                 'xi_3_3': 0.3,
                 }
    for i in range(27):
        scenario_names.append('scenario-' + str(i + 1))
    all_scenarios = pd.DataFrame(
        data={'xi_1': [3, 5, 7], 'xi_2': [2, 3, 4], 'xi_3': [1, 2, 3]})  # Scenario matrix as a data frame

    all_scenarios = pd.DataFrame(index=scenario_names, data=list(product(*all_scenarios.values.T)))
    all_scenarios_prob = []
    for s in all_scenarios.iterrows():
        all_scenarios_prob.append(prob_dict['xi_1_' + str(s[1][0])] *
                                  prob_dict['xi_2_' + str(s[1][1])] *
                                  prob_dict['xi_3_' + str(s[1][2])])

    fig = plt.figure(figsize=(15, 8))
    plt.stem(np.arange(27) + 1, all_scenarios_prob)
    for i in range(27):
        plt.text(i + 1, all_scenarios_prob[i],
                 '[' + str(all_scenarios[0][i]) + ','
                 + str(all_scenarios[1][i]) + ','
                 + str(all_scenarios[2][i]) + ']',
                 rotation=45)
    plt.xlabel('Scenarios')
    plt.ylabel('Probability')
    plt.xticks(np.arange(27) + 1)
    plt.grid()
    plt.savefig('discrete_scenario.png', dpi=300)
    plt.show()

    scen_prob_df = pd.DataFrame(index=scenario_names, data={'Probability': all_scenarios_prob})
    return scenario_names, all_scenarios, scen_prob_df


def scenario_creator_qa(scenario_name, S = '', prob_df='' ):

    # This function is necessary to solve model1 and obtain stochastic (TS) solution using the MPI-SSPY package
    # compatible with pyomo

    try:
        mode_demand = S.loc[scenario_name].to_numpy()
    except:
        raise ValueError("Unrecognized scenario name")

    model = create_model_qa(capacity_per_mode=mode_demand,capacity_max = [5,3,2])
    sputils.attach_root_node(model, model.ObjCost_first_stage, [model.x])  # It calls the attach_root_node function.
    # We tell this function which part of the objective function (model.ObjCost_first_stage) and
    # which set of variables (model.x) belong to the first stage.
    # In this case, the problem is only two stages, so we need only specify the root node and the
    # first-stage information–MPI-SPPy assumes the remainder of the model belongs to the second stage.
    model._mpisppy_probability = prob_df.loc[scenario_name].to_numpy() # Equal probability
    return model


def generate_combinations(arr, total_combinations, column_names):
    """
    This function returns all the permutations of choosing one one eleme nt from each given arays of list.
        Function to produce all possible combuination from each individual arrays
    """
    # number of arrays
    n = len(arr)
    # to keep track of next element
    # in each of the n arrays
    indices = [0 for i in range(n)]
    arr_fin_ = []
    s_names = []
    for i in range(total_combinations):
        s_names.append('scenario_' + str(i))
    while (1):

        # print current combination
        arr_ = []
        for i in range(n):
            arr_.append(arr[i][indices[i]])
        arr_fin_.append(arr_)
        next = n - 1
        while (next >= 0 and
               (indices[next] + 1 >= len(arr[next]))):
            next -= 1
        if (next < 0):
            df_ = pd.DataFrame(index=s_names, data=arr_fin_)
            df_.columns = column_names
            return df_, s_names
        indices[next] += 1
        for i in range(next + 1, n):
            indices[i] = 0


def generate_scenario_qb(number_of_samples, plot=True, seed = None):

    # This functiion generates the scenarios based on \xi_i and alpha_i. The total number of scenarios generated
    #    is: 27*number_of_samples**4

    rng = default_rng(seed)
    xi1 = np.array([3, 5, 7])
    xi2 = np.array([2, 3, 4])
    xi3 = np.array([1, 2, 3])
    prob_dict = {'xi_1_3': 0.3,
                 'xi_1_5': 0.4,
                 'xi_1_7': 0.3,
                 'xi_2_2': 0.3,
                 'xi_2_3': 0.4,
                 'xi_2_4': 0.3,
                 'xi_3_1': 0.3,
                 'xi_3_2': 0.4,
                 'xi_3_3': 0.3,
                 }

    # Sampling from a uniform distribution
    alpha_1 = rng.uniform(0.6, 0.9, number_of_samples)  # Generating points between lower and higher limits.
    alpha_2 = rng.uniform(0.7, 0.8, number_of_samples)  # Generating points between lower and higher limits.
    alpha_3 = rng.uniform(0.5, 0.8, number_of_samples)  # Generating points between lower and higher limits.
    alpha_4 = rng.uniform(0.9, 1, number_of_samples)  # Generating points between lower and higher limits.

    s_all, scenario_names__ = generate_combinations([alpha_1, alpha_2, alpha_3, alpha_4, xi1, xi2, xi3],
                                                    (number_of_samples ** 4) * 27,
                                                    ['alpha_1', 'alpha_2', 'alpha_3', 'alpha_4', 'xi_1', 'xi_2',
                                                     'xi_3'])

    s_prob = []
    for s in s_all.iterrows():
        s_prob.append(((1 / number_of_samples) ** 4) * prob_dict['xi_1_' + str(int(s[1][4]))] *
                      prob_dict['xi_2_' + str(int(s[1][5]))] *
                      prob_dict['xi_3_' + str(int(s[1][6]))])
    s_prob_df = pd.DataFrame(index=scenario_names__, data=s_prob)
    if plot:
        fig, ax = plt.subplot_mosaic([['1', '2'], ['3', '4']], constrained_layout=True)
        ax['1'].hist(s_all['alpha_1'], 40, facecolor='green', density=True, )
        ax['2'].hist(s_all['alpha_2'], 40, facecolor='red')
        ax['3'].hist(s_all['alpha_3'], 40, facecolor='blue')
        ax['4'].hist(s_all['alpha_4'], 40, facecolor='brown')
        ax['1'].set_xlabel(r'$\alpha_1:=\mathcal{U}(0.6,0.9)$')
        ax['2'].set_xlabel(r'$\alpha_2:=\mathcal{U}(0.7,0.8)$')
        ax['3'].set_xlabel(r'$\alpha_3:=\mathcal{U}(0.5,0.8)$')
        ax['4'].set_xlabel(r'$\alpha_4:=\mathcal{U}(0.9,1)$')
        ax['1'].set_ylabel('count')
        ax['2'].set_ylabel('count')
        ax['3'].set_ylabel('count')
        ax['4'].set_ylabel('count')

        ax['1'].grid(True)
        ax['2'].grid(True)
        ax['3'].grid(True)
        ax['4'].grid(True)

        # plt.tight_layout()
        plt.suptitle(r'Histograms for $\alpha_i$')
        plt.savefig('mod2_scenario.png', dpi=300)
        plt.show()
    return s_all, s_prob_df


def generate_scenario_qc(number_of_samples, plot=True, seed = None):

    # This functiion generates the scenarios based on \xi_i and alpha_i. The total number of scenarios generated
    #    is: 27*number_of_samples**4

    rng = default_rng(seed)
    xi1 = np.array([3, 5, 7])
    xi2 = np.array([2, 3, 4])
    xi3 = np.array([1, 2, 3])
    tau_2 = np.array([5, 7.5])
    tau_3 = np.array([0.5, 1.75])
    prob_dict = {'xi_1_3': 0.3,
                 'xi_1_5': 0.4,
                 'xi_1_7': 0.3,
                 'xi_2_2': 0.3,
                 'xi_2_3': 0.4,
                 'xi_2_4': 0.3,
                 'xi_3_1': 0.3,
                 'xi_3_2': 0.4,
                 'xi_3_3': 0.3,
                 'tau_2_5.0': 0.6,
                 'tau_2_7.5': 0.6,
                 'tau_3_0.5': 0.6,
                 'tau_3_1.75': 0.6
                 }

    # Sampling from a uniform distribution
    alpha_1 = rng.uniform(0.6, 0.9, number_of_samples)  # Generating points between lower and higher limits.
    alpha_2 = rng.uniform(0.7, 0.8, number_of_samples)  # Generating points between lower and higher limits.
    alpha_3 = rng.uniform(0.5, 0.8, number_of_samples)  # Generating points between lower and higher limits.
    alpha_4 = rng.uniform(0.9, 1, number_of_samples)  # Generating points between lower and higher limits.

    s_all, scenario_names__ = generate_combinations([tau_2,tau_3, alpha_1, alpha_2, alpha_3, alpha_4, xi1, xi2, xi3],
                                                    (number_of_samples ** 4) * 27*4,
                                                    ['tau_1','tau_2','alpha_1', 'alpha_2', 'alpha_3', 'alpha_4', 'xi_1', 'xi_2',
                                                     'xi_3'])

    s_prob = []
    for s in s_all.iterrows():
        s_prob.append(((1 / number_of_samples) ** 4) * prob_dict['xi_1_' + str(int(s[1][6]))] *
                      prob_dict['xi_2_' + str(int(s[1][7]))] *
                      prob_dict['xi_3_' + str(int(s[1][8]))]*
                      prob_dict['tau_2_' + str(float(s[1][0]))]*
                      prob_dict['tau_3_' + str(float(s[1][1]))])
    s_prob_df = pd.DataFrame(index=scenario_names__, data=s_prob)
    if plot:
        fig, ax = plt.subplot_mosaic([['1', '2'], ['3', '4']], constrained_layout=True)
        ax['1'].hist(s_all['alpha_1'], 40, facecolor='green', density=True, )
        ax['2'].hist(s_all['alpha_2'], 40, facecolor='red')
        ax['3'].hist(s_all['alpha_3'], 40, facecolor='blue')
        ax['4'].hist(s_all['alpha_4'], 40, facecolor='brown')
        ax['1'].set_xlabel(r'$\alpha_1:=\mathcal{U}(0.6,0.9)$')
        ax['2'].set_xlabel(r'$\alpha_2:=\mathcal{U}(0.7,0.8)$')
        ax['3'].set_xlabel(r'$\alpha_3:=\mathcal{U}(0.5,0.8)$')
        ax['4'].set_xlabel(r'$\alpha_4:=\mathcal{U}(0.9,1)$')
        ax['1'].set_ylabel('count')
        ax['2'].set_ylabel('count')
        ax['3'].set_ylabel('count')
        ax['4'].set_ylabel('count')

        ax['1'].grid(True)
        ax['2'].grid(True)
        ax['3'].grid(True)
        ax['4'].grid(True)

        # plt.tight_layout()
        plt.suptitle(r'Histograms for $\alpha_i$')
        plt.savefig('mod2_scenario.png', dpi=300)
        plt.show()
    return s_all, s_prob_df


def scenario_creator_qb(scenario_name, S = '', prob_df='' ):
    """
              This function is necessary to solve model2 and obtain stochastic (TS) solution using the MPI-SSPY package
              compatible with pyomo
    """
    try:
        mode_demand = S.loc[scenario_name].to_numpy()
    except:
        raise ValueError("Unrecognized scenario name")

    model = create_model_qb(random_parameters = mode_demand)
    sputils.attach_root_node(model, model.ObjCost_first_stage, [model.x])
    model._mpisppy_probability = prob_df.loc[scenario_name].to_numpy()
    return model


# --------------------- Solving Wait-and-See Problem -----------------------------------------------
# Solving wait and see

def solve_ws_qa(demandmax=[],all_scenarios=pd.DataFrame(), create_model=ConcreteModel()):

    x1_ws = []
    x2_ws = []
    x3_ws = []
    x4_ws = []
    total_investment_ws = []
    total_operation_cost_ws = []
    for s in all_scenarios.iterrows():
        model = create_model(capacity_per_mode=[s[1][0], s[1][1], s[1][2]],capacity_max = demandmax)
        SolverFactory('gurobi').solve(model)
        x1_ws.append(model.x[0].value)
        x2_ws.append(model.x[1].value)
        x3_ws.append(model.x[2].value)
        x4_ws.append(model.x[3].value)
        total_investment_ws.append(value(model.ObjCost_first_stage))
        total_operation_cost_ws.append(value(model.ObjCost_second_stage))
    res_pd = pd.DataFrame(data={'Scenarios':all_scenarios.index ,
                                'Technology-1':x1_ws ,
                                'Technology-2':x2_ws ,
                                'Technology-3':x3_ws ,
                                'Technology-4':x4_ws ,
                                'Total Investment':total_investment_ws ,
                                'Total Operation Cost':total_operation_cost_ws ,
                                'Total objective cost':np.array(total_investment_ws) +
                                                    np.array(total_operation_cost_ws)})
    res=[['Technology-1',np.mean(x1_ws)] ,
        ['Technology-2',np.mean(x2_ws)] ,
        ['Technology-3',np.mean(x3_ws)] ,
        ['Technology-4',np.mean(x4_ws)] ,
        ['Total Investment',np.mean(total_investment_ws)] ,
        ['Total Operation Cost',np.mean(total_operation_cost_ws)] ,
        ['Total objective cost (WS)', np.mean(total_operation_cost_ws)+np.mean(total_investment_ws)]]
    print(res_pd)
    print('------------------------- Average value of wait and see -------------------------------')
    print(tabulate(res, headers =['Items', 'Average Values']))


def solve_ws_qb(all_scenarios=pd.DataFrame(), create_model=ConcreteModel()):
    x1_ws_qb = []
    x2_ws_qb = []
    x3_ws_qb = []
    x4_ws_qb = []
    x5_ws_qb = []
    total_investment_ws_qb = []
    total_operation_cost_ws_qb = []
    for s in tqdm(all_scenarios.iterrows()):
        model = create_model(random_parameters=
                                        [s[1][0], s[1][1], s[1][2],s[1][3], s[1][4], s[1][5], s[1][6]])

        SolverFactory('gurobi').solve(model)
        x1_ws_qb.append(model.x[0].value)
        x2_ws_qb.append(model.x[1].value)
        x3_ws_qb.append(model.x[2].value)
        x4_ws_qb.append(model.x[3].value)
        x5_ws_qb.append(model.x[4].value)
        total_investment_ws_qb.append(value(model.ObjCost_first_stage))
        total_operation_cost_ws_qb.append(value(model.ObjCost_second_stage))
    res_pd = pd.DataFrame(data={'Scenarios':all_scenarios.index ,
                                'Technology-1':x1_ws_qb ,
                                'Technology-2':x2_ws_qb ,
                                'Technology-3':x3_ws_qb ,
                                'Technology-4':x4_ws_qb ,
                                'Technology-5':x5_ws_qb ,
                                'Total Investment':total_investment_ws_qb ,
                                'Total Operation Cost':total_operation_cost_ws_qb ,
                                'Total objective cost': np.array(total_investment_ws_qb)+
                                                        np.array(total_operation_cost_ws_qb)})
    res=[['Technology-1',np.mean(x1_ws_qb)] ,
        ['Technology-2',np.mean(x2_ws_qb)] ,
        ['Technology-3',np.mean(x3_ws_qb)] ,
        ['Technology-4',np.mean(x4_ws_qb)] ,
        ['Technology-5',np.mean(x5_ws_qb)],
        ['Total Investment',np.mean(total_investment_ws_qb)] ,
        ['Total Operation Cost',np.mean(total_operation_cost_ws_qb)] ,
        ['Total objective cost (WS)', np.mean(total_operation_cost_ws_qb)+np.mean(total_investment_ws_qb)]]
    print(res_pd)
    print('------------------------- Average value of wait and see -------------------------------')
    print(tabulate(res, headers =['Items', 'Average Values']))

def solve_ws_qc(all_scenarios=pd.DataFrame(), create_model=ConcreteModel()):
    x1_ws_qb = []
    x2_ws_qb = []
    x3_ws_qb = []
    x4_ws_qb = []
    x5_ws_qb = []
    total_investment_ws_qb = []
    total_operation_cost_ws_qb = []
    for s in tqdm(all_scenarios.iterrows()):
        model = create_model(random_parameters=
                                        [s[1][0], s[1][1], s[1][2],s[1][3], s[1][4], s[1][5], s[1][6], s[1][7], s[1][8]])

        SolverFactory('gurobi').solve(model)
        x1_ws_qb.append(model.x[0].value)
        x2_ws_qb.append(model.x[1].value)
        x3_ws_qb.append(model.x[2].value)
        x4_ws_qb.append(model.x[3].value)
        x5_ws_qb.append(model.x[4].value)
        total_investment_ws_qb.append(value(model.ObjCost_first_stage))
        total_operation_cost_ws_qb.append(value(model.ObjCost_second_stage))
    res_pd = pd.DataFrame(data={'Scenarios':all_scenarios.index ,
                                'Technology-1':x1_ws_qb ,
                                'Technology-2':x2_ws_qb ,
                                'Technology-3':x3_ws_qb ,
                                'Technology-4':x4_ws_qb ,
                                'Technology-5':x5_ws_qb ,
                                'Total Investment':total_investment_ws_qb ,
                                'Total Operation Cost':total_operation_cost_ws_qb ,
                                'Total objective cost': np.array(total_investment_ws_qb)+
                                                        np.array(total_operation_cost_ws_qb)})
    res=[['Technology-1',np.mean(x1_ws_qb)] ,
        ['Technology-2',np.mean(x2_ws_qb)] ,
        ['Technology-3',np.mean(x3_ws_qb)] ,
        ['Technology-4',np.mean(x4_ws_qb)] ,
        ['Technology-5',np.mean(x5_ws_qb)],
        ['Total Investment',np.mean(total_investment_ws_qb)] ,
        ['Total Operation Cost',np.mean(total_operation_cost_ws_qb)] ,
        ['Total objective cost (WS)', np.mean(total_operation_cost_ws_qb)+np.mean(total_investment_ws_qb)]]
    print(res_pd)
    print('------------------------- Average value of wait and see -------------------------------')
    print(tabulate(res, headers =['Items', 'Average Values']))