import numpy as np
import time
import random
import torch
from typing import Callable, List
from mcbo.utils.initial_design import generate_initial_design
from os_cbo.dtos.cbo.cbo_results import CBOResults
from os_cbo.dtos.cbo.cbo_config import CBOConfig
import os_cbo.constants.constants as constants
from mcbo.optimal_stopping.oscbo import OSCBO
from os_cbo.util.cbo_util import CBOUtil
from mcbo.utils.initial_design import random_causal
from mcbo.utils.optimisation_policy import get_new_suggested_point, obj_mean, get_acq_fun

def mcbo_trial(
    algo_profile: dict,
    env_profile: dict,
    budget: float,
    variables: List[str],
    optimal_value: float,
    cbo_config : CBOConfig,
    function_network: Callable,
    network_to_objective_transform: Callable,
) -> None:
    r'''
    Interacts with the environment specified by env_profile and function_network for 
    algo_profile['n_bo_iters'] rounds using the algorithm specified in algo_profile. 
    Logs to wandb the average and best score across rounds. 
    '''

    # Set seed
    torch.manual_seed(algo_profile["seed"])
    np.random.seed(algo_profile["seed"])
    random.seed(algo_profile["seed"])

    # Initial evaluations
    X = generate_initial_design(algo_profile, env_profile)
    mean_at_X = obj_mean(X, function_network, network_to_objective_transform)
    network_observation_at_X = function_network(X)
    observation_at_X = network_to_objective_transform(network_observation_at_X)
    # Current best objective value.
    best_obs_index = np.argmax(observation_at_X).item()
    best_obs_val = observation_at_X.max().item()

    # Historical best observed objective values and running times.
    hist_best_obs_vals = [best_obs_val]
    runtimes = []

    init_batch_id = 1

    results = CBOResults(remaining_budget=budget, exploration_set=cbo_config.scm.exploration_set)

    # Initialize the best intervention set, level and target
    results.initialize_best_intervention_set_level_target(cbo_config=cbo_config)

    # Compute the optimal intervention mean
    results.optimal_intervention_mean = CBOUtil.estimate_optimal_intervention_mean(cbo_config=cbo_config)
    i =0
    for i in range(len(X)):
        intervention_set_identifiers = list(map(lambda x: int(x), X[i].numpy()[0:3].tolist()))
        if 1 in intervention_set_identifiers:
            intervention_set_idx = intervention_set_identifiers.index(1)
            intervention_set = cbo_config.scm.exploration_set[intervention_set_idx]
            intervention_level = network_observation_at_X[i][intervention_set_idx]
            cost = cbo_config.scm.costs[constants.CBO.INTERVENE](intervention_set=intervention_set,
                                                                 intervention_levels = [intervention_level])
            target_interventional_mean = -mean_at_X[i]
            # Add the new x-value to the intervention dataset
            if len(cbo_config.scm.interventions[
                       cbo_config.scm.exploration_set.index(intervention_set)][
                       constants.CBO.INTERVENTIONS_X_INDEX]) == 0:
                cbo_config.scm.interventions[cbo_config.scm.exploration_set.index(intervention_set)][
                    constants.CBO.INTERVENTIONS_X_INDEX] = \
                    np.array([[intervention_level]])
            else:
                cbo_config.scm.interventions[
                    cbo_config.scm.exploration_set.index(intervention_set)][
                    constants.CBO.INTERVENTIONS_X_INDEX] = np.append(
                    cbo_config.scm.interventions[
                        cbo_config.scm.exploration_set.index(intervention_set)][
                        constants.CBO.INTERVENTIONS_X_INDEX],
                    np.array([[intervention_level]]), axis=0)
            # Add the new y-value to the intervention dataset
            if len(cbo_config.scm.interventions[
                       cbo_config.scm.exploration_set.index(intervention_set)][
                       constants.CBO.INTERVENTIONS_Y_INDEX]) == 0:
                cbo_config.scm.interventions[cbo_config.scm.exploration_set.index(intervention_set)][
                    constants.CBO.INTERVENTIONS_Y_INDEX] = \
                    np.array([[target_interventional_mean]])
            else:
                cbo_config.scm.interventions[cbo_config.scm.exploration_set.index(intervention_set)][
                    constants.CBO.INTERVENTIONS_Y_INDEX] = \
                    np.append(cbo_config.scm.interventions[
                                  cbo_config.scm.exploration_set.index(intervention_set)][
                                  constants.CBO.INTERVENTIONS_Y_INDEX],
                              np.array([[target_interventional_mean]]), axis=0)

            # Record the new intervention in the state:
            results.record_intervention(intervention_set=intervention_set)
        else:
            results.num_observe_decisions += 1
            results.actions.append(0)
            new_observations = np.array([network_observation_at_X[i].numpy()])
            observation_set = variables
            # Update the observational GPs with the new data
            cbo_config.scm.update_observational_gps(new_observations=new_observations,
                                                    observation_set=observation_set)

            # Update the causal prior with the new data
            mean_functions, var_functions = CBOUtil.update_causal_prior(
                cbo_config=cbo_config, F=CBOUtil.copy_observational_gps(cbo_config=cbo_config))
            cbo_config.scm.mean_functions_causal_do_prior = mean_functions
            cbo_config.scm.var_functions_causal_do_prior = var_functions

            # Cost
            cost = cbo_config.scm.costs[constants.CBO.OBSERVE](observation_set=cbo_config.scm.variables)

        # Update cost in the state
        results.update_cost(cost=cost)

        # Update current optimum in the state
        results.update_best_intervention_set_level_target(cbo_config=cbo_config)

        if len(results.actions) > 0 and results.actions[-1] == 0:
            # If the previous action was to observe, update all interventional GPs with the new causal prior
            for intervention_set_idx in range(len(cbo_config.scm.exploration_set)):
                cbo_config.scm.update_interventional_gp(
                    intervention_set=cbo_config.scm.exploration_set[intervention_set_idx])
        else:
            # If the previous action was an intervention, update only the interventional GP of that particular
            # intervention
            if len(results.interventions) > 0:
                cbo_config.scm.update_interventional_gp(intervention_set=results.interventions[-1])

    old_nets = []  # only used by NMCBO to reuse old computation
    print("Starting MCBO execution")
    iteration = 1
    while results.remaining_budget > 0:
        results.iteration = iteration

        if len(results.actions) > 0 and results.actions[-1] == 0:
            # If the previous action was to observe, update all interventional GPs with the new causal prior
            for intervention_set_idx in range(len(cbo_config.scm.exploration_set)):
                cbo_config.scm.update_interventional_gp(
                    intervention_set=cbo_config.scm.exploration_set[intervention_set_idx])
        else:
            # If the previous action was an intervention, update only the interventional GP of that particular
            # intervention
            if len(results.interventions) > 0:
                cbo_config.scm.update_interventional_gp(intervention_set=results.interventions[-1])

        # New suggested point
        t0 = time.time()
        new_x, new_net = get_new_suggested_point(
            X=X,
            network_observation_at_X=network_observation_at_X,
            observation_at_X=observation_at_X,
            algo_profile=algo_profile,
            env_profile=env_profile,
            function_network=function_network,
            network_to_objective_transform=network_to_objective_transform,
            old_nets=old_nets,
        )
        if new_net is not None:
            old_nets.append(new_net)
        t1 = time.time()
        runtimes.append(t1 - t0)

        # Evaluate network at new point
        network_observation_at_new_x = function_network(new_x)

        # The mean value of the new action.
        mean_at_new_x = obj_mean(
            new_x, function_network, network_to_objective_transform
        )

        # Evaluate objective at new point
        observation_at_new_x = network_to_objective_transform(
            network_observation_at_new_x
        )

        intervention_set_identifiers = list(map(lambda x: int(x), new_x.numpy()[0][0:3].tolist()))
        if 1 in intervention_set_identifiers:
            intervention_set_idx = intervention_set_identifiers.index(1)
        intervention_set = cbo_config.scm.exploration_set[intervention_set_idx]
        intervention_level = np.array([network_observation_at_new_x.numpy()[0][intervention_set_idx]])

        # Implement OS decision here
        intervene, observation_set = OSCBO.os_cbo(state=results, cbo_config=cbo_config, X=X, network_observation_at_X=network_observation_at_X,
                     observation_at_X=observation_at_X, algo_profile=algo_profile, env_profile=env_profile,
                     function_network=function_network, network_to_objective_transform=network_to_objective_transform,
                     old_nets=old_nets, selected_intervention_set=intervention_set, selected_intervention_level=intervention_level,
                                                  selected_intervention_set_idx=intervention_set_idx)
        observation_probability = 0.0
        observation_set = ["X", "Z", "Y"]
        if intervene:
            # Update training data
            if mean_at_X is None:
                mean_at_X = mean_at_new_x
            else:                mean_at_X = torch.cat([mean_at_X, mean_at_new_x], 0)
            X = torch.cat([X, new_x], 0)
            network_observation_at_X = torch.cat(
                [network_observation_at_X, network_observation_at_new_x], 0
            )
            observation_at_X = torch.cat([observation_at_X, observation_at_new_x], 0)
            best_obs_val = observation_at_X.max().item()
            best_obs_index = np.argmax(observation_at_X).item()
            hist_best_obs_vals.append(best_obs_val)

            # OSCO
            intervention_set_idx = intervention_set_identifiers.index(1)
            intervention_set = cbo_config.scm.exploration_set[intervention_set_idx]
            intervention_level = network_observation_at_X[-1][intervention_set_idx]
            cost = cbo_config.scm.costs[constants.CBO.INTERVENE](intervention_set=intervention_set,
                                                                 intervention_levels = [intervention_level])
            target_interventional_mean = -mean_at_X[-1]
            # Add the new x-value to the intervention dataset
            if len(cbo_config.scm.interventions[
                       cbo_config.scm.exploration_set.index(intervention_set)][
                       constants.CBO.INTERVENTIONS_X_INDEX]) == 0:
                cbo_config.scm.interventions[cbo_config.scm.exploration_set.index(intervention_set)][
                    constants.CBO.INTERVENTIONS_X_INDEX] = \
                    np.array([[intervention_level]])
            else:
                cbo_config.scm.interventions[
                    cbo_config.scm.exploration_set.index(intervention_set)][
                    constants.CBO.INTERVENTIONS_X_INDEX] = np.append(
                    cbo_config.scm.interventions[
                        cbo_config.scm.exploration_set.index(intervention_set)][
                        constants.CBO.INTERVENTIONS_X_INDEX],
                    np.array([[intervention_level]]), axis=0)

            # Add the new y-value to the intervention dataset
            if len(cbo_config.scm.interventions[
                       cbo_config.scm.exploration_set.index(intervention_set)][
                       constants.CBO.INTERVENTIONS_Y_INDEX]) == 0:
                cbo_config.scm.interventions[cbo_config.scm.exploration_set.index(intervention_set)][
                    constants.CBO.INTERVENTIONS_Y_INDEX] = \
                    np.array([[target_interventional_mean]])
            else:
                cbo_config.scm.interventions[cbo_config.scm.exploration_set.index(intervention_set)][
                    constants.CBO.INTERVENTIONS_Y_INDEX] = \
                    np.append(cbo_config.scm.interventions[
                                  cbo_config.scm.exploration_set.index(intervention_set)][
                                  constants.CBO.INTERVENTIONS_Y_INDEX],
                              np.array([[target_interventional_mean]]), axis=0)

            # Record the new intervention in the state:
            results.record_intervention(intervention_set=intervention_set)
        else:
            new_observations = np.array(function_network(random_causal(
                torch.zeros(env_profile["valid_targets"][0].shape),
                env_profile,
                N=1,
            )).numpy())

            cbo_config.scm.update_observational_gps(new_observations=new_observations,
                                                    observation_set=observation_set)

            # Update the causal prior with the new data
            mean_functions, var_functions = CBOUtil.update_causal_prior(
                cbo_config=cbo_config, F=CBOUtil.copy_observational_gps(cbo_config=cbo_config))
            cbo_config.scm.mean_functions_causal_do_prior = mean_functions
            cbo_config.scm.var_functions_causal_do_prior = var_functions

            results.num_observe_decisions += 1
            results.actions.append(0)
            cost = cbo_config.scm.costs[constants.CBO.OBSERVE](observation_set=cbo_config.scm.variables)

        # Update cost in the state
        results.update_cost(cost=cost)

        # Update current optimum in the state
        results.update_best_intervention_set_level_target(cbo_config=cbo_config)

        best_intervention_set_idx = cbo_config.scm.exploration_set.index(results.best_intervention_set[-1])
        print(
            f"i:{results.iteration}, budget:{round(results.remaining_budget, 2)}, "
            f"X'_t: {results.best_intervention_set[-1]}, "
            f"x'_t: {results.best_intervention_levels[best_intervention_set_idx][-1]}, "
            f", E[Y|Do(X'_t=x'_t)]: ",
            f"{round(results.best_intervention_targets[best_intervention_set_idx][-1], 2)},"            
            f"latest X'_t: {intervention_set}, latest x'_t: {intervention_level}, latest E[Y|Do(X'_t=x'_t)]: "
            f"{round(target_interventional_mean.item(), 2)}, "            
            f"E[Y|Do(X^*=x^*)]: {round(results.optimal_intervention_mean, 2)}, "
            f"P[observe]={round(observation_probability, 4)}, # observations: {cbo_config.scm.num_observations()}, "
            f"# interventions: {results.num_intervene_decisions}, observation_set: {observation_set}, "
            f"intervention_set: {intervention_set}")
        iteration+=1

    results.observation_data = cbo_config.scm.observations
    results.intervention_data = cbo_config.scm.interventions
    results.total_time = time.time() - results.start_time
    print(f"MCBO execution complete, total time: {round(results.total_time, 2)}s")

    results_file = f"mcbo_seed_{algo_profile['seed']}.json"
    print(f"Saving the results to file: {results_file}")
    results.to_json_file(results_file)
