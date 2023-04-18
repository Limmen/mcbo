from typing import List, Dict, Callable, Tuple
import numpy as np
import math
import logging
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper
from os_cbo.dtos.cbo.cbo_config import CBOConfig
from os_cbo.dtos.cbo.cbo_results import CBOResults
from os_cbo.util.cbo_util import CBOUtil
import os_cbo.constants.constants as constants
from mcbo.utils.optimisation_policy import get_new_suggested_point, obj_mean

class OSCBO:
    """
    Class implementing the Optimal stopping module described in (Hammar & Dhir 2023).
    """

    @staticmethod
    def os_cbo(state: CBOResults, cbo_config: CBOConfig, X, network_observation_at_X, observation_at_X, algo_profile,
               env_profile, function_network, network_to_objective_transform, old_nets, selected_intervention_set,
               selected_intervention_set_idx, selected_intervention_level) -> Tuple[bool, List[str]]:
        """
        Decides whether to intervene or observe in CBO using optimal stopping

        :param state: the current state of the CBO execution
        :param cbo_config: the configuration of the CBO execution
        :return: True if intervene (stop) else False, observation set
        """
        # Always observe if we have zero observations, to initialize \widehat{\mathbf{F}} and \widehat{\mu}
        if cbo_config.scm.num_observations() == 0:
            return False, cbo_config.scm.variables

        # Next intervention to evaluate
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
        # network_observation_at_new_x = function_network(new_x)
        # mean_at_new_x = obj_mean(
        #     new_x, function_network, network_to_objective_transform
        # )
        # observation_at_new_x = network_to_objective_transform(
        #     network_observation_at_new_x
        # )
        # intervention_set_identifiers = list(map(lambda x: int(x), new_x.numpy()[0][0:3].tolist()))
        # if 1 in intervention_set_identifiers:
        #     selected_intervention_set_idx = intervention_set_identifiers.index(1)
        # selected_intervention_set = cbo_config.scm.exploration_set[selected_intervention_set_idx]
        # selected_intervention_level = np.array([network_observation_at_new_x.numpy()[0][selected_intervention_set_idx]])

        # selected_intervention_set, selected_intervention_set_idx, selected_intervention_level, \
        # intervention_levels, _ = \
        #     CBOUtil.optimization_policy(cbo_config=cbo_config, state=state,
        #                                 interventional_gps=cbo_config.scm.interventional_gps)

        # MOS
        # observation_set = cbo_config.scm.mos(selected_intervention_set)
        observation_set = cbo_config.scm.variables

        # Always intervene if we have never tried the intervention set before, so that we can initialize \widehat{mu}
        if len(cbo_config.scm.interventions[selected_intervention_set_idx][constants.CBO.INTERVENTIONS_X_INDEX]) == 0:
            return True, observation_set

        # Always intervene if the intervention is not identifiable from observational data
        if not cbo_config.scm.identifiable(selected_intervention_set):
            return True, cbo_config.scm.variables

        # Costs
        intervention_cost = cbo_config.scm.costs[constants.CBO.INTERVENE](
            intervention_set=selected_intervention_set,
            intervention_levels=selected_intervention_level)
        observation_cost = cbo_config.scm.costs[constants.CBO.OBSERVE](observation_set=observation_set)

        # Observation bonus
        obs_bonus = cbo_config.observation_acquisition_config.delta * (1 - (min(state.num_observe_decisions
                                                                                / cbo_config.max_num_obs, 1)))

        # If we cannot afford to intervene, always observe
        if intervention_cost >= state.remaining_budget:
            # Save metrics
            state.information_gains.append(0)
            state.observation_gains.append(0)
            state.mean_gains.append(0)
            return False, observation_set

        # Copy probabilsitic models for look-ahead simulation
        mu = OSCBO.fit_mu(cbo_config=cbo_config, mean_functions=cbo_config.scm.mean_functions_causal_do_prior,
                          var_functions=cbo_config.scm.var_functions_causal_do_prior)
        F = OSCBO.fit_F(cbo_config=cbo_config, observations=cbo_config.scm.observations)

        # Compute reward of stopping
        r_stop, ig_stop, mean_stop = OSCBO.stopping_reward(
            interventional_samples=np.array([selected_intervention_level]), mu=mu[selected_intervention_set_idx],
            next_intervention_level=np.array([selected_intervention_level]),
            intervention_cost=intervention_cost,
            kappa=cbo_config.observation_acquisition_config.kappa,
            eta=cbo_config.observation_acquisition_config.eta
        )
        # Compute expected reward of observing and stopping in the next stage
        v_1, ig_observe, mean_observe, mean_obs_reward = OSCBO.one_step_lookahead_stopping_reward(
            cbo_config=cbo_config, F=F, state=state, observations=cbo_config.scm.observations,
            observation_set=observation_set, mu=mu, X=X, network_observation_at_X=network_observation_at_X,
            observation_at_X=observation_at_X, algo_profile=algo_profile, env_profile=env_profile,
            function_network=function_network, network_to_objective_transform=network_to_objective_transform,
            old_nets=old_nets, selected_intervention_set=selected_intervention_set,
            selected_intervention_set_idx=selected_intervention_set_idx,
            selected_intervention_level=selected_intervention_level)
        r_observe = v_1 - observation_cost + obs_bonus + mean_obs_reward

        # OSLA rule (Corollary 1, Hammar & Dhir)
        stop = False
        if r_stop >= r_observe:
            stop = True

        # Save metrics
        state.information_gains.append(ig_observe - ig_stop)
        state.observation_gains.append(r_observe - r_stop)
        state.mean_gains.append(mean_observe - mean_stop)
        state.exp_bonuses.append(mean_obs_reward)

        # Logging
        print(f"R_intervene:{round(r_stop, 3)}, R_obs:{round(r_observe, 3)}, IG_observe: {round(ig_observe, 3)},"
                    f" ig_stop: {round(ig_stop, 3)}, "
                    f"mean_observe: {round(mean_observe, 3)}, mean_stop: {round(mean_stop, 3)}, "
                    f"cost_stop: {round(intervention_cost, 3)}, cost_observe: {round(observation_cost, 3)}, "
                    f"obs_reward: {mean_obs_reward}")
        return stop, cbo_config.scm.mos(selected_intervention_set)

    @staticmethod
    def observation_span(observations_1d, var, cbo_config: CBOConfig):
        unexplored_regions = 0
        left_unexplored = np.min(observations_1d) - cbo_config.scm.intervention_spaces[var].min_intervention_level
        right_unexplored = cbo_config.scm.intervention_spaces[var].max_intervention_level - np.max(observations_1d)
        unexplored_regions += left_unexplored
        unexplored_regions += right_unexplored
        largest_interval = max(left_unexplored, right_unexplored)
        if len(observations_1d) > 1:
            sorted_obs = np.sort(observations_1d)
            for i in range(1, len(sorted_obs)):
                if sorted_obs[i] - sorted_obs[i - 1] > largest_interval:
                    largest_interval = sorted_obs[i] - sorted_obs[i - 1]
        unexplored_regions += largest_interval
        return unexplored_regions

    @staticmethod
    def model_descriptive_statistics(mu: Dict, F: Dict, cbo_config: CBOConfig):
        num_samples = 200
        total_var = 0
        mu_var = 0
        obs_var = 0
        total_mean = 0
        mu_mean = 0
        obs_mean = 0
        for idx, intervention_set in enumerate(cbo_config.scm.exploration_set):
            var_spaces = []
            for var in intervention_set:
                var_spaces.append(np.linspace(
                    cbo_config.scm.intervention_spaces[var].min_intervention_level,
                    cbo_config.scm.intervention_spaces[var].max_intervention_level, num=num_samples))
            input_space = np.array(var_spaces).reshape(num_samples, len(intervention_set))
            mean_pred, var_pred = mu[idx].predict(input_space)
            sum_var = sum(var_pred.flatten())
            sum_mean = sum(mean_pred.flatten())
            total_var += sum_var
            mu_var += sum_var
            total_mean += sum_mean
            mu_mean += sum_mean

        for k, v in F.items():
            condition_var, target_var = k
            var_spaces = np.linspace(
                cbo_config.scm.intervention_spaces[condition_var].min_intervention_level,
                cbo_config.scm.intervention_spaces[condition_var].max_intervention_level, num=num_samples)
            input_space = np.array(var_spaces).reshape(num_samples, 1)
            mean_pred, var_pred = v.predict(input_space)
            sum_var = sum(var_pred.flatten())
            sum_mean = sum(mean_pred.flatten())
            total_var += sum_var
            obs_var += sum_var
            total_mean += sum_mean
            obs_mean += sum_mean
        return total_var, mu_var, obs_var, total_mean, mu_mean, obs_mean

    @staticmethod
    def model_uncertainty(mu: Dict, F: Dict, cbo_config: CBOConfig, obs, observation_set, intervention_set_idx):
        obs_mean = 0
        total_var = 0
        mu_var = 0
        mu_mean = 0
        mu_mean_pred, mu_var_pred = mu[intervention_set_idx].predict(np.array([[obs[0]]]))
        total_var += mu_var_pred[0][0]
        mu_var += mu_var_pred[0][0]
        mu_mean += mu_mean_pred[0][0]
        obs_var = 0
        for k, v in F.items():
            condition_var, target_var = k
            if condition_var in observation_set:
                obs_mean_pred, obs_var_pred = v.predict(np.array([[obs[observation_set.index(condition_var)]]]))
                obs_var += obs_var_pred[0][0]
                total_var += obs_var_pred[0][0]
                obs_mean += obs_mean_pred[0][0]
        return total_var, obs_var, mu_var, obs_mean, mu_mean

    @staticmethod
    def gaussian_KL(mu_1: np.ndarray, mu_2: np.ndarray, K_1: np.ndarray, K_2: np.ndarray, n) -> float:
        """
        Calculates the KL divergence betweeen two multivariate Gaussians

        :param mu_1: mean vector for the first Gaussian
        :param mu_2: mean vector for the second Gaussian
        :param K_1: covariance matrix for the first Gaussian
        :param K_2: covariance matrix for the second Gaussian
        :param n: dimension of the multivariate Gaussian
        :return: the KL divergence
        """
        if np.linalg.det(K_1) <=0:
            return 0
        try:
            kl = max(0, (1 / 2) * (np.trace(np.dot(np.linalg.inv(K_2), K_1)) +
                                   np.dot(np.transpose(mu_2 - mu_1), (np.dot(np.linalg.inv(K_2), (mu_2 - mu_1)))) -
                                   n + math.log(np.linalg.det(K_2) / np.linalg.det(K_1)))[0][0])
            return 1 - math.exp(-kl)
        except Exception:
            return 1

    @staticmethod
    def model_KL(mu: Dict, F: Dict, new_mu: Dict, new_F: Dict, cbo_config: CBOConfig) -> float:
        """
        Computes the total KL divergence between two models (mu, F) and (new_mu, new_F)

        :param mu: the probabilistic model over Mu for the first model
        :param F: the probabilistic model over F for the first model
        :param new_mu: the probabilistic model over Mu for the second model
        :param new_F: the probabilistic model over F for the second model
        :param cbo_config: the CBO config
        :return: The total KL divergence
        """
        kl = 0
        for idx, intervention_set in enumerate(cbo_config.scm.exploration_set):
            input_space = mu[idx].model.X
            mu_1, K_1 = mu[idx].predict_with_full_covariance(input_space)
            mu_2, K_2 = new_mu[idx].predict_with_full_covariance(input_space)
            kl += OSCBO.gaussian_KL(mu_1=mu_1, mu_2=mu_2, K_1=K_1, K_2=K_2, n=len(input_space))
            # kl = max(kl, OSCBO.gaussian_KL(mu_1=mu_1, mu_2=mu_2, K_1=K_1, K_2=K_2, n=len(input_space)))

        for k, v in F.items():
            input_space = new_F[k].model.X
            mu_1, K_1 = F[k].predict_with_full_covariance(input_space)
            mu_2, K_2 = new_F[k].predict_with_full_covariance(input_space)
            kl += OSCBO.gaussian_KL(mu_1=mu_1, mu_2=mu_2, K_1=K_1, K_2=K_2, n=len(input_space))
            # kl = max(kl, OSCBO.gaussian_KL(mu_1=mu_1, mu_2=mu_2, K_1=K_1, K_2=K_2, n=len(input_space)))
        kl = kl*(1/(len(cbo_config.scm.exploration_set)+len(F.items())))
        return kl

    @staticmethod
    def one_step_lookahead_stopping_reward(
            cbo_config: CBOConfig, F: Dict, mu: Dict, state: CBOResults, observations: np.ndarray,
            observation_set: List[str], X, network_observation_at_X, observation_at_X, algo_profile,
            env_profile, function_network, network_to_objective_transform, old_nets,
            selected_intervention_set, selected_intervention_set_idx, selected_intervention_level) \
            -> Tuple[float, float, float, float]:
        """
        Computes the reward of observing and then stopping in the next time-step.
        This value is then used in the # OSLA rule (Corollary 1, Hammar & Dhir).

        :param cbo_config: the CBO config
        :param F: the probabilistic model over the causal functions
        :param state: the current state of the CBO execution
        :param observations: the current set of observations
        :param observation_set: the observation set
        :return: The one-step lookahead stopping reward, the information gain, mean next intervention, exp bonus mean
        """
        stopping_rewards = []
        information_gains = []
        means = []
        observation_rewards = []
        unexplored_regions = sum(list(map(lambda var: OSCBO.observation_span(
            observations_1d=cbo_config.scm.get_observations(observation_set=[var]),
            cbo_config=cbo_config, var=var), cbo_config.scm.manipulative_variables)))

        # total_var, mu_var, obs_var, total_mean, mu_mean, obs_mean = \
        #     OSCBO.model_descriptive_statistics(mu=mu, F=F, cbo_config=cbo_config)

        for i in range(cbo_config.observation_acquisition_config.samples_for_mc_estimation):
            # Sample a new observation
            obs, sigma = cbo_config.scm.simulate_observation(observational_gps=F,
                                                             eps_noise_terms=cbo_config.scm.eps_noise_terms,
                                                             observation_set=observation_set)

            # Update \widehat{F} with the new observation
            new_observations = observations.copy()
            if len(new_observations[tuple(observation_set)]) > 0:
                new_observations[tuple(observation_set)] = np.append(new_observations[tuple(observation_set)],
                                                                     np.array([obs]), axis=0)
            else:
                new_observations[tuple(observation_set)] = np.array([obs])
            new_F = OSCBO.fit_F(cbo_config=cbo_config, observations=new_observations)

            # Update \widehat{\mu} with the new observation
            mean_functions, var_functions = CBOUtil.update_causal_prior(cbo_config=cbo_config, F=new_F)
            new_mu = OSCBO.fit_mu(cbo_config=cbo_config, mean_functions=mean_functions,
                                  var_functions=var_functions)

            # Calculate KL divergence between the new and previous model
            kl = OSCBO.model_KL(mu=mu, new_mu=new_mu, F=F, new_F=new_F, cbo_config=cbo_config)

            # Calculate reduction in uncertainty of the interventional spaces
            new_unexplored_regions = sum(list(map(lambda var: OSCBO.observation_span(
                observations_1d=cbo_config.scm.get_observations(observation_set=[var],
                                                                observations=new_observations),
                cbo_config=cbo_config, var=var), cbo_config.scm.manipulative_variables)))

            # Calculate improvement in optimal intervention
            best_intervention_value, _, _, _ = cbo_config.scm.calculate_best_intervention(interventional_gps=new_mu)

            # Calculate observation reward
            region_bonus = cbo_config.observation_acquisition_config.tau_1 * \
                           abs(unexplored_regions - new_unexplored_regions)
            model_bonus = cbo_config.observation_acquisition_config.tau_2 * kl
            best_bonus = cbo_config.observation_acquisition_config.tau_3 * \
                         max(0, cbo_config.scm.best_intervention_target - best_intervention_value)
            observation_reward = region_bonus + model_bonus + best_bonus

            # Compute the next intervention to evaluate based on the updated models
            # selected_intervention_set, selected_intervention_set_idx, selected_intervention_level, \
            # intervention_levels, _ = \
            #     CBOUtil.optimization_policy(cbo_config=cbo_config, interventional_gps=new_mu, state=state)

            # new_x, new_net = get_new_suggested_point(
            #     X=X,
            #     network_observation_at_X=network_observation_at_X,
            #     observation_at_X=observation_at_X,
            #     algo_profile=algo_profile,
            #     env_profile=env_profile,
            #     function_network=function_network,
            #     network_to_objective_transform=network_to_objective_transform,
            #     old_nets=old_nets,
            # )
            # network_observation_at_new_x = function_network(new_x)
            # mean_at_new_x = obj_mean(
            #     new_x, function_network, network_to_objective_transform
            # )
            # observation_at_new_x = network_to_objective_transform(
            #     network_observation_at_new_x
            # )
            # intervention_set_identifiers = list(map(lambda x: int(x), new_x.numpy()[0][0:3].tolist()))
            # if 1 in intervention_set_identifiers:
            #     selected_intervention_set_idx = intervention_set_identifiers.index(1)
            # selected_intervention_set = cbo_config.scm.exploration_set[selected_intervention_set_idx]
            # selected_intervention_level = np.array([network_observation_at_new_x.numpy()[0][selected_intervention_set_idx]])


            # Compute the stopping reward based on the updated \widehat{P} and \widehat{\mu} and
            # the interventional samples
            intervention_cost = cbo_config.scm.costs[constants.CBO.INTERVENE](
                intervention_set=selected_intervention_set,
                intervention_levels=selected_intervention_level)
            interventional_samples = np.array([selected_intervention_level])
            r_stop, ig_stop, mean_stop = OSCBO.stopping_reward(
                interventional_samples=interventional_samples,
                mu=new_mu[selected_intervention_set_idx],
                next_intervention_level=np.array([selected_intervention_level]),
                intervention_cost=intervention_cost,
                kappa=cbo_config.observation_acquisition_config.kappa,
                eta=cbo_config.observation_acquisition_config.eta
            )

            # Save metrics
            stopping_rewards.append(r_stop)
            information_gains.append(ig_stop)
            means.append(mean_stop)
            observation_rewards.append(observation_reward)

        # Average to estimate the mean
        stopping_rewards_mean = float(np.mean(stopping_rewards))
        information_gains_mean = float(np.mean(information_gains))
        means_mean = float(np.mean(means))
        observation_reward_mean = float(np.mean(observation_rewards))

        return stopping_rewards_mean, information_gains_mean, means_mean, observation_reward_mean

    @staticmethod
    def stopping_reward(interventional_samples: np.ndarray, mu: GPyModelWrapper, next_intervention_level: np.ndarray,
                        intervention_cost: float, eta: float = 1, kappa: float = 1) -> Tuple[float, float, float]:
        """
        Computes the stopping Reward (Hammar & Dhir, Section 4).

        :param interventional_samples: the new interventional sample to compute the information gain
        :param mu: the probabilistic model over the objective function
        :param next_intervention_level: the next intervention level according to the optimsation policy
        :param intervention_cost: the intervention cost
        :param eta: the eta hyperparameter
        :param kappa: the kappa hyperparameter
        :return: The stopping reward, the information gain, and the estimated mean of the next intervention
        """
        ig = OSCBO.information_gain(intervention_samples=interventional_samples, mu=mu,
                                    intervention_dim=len(interventional_samples[0]))
        mean, _ = mu.predict(next_intervention_level)
        mean = mean[0][0]
        return eta * ig - kappa * mean - intervention_cost, eta * ig, kappa * mean

    @staticmethod
    def information_gain(intervention_samples: np.ndarray, mu: GPyModelWrapper, intervention_dim: int) -> float:
        """
        Computes the intervention gain of <interventional samples> given the model mu. I.e quantifying
        the reduction in uncertainty about mu from revealing <interventional samples>.
        (Based on Section 2.2 in "Gaussian Process Optimization in the Bandit Setting:
                                  No Regret and Experimental Design")

        :param intervention_samples: the new interventional samples
        :param mu: the probabilistic model
        :param intervention_dim: dimension of the intervention set
        :return: the information gain
        """
        if len(intervention_samples) == 0:
            return 0.0
        A = intervention_samples.reshape(len(intervention_samples), intervention_dim)
        I = np.identity(len(A))
        _, sigma = mu.predict(A)
        _, K_A = mu.predict_with_full_covariance(A)
        try:
            return (1 / 2) * math.log(np.linalg.det(I + K_A.dot(sigma)))
        except Exception:
            return 0.0

    @staticmethod
    def fit_mu(cbo_config: CBOConfig, mean_functions: List[Callable], var_functions: List[Callable]) -> Dict:
        """
        Updates the probabilistic model mu of the objective function

        :param cbo_config: the CBO config
        :param mean_functions: the mean functions in the causal prior of mu
        :param var_functions: the variance functions in the causal prior of mu
        :return: the fitted model
        """
        mu = {}
        for i, intervention_set in enumerate(cbo_config.scm.exploration_set):
            mu[i] = cbo_config.scm.causal_gp_config.create_gp(
                X=cbo_config.scm.interventions[i][constants.CBO.INTERVENTIONS_X_INDEX],
                Y=cbo_config.scm.interventions[i][constants.CBO.INTERVENTIONS_Y_INDEX],
                input_dim=cbo_config.scm.interventions[i][constants.CBO.INTERVENTIONS_X_INDEX][0].shape[0],
                mean_function=mean_functions[i],
                var_function=var_functions[i]
            )
            mu[i].optimize()
        return mu

    @staticmethod
    def fit_F(cbo_config: CBOConfig, observations) -> Dict:
        """
        Updates the probabilistic model over the causal functions F

        :param cbo_config: the CBO config
        :param observations: the observations to use for fitting the model
        :return: the fitted model
        """
        p = {}
        if cbo_config.scm.name == "ToySCM":
            for k, v in cbo_config.scm.observational_gps.items():
                obs = cbo_config.scm.get_observations(observation_set=list(k), observations=observations)
                p[k] = cbo_config.scm.gp_config.create_gp(
                    X=obs[:, 0].reshape(len(obs), 1), Y=obs[:, 1].reshape(len(obs), 1),
                    input_dim=cbo_config.scm.intervention_spaces[k[0]].input_dim
                )
                p[k].optimize()
        else:
            for key, gp in cbo_config.scm.observational_gps.items():
                # conditional_var, target_var = key
                obs = cbo_config.scm.get_observations(list(key), observations=observations)
                columns = []
                for i in range(len(key)-1):
                    columns.append(cbo_config.scm.variables.index(key[i]))
                target_index = cbo_config.scm.variables.index(key[-1])
                p[key] = cbo_config.scm.gp_config.create_gp(
                    X=obs[:, columns].reshape((len(obs), len(columns))),
                    Y=obs[:, target_index].reshape((len(obs), 1)),
                    input_dim=len(columns))
                p[key].optimize()
        return p
