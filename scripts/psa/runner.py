import os
import torch
import botorch
from botorch.acquisition.objective import GenericMCObjective
from botorch.settings import debug
from torch import Tensor
import wandb
import math
import datetime
import logging
import random
import argparse
from functions import Dropwave, Alpine2, Ackley, Rosenbrock, ToyGraph, PSAGraph
import warnings
warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)
debug._set_state(True)
from mcbo.mcbo_trial_two import mcbo_trial
from mcbo.utils import runner_utils
from os_cbo.algorithms.cbo import CBO
from os_cbo.scms.psa_scm import PSASCM
from os_cbo.dtos.acquisition.acquisition_function_type import AcquisitionFunctionType
from os_cbo.dtos.acquisition.acquisition_optimizer_type import AcquisitionOptimizerType
from os_cbo.dtos.acquisition.observation_acquisition_function_type import ObservationAcquisitionFunctionType
from os_cbo.dtos.gp.gp_config import GPConfig
from os_cbo.dtos.gp.causal_gp_config import CausalGPConfig
from os_cbo.dtos.kernel.rbf_kernel_config import RBFKernelConfig
from os_cbo.dtos.cbo.exploration_set_type import ExplorationSetType
from os_cbo.dtos.cbo.cbo_config import CBOConfig
from os_cbo.plotting.plot_cbo_toygraph_results import plot_cbo_toygraph_gps, plot_toygraph_cbo_results, \
    plot_toygraph_cbo_optimal_stopping, plot_toygraph_trajectories
from os_cbo.dtos.acquisition.observation_acquistion_function_config import ObservationAcquisitionFunctionConfig
from os_cbo.dtos.kernel.causal_rbf_kernel_config import CausalRBFKernelConfig
from os_cbo.dtos.optimization.objective_type import ObjectiveType


def psa_config() -> CBOConfig:
    # Static hyperparameter constants
    num_obs_per_observation = 1
    evaluation_budget = 300
    num_samples_for_evaluation = 100000
    maximum_number_of_observations = 20
    kappa = 1
    eta = 2
    delta = 0
    samples_for_mc_estimation = 10
    epsilon = 1.0
    eps_noise_terms = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1] # Gaussian white noise, the numbers are standard deviations
    # eps_noise_terms = [0.1, 0, 0] # Gaussian white noise, the numbers are standard deviations
    lengthscale_rbf_kernel = 1. # Larger lengthscale means correlation of close points decays quicker
    variance_rbf_kernel = 1  # Variance term multiplying the exponential in the rbf kernel
    # obs_likelihood_variance=1e-10
    obs_likelihood_variance=1e-5
    # obs_likelihood_variance = 1
    causal_prior = True
    num_initial_observations = 0
    beta = 5
    lamb = 0.5  # Factor for UCB exploration bonus
    tau_1 = 5 # Factor for region bonus
    tau_2 = 5 # Factor for model bonus
    tau_3 = 10 # Factor for best bonus
    seed = 0

    # GP and kernels configuration
    kernel_config = RBFKernelConfig(lengthscale_rbf_kernel=lengthscale_rbf_kernel,
                                    variance_rbf_kernel=variance_rbf_kernel)
    causal_kernel_config = CausalRBFKernelConfig(lengthscale_rbf_kernel=lengthscale_rbf_kernel,
                                                 variance_rbf_kernel=variance_rbf_kernel, ARD=False)
    gp_config = GPConfig(kernel_config=kernel_config, obs_likelihood_variance=obs_likelihood_variance)
    causal_gp_config = CausalGPConfig(gp_config=gp_config, causal_prior=causal_prior,
                                      kernel_config=causal_kernel_config)

    # Costs
    obs_cost = math.pow(2, 2)
    intervene_cost = math.pow(2,4)
    intervention_vars_costs = {"C": intervene_cost, "D": intervene_cost}
    observation_vars_costs = {
        "A": obs_cost, "B": obs_cost, "C": obs_cost, "D": obs_cost, "E": obs_cost, "F": obs_cost
    }

    # Initialize SCM
    psa_scm = PSASCM(gp_config=gp_config,
                     exploration_set_type=ExplorationSetType.POMIS,
                     seed=seed, causal_gp_config=causal_gp_config, eps_noise_terms=eps_noise_terms,
                     intervention_vars_costs=intervention_vars_costs, observation_vars_costs=observation_vars_costs)
    # toy_scm.objective_type = ObjectiveType.MAX


    # Specify observation acquisition config
    observation_acquisition_config = ObservationAcquisitionFunctionConfig(
        kappa=kappa, eta=eta, type=ObservationAcquisitionFunctionType.OPTIMAL_STOPPING, epsilon=epsilon,
        samples_for_mc_estimation=samples_for_mc_estimation, initial_collection_size=num_initial_observations,
        delta=delta, tau_1=tau_1, tau_2=tau_2, tau_3=tau_3)

    # Specify CBO config
    date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_file_name = f"{date_str}_obs_cost_{math.log(obs_cost, 2)}_int_cost_" \
                    f"{math.log(intervene_cost, 2)}_eps_{'_'.join(list(map(lambda x: str(x), eps_noise_terms)))}" \
                    f"_cbo.log"
    cbo_config = CBOConfig(scm=psa_scm, num_obs_per_observation=num_obs_per_observation,
                           evaluation_budget=evaluation_budget,
                           max_num_obs=maximum_number_of_observations,
                           acquisition_function_type=AcquisitionFunctionType.NEGATIVE_LOWER_CONFIDENCE_BOUND,
                           acquisition_optimizer_type=AcquisitionOptimizerType.GRADIENT,
                           num_samples_for_evaluation=num_samples_for_evaluation,
                           log_file=log_file_name, log_level=logging.INFO,
                           observation_acquisition_config=observation_acquisition_config,
                           lamb=lamb, beta=beta)
    return cbo_config


if __name__ == '__main__':
    noise_scale = torch.tensor([1, 0.5, 0.1])
    # Function that maps the network output to the objective value
    network_to_objective_transform = lambda Y: Y[..., -1]
    network_to_objective_transform = GenericMCObjective(network_to_objective_transform)
    env = PSAGraph()
    env_profile = env.get_env_profile()
    def function_network(X: Tensor):
        return env.evaluate(X=X)
    algo_profile = {
        "algo": "NMCBO",
        "seed": 61752433,
        "n_init_evals": 2 * (env_profile["input_dim"] + 1),
        "n_bo_iter": 100,
        "beta": 0.5,
        "initial_obs_samples": 1,
        "initial_int_samples": 1,
        "batch_size": 32,
    }

    cbo_config = psa_config()

    mcbo_trial(
        algo_profile=algo_profile,
        env_profile=env_profile,
        function_network=function_network,
        network_to_objective_transform=network_to_objective_transform,
        budget = 300, optimal_value = -5.16,
        variables = ["A", "B", "C", "D", "E", "F"], cbo_config=cbo_config
    )
