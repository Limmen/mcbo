import torch
from botorch.acquisition.objective import GenericMCObjective
from botorch.settings import debug
from torch import Tensor
import math
import datetime
import logging
import random
from mcbo.scms.functions import ToyGraph
import warnings
warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)
debug._set_state(True)
import numpy as np
from mcbo.mcbo_trial import mcbo_trial
from os_cbo.scms.toy_scm import ToySCM
from os_cbo.dtos.acquisition.acquisition_function_type import AcquisitionFunctionType
from os_cbo.dtos.acquisition.acquisition_optimizer_type import AcquisitionOptimizerType
from os_cbo.dtos.acquisition.observation_acquisition_function_type import ObservationAcquisitionFunctionType
from os_cbo.dtos.gp.gp_config import GPConfig
from os_cbo.dtos.gp.causal_gp_config import CausalGPConfig
from os_cbo.dtos.kernel.rbf_kernel_config import RBFKernelConfig
from os_cbo.dtos.cbo.exploration_set_type import ExplorationSetType
from os_cbo.dtos.cbo.observation_set_type import ObservationSetType
from os_cbo.dtos.cbo.cbo_config import CBOConfig
from os_cbo.dtos.acquisition.observation_acquistion_function_config import ObservationAcquisitionFunctionConfig
from os_cbo.dtos.kernel.causal_rbf_kernel_config import CausalRBFKernelConfig


def toygraph_config(budget: int, kappa: int, mc_samples: int, l :int, obs_cost: float,
                    intervene_cost: float, seed: int, epsilon: float,
                    observation_acquisition: ObservationAcquisitionFunctionType) -> CBOConfig:
    """
    Gets the SCM config
    """

    # Static hyperparameter constants
    num_obs_per_observation = 1
    num_samples_for_evaluation = 100000
    maximum_number_of_observations = 10
    eps_noise_terms = [0.5, 4, 0.05]
    lengthscale_rbf_kernel = 1.0
    variance_rbf_kernel = 1
    obs_likelihood_variance = 1e-5
    causal_prior = True
    num_initial_observations = 0
    beta = 1
    lamb = 0.5  # Factor for UCB exploration bonus
    max_horizon = 1
    seed = seed

    # GP and kernels configuration
    kernel_config = RBFKernelConfig(lengthscale_rbf_kernel=lengthscale_rbf_kernel,
                                    variance_rbf_kernel=variance_rbf_kernel)
    causal_kernel_config = CausalRBFKernelConfig(lengthscale_rbf_kernel=lengthscale_rbf_kernel,
                                                 variance_rbf_kernel=variance_rbf_kernel, ARD=False)
    gp_config = GPConfig(kernel_config=kernel_config, obs_likelihood_variance=obs_likelihood_variance)
    causal_gp_config = CausalGPConfig(gp_config=gp_config, causal_prior=causal_prior,
                                      kernel_config=causal_kernel_config)

    # Costs
    intervention_vars_costs = {"X": intervene_cost, "Z": intervene_cost, "Y": intervene_cost}
    observation_vars_costs = {"X": obs_cost, "Z": obs_cost, "Y": obs_cost}

    # Initialize SCM
    toy_scm = ToySCM(gp_config=gp_config,
                     observation_set_type=ObservationSetType.FULL,
                     exploration_set_type=ExplorationSetType.MIS,
                     seed=seed, causal_gp_config=causal_gp_config, eps_noise_terms=eps_noise_terms,
                     intervention_vars_costs=intervention_vars_costs, observation_vars_costs=observation_vars_costs)
    # toy_scm.objective_type = ObjectiveType.MAX


    # Specify observation acquisition config
    observation_acquisition_config = ObservationAcquisitionFunctionConfig(
        kappa=kappa, type=observation_acquisition, epsilon=epsilon,
        samples_for_mc_estimation=mc_samples, initial_collection_size=num_initial_observations)

    # Specify CBO config
    date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_file_name = f"{date_str}_obs_cost_{math.log(obs_cost, 2)}_int_cost_" \
                    f"{math.log(intervene_cost, 2)}_eps_{'_'.join(list(map(lambda x: str(x), eps_noise_terms)))}" \
                    f"_cbo.log"
    cbo_config = CBOConfig(scm=toy_scm, num_obs_per_observation=num_obs_per_observation,
                           evaluation_budget=budget,
                           max_num_obs=maximum_number_of_observations,
                           acquisition_function_type=AcquisitionFunctionType.NEGATIVE_LOWER_CONFIDENCE_BOUND,
                           acquisition_optimizer_type=AcquisitionOptimizerType.GRADIENT,
                           num_samples_for_evaluation=num_samples_for_evaluation,
                           log_file=log_file_name, log_level=logging.INFO,
                           observation_acquisition_config=observation_acquisition_config,
                           lamb=lamb, beta=beta, l=l, mc_samples=mc_samples, max_horizon=max_horizon)
    return cbo_config


if __name__ == '__main__':
    noise_scale = torch.tensor([0.5, 4, 0.05])
    # Function that maps the network output to the objective value
    network_to_objective_transform = lambda Y: Y[..., -1]
    network_to_objective_transform = GenericMCObjective(network_to_objective_transform)
    env = ToyGraph(noise_scales=noise_scale)
    env_profile = env.get_env_profile()
    def function_network(X: Tensor):
        return env.evaluate(X=X)
    seeds = [561512, 351, 5126, 2350, 16391, 52101, 3520210, 11124, 61912, 888812, 235610, 12511,
             44102, 21501, 5112, 35011,
             7776612, 22212, 2019850, 98212, 333901]
    intervene_costs = [math.pow(2, 4)]
    observation_costs = [math.pow(2, -2), math.pow(2, 0), math.pow(2, 2), math.pow(2, 4)]
    observation_acquisitions = [
        ObservationAcquisitionFunctionType.EPSILON_GREEDY,
        ObservationAcquisitionFunctionType.EPSILON_GREEDY,
        ObservationAcquisitionFunctionType.EPSILON_GREEDY,
        ObservationAcquisitionFunctionType.EPSILON_GREEDY_CONVEX_HULL,
        ObservationAcquisitionFunctionType.OSCO]
    acquisitions = [AcquisitionFunctionType.CAUSAL_EXPECTED_IMPROVEMENT]
    epsilons = [0, 0.99, 0.5, 1, 1]
    evaluation_budgets = [75, 150, 300]
    l_values = [1]
    mc_samples_values = [1]
    kappa_values = [10]
    exploration_sets = [ExplorationSetType.MIS]
    observation_sets = [ObservationSetType.FULL]
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        for budget in evaluation_budgets:
            for i, obs_acquisition in enumerate(observation_acquisitions):
                epsilon = epsilons[i]
                for exploration_set in exploration_sets:
                    for observation_set in observation_sets:
                        for j, acquisition in enumerate(acquisitions):
                            for intervene_cost in intervene_costs:
                                for l in l_values:
                                    for mc_samples in mc_samples_values:
                                        for observation_cost in observation_costs:
                                            for kappa in kappa_values:
                                                cbo_config = toygraph_config(budget=budget, kappa=kappa, l=l,
                                                                             obs_cost=observation_cost,
                                                                             intervene_cost=intervene_cost, seed=seed,
                                                                             mc_samples=mc_samples, epsilon=epsilon,
                                                                             observation_acquisition=obs_acquisition)
                                                initial_obs_samples = 0
                                                initial_int_samples = 0
                                                if obs_acquisition != ObservationAcquisitionFunctionType.OSCO:
                                                    initial_obs_samples = 2
                                                algo_profile = {
                                                    "algo": "NMCBO",
                                                    "seed": seed,
                                                    "n_init_evals": 2 * (env_profile["input_dim"] + 1),
                                                    "n_bo_iter": 100,
                                                    "beta": 0.5,
                                                    "initial_obs_samples": initial_obs_samples,
                                                    "initial_int_samples": initial_int_samples,
                                                    "batch_size": 32,
                                                }
                                                mcbo_trial(
                                                    algo_profile=algo_profile,
                                                    env_profile=env_profile,
                                                    function_network=function_network,
                                                    network_to_objective_transform=network_to_objective_transform,
                                                    budget = budget,
                                                    variables = ["X", "Z", "Y"], cbo_config=cbo_config
                                                )
