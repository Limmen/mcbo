import torch
from botorch.acquisition.objective import GenericMCObjective
from botorch.settings import debug
from torch import Tensor
import math
import datetime
import logging
from mcbo.scms.functions import ToyUCGraph
import warnings
warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)
debug._set_state(True)
from mcbo.mcbo_trial import mcbo_trial
from os_cbo.scms.toy_uc_scm import ToyUCSCM
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


def toygraph_config() -> CBOConfig:
    """
    Gets the CBO  config of the SCM
    """

    # Static hyperparameter constants
    num_obs_per_observation = 1
    evaluation_budget = 200
    num_samples_for_evaluation = 100000
    maximum_number_of_observations = 10
    kappa = 10
    samples_for_mc_estimation = 1
    epsilon = 1.0
    eps_noise_terms = [0.5, 4, 0.05, 0.1]
    lengthscale_rbf_kernel = 1.0
    variance_rbf_kernel = 1
    obs_likelihood_variance = 1e-5
    causal_prior = True
    num_initial_observations = 0
    beta = 1
    lamb = 0.5  # Factor for UCB exploration bonus
    l = 1
    max_horizon = 1
    mc_samples = 1
    seed = 999

    # GP and kernels configuration
    kernel_config = RBFKernelConfig(lengthscale_rbf_kernel=lengthscale_rbf_kernel,
                                    variance_rbf_kernel=variance_rbf_kernel)
    causal_kernel_config = CausalRBFKernelConfig(lengthscale_rbf_kernel=lengthscale_rbf_kernel,
                                                 variance_rbf_kernel=variance_rbf_kernel, ARD=False)
    gp_config = GPConfig(kernel_config=kernel_config, obs_likelihood_variance=obs_likelihood_variance)
    causal_gp_config = CausalGPConfig(gp_config=gp_config, causal_prior=causal_prior,
                                      kernel_config=causal_kernel_config)

    # Costs
    obs_cost = math.pow(2, -2)
    # obs_cost = 1
    intervene_cost = math.pow(2,4)
    intervention_vars_costs = {"X": intervene_cost, "Z": intervene_cost, "Y": intervene_cost}
    observation_vars_costs = {"X": obs_cost, "Z": obs_cost, "Y": obs_cost}

    # Initialize SCM
    toy_scm = ToyUCSCM(gp_config=gp_config,
                     observation_set_type=ObservationSetType.FULL,
                     exploration_set_type=ExplorationSetType.MIS,
                     seed=seed, causal_gp_config=causal_gp_config, eps_noise_terms=eps_noise_terms,
                     intervention_vars_costs=intervention_vars_costs, observation_vars_costs=observation_vars_costs)
    # toy_scm.objective_type = ObjectiveType.MAX


    # Specify observation acquisition config
    observation_acquisition_config = ObservationAcquisitionFunctionConfig(
        kappa=kappa, type=ObservationAcquisitionFunctionType.OSCO, epsilon=epsilon,
        samples_for_mc_estimation=samples_for_mc_estimation, initial_collection_size=num_initial_observations)

    # Specify CBO config
    date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_file_name = f"{date_str}_obs_cost_{math.log(obs_cost, 2)}_int_cost_" \
                    f"{math.log(intervene_cost, 2)}_eps_{'_'.join(list(map(lambda x: str(x), eps_noise_terms)))}" \
                    f"_cbo.log"
    cbo_config = CBOConfig(scm=toy_scm, num_obs_per_observation=num_obs_per_observation,
                           evaluation_budget=evaluation_budget,
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
    env = ToyUCGraph(uc_noise=0.1, noise_scales=noise_scale)
    env_profile = env.get_env_profile()
    def function_network(X: Tensor):
        return env.evaluate(X=X)
    algo_profile = {
        "algo": "NMCBO",
        "seed": 15781,
        "n_init_evals": 2 * (env_profile["input_dim"] + 1),
        "n_bo_iter": 100,
        "beta": 0.5,
        "initial_obs_samples": 0,
        "initial_int_samples": 0,
        "batch_size": 32,
    }

    cbo_config = toygraph_config()

    mcbo_trial(
        algo_profile=algo_profile,
        env_profile=env_profile,
        function_network=function_network,
        network_to_objective_transform=network_to_objective_transform,
        budget = 300,
        variables = ["X", "Z", "Y"], cbo_config=cbo_config
    )
