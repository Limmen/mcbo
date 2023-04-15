import os
import torch
import botorch
from botorch.acquisition.objective import GenericMCObjective
from botorch.settings import debug
from torch import Tensor
import wandb
import argparse
from functions import Dropwave, Alpine2, Ackley, Rosenbrock, ToyGraph, PSAGraph
import warnings

warnings.filterwarnings("ignore")

torch.set_default_dtype(torch.float64)
debug._set_state(True)

from mcbo.mcbo_trial import mcbo_trial
from mcbo.utils import runner_utils


if __name__ == '__main__':
    noise_scale = 0.05
    # Function that maps the network output to the objective value
    network_to_objective_transform = lambda Y: Y[..., -1]
    network_to_objective_transform = GenericMCObjective(network_to_objective_transform)
    env = ToyGraph(noise_scales=noise_scale)
    env_profile = env.get_env_profile()
    def function_network(X: Tensor):
        return env.evaluate(X=X)
    algo_profile = {
        "algo": "NMCBO",
        "seed": 3561912,
        "n_init_evals": 2 * (env_profile["input_dim"] + 1),
        "n_bo_iter": 100,
        "beta": 0.5,
        "initial_obs_samples": 1,
        "initial_int_samples": 1,
        "batch_size": 32,
    }

    mcbo_trial(
        algo_profile=algo_profile,
        env_profile=env_profile,
        function_network=function_network,
        network_to_objective_transform=network_to_objective_transform,
        budget = 300, optimal_value = 2.17,
        variables = ["X", "Z", "Y"]
    )
