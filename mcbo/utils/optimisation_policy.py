import numpy as np
import torch
from botorch.acquisition import (
    ExpectedImprovement,
    qExpectedImprovement,
    qKnowledgeGradient,
    UpperConfidenceBound,
    qSimpleRegret,
)
from botorch.acquisition import PosteriorMean as GPPosteriorMean
from botorch.sampling import SobolQMCNormalSampler
from torch import Tensor
from typing import Callable, List
from mcbo.acquisition_function_optimization.optimize_acqf import (
    optimize_acqf_and_get_suggested_point,
)
from mcbo.utils.fit_gp_model import fit_gp_model
from mcbo.models.gp_network import GaussianProcessNetwork
from mcbo.utils.posterior_mean import PosteriorMean
from mcbo.models import eta_network

def get_new_suggested_point(
        X: Tensor,
        network_observation_at_X: Tensor,
        observation_at_X: Tensor,
        algo_profile: dict,
        env_profile: dict,
        function_network: Callable,
        network_to_objective_transform: Callable,
        old_nets: List,
) -> Tensor:

    algo = algo_profile["algo"]

    algos_interventions_not_implemented = ["UCB, KG, EI, EICF"]

    if env_profile["interventional"] and algo in algos_interventions_not_implemented:
        raise ValueError(
            "This algorithm is not implemented for the interventional setting"
        )

    if algo == "Random":
        return get_new_suggested_point_random(env_profile)

    model, acq_fun_input_dim = get_model(
        X, network_observation_at_X, observation_at_X, algo_profile, env_profile
    )

    if algo in algos_interventions_not_implemented:
        acquisition_function, posterior_mean_function = get_acq_fun(
            model,
            network_to_objective_transform,
            observation_at_X,
            algo_profile,
            env_profile,
        )
        new_x, new_score = optimize_acqf_and_get_suggested_point(
            acq_func=acquisition_function,
            bounds=torch.tensor(
                [
                    [0.0 for i in range(acq_fun_input_dim)],
                    [1.0 for i in range(acq_fun_input_dim)],
                ]
            ),
            batch_size=1,
            posterior_mean=posterior_mean_function,
        )
        return new_x, None

    r"""
    The approach is general for both causal BO environments (interventional=True) and 
    function network environments. For both we loop of the possible intervention targets.
    The 'target' for function networks makes no difference to the output and only a 
    single set of targets is contained in valid_targets. 
    """

    best_x = None
    best_score = -torch.inf
    best_target = None

    for target in env_profile["valid_targets"]:
        try:
            model.set_target(target)
        except:
            raise ValueError("Model class isn't able to have interventional targets")
        ## Use a different training procedure if noisy MCBO is used.
        if algo == "NMCBO":
            new_x, new_score, new_net = eta_network.train(
                model,
                network_to_objective_transform,
                env_profile,
                acq_fun_input_dim,
                old_nets,
                batch_size=algo_profile["batch_size"],
            )
        else:
            acquisition_function, posterior_mean_function = get_acq_fun(
                model,
                network_to_objective_transform,
                observation_at_X,
                algo_profile,
                env_profile,
            )

            new_x, new_score = optimize_acqf_and_get_suggested_point(
                acq_func=acquisition_function,
                bounds=torch.tensor(
                    [
                        [0.0 for i in range(acq_fun_input_dim)],
                        [1.0 for i in range(acq_fun_input_dim)],
                    ]
                ),
                batch_size=1,
                posterior_mean=posterior_mean_function,
            )
        if new_score > best_score:
            best_score = new_score
            best_x = new_x
            best_target = torch.tensor(target)

    r"""
    For algorithms that hallucinate inputs, we need to remove these parts of the action
    because they are not used in the real environment. 
    """

    if algo in ["NMCBO", "MCBO"]:
        if env_profile["interventional"]:
            r"""
            For causal BO we also ignore the target dimension which the env_profile
            counts in the input_dim.
            """
            X_dim = env_profile["input_dim"] - env_profile["dag"].get_n_nodes()
            best_x = best_x[:, 0:X_dim]
        else:
            X_dim = env_profile["input_dim"]
            best_x = best_x[:, 0:X_dim]

    # If we're in a causal BO setting to need to prepend the best targets to our ation.
    if env_profile["interventional"]:
        best_x = torch.cat([best_target.unsqueeze(0), best_x], dim=-1)

    # Only NMCBO stores the action and eta networks previously used.
    if algo != "NMCBO":
        new_net = None

    return best_x, new_net


def get_acq_fun(
        model, network_to_objective_transform, observation_at_X, algo_profile, env_profile
):
    algo = algo_profile["algo"]
    if algo == "EIFN":
        # Sampler
        qmc_sampler = SobolQMCNormalSampler(num_samples=128)
        # Acquisition function
        acquisition_function = qExpectedImprovement(
            model=model,
            best_f=observation_at_X.max().item(),
            sampler=qmc_sampler,
            objective=network_to_objective_transform,
        )
        posterior_mean_function = PosteriorMean(
            model=model,
            sampler=qmc_sampler,
            objective=network_to_objective_transform,
        )
    elif algo == "EICF":
        qmc_sampler = SobolQMCNormalSampler(num_samples=128)
        acquisition_function = qExpectedImprovement(
            model=model,
            best_f=observation_at_X.max().item(),
            sampler=qmc_sampler,
            objective=network_to_objective_transform,
        )
        posterior_mean_function = PosteriorMean(
            model=model,
            sampler=qmc_sampler,
            objective=network_to_objective_transform,
        )
    elif algo == "EI":
        acquisition_function = ExpectedImprovement(
            model=model, best_f=observation_at_X.max().item()
        )
        posterior_mean_function = GPPosteriorMean(model=model)
    elif algo == "KG":
        acquisition_function = qKnowledgeGradient(model=model, num_fantasies=8)
        posterior_mean_function = GPPosteriorMean(model=model)
    elif algo == "UCB":
        acquisition_function = UpperConfidenceBound(
            model=model, beta=algo_profile["beta"]
        )
        posterior_mean_function = None
    elif algo == "MCBO":
        qmc_sampler = SobolQMCNormalSampler(num_samples=128)
        # Acquisition function
        acquisition_function = qSimpleRegret(
            model=model,
            sampler=qmc_sampler,
            objective=network_to_objective_transform,
        )

        posterior_mean_function = None

    return acquisition_function, posterior_mean_function


def get_new_suggested_point_random(env_profile) -> Tensor:
    r"""Returns a new suggested point randomly."""
    if env_profile["interventional"]:
        # For interventional random = random targets, random values
        target = np.random.choice(env_profile["valid_targets"])
        return mcbo.utils.initial_design.random_causal(target, env_profile), None
    else:
        return torch.rand([1, env_profile["input_dim"]]), None


def obj_mean(
        X: Tensor, function_network: Callable, network_to_objective_transform: Callable
) -> Tensor:
    '''
    Estimates the mea value of a noisy objective by computing it a lot of times and
    averaging.
    '''
    X = X[None]  # create new 0th dim
    Ys = function_network(X.repeat(100000, 1, 1))
    return torch.mean(network_to_objective_transform(Ys), dim=0)


def get_model(
        X: Tensor,
        network_observation_at_X: Tensor,
        observation_at_X: Tensor,
        algo_profile: dict,
        env_profile: dict,
):
    input_dim = env_profile["input_dim"]
    algo = algo_profile["algo"]
    if algo == "EIFN":
        model = GaussianProcessNetwork(
            train_X=X,
            train_Y=network_observation_at_X,
            algo_profile=algo_profile,
            env_profile=env_profile,
        )
    elif algo == "EICF":
        model = fit_gp_model(X=X, Y=network_observation_at_X)
    elif algo == "EI":
        model = fit_gp_model(X=X, Y=observation_at_X)
    elif algo == "KG":
        model = fit_gp_model(X=X, Y=observation_at_X)
    elif algo == "UCB":
        model = fit_gp_model(X=X, Y=observation_at_X)
    elif algo == "MCBO":
        model = GaussianProcessNetwork(
            train_X=X,
            train_Y=network_observation_at_X,
            algo_profile=algo_profile,
            env_profile=env_profile,
        )
        # Make the input dimension bigger to account for the hallucination inputs.
        input_dim = input_dim + env_profile["dag"].get_n_nodes()
    elif algo == "NMCBO":
        model = GaussianProcessNetwork(
            train_X=X,
            train_Y=network_observation_at_X,
            algo_profile=algo_profile,
            env_profile=env_profile,
        )
    else:
        raise ValueError("No algorithm of this name implemented")
    # Remove the binary target inputs for interventions
    if env_profile["interventional"]:
        input_dim = input_dim - env_profile["dag"].get_n_nodes()
    return model, input_dim

