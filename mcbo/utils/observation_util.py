from typing import Tuple, List
import numpy as np
import logging
import os_cbo.constants.constants as constants
from os_cbo.dtos.cbo.cbo_config import CBOConfig
from os_cbo.dtos.cbo.cbo_results import CBOResults
from os_cbo.dtos.acquisition.observation_acquisition_function_type import ObservationAcquisitionFunctionType
from os_cbo.algorithms.oscbo import OSCBO


class ObservationUtil:
    """
    Class with utility functions related to the intervention-observation trade off
    """

    @staticmethod
    def observation_policy(cbo_config: CBOConfig, state: CBOResults, logger: logging.Logger) \
            -> Tuple[bool, float, List[str]]:
        """
        Function representing the observation policy, which decides if the next evaluation should be an
        intervention or an observation in CBO

        :param cbo_config: the CBO config
        :param cbo_results: the CBO results
        :param logger: the logger to use for logging
        :return: A tuple with (True if observe, probability, observation set)
        """
        if cbo_config.observation_acquisition_config.type == \
                ObservationAcquisitionFunctionType.EPSILON_GREEDY_CONVEX_HULL:
            if cbo_config.scm.num_observations() < 3:
                return True, 1, cbo_config.scm.variables
            try:
                epsilon = cbo_config.compute_epsilon_from_convex_hull()
            except Exception as _:
                epsilon = 0.5
            return np.random.uniform(0., 1.) < epsilon, epsilon, cbo_config.scm.variables
        elif cbo_config.observation_acquisition_config.type == \
                ObservationAcquisitionFunctionType.EPSILON_GREEDY:
            if cbo_config.scm.num_observations()  == 0:
                return True, 1, cbo_config.scm.variables
            for intervention_set_idx in range(len(cbo_config.scm.exploration_set)):
                if len(cbo_config.scm.interventions[intervention_set_idx][constants.CBO.INTERVENTIONS_X_INDEX]) == 0:
                    return False, 1, cbo_config.scm.mos(
                        intervention_set=cbo_config.scm.exploration_set[intervention_set_idx])
            epsilon = cbo_config.observation_acquisition_config.epsilon
            return np.random.uniform(0., 1.) < epsilon, epsilon, cbo_config.scm.variables
        elif cbo_config.observation_acquisition_config.type == \
                ObservationAcquisitionFunctionType.OPTIMAL_STOPPING:
            stop, observation_set = OSCBO.os_cbo(state=state, cbo_config=cbo_config, logger=logger)
            return not stop, 1, observation_set
        elif cbo_config.observation_acquisition_config.type == \
                ObservationAcquisitionFunctionType.INITIAL_COLLECTION:
            if state.iteration <= cbo_config.observation_acquisition_config.initial_collection_size:
                return True, 1.0, cbo_config.scm.variables
            else:
                return False, 0.0, cbo_config.scm.variables
        else:
            raise ValueError(f"Observation acquisition policy type: {cbo_config.observation_acquisition_config.type} "
                             f"not recognized")