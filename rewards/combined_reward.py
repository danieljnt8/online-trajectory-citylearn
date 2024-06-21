from typing import Any, List, Mapping, Tuple, Union

from agents.rbc import RBCAgent1 as Agent
from citylearn.citylearn import CityLearnEnv
import numpy as np

class CombinedReward:
    r"""Base and default reward function class.

    The default reward is the electricity consumption from the grid at the current time step returned as a negative value.

    Parameters
    ----------
    env_metadata: Mapping[str, Any]:
        General static information about the environment.
    **kwargs : dict
        Other keyword arguments for custom reward calculation.
    """
    
    def __init__(self, env_metadata: Mapping[str, Any], exponent: float = None, **kwargs):
        self.env_metadata = env_metadata
        self.exponent = exponent
        self.electricity_consumption_history = []

    @property
    def env_metadata(self) -> Mapping[str, Any]:
        """General static information about the environment."""

        return self.__env_metadata
    
    @property
    def central_agent(self) -> bool:
        """Expect 1 central agent to control all buildings."""

        return self.env_metadata['central_agent']
    
    @property
    def exponent(self) -> float:
        return self.__exponent
    
    @env_metadata.setter
    def env_metadata(self, env_metadata: Mapping[str, Any]):
        self.__env_metadata = env_metadata

    @exponent.setter
    def exponent(self, exponent: float):
        self.__exponent = 1.0 if exponent is None else exponent

    def reset(self):
        """Use to reset variables at the start of an episode."""

        self.electricity_consumption_history = []


    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        r"""Calculates reward.

        Parameters
        ----------
        observations: List[Mapping[str, Union[int, float]]]
            List of all building observations at current :py:attr:`citylearn.citylearn.CityLearnEnv.
            time_step` that are got from calling :py:meth:`citylearn.building.Building.observations`.

        Returns
        -------
        reward: List[float]
            Reward for transition to current timestep.
        """
        #print(observations)

        net_electricity_consumption = [o['net_electricity_consumption'] for o in observations]
        carbon_emission = [o['net_electricity_consumption'] * o['carbon_intensity'] for o in observations]
        electricity_cost = [o['net_electricity_consumption'] * o['electricity_pricing'] for o in observations]


        net_cost = -1 * np.array(net_electricity_consumption).clip(min=0)
        carbon_cost = -1 * np.array(carbon_emission).clip(min=0)
        price_cost = -1 * np.array(electricity_cost).clip(min=0)

        self.electricity_consumption_history.append(net_electricity_consumption)
        ramping_cost = np.zeros(len(observations))
        if len(self.electricity_consumption_history) >= 2:
            ramping_cost = - 1 * abs(
                np.array(self.electricity_consumption_history[-1]) - np.array(self.electricity_consumption_history[-2]))

        load_factor_cost = -1 * np.array(net_electricity_consumption).clip(min=0) ** 2

        reward_list = 1 / 4 * net_cost + 1 / 4 * carbon_cost + 1 / 4 * price_cost + 1 / 8 * ramping_cost + 1 / 8 * load_factor_cost


        #reward_list = [-(max(o, 0)**self.exponent) for o in net_electricity_consumption]
        
        
        

        if self.central_agent:
            reward = [sum(reward_list)]
        else:
            reward = reward_list

        return reward
