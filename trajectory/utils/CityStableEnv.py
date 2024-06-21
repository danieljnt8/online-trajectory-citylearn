import gym
from citylearn.citylearn import CityLearnEnv
import itertools
import numpy as np
from citylearn.agents.base import BaselineAgent as Agent
## current schema 
schema = "citylearn_challenge_2022_phase_2"

def action_space_to_dict(aspace):
    """ Only for box space """
    return { "high": aspace.high,
             "low": aspace.low,
             "shape": aspace.shape,
             "dtype": str(aspace.dtype)
    }

def env_reset(env):
    observations = env.reset()
    action_space = env.action_space
    observation_space = env.observation_space
    #building_info = env.get_building_information()
    #building_info = list(building_info.values())
    action_space_dicts = [action_space_to_dict(asp) for asp in action_space]
    observation_space_dicts = [action_space_to_dict(osp) for osp in observation_space]
    obs_dict = {"action_space": action_space_dicts,
                "observation_space": observation_space_dicts,
              #  "building_info": building_info,
                "observation": observations }
    return obs_dict


index_commun = [0, 2, 19, 4, 8, 24]
index_particular = [20, 21, 22, 23]

normalization_value_commun = [12, 24, 2, 100, 100, 1]
normalization_value_particular = [5, 5, 5, 5]

len_tot_index = len(index_commun) + len(index_particular) * 5

## env wrapper for stable baselines
class EnvCityGym(gym.Env):
    """
    Env wrapper coming from the gym library.
    """
    def __init__(self, env):
        self.env = env

        # get the number of buildings
        self.num_buildings = len(env.action_space)

        # define action and observation space
        self.action_space = gym.spaces.Box(low=np.array([-1] * self.num_buildings), high=np.array([1] * self.num_buildings), dtype=np.float32)

        # define the observation space
        self.observation_space = gym.spaces.Box(low=np.array([0] * len_tot_index), high=np.array([1] * len_tot_index), dtype=np.float32)

        # TO THINK : normalize the observation space
        self.current_obs = None
    def reset(self):
        obs_dict = env_reset(self.env)
        obs = self.env.reset()

        observation = self.get_observation(obs)
        
        self.current_obs = observation
        self.interactions = []

        return observation

    def get_observation(self, obs):
        """
        We retrieve new observation from the building observation to get a proper array of observation
        Basicly the observation array will be something like obs[0][index_commun] + obs[i][index_particular] for i in range(5)

        The first element of the new observation will be "commun observation" among all building like month / hour / carbon intensity / outdoor_dry_bulb_temperature_predicted_6h ...
        The next element of the new observation will be the concatenation of certain observation specific to buildings non_shiftable_load / solar_generation / ...  
        """
        
        # we get the observation commun for each building (index_commun)
        observation_commun = [obs[0][i]/n for i, n in zip(index_commun, normalization_value_commun)]
        observation_particular = [[o[i]/n for i, n in zip(index_particular, normalization_value_particular)] for o in obs]

        observation_particular = list(itertools.chain(*observation_particular))
        # we concatenate the observation
        observation = observation_commun + observation_particular

        return observation

    def step(self, action):
        """
        we apply the same action for all the buildings
        """
        # reprocessing action
        action = [[act] for act in action]

        # we do a step in the environment
        obs, reward, done, info = self.env.step(action)
        
        observation = self.get_observation(obs)
        
        
        self.interactions.append({
            "observations": self.current_obs,
            "next_observations": self.get_observation(obs),  # Assuming next observation is same as current for simplicity
            "actions": action,
            "rewards": reward,
            "dones": done,
            "info": info
        })
        
        self.current_obs = observation
        
        

        return observation, sum(reward), done, info
        
    def render(self, mode='human'):
        return self.env.render(mode)