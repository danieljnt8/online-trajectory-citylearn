"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import numpy as np
from tqdm import trange


class ReplayBuffer(object):
    def __init__(self, capacity, trajectories=[]):
        self.capacity = capacity
        if len(trajectories) <= self.capacity:
            self.trajectories = trajectories
        else:
            returns = [traj["rewards"].sum() for traj in trajectories]
            sorted_inds = np.argsort(returns)  # lowest to highest
            self.trajectories = [
                trajectories[ii] for ii in sorted_inds[-self.capacity :]
            ]

        self.start_idx = 0

    def __len__(self):
        return len(self.trajectories)

    def add_new_trajs(self, new_trajs):
        if len(self.trajectories) < self.capacity:
            self.trajectories.extend(new_trajs)
            self.trajectories = self.trajectories[-self.capacity :]
        else:
            self.trajectories[
                self.start_idx : self.start_idx + len(new_trajs)
            ] = new_trajs
            self.start_idx = (self.start_idx + len(new_trajs)) % self.capacity

        assert len(self.trajectories) <= self.capacity


class ReplayBufferTrajectory(object):
    def __init__(self, capacity, dataset):
        self.capacity = capacity
        #trajectories must be segment here 
        trajectories, traj_lens = self.segment(dataset["observations"],dataset["actions"],dataset["rewards"],dataset["dones"])
        
        
        if len(trajectories) <= self.capacity:
            self.trajectories = trajectories
            self.traj_lens = traj_lens
        else:
            returns = [traj["rewards"].sum() for traj in trajectories]
            sorted_inds = np.argsort(returns)  # lowest to highest
            self.trajectories = [
                trajectories[ii] for ii in sorted_inds[-self.capacity :]
            ]
            self.traj_lens = [
                traj_lens[ii] for ii in sorted_inds[-self.capacity :]
            ]
            

        self.start_idx = 0

    def __len__(self):
        return len(self.trajectories)

    def segment(self,states, actions, rewards, terminals):
        assert len(states) == len(terminals)
        
        trajectories = []
        episode = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones":[]
        }
    
        for t in trange(len(terminals), desc="Segmenting"):
            episode["observations"].append(states[t])
            episode["actions"].append(actions[t])
            episode["rewards"].append(rewards[t])
            episode["dones"].append(terminals[t])
    
            if terminals[t]:
                # Convert lists to numpy arrays
                episode["observations"] = np.array(episode["observations"])
                episode["actions"] = np.array(episode["actions"])
                episode["rewards"] = np.array(episode["rewards"])
                episode["dones"] = np.array(episode["dones"])
                # Append the current episode to the trajectories list
                trajectories.append(episode)
                # Reset episode
                episode = {
                    "observations": [],
                    "actions": [],
                    "rewards": [],
                    "dones":[]
                }
    
        # If there are any remaining observations, actions, and rewards in the current episode
        if episode["observations"]:
            episode["observations"] = np.array(episode["observations"])
            episode["actions"] = np.array(episode["actions"])
            episode["rewards"] = np.array(episode["rewards"])
            episode["dones"] = np.array(episode["dones"])
            trajectories.append(episode)
        
        trajectories_lens = [len(episode["observations"]) for episode in trajectories]
    
        return trajectories, trajectories_lens

    

    def add_new_dataset(self,dataset ):
        new_trajs, new_traj_lens = self.segment(dataset["observations"],dataset["actions"],dataset["rewards"],dataset["dones"])
        print(len(self.trajectories))
        print(self.capacity)
        if len(self.trajectories) <= self.capacity:
            self.trajectories.extend(new_trajs)
            self.trajectories = self.trajectories[-self.capacity :]

            self.traj_lens.extend(new_traj_lens)
            self.traj_lens = self.traj_lens[-self.capacity:]
            #print(len(self.trajectories))
        else:
            self.trajectories[
                self.start_idx : self.start_idx + len(new_trajs)
            ] = new_trajs

            self.traj_lens[
                self.start_idx : self.start_idx + len(new_traj_lens)
            ] = new_traj_lens
            
            
            self.start_idx = (self.start_idx + len(new_trajs)) % self.capacity

        print(len(self.trajectories))
        print(self.capacity)

        assert len(self.trajectories) <= self.capacity
