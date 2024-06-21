"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

from torch.utils.tensorboard import SummaryWriter
import argparse
import pickle
import random
import time
import gym
import wandb
import torch
import numpy as np

from datasets import load_from_disk
import datasets
from omegaconf import OmegaConf
from trajectory.utils.dataset_helper import DiscretizedDataset

from trajectory.models.gpt import GPT, GPTTrainer

from trajectory.utils.common import pad_along_axis
from trajectory.utils.discretization import KBinsDiscretizer

import utils
from replay_buffer import ReplayBuffer,ReplayBufferTrajectory
from lamb import Lamb
#from stable_baselines3.common.vec_env import SubprocVecEnv
from pathlib import Path
from data import create_dataloader
from decision_transformer.models.decision_transformer import DecisionTransformer
#from evaluation import create_vec_eval_episodes_fn, vec_evaluate_episode_rtg
from evaluation_citylearn import create_eval_episodes_fn,evaluate_episode_rtg
from trainer import SequenceTrainerCustom, SequenceTrainerCustom
from logger import Logger
from wrappers_custom import *
from utils_.helpers import *
from torch.utils.data import DataLoader

from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import *
from utils_.variant_dict import variant
from trajectory.datasets.discretized import DiscretizedDatasetTrajectory
from eval_traj import aug_trajectory


MAX_EPISODE_LEN = 8760


class Experiment:
    def __init__(self, variant,dataset_path,config_path = "configs/medium/city_learn_traj_testing.yaml"):

        config = OmegaConf.load(config_path)
        self.config = config
        self.eval_config = OmegaConf.load(variant["eval_config"])

        self.device = variant.get("device", "cuda")

        

        if dataset_path is None :
            offline_data_path = config.trainer.offline_data_path
        else:
            offline_data_path = dataset_path
        #if torch.cuda.is_available():
        #    device = "cuda:1"

  
        wandb.init(
                **config.wandb,
                config=dict(OmegaConf.to_container(config, resolve=True))
            )
        
        offline_data_path = offline_data_path
        dataset = load_from_disk(offline_data_path)
        self.replay_buffer = ReplayBufferTrajectory(8,dataset)

        datasets = DiscretizedDatasetTrajectory(replay_buffer=self.replay_buffer,discount = config.dataset.discount, seq_len = config.dataset.seq_len, strategy = config.dataset.strategy)
        if self.device == "cpu":
            dataloader = DataLoader(datasets,  batch_size=1, shuffle=True)
        else:
            dataloader = DataLoader(datasets,  batch_size=config.dataset.batch_size, shuffle=True, num_workers=8, pin_memory=True)


        trainer_conf = config.trainer
        data_conf = config.dataset
        self.model = GPT(**config.model)
        self.model.to(self.device)
        num_epochs = config.trainer.num_epochs
        warmup_tokens = len(datasets) * data_conf.seq_len * config.model.transition_dim
        final_tokens = warmup_tokens * num_epochs


        self.trainer = GPTTrainer(
            final_tokens=final_tokens,
            warmup_tokens=warmup_tokens,
            action_weight=trainer_conf.action_weight,
            value_weight=trainer_conf.value_weight,
            reward_weight=trainer_conf.reward_weight,
            learning_rate=trainer_conf.lr,
            betas=trainer_conf.betas,
            weight_decay=trainer_conf.weight_decay,
            clip_grad=trainer_conf.clip_grad,
            eval_seed=trainer_conf.eval_seed,
            eval_every=trainer_conf.eval_every,
            eval_episodes=trainer_conf.eval_episodes,
            eval_temperature=trainer_conf.eval_temperature,
            eval_discount=trainer_conf.eval_discount,
            eval_plan_every=trainer_conf.eval_plan_every,
            eval_beam_width=trainer_conf.eval_beam_width,
            eval_beam_steps=trainer_conf.eval_beam_steps,
            eval_beam_context=trainer_conf.eval_beam_context,
            eval_sample_expand=trainer_conf.eval_sample_expand,
            eval_k_obs=trainer_conf.eval_k_obs,  # as in original implementation
            eval_k_reward=trainer_conf.eval_k_reward,
            eval_k_act=trainer_conf.eval_k_act,
            checkpoints_path=trainer_conf.checkpoints_path,
            save_every=1,
            device=self.device
        )

        

        env = CityLearnEnv(schema="citylearn_challenge_2022_phase_2")
        env.central_agent = True
        env = NormalizedObservationWrapper(env)
        env = StableBaselines3WrapperCustom(env)

       
       

        self.aug_trajs = []

      
        
        # track the training progress and
        # training/evaluation/online performance in all the iterations
        self.pretrain_iter = 0
        self.online_iter = 0
        self.total_transitions_sampled = 0
        self.variant = variant
        
        self.logger = Logger(variant)
    
    def _get_initial_trajectories(self,dataset_path):
        dataset = load_from_disk(dataset_path)
        dataset,_ = segment_v2(dataset["observations"],dataset["actions"],dataset["rewards"],dataset["dones"])
        trajectories = datasets.Dataset.from_dict({k: [s[k] for s in dataset] for k in dataset[0].keys()})

        return trajectories
    
    
   
    def _get_env_spec(self,env):
        state_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        action_range = [
                float(env.action_space.low.min()) ,
                float(env.action_space.high.max()) ,
            ]
        return state_dim,act_dim, action_range

    def _save_model(self, path_prefix, is_pretrain_model=False):
        to_save = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "pretrain_iter": self.pretrain_iter,
            "online_iter": self.online_iter,
            "args": self.variant,
            "total_transitions_sampled": self.total_transitions_sampled,
            "np": np.random.get_state(),
            "python": random.getstate(),
            "pytorch": torch.get_rng_state(),
            "log_temperature_optimizer_state_dict": self.log_temperature_optimizer.state_dict(),
        }

        with open(f"{path_prefix}/model.pt", "wb") as f:
            torch.save(to_save, f)
        print(f"\nModel saved at {path_prefix}/model.pt")

        if is_pretrain_model:
            with open(f"{path_prefix}/pretrain_model.pt", "wb") as f:
                torch.save(to_save, f)
            print(f"Model saved at {path_prefix}/pretrain_model.pt")




    def pretrain(self,schema_eval = "citylearn_challenge_2022_phase_2"):
        print("\n\n\n*** Pretrain ***")


        datasets = DiscretizedDatasetTrajectory(replay_buffer=self.replay_buffer,discount = self.config.dataset.discount, seq_len = self.config.dataset.seq_len, strategy = self.config.dataset.strategy)
        if self.device == "cpu":
            dataloader = DataLoader(datasets,  batch_size=1, shuffle=True)
        else:
            dataloader = DataLoader(datasets,  batch_size=self.config.dataset.batch_size, shuffle=True, num_workers=8, pin_memory=True)

        self.discretizer = datasets.get_discretizer()

        self.trainer.train(
            model=self.model,
            dataloader=dataloader,
            num_epochs = self.variant["max_pretrain_iters"]
        )

        env_dataset,df_evaluate = aug_trajectory(model = self.model,discretizer = self.discretizer,config = self.eval_config,schema= schema_eval,device = self.device)


        torch.save(self.discretizer, self.logger.log_path+f"/discretizer_pretrain.pt")
        torch.save(self.model.state_dict(), self.logger.log_path+f"/model_pretrain.pt")
        df_evaluate.to_csv(self.logger.log_path+f"/pretrained_eval.csv")

            




    def online_tuning(self,  online_schema, schema_eval):

        print("\n\n\n*** Online Finetuning ***")
        
        """
        writer = (
            SummaryWriter(self.logger.log_path) if self.variant["log_to_tb"] else None
        )
        """

        
        while self.online_iter < self.variant["max_online_iters"]:

            ## in every iteration add new_trajectory and train the model
            env_dataset,df_evaluate = aug_trajectory(model = self.model,discretizer = self.discretizer,config = self.eval_config,schema= online_schema,device = self.device)
            self.replay_buffer.add_new_dataset(env_dataset)

            

            datasets = DiscretizedDatasetTrajectory(replay_buffer=self.replay_buffer,discount = self.config.dataset.discount, seq_len = self.config.dataset.seq_len, strategy = self.config.dataset.strategy)
            
            if self.device == "cpu":
                dataloader = DataLoader(datasets,  batch_size=1, shuffle=True)
            else:
                dataloader = DataLoader(datasets,  batch_size=self.config.dataset.batch_size, shuffle=True, num_workers=8, pin_memory=True)

            self.discretizer = datasets.get_discretizer() #update discretizer because new dataset

            self.trainer.train(
                model=self.model,
                dataloader=dataloader,
                num_epochs = 1


            )

            torch.save(self.discretizer, self.logger.log_path+f"/discretizer_online_iter_{self.online_iter}.pt")
            torch.save(self.model.state_dict(), self.logger.log_path+f"/model_online_iter_{self.online_iter}.pt")

            df_evaluate.to_csv(self.logger.log_path+f"/aug_iter_{self.online_iter}_eval.csv")


            self.online_iter += 1

            #df_evaluate.to_csv(self.logger.log_path+f"/aug_iter_{iter}.csv")


       

            

    def __call__(self):

        utils.set_seed_everywhere(args.seed)
        print("\n\nMaking Eval Env.....")
        
        eval_env_schema = "citylearn_challenge_2022_phase_2"

        self.start_time = time.time()
        if self.variant["max_pretrain_iters"]:
            self.pretrain(schema_eval=eval_env_schema)

        if self.variant["max_online_iters"]:
            print("\n\nMaking Online Env.....")
            

            online_schema = "citylearn_challenge_2022_phase_1"
            schema_eval = "citylearn_challenge_2022_phase_2"
            #online_schema = "citylearn_challenge_2022_phase_1"

            self.online_tuning(online_schema,schema_eval)
            #online_env.close()

        #eval_envs.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--env", type=str, default="hopper-medium-v2")

  

  
    # pretraining options
    parser.add_argument("--max_pretrain_iters", type=int, default=1)
    parser.add_argument("--num_updates_per_pretrain_iter", type=int, default=1)

    # finetuning options
    parser.add_argument("--max_online_iters", type=int, default=2)
    parser.add_argument("--online_rtg", type=int, default=7200)
    parser.add_argument("--num_online_rollouts", type=int, default=1)
    parser.add_argument("--replay_size", type=int, default=1000)
    parser.add_argument("--num_updates_per_online_iter", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=10)

    # environment options
    parser.add_argument("--device", type=str, default="cpu") ##cuda 
    parser.add_argument("--log_to_tb", "-w", type=bool, default=True)
    parser.add_argument("--save_dir", type=str, default="./exp")
    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument("--eval_config",type=str, default="configs/eval_base.yaml")

    args = parser.parse_args()

    utils.set_seed_everywhere(args.seed)
    experiment = Experiment(vars(args),dataset_path="./data_interactions/sac_dataset.pkl")

    print("=" * 50)
    experiment()
