import os
import torch
import wandb
import numpy as np
import torch.nn.functional as F

from tqdm.auto import tqdm, trange
from stable_baselines3.common.vec_env import DummyVecEnv

from trajectory.utils.env import vec_rollout, create_env
from trajectory.models.gpt.ein_linear import EinLinear
from trajectory.utils.scheduler import GPTScheduler
from trajectory.utils.common import weight_decay_groups, set_seed
from trajectory.utils.training import update_loss_csv
from trajectory.models.sample import sample_rollout, top_k_logits, filter_cdf
from torch.utils.data.dataloader import DataLoader
from eval_traj import aug_trajectory,eval_trajectory


import pickle

class GPTTrainerBoot:
    def __init__(
            self,
            final_tokens,
            warmup_tokens=1_000_000,
            learning_rate=1e-4,
            betas=(0.9, 0.999),
            weight_decay=0.0,
            clip_grad=None,
            eval_config = "configs/eval_base.yaml",
            eval_seed=42,
            eval_every=10,
            eval_episodes=10,
            eval_plan_every=1,
            eval_beam_width=256,
            eval_beam_steps=64,
            eval_beam_context=16,
            eval_sample_expand=1,
            eval_temperature=1,
            eval_discount=0.99,
            eval_k_act=None,
            eval_k_obs=1,
            eval_k_reward=1,
            action_weight=1,
            value_weight=1,
            reward_weight=1,
            save_every=5,
            checkpoints_path=None,
            device="cpu",
            logger = None
    ):
        # optimizer params
        self.betas = betas
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.clip_grad = clip_grad
        # loss params
        self.action_weight = action_weight
        self.reward_weight = reward_weight
        self.value_weight = value_weight
        # scheduler params
        self.warmup_tokens = warmup_tokens
        self.final_tokens = final_tokens
        # eval params
        self.eval_config = eval_config
        self.eval_seed = eval_seed
        self.eval_every = eval_every
        self.eval_episodes = eval_episodes
        self.eval_plan_every = eval_plan_every
        self.eval_beam_width = eval_beam_width
        self.eval_beam_steps = eval_beam_steps
        self.eval_beam_context = eval_beam_context
        self.eval_sample_expand = eval_sample_expand
        self.eval_temperature = eval_temperature
        self.eval_discount = eval_discount
        self.eval_k_act = eval_k_act
        self.eval_k_obs = eval_k_obs
        self.eval_k_reward = eval_k_reward
        # checkpoints
        self.save_every = save_every
        self.checkpoints_path = checkpoints_path
        self.batch_size = 256
        self.device = device

        self.n_epochs = 0 

        self.generated_dataset =[]
        self.logger = logger


        #create dataset folder
        os.makedirs(self.logger.log_path+"/dataset",exist_ok=True)


        #create discritzer folder
        os.makedirs(self.logger.log_path+"/discretizer",exist_ok=True)

        

        #create model folder
        os.makedirs(self.logger.log_path+"/model",exist_ok=True)

        #create loss folder
        os.makedirs(self.logger.log_path+"/loss_info",exist_ok=True)

        #create eval folder
        os.makedirs(self.logger.log_path+"/eval_info",exist_ok=True)

        #create discretizer
        os.makedirs(self.logger.log_path+"/discretizer",exist_ok=True)

        



        

        



    def get_optimizer(self, model):
        param_groups = weight_decay_groups(
            model=model,
            whitelist_modules=(torch.nn.Linear,  torch.nn.MultiheadAttention, EinLinear),
            blacklist_modules=(torch.nn.LayerNorm, torch.nn.Embedding),
            blacklist_named=("pos_emb",)
        )
        optim_groups = [
            {"params": param_groups["decay"], "weight_decay": self.weight_decay},
            {"params": param_groups["nodecay"], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=self.betas)

        return optimizer

    def get_scheduler(self, optimizer):
        scheduler = GPTScheduler(
            optimizer,
            warmup_tokens=self.warmup_tokens,
            final_tokens=self.final_tokens,
            decay=True,
        )
        return scheduler

    def __get_loss(self, model, batch):
        tokens, targets, loss_pad_mask = batch
        logits, state = model(tokens)

        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none")
        if self.action_weight != 1 or self.value_weight != 1 or self.reward_weight != 1:
            n_states = int(np.ceil(tokens.shape[1] / model.transition_dim))
            weights = torch.cat([
                torch.ones(model.observation_dim, device=tokens.device),
                torch.ones(model.action_dim, device=tokens.device) * self.action_weight,
                torch.ones(1, device=tokens.device) * self.reward_weight,
                torch.ones(1, device=tokens.device) * self.value_weight,
            ])
            weights = weights.repeat(n_states)[1:].repeat(tokens.shape[0], 1)
            loss = loss * weights.view(-1)

        loss = (loss * loss_pad_mask.view(-1)).mean()

        return loss

    def eval(self, env_name, model, discretizer, seed=None):
        model.eval()
        set_seed(seed=seed)

        vec_env = DummyVecEnv([lambda: create_env(env_name) for _ in range(self.eval_episodes)])
        rewards = vec_rollout(
            vec_env=vec_env,
            model=model,
            discretizer=discretizer,
            beam_context_size=self.eval_beam_context,
            beam_width=self.eval_beam_width,
            beam_steps=self.eval_beam_steps,
            plan_every=self.eval_plan_every,
            sample_expand=self.eval_sample_expand,
            k_act=self.eval_k_act,
            k_obs=self.eval_k_obs,
            k_reward=self.eval_k_reward,
            temperature=self.eval_temperature,
            discount=self.eval_discount,
            max_steps=vec_env.envs[0].max_episode_steps,
            device=self.device
        )
        scores = [vec_env.envs[0].get_normalized_score(r) for r in rewards]

        model.train()
        return np.mean(rewards), np.std(rewards), np.mean(scores), np.std(scores)
    
    def filter_by_confidence_ratio(self, x, mask, confidence, confidence_ratio):
        rollout_num = x.shape[0]
        chosen_num = int(rollout_num * confidence_ratio)
        _, chosen = torch.topk(confidence, chosen_num)
        return x[chosen], mask[chosen], confidence[chosen]
    
    def generate_autoregressive(self,model,x,y,mask,**kwargs):
        obs_dim, act_dim, trans_dim = model.observation_dim, model.action_dim, model.transition_dim

        x_crop = torch.cat([x, y[:, [-1]]], dim=1).repeat_interleave(kwargs.get("generation_num", 2), dim=0)
        
        start_idx = x_crop.shape[1] - kwargs.get("generation_len", 1) * trans_dim
        x_generate = x_crop[:, :start_idx]
        bs = x_generate.shape[0]
        print(bs)
        print(obs_dim)

        obs_row_indices = torch.arange(bs).repeat_interleave(obs_dim, dim=0)
        obs_dim_indices = torch.arange(obs_dim).unsqueeze(0).repeat(bs, 1).flatten()
        act_row_indices = torch.arange(bs).repeat_interleave(act_dim, dim=0)
        act_dim_indices = torch.arange(act_dim).unsqueeze(0).repeat(bs, 1).flatten()
        rew_row_indices = torch.arange(bs)
        rew_dim_indices = torch.zeros(bs).long()

        confidence_sum = torch.zeros(bs, device=x.device)
        confidence_nums = torch.zeros(1, device=x.device)

        generation_len = kwargs.get("generation_len", 1)

        for i in range(generation_len):
            r_idx = start_idx + (i + 1) * trans_dim - 2

            x_generate, obs_probs = sample_rollout(
                model, x_generate, obs_dim, topk=kwargs.get("k_obs", 1), cdf=kwargs.get("cdf_obs", None)
            )
            obs_indices = x_generate[:, -obs_dim:].flatten()
            obs_probs = obs_probs[obs_row_indices, obs_dim_indices, obs_indices].reshape(bs, obs_dim)
            confidence_sum += torch.log10(obs_probs + 1e-15).sum(dim=-1)
            confidence_nums += obs_dim

            x_generate, act_probs = sample_rollout(
                model, x_generate, act_dim, topk=kwargs.get("k_act", None), cdf=kwargs.get("cdf_act", 0.6)
            )
            act_indices = x_generate[:, -act_dim:].flatten()
            act_probs = act_probs[act_row_indices, act_dim_indices, act_indices].reshape(bs, act_dim)
            confidence_sum += torch.log10(act_probs + 1e-15).sum(dim=-1)
            confidence_nums += act_dim

            if kwargs.get("generation_real_r", True):
                x_generate = torch.cat([x_generate, x_crop[:, r_idx: r_idx + 1]], dim=1)
            else:
                x_generate, rew_probs = sample_rollout(
                    model, x_generate, 1, topk=kwargs.get("k_rew", 1), cdf=kwargs.get("cdf_rew", None)
                )
                rew_indices = x_generate[:, -1:].flatten()
                rew_probs = rew_probs[rew_row_indices, rew_dim_indices, rew_indices].reshape(bs, 1)
                confidence_sum += torch.log10(rew_probs + 1e-15).sum(dim=-1)
                confidence_nums += 1

            if kwargs.get("generation_real_R", True):
                x_generate = torch.cat([x_generate, x_crop[:, r_idx + 1: r_idx + 2]], dim=1)
            else:
                x_generate, val_probs = sample_rollout(
                    model, x_generate, 1, topk=kwargs.get("k_rew", 1), cdf=kwargs.get("cdf_rew", None)
                )
                val_indices = x_generate[:, -1:].flatten()
                val_probs = val_probs[rew_row_indices, rew_dim_indices, val_indices].reshape(bs, 1)
                confidence_sum += torch.log10(val_probs + 1e-15).sum(dim=-1)
                confidence_nums += 1
            
        confidence = confidence_sum / confidence_nums
        mask = mask.repeat_interleave(kwargs.get("generation_num", 4), dim=0)

        if kwargs.get("generation_confidence_type") == "thresh":
            # Filter by a hard confidence threshold
            confidence_thresh = kwargs.get("generation_confidence_factor", -0.4)
            if confidence_thresh > 0:
                confidence_thresh = -confidence_thresh
            x_generate, mask, confidence = self.filter_by_confidence_thresh(
                x_generate, mask, confidence, confidence_thresh
            )
        elif kwargs.get("generation_confidence_type") == "ratio":
            # Filter by a percentage of trajectories with top confidence
            confidence_ratio = kwargs.get("generation_confidence_factor", 0.08)
            x_generate, mask, confidence = self.filter_by_confidence_ratio(
                x_generate, mask, confidence, confidence_ratio
            )
        elif kwargs.get("generation_confidence_type") is None or kwargs.get("generation_confidence_type") == "none":
            pass
        else:
            raise NotImplementedError()

        return x_generate, mask, confidence





    def train(self, model, dataloader, num_epochs=1, log_every=100, bootstrap_kwargs = None):

        if bootstrap_kwargs is None:
            bootstrap_kwargs = {}

        bootstrap = bootstrap_kwargs.get("bootstrap", True)
        bootstrap_type = bootstrap_kwargs.get("bootstrap_type", "repeat")
        generation_type = bootstrap_kwargs.get("generation_type", "autoregressive")
        generation_epoch_thresh = bootstrap_kwargs.get("generation_epoch_thresh", 10)
        generation_len = bootstrap_kwargs.get("generation_len", 1)
        generation_num = bootstrap_kwargs.get("generation_num", 1)
        generation_confidence_type = bootstrap_kwargs.get("generation_confidence_type", "ratio")
        generation_confidence_factor = bootstrap_kwargs.get("generation_confidence_factor", 0.2)
        generation_real_r = bootstrap_kwargs.get("generation_real_r", False)
        generation_real_R = bootstrap_kwargs.get("generation_real_R", False)

        perform_bootstrap = (bootstrap and self.n_epochs >= generation_epoch_thresh)


        discretizer = dataloader.dataset.get_discretizer()

        model.train()

        optimizer = self.get_optimizer(model)
        scheduler = self.get_scheduler(optimizer)
        checkpoints_path = self.checkpoints_path
        if os.path.exists(checkpoints_path):
            suffix = 2
            while True:
                new_path = f"{checkpoints_path}_{suffix}"
                if not os.path.exists(new_path):
                    checkpoints_path = new_path
                    break
                suffix += 1
        os.makedirs(checkpoints_path, exist_ok=True)
       
        torch.save(dataloader.dataset.get_discretizer(), os.path.join(self.logger.log_path, "discretizer/discretizer.pt"))

        

        for epoch in trange(1, num_epochs + 1, desc="Training"):
            self.total_generated_data= []
            
            epoch_losses = []

            perform_bootstrap = (bootstrap and self.n_epochs >= generation_epoch_thresh and self.n_epochs % 10 ==0 )

            if  bootstrap_type == "repeat" and len(self.generated_dataset) > 0:

                with open(self.logger.log_path+f"/dataset/generated_dataset_epoch_{epoch}.pkl",'wb') as f :
                    pickle.dump(self.generated_dataset, f)


                #print("Batch Size for Training in Generated Dataset "+str(self.batch_size))

                generated_dataset_loader = DataLoader(
                    self.generated_dataset, shuffle=True, pin_memory=True,
                    batch_size=self.batch_size, num_workers=8 ## remember to change 
                )

                for it,batch in enumerate(tqdm(generated_dataset_loader, desc="Epoch", leave=False)):
                    batch = [b.to(self.device) for b in batch]
                    with torch.set_grad_enabled(True):
                        loss = self.__get_loss(model, batch)
                        loss = loss.mean()
                    
                    scheduler.step(batch_size=batch[0].reshape(-1).shape[0])
                    optimizer.zero_grad()
                    loss.backward()

                    if self.clip_grad is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad)

                    optimizer.step()


                    update_loss_csv(iter_value = i , loss = loss.item(),filename=os.path.join(self.logger.log_path,f"loss_info/batch_loss_generated_dataset_{epoch}.csv"),type_name ="Batch")

                    # log here!!
                    epoch_losses.append(loss.item())
                    if i % log_every == 0:
                        wandb.log({
                            "train/loss_batch": loss.item(),
                            "train/lr": scheduler.get_current_lr()
                        })

                    #print("Updated using Generated Data")

                
                 
                

                #env_dataset_1,df_evaluate_1 = eval_trajectory(model = model,discretizer = discretizer,config = self.eval_config,schema= "citylearn_challenge_2022_phase_1",device = self.device)
                #env_dataset_2,df_evaluate_2 = eval_trajectory(model = model,discretizer = discretizer,config = self.eval_config,schema= "citylearn_challenge_2022_phase_2",device = self.device)
                
                #df_evaluate_1.to_csv(self.logger.log_path+f"/eval_info/eval_schema_1_epoch_{epoch}_with_generated_dataset.csv")
                #df_evaluate_2.to_csv(self.logger.log_path+f"/eval_info/eval_schema_2_epoch_{epoch}_with_generated_dataset.csv")

                
                torch.save(model.state_dict(), os.path.join(self.logger.log_path, f"model/model_{epoch}_with_generated_dataset.pt"))

            for i, batch in enumerate(tqdm(dataloader, desc="Epoch", leave=False)):
                batch = [b.to(self.device) for b in batch]
                batch_X, batch_Y, mask = batch

                loss = self.__get_loss(model, batch)

                ## batch_size will be checked first 

                #print(batch[0].size())
                
                self.batch_size = batch[0].shape[0]
                
                #print("Batch Size during normal training" +str(self.batch_size))
                
                scheduler.step(batch_size=batch[0].reshape(-1).shape[0])
                optimizer.zero_grad()
                loss.backward()
                if self.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad)
                optimizer.step()

                update_loss_csv(iter_value = i , loss = loss.item(),filename=os.path.join(self.logger.log_path,f"loss_info/batch_loss_{epoch}.csv"),type_name ="Batch")
                

                
                if perform_bootstrap :
                    with torch.set_grad_enabled(False):
                        batch_generated, mask_generated, confidence_generated = self.generate_autoregressive(
                                model, batch_X, batch_Y, mask, 
                                generation_type=generation_type,
                                generation_len=generation_len,
                                generation_num=generation_num,
                                generation_confidence_type=generation_confidence_type,
                                generation_confidence_factor=generation_confidence_factor,
                                generation_real_r=generation_real_r,
                                generation_real_R=generation_real_R
                            )
                        
                        #print("Generated Dataset" + str(batch_generated.size()))
                       
                        batch_X_generated = batch_generated[:, :-1].clone().detach()
                        
                        batch_Y_generated = batch_generated[:, 1:].clone().detach()

                        if bootstrap_type == "once":
                            with torch.set_grad_enabled(True):
                                loss_generated = self.__get_loss(model,batch)
                                
                                scheduler.step(batch_size=batch[0].reshape(-1).shape[0])

                                optimizer.zero_grad()
                                loss.backward()  
                                if self.clip_grad is not None:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad)
                                optimizer.step() 

                        elif bootstrap_type == "repeat":
                            
                            batch_X_generated = batch_X_generated.cpu()
                            batch_Y_generated = batch_Y_generated.cpu()
                            mask_generated = mask_generated.cpu()

                            augmented_data = [
                                (batch_X_generated[i], batch_Y_generated[i], mask_generated[i]) 
                                for i in range(batch_X_generated.shape[0])
                            ]
                            self.generated_dataset.extend(augmented_data)

                            print("Generated Dataset "+str(len(self.generated_dataset)))
                            #self.total_generated_data.append(batch_X_generated.shape[0])
                            #with open(self.logger.log_path+f"/dataset/generated_dataset.pkl",'wb') as f :
                            #    pickle.dump(self.generated_dataset, f)
                        else:
                            raise NotImplementedError()
                        
              
            

            self.n_epochs += 1        

            """
            if epoch % self.eval_every == 0:
                env_dataset_1,df_evaluate_1 = eval_trajectory(model = model,discretizer = discretizer,config = self.eval_config,schema= "citylearn_challenge_2022_phase_1",device = self.device)
                #env_dataset_2,df_evaluate_2 = eval_trajectory(model = model,discretizer = discretizer,config = self.eval_config,schema= "citylearn_challenge_2022_phase_2",device = self.device)
                
                df_evaluate_1.to_csv(self.logger.log_path+f"/eval_info/eval_schema_1_epoch_{epoch}.csv")
                #df_evaluate_2.to_csv(self.logger.log_path+f"/eval_info/eval_schema_2_epoch_{epoch}.csv")

                torch.save(model.state_dict(), os.path.join(self.logger.log_path, f"model/model_{epoch}.pt"))
            """
            
                
                    
        return model