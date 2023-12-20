import os
import time
from webbrowser import get
import ipdb
from gym.spaces import Space, Box
import math
import numpy as np
import statistics
from collections import deque
import collections
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from isaacgym import gymapi
from algorithms.ppo import RolloutStorage
from algorithms.ppo import RolloutStoragePC
from .module import Normalization, RewardScaling
from envs.utils.misc import _draw_line
from isaacgym.torch_utils import quat_rotate, quat_apply, quat_conjugate,\
     quat_mul, tensor_clamp
import torch    
from isaacgym.gymtorch import wrap_tensor
from envs.utils.get_running_bbox import get_bbox_isaac_tensor, _draw_bbox_tensor, get_bbox_pt, _draw_bbox_pt, get_bbox_isaac_tensor_nohandle, get_bbox_pt_nohandle
from envs.utils.get_reward import quat_axis
from ..ppo_utils.io_util import load, save, load_backbone_only
from ..ppo_utils.misc_util import lr_decay
from ..ppo_utils.log_util import log, log_test
from data_structure.observation import Observations

class PPO:
    def __init__(self,
                 vec_env,
                 actor_critic_class,
                 num_transitions_per_env,
                 num_learning_epochs,
                 num_mini_batches,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 init_noise_std=1.0,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=0.5,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=None,
                 max_lr=1e-3,
                 min_lr=1e-7,
                 model_cfg=None,
                 device='cpu',
                 sampler='sequential',
                 log_dir='run',
                 is_testing=False,
                 eval_round=1,
                 eval_freq = 50,
                 print_log=True,

                 max_iterations = 50000,
                 wandb_writer = None,
                 checkpoint_path = ' ',

                 use_adv_norm = True,          ### trick 1
                 adv_norm_epsilon= 1e-8,
                 learning_rate_decay = False,   ### trick 6
                 use_grad_clip = True,          ###trick 7
                 adam_epsilon = 1e-8,           ### trick 9
                 ):

        
        self.vec_env = vec_env
        if not isinstance(vec_env.observation_space, Space):
            raise TypeError("vec_env.observation_space must be a gym Space")
        if not isinstance(vec_env.state_space, Space):
            raise TypeError("vec_env.state_space must be a gym Space")
        if not isinstance(vec_env.action_space, Space):
            raise TypeError("vec_env.action_space must be a gym Space")
        self.is_testing = is_testing
        self.vec_env.task.env.cfg["test_save_video"] = False
        self.observation_space = vec_env.observation_space
        self.action_space = vec_env.action_space
        self.state_space = vec_env.state_space
        self.env_num = vec_env.env_num
        self.max_episode_length = vec_env.task.env.max_episode_length
        self.eval_round = eval_round
        self.eval_freq = eval_freq

        self.device = device
        self.desired_kl = desired_kl
        self.lr_upper = float(max_lr)
        self.lr_lower = float(min_lr)
        self.schedule = schedule
        self.step_size = learning_rate
        
        self.init_obs = None

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.num_transitions_per_env = num_transitions_per_env
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.max_iterations = max_iterations
        

        assert(self.max_episode_length % self.num_transitions_per_env == 0)
        self.actor_critic = actor_critic_class(self.observation_space.shape, self.action_space.shape,
                                               init_noise_std, model_cfg, device = self.device)
        
        if model_cfg["ckpt"] != "None" and model_cfg["ckpt"] != None:
            path = model_cfg["ckpt"]
            checkpoint_dict= torch.load(path, map_location=self.device)
            try:
                self.actor_critic.load_state_dict(checkpoint_dict["model_state_dict"], strict = False)
            except:
                self.actor_critic.backbone.load_state_dict(checkpoint_dict["model_state_dict"], strict = False)
            print(f"Loading from ckpt {path}")

        
        print(self.actor_critic)
        self.actor_critic.to(self.device)
        self.storage = RolloutStorage(self.env_num, num_transitions_per_env, self.observation_space.shape,
                                self.state_space.shape, self.action_space.shape, self.device, sampler)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate, eps= adam_epsilon)   # , weight_decay=float(self.weight_decay), trick 9

        
        # Log
        self.log_dir = log_dir
        self.print_log = print_log
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10) 
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.wandb_writer = wandb_writer
        self.checkpoint_path = checkpoint_path
        self.order_part_pos = 22
            
        # Trick
        self.use_adv_norm = use_adv_norm                    ###trick 1
        self.adv_norm_epsilon = adv_norm_epsilon
        self.learning_rate_decay = learning_rate_decay      ###trick 6
        self.use_grad_clip = use_grad_clip                  ###trick 7
        
        self.episode_step_for_now = 0

    def eval(self, it) :
        self.vec_env.task.eval()
        self.actor_critic.eval()
        current_obs = self.vec_env.reset()
        self.episode_step_for_now = 0
        total_reward = torch.zeros((self.env_num), device=self.device)
        total_success = torch.zeros((self.env_num, self.eval_round), device=self.device)
        
        with tqdm(total=self.eval_round) as pbar:
            pbar.set_description('Validating:')
            with torch.no_grad() :
                for r in range(self.eval_round) :
                    for i in range(self.max_episode_length) :

                        self.vec_env.task.current_eval_round = r
                        self.vec_env.task.current_step_num = i

                        actions, _ = self.actor_critic.act_inference(current_obs)

                        next_obs, rews, dones, infos = self.vec_env.step(actions)
                        current_obs = Observations(state=next_obs.state, obs = next_obs.obs, points=next_obs.points)
                      
                        total_reward += rews.to(self.device)
                        total_success[:, r] = torch.logical_or(infos["successes"].to(self.device), total_success[:, r])
                    # Store Env State for Next Task
                    self.vec_env.task.end_task()
                    pbar.update(1)
                    
        
        train_reward = total_reward.mean() / self.max_episode_length / self.eval_round
        train_min_reward = total_reward.min()
        train_max_reward = total_reward.max()
    
        train_success = total_success.float().mean()
        train_max_success = total_success.float().max()
        train_min_success = total_success.float().min()

        train_reward = train_reward.cpu().item()
        train_success = train_success.cpu().item()

        status_dict = {
            "train_reward": train_reward,
            "train_reward_min": train_min_reward.cpu().item(),
            "train_reward_max": train_max_reward.cpu().item(),
            "train_success": train_success,
            "train_success_max": train_max_success.cpu().item(),
            "train_success_min": train_min_success.cpu().item()
        }
        self.vec_env.task.status_dict = status_dict

        #if self.is_testing:
        print("Training set average reward:     ", train_reward)
        print("Training set average success:    ", train_success)
        print("per eval_round success")
        print(torch.mean(total_success.float(), dim = 0))
        print("per asset success")
        print(torch.mean(total_success.float().reshape(1, self.env_num * self.eval_round),dim = 1))
        asset_train_mean = torch.mean(total_success.float().reshape(1, self.env_num * self.eval_round),dim = 1)

        self.writer.add_scalar('Test/' + 'TestSuccessRate/TrainSet', train_success, it)
        self.writer.add_scalar('Test/' + 'TestReward/TrainSet', train_reward, it)
        
        self.vec_env.task.train()
        self.actor_critic.train()
        return (train_reward,  train_success)

    def run(self, num_learning_iterations, log_interval=1):
        self.episode_step_for_now = 0 
        
        if self.is_testing:

            self.vec_env.task.eval()

            _ = self.eval(self.current_learning_iteration)

        else:
            self.vec_env.task.eval()

            _ = self.eval(self.current_learning_iteration)
            
            self.vec_env.task.train()

            rewbuffer = deque(maxlen=200)
            lenbuffer = deque(maxlen=200)
            cur_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            cur_episode_length = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            success_rate = []
            reward_sum = []
            episode_length = []

            for it in range(self.current_learning_iteration, num_learning_iterations):
                current_obs = self.vec_env.reset() 
                self.pre_grasp()
                start = time.time()
                ep_infos = []
                task_info = {}
                
                if it % self.eval_freq == 0:
                    train_reward,  train_success = self.eval(it)
                    if self.wandb_writer is not None:
                        self.wandb_writer.log({
                            "Val/train_reward": train_reward,
                            "Val/train_success": train_success,
                        })
                    active_task_name = self.vec_env.task.__class__.__name__
                    self.save(os.path.join(self.log_dir,  '{}_model_{}.tar'.format(active_task_name, it)), it)
                    

                for i in range(self.max_episode_length):
                    self.episode_step_for_now = i

                    current_train_obs = Observations(state=current_obs.state[:self.env_num], obs=current_obs.obs[:self.env_num])

                    train_actions, train_actions_log_prob, train_values, train_mu, train_sigma, _ = self.actor_critic.act(current_train_obs, require_grad = False)
                    actions = train_actions

                    # Step the vec_environment
                    next_obs, rews, dones, infos = self.vec_env.step(actions)
                    next_states = self.vec_env.get_state()
                    train_rews = rews[:self.env_num]
                    train_dones = dones[:self.env_num]
                    # Record the transition
                    self.storage.add_transitions(
                        observations = current_train_obs.obs,
                        states = current_train_obs.state,
                        actions = train_actions,
                        rewards = train_rews, 
                        dones = train_dones,
                        values = train_values, 
                        actions_log_prob = train_actions_log_prob,
                        mu = train_mu,
                        sigma = train_sigma,
                    )

                    current_obs = Observations(state=next_obs.state, obs = next_obs.obs, points=next_obs.points)
                    
                    # Book keeping

                    ep_infos.append(infos)
                    torch.cuda.synchronize()

                    if self.print_log:
                        cur_reward_sum[:] += rews
                        cur_episode_length[:] += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)

                        reward_sum.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        episode_length.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())

                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                    
                    self.actor_critic.train() # new
                    if (i + 1) % self.num_transitions_per_env == 0  and  (i + 1) != self.max_episode_length:
                        _, _, last_values, _, _, _ = self.actor_critic.act(current_obs)
                        torch.cuda.synchronize()
                        stop = time.time()
                        collection_time = stop - start

                        # mean_trajectory_length, mean_reward = self.storage.get_statistics()
                        mean_reward = self.storage.rewards.mean()
                        # Learning step
                        start = stop
                        self.storage.compute_returns(last_values[:self.env_num], self.gamma, self.lam)
                        #(self.storage.observations[:,:,14])
                        mean_value_loss, mean_surrogate_loss = self.update(it)
                        self.storage.clear()
                        torch.cuda.synchronize()
                        stop = time.time()
                        learn_time = stop - start
                        start = stop
                        if self.print_log:
                            self.log(locals())

                if self.print_log:
                    rewbuffer.extend(reward_sum)
                    lenbuffer.extend(episode_length)

                self.actor_critic.train() # new
                _, _, last_values, _, _ , _= self.actor_critic.act(current_obs)
                torch.cuda.synchronize()
                stop = time.time()
                collection_time = stop - start

                # mean_trajectory_length, mean_reward = self.storage.get_statistics()
                mean_reward = self.storage.rewards.mean()
                # Learning step
                start = stop
                self.storage.compute_returns(last_values[:self.env_num], self.gamma, self.lam)
                mean_value_loss, mean_surrogate_loss = self.update(it)

                self.storage.clear()
                torch.cuda.synchronize()
                stop = time.time()
                learn_time = stop - start
                start = stop
            
                if self.print_log:
                    self.log(locals())
                ep_infos.clear()

    def load(self, path):
        load(self, path)

    def save(self, path, it):    
        save(self, path, it)
    
    def lr_decay(self, total_steps):
        lr_decay(self, total_steps)

    def pre_grasp(self):
        return
        
 
    def log_test(self, locs, width=80, pad=35) :
        return log_test(self, locs, width, pad)

    def log(self, locs, width=80, pad=35):
        return log(self, locs, width, pad)

    def update(self, it):
        mean_value_loss = 0
        mean_surrogate_loss = 0

        batch = self.storage.mini_batch_generator(self.num_mini_batches)
        for epoch in range(self.num_learning_epochs):
            for indices in batch:
            
                state_batch = self.storage.states.view(-1, *self.storage.states.size()[2:])[indices]
                observations_batch = self.storage.observations.view(-1, *self.storage.observations.size()[2:])[indices]
                obs_batch = Observations(state = state_batch, obs=observations_batch)
                actions_batch = self.storage.actions.view(-1, self.storage.actions.size(-1))[indices]
                target_values_batch = self.storage.values.view(-1, 1)[indices]
                returns_batch = self.storage.returns.view(-1, 1)[indices]
                old_actions_log_prob_batch = self.storage.actions_log_prob.view(-1, 1)[indices]
                advantages_batch = self.storage.advantages.view(-1, 1)[indices]
                old_mu_batch = self.storage.mu.view(-1, self.storage.actions.size(-1))[indices]
                old_sigma_batch = self.storage.sigma.view(-1, self.storage.actions.size(-1))[indices]
                actions_log_prob_batch, entropy_batch, value_batch, mu_batch, sigma_batch = self.actor_critic.evaluate(obs_batch, actions_batch)

                ###Trick 1 advantage normalization
                if self.use_adv_norm:
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + self.adv_norm_epsilon)

                # KL learning rate schedule
                if self.desired_kl > 0 and self.schedule == 'adaptive':

                    kl = torch.sum(sigma_batch - old_sigma_batch + (torch.square(old_sigma_batch.exp()) + torch.square(old_mu_batch - mu_batch)) /\
                        (2.0 * torch.square(sigma_batch.exp())) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.step_size = max(self.lr_lower, self.step_size / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.step_size = min(self.lr_upper, self.step_size * 1.5)
                    
                    # if it > 2000 :
                    #     self.step_size = max(min(self.step_size, 3e-4 - (it-2000)/1000*3e-4), 0.0)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.step_size
                
                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
                # #!
                # surrogate_loss = surrogate.mean()


                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()
                # ipdb.set_trace()
                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                
                if self.use_grad_clip:   ###trick 7
                    nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                # self.network_lr_scheduler.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        
        if self.learning_rate_decay: 
            self.lr_decay(it)

        return mean_value_loss, mean_surrogate_loss
    
    
def space_add(a, b):

    if len(a.shape) != 1 or len(b.shape) != 1 :
        
        raise TypeError("Shape of two spaces need to be 1d")
    
    elif not isinstance(a, Box) or not isinstance(b, Box) :

        raise TypeError("Type of two spaces need to be Box")
    
    else :

        low = np.concatenate((a.low, b.low))
        high = np.concatenate((a.high, b.high))
        return Box(low=low, high=high)

