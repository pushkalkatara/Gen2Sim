import torch
import numpy as np

from .utils.get_reward import get_reward_done
from .utils.reset import _partial_reset, reset
from .utils.get_observation import _get_base_observation,_refresh_observation
from .utils.step import step
from .utils.perform_action import _perform_actions


class BaseTask():
    def __init__(
        self,
        env,
        active_env_state,
        active_task_type='articulate'
    ):
        self.env = env
        self.active_task_type = active_task_type

        self.init_task_buffers()
        self.init_success_vars()
        self.init_env(active_env_state)

        self.train_mode = True
        self.end_task_dict = None

    def init_task_buffers(self):
        if self.active_task_type == 'articulate':
            self.obs_dim = (self.env.num_qpose_actions * 2) + 2 + 16 + 24
            self.state_dim = (self.env.num_qpose_actions * 2) + 2 + 16 + 24
            # franka action space, 2 for cabinet qpos, 16 for hand root pose, 24 for cabinet bbox
        elif self.active_task_type == 'pick_and_place':
            self.obs_dim = self.env.num_qpose_actions * 2 + 16 + 13
            self.state_dim = self.env.num_qpose_actions * 2 + 16 + 13
            # franka action space, 16 for hand root pose
            # 16 for object root pose
            # suction gripper
        
        # allocate buffers
        self.obs_buf = torch.zeros(
            (self.env.env_num, self.obs_dim), device=self.env.device, dtype=torch.float)
        self.states_buf = torch.zeros(
            (self.env.env_num, self.state_dim), device=self.env.device, dtype=torch.float)
        self.rew_buf = torch.zeros(
            self.env.env_num, device=self.env.device, dtype=torch.float)
        self.reset_buf = torch.ones(
            self.env.env_num, device=self.env.device, dtype=torch.long)
        self.progress_buf = torch.zeros(
            self.env.env_num, device=self.env.device, dtype=torch.long)
        self.randomize_buf = torch.zeros(
            self.env.env_num, device=self.env.device, dtype=torch.long)

        self.extras = {}
        self.average_reward = None

    def init_success_vars(self):
        # params for success rate
        self.success = torch.zeros((self.env.env_num,), device=self.env.device)
        self.success_rate = torch.zeros((self.env.env_num,), device=self.env.device)
        self.current_success_rate = torch.zeros((self.env.env_num,), device = self.env.device)
        self.success_queue = torch.zeros((self.env.env_num, 1), device=self.env.device)
        self.success_idx = torch.zeros((self.env.env_num,), device=self.env.device).long()
        self.success_buf = torch.zeros((self.env.env_num,), device=self.env.device).long()
        self.success_grasp_buf = torch.zeros((self.env.env_num,), device=self.env.device).long()
        self.grasp_success_condition_1_buf = torch.zeros((self.env.env_num,), device=self.env.device).long()
        self.grasp_success_condition_2_buf = torch.zeros((self.env.env_num,), device=self.env.device).long()

    def init_env(self, active_env_state):
        if active_env_state is None:
            return
        self.env.update_init_states(active_env_state)
        self.reset()

    def end_task(self):
        # Store Env state for next task
        self.end_task_dict = {
            'dof_states': self.env.dof_state_tensor_all.clone(),
            'root_states': self.env.root_tensor.clone(),
            'rigid_body_states': self.env.rigid_body_tensor_all.clone()
        }

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

    def _get_reward_done(self):
        return get_reward_done(self)

    def _partial_reset(self, to_reset):
        return _partial_reset(self, to_reset)
   
    def reset(self, to_reset = "all"):
        return reset(self, to_reset)

    def _perform_actions(self, actions):
        return _perform_actions(self, actions)

    def _get_base_observation(self, suggested_gt=None, pregrasp=False) :
        return _get_base_observation(self, suggested_gt)
    
    def _refresh_observation(self, pregrasp=False) :
        return _refresh_observation(self)

    def step(self, actions):
        return step(self, actions)