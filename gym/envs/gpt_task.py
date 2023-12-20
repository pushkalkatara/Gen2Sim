import numpy as np
from os.path import join as pjoin
from pathlib import Path

import torch

from isaacgym import gymtorch
from .base_task import BaseTask
from .utils.get_running_bbox import get_bbox_isaac_tensor, _draw_bbox_tensor


class OpenMicrowaveDoor(BaseTask):
    def __init__(
        self,
        env,
        active_env_state
    ):
        super().__init__(
            env=env,
            active_env_state=active_env_state
        )   

        # Boilerplate code to set the initial state of the environment, not needed while
        # generating tasks with LLM. 

        self.env.initial_dof_states = self.env.dof_state_tensor_all.clone()
        self.env.initial_dof_states = self.env.initial_dof_states.view(
            self.env.env_num, 
            self.env.franka_num_dofs+self.env.cabinet_dof_num+self.env.static_asset_dof_num,
            self.env.initial_dof_states.shape[-1]
        )

        # Set initial state as closed
        self.env.initial_dof_states[:, self.env.franka_num_dofs+self.env.part_dof_id, 0] = 0.0
        self.env.gym.set_dof_state_tensor(self.env.sim, gymtorch.unwrap_tensor(self.env.initial_dof_states))
        self.env.gym.refresh_actor_root_state_tensor(self.env.sim)
        self.env.gym.refresh_dof_state_tensor(self.env.sim)
        self.env.gym.refresh_rigid_body_state_tensor(self.env.sim)

        # initialise robot close to the asset for faster convergence. 
        self.env.cabinet_dof_tensor = self.env.dof_state_tensor_used[:, self.env.franka_num_dofs + self.env.part_dof_id, :]
        part_bbox_tensor, handle_bbox_tensor = get_bbox_isaac_tensor(self.env, self.env.cabinet_dof_tensor[:,0], 0)
        handle_center_xy = handle_bbox_tensor.mean(1)[:, :2]

        self.env.rigid_body_tensor_all = gymtorch.wrap_tensor(self.env.gym.acquire_rigid_body_state_tensor(self.env.sim))
        self.env.rigid_body_tensor_used = self.env.rigid_body_tensor_all.reshape([self.env.env_num, self.env.franka_rigid_num+self.env.cabinet_rigid_num+self.env.static_asset_rigid_num+self.env.distractor_rig_num+self.env.distractor_1_rig_num, self.env.rigid_body_tensor_all.shape[-1]])
        self.env.hand_rigid_body_tensor = self.env.rigid_body_tensor_used[:, self.env.hand_rigid_body_index, :] # N*13
        current_gripper_position = self.env.get_robot_gripper_position()

        self.env.initial_dof_states.view([self.env.env_num, self.env.franka_num_dofs+self.env.cabinet_dof_num, self.env.dof_state_tensor_all.shape[-1]])[:, 0, 0] -= handle_center_xy[:, 0] - current_gripper_position[:, 0] + 0.4
        self.env.initial_dof_states.view([self.env.env_num, self.env.franka_num_dofs+self.env.cabinet_dof_num, self.env.dof_state_tensor_all.shape[-1]])[:, 1, 0] -= handle_center_xy[:, 1] - current_gripper_position[:, 1]

        self.env.gym.set_dof_state_tensor(self.env.sim, gymtorch.unwrap_tensor(self.env.initial_dof_states))
        self.env.gym.refresh_actor_root_state_tensor(self.env.sim)
        self.env.gym.refresh_dof_state_tensor(self.env.sim)
        self.env.gym.refresh_rigid_body_state_tensor(self.env.sim)

    def compute_reward(self):
        current_handle_position = self.env.get_position_by_link_name()
        current_gripper_position = self.env.get_robot_gripper_position()
        distance_gripper_to_handle = torch.norm(current_gripper_position - current_handle_position, dim=-1)
        current_door_state = self.env.get_state_by_link_name()

        reward = - distance_gripper_to_handle + current_door_state

        target_door_state = self.env.get_limits_by_joint_name()["upper"]

        success = torch.abs((target_door_state - current_door_state)) < 0.1

        # For logging
        self.extras["target_state"] = target_door_state
        self.extras["achieved_state"] = current_door_state 

        return reward, success


class CloseMicrowaveDoor(BaseTask):
    def __init__(
        self,
        env,
        active_env_state
    ):
        super().__init__(
            env=env,
            active_env_state=active_env_state,
        )

        self.part_target_state = self.env.cabinet_target_joint_lower_limits_tensor.clone()

    def compute_reward(self):
        # Parse Door Handle Position
        current_handle_position = self.env.get_position_by_link_name()

        # Parse Current Robot Gripper Position
        current_gripper_position = self.env.get_robot_gripper_position()

        # Estimate the distance between the Robot Gripper and the Door handle
        distance_gripper_to_handle = torch.norm(current_gripper_position - current_handle_position, dim=-1)

        # Parse Current Door State
        current_door_state = self.env.get_state_by_link_name()

        # Parse Target Door State
        #target_door_state = self.get_joint_upper_limit_by_part_id(self.door_part_id)

        # Estimate the distance between the target door state and the current door state
        distance_current_to_target = current_door_state

        # The cost is the sum of the distance of the gripper to the handle and
        # the distance of the current door state to the target door state
        cost = distance_gripper_to_handle + distance_current_to_target

        # The reward is the negative of the cost
        reward = -cost

        self.extras["target_state"] = self.part_target_state.view(1, -1)
        self.extras["achieved_state"] = self.env.cabinet_dof_tensor_spec[:, :, 0].view(-1)

        diff_from_success = torch.abs((self.part_target_state.view(1, -1) - self.env.cabinet_dof_tensor_spec[:, :, 0]).view(-1))
        success = (diff_from_success < 0.1)

        return reward, success
