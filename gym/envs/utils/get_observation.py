from random import shuffle
from utils.gym_info import *
from pathlib import Path
import torch
from os.path import join as pjoin
import numpy as np
from isaacgym.gymtorch import wrap_tensor
from .compute import *
from .get_running_bbox import get_bbox_isaac_tensor, _draw_bbox_tensor,  get_bbox_pt, get_bbox_pt_nohandle ,get_bbox_isaac_tensor_nohandle
from data_structure.observation import Observations


def _get_base_observation(task, suggested_gt=None) :
    task.env.dof_state_tensor_all = wrap_tensor(task.env.gym.acquire_dof_state_tensor(task.env.sim))
    task.env.rigid_body_tensor_all = wrap_tensor(task.env.gym.acquire_rigid_body_state_tensor(task.env.sim))
    task.env.root_tensor = wrap_tensor(task.env.gym.acquire_actor_root_state_tensor(task.env.sim))

    task.env.dof_state_tensor_used = task.env.dof_state_tensor_all.reshape([task.env.env_num, task.env.franka_num_dofs+task.env.cabinet_dof_num+task.env.static_asset_dof_num, task.env.dof_state_tensor_all.shape[-1]])
    task.env.rigid_body_tensor_used = task.env.rigid_body_tensor_all.reshape([task.env.env_num, task.env.franka_rigid_num+task.env.cabinet_rigid_num+task.env.static_asset_rigid_num+task.env.distractor_rig_num+task.env.distractor_1_rig_num, task.env.rigid_body_tensor_all.shape[-1]])
    task.env.root_tensor = task.env.root_tensor.view(task.env.env_num, -1, 13)

    task.env.hand_rigid_body_tensor = task.env.rigid_body_tensor_used\
        [:, task.env.hand_rigid_body_index, :] # N*13
    task.env.franka_dof_tensor = task.env.dof_state_tensor_used\
        [:, :task.env.franka_num_dofs, :] # N*11*2
    task.env.cabinet_dof_tensor = task.env.dof_state_tensor_used\
        [:, task.env.franka_num_dofs + task.env.part_dof_id, :] # N*2
    task.env.cabinet_dof_tensor_spec = task.env.cabinet_dof_tensor.view(
        1, task.env.env_num, -1) # M*(N/M)*2
        

    task.env.cabinet_part_rigid_body_tensor = task.env.rigid_body_tensor_used[:, task.env.franka_rigid_num+task.env.part_rigid_id, :]
    task.env.cabinet_handle_rigid_body_tensor = task.env.rigid_body_tensor_used[:, task.env.franka_rigid_num+task.env.handle_rigid_id, :]
    task.env.cabinet_handle_pos_tensor = task.env.cabinet_handle_rigid_body_tensor[:,:3]

    hand_rot = task.env.hand_rigid_body_tensor[..., 3:7]
    hand_down_dir = quat_axis(hand_rot, 2)
    task.env.hand_tip_pos = task.env.hand_rigid_body_tensor[..., 0:3] + hand_down_dir * 0.130    # calculating middle of two fingers
    task.env.hand_mid_pos = task.env.hand_rigid_body_tensor[..., 0:3] + hand_down_dir * 0.07 # TODO modify
    task.env.hand_rot = hand_rot


    if task.env.target_part in ["door", "drawer", "handle"]:
        if task.env.target_part in ["door", "handle"]:
            part_bbox_tensor, handle_bbox_tensor = get_bbox_isaac_tensor(task.env, task.env.cabinet_dof_tensor[:,0], 0)
        elif task.env.target_part == "drawer":
            part_bbox_tensor, handle_bbox_tensor = get_bbox_isaac_tensor(task.env, task.env.cabinet_dof_tensor[:,0], 1)

        handle_out = handle_bbox_tensor[:, 0] - handle_bbox_tensor[:, 4]
        handle_long = handle_bbox_tensor[:, 1] - handle_bbox_tensor[:, 0]
        handle_short = handle_bbox_tensor[:, 3] - handle_bbox_tensor[:, 0]
        handle_center = (handle_bbox_tensor[:, 0] + handle_bbox_tensor[:, 6]) / 2
        part_out = part_bbox_tensor[:, 0] - part_bbox_tensor[:, 4]
        part_long = part_bbox_tensor[:, 1] - part_bbox_tensor[:, 0]
        part_short = part_bbox_tensor[:, 3] - part_bbox_tensor[:, 0]
        part_center = (part_bbox_tensor[:, 0] + part_bbox_tensor[:, 6]) / 2

    hand_pose = relative_pose(task.env.franka_root_tensor, task.env.hand_rigid_body_tensor).view(task.env.env_num, -1)
    
    root_tensor = task.env.franka_root_tensor[:, :3]

    task.env.static_asset_rigid_body_tensor = task.env.rigid_body_tensor_used[:, task.env.franka_rigid_num+task.env.cabinet_rigid_num, :]
    
    # Prepare State, Obs
    state = torch.zeros((task.env.env_num, 0), device = task.env.device, dtype=torch.float32)
    obs = torch.zeros((task.env.env_num, 0), device = task.env.device, dtype=torch.float32)
    robot_qpose = (2 * (task.env.franka_dof_tensor[:, :, 0]-task.env.franka_dof_lower_limits_tensor[:])/(task.env.franka_dof_upper_limits_tensor[:] - task.env.franka_dof_lower_limits_tensor[:])) - 1
    robot_qvel = task.env.franka_dof_tensor[:, :, 1]
    cabinet_qpose = task.env.cabinet_dof_tensor
    hand_root_pose = torch.cat((root_tensor, hand_pose), dim=1)
    # bbox

    if task.active_task_type == 'articulate':
        state = torch.cat((state, robot_qpose, robot_qvel, cabinet_qpose, hand_root_pose), dim = 1)
        if task.env.target_part in ["door", "drawer", "handle"]:
            state = torch.cat((state, handle_out, handle_long, handle_short, handle_center, part_out, part_long, part_short, part_center), dim = 1)
        else:
            state = torch.cat([state, torch.zeros(task.env.env_num, 24, device=task.env.device)], dim=1)
    elif task.active_task_type == 'pick_and_place':
        state = torch.cat((state, robot_qpose, robot_qvel, hand_root_pose), dim = 1)
        state = torch.cat((state, task.env.static_asset_rigid_body_tensor), dim=1)

    obs = state.clone()

    return Observations(state = state, obs = obs)

def _refresh_observation(task) :
    task.env.gym.refresh_actor_root_state_tensor(task.env.sim)
    task.env.gym.refresh_dof_state_tensor(task.env.sim)
    task.env.gym.refresh_rigid_body_state_tensor(task.env.sim)
    task.env.gym.refresh_jacobian_tensors(task.env.sim)

    task.obs_buf = task._get_base_observation()
    