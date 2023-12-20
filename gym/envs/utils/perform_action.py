import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate, quat_apply, quat_conjugate,\
    quat_mul, tensor_clamp
from .misc import _draw_line
from .compute import *
from utils.gym_info import clip_actions, clip_observations, clip_error_pos, clip_error_rot, clip_delta
from utils.tools import axis_angle_to_matrix, matrix_to_quaternion

GRASP_COEFF = 0.1

def update_grasp(task, env_ids_grasp, env_ids_ungrasp, gripper_pose):
    # grasp
    if env_ids_grasp.any():
        task.env.root_tensor[env_ids_grasp, -3, :3] = gripper_pose[env_ids_grasp]
        task.env.root_tensor[env_ids_grasp, -3, 7:13] = 0.0
        task.env.gym.set_actor_root_state_tensor(
            task.env.sim,
            gymtorch.unwrap_tensor(task.env.root_tensor)         
        )
    
    #ungrasp
    if env_ids_ungrasp is not None and env_ids_ungrasp.any():
        task.env.root_tensor[env_ids_ungrasp, -3, :3] = task.goal[env_ids_ungrasp]
        task.env.root_tensor[env_ids_ungrasp, -3, 7:13] = 0.0
        task.env.gym.set_actor_root_state_tensor(
            task.env.sim,
            gymtorch.unwrap_tensor(task.env.root_tensor)
        )

def _grasp(task, actions):

    # gripper_pos = actions[:, -2:] > 0.5

    task.env.pos_act[:, -3:-1] = 0.0 # fully open gripper

    current_pos = task.env.get_robot_gripper_position()    
    asset_pos = task.env.static_asset_root_tensor[:, :3]
    dist = (asset_pos - current_pos).pow(2).sum(1).sqrt()
    env_ids_grasp = dist < GRASP_COEFF

    env_ids_ungrasp = None
    if hasattr(task, 'goal'):
        dist_goal = (task.goal - asset_pos).pow(2).sum(1).sqrt()
        env_ids_ungrasp = dist_goal < 0.1

    # gripper pose True, gripper near object: Attach object to suction
    # gripper pose True, object near goal: Detach object
    # gripper pose False, 
    update_grasp(task, env_ids_grasp, env_ids_ungrasp, current_pos)


def _perform_actions(task, actions):
    actions = torch.clamp(actions, -clip_actions, clip_actions)
    # actions[:, :3] = torch.tensor([1,1,1]).cuda()

    # actions
    task.env.pos_act[:] = 0
    task.env.eff_act[:] = 0
    task.env.vel_act[:] = 0

    dof_pos = task.env.franka_dof_tensor[:, :, 0]
    target_pos = actions[:, :3]
    target_rot = matrix_to_quaternion(axis_angle_to_matrix(actions[:, 3:6]))
    pos_err = target_pos - task.env.hand_rigid_body_tensor[:, :3]
    rot_err = orientation_error(target_rot, task.env.hand_rigid_body_tensor[:, 3:7])

    dpose = torch.cat([pos_err, rot_err], -1).unsqueeze(-1)
    # delta = torch.clip(control_ik(task.jacobian_tensor[:, task.hand_rigid_body_index - 1, :, :-2], task.device, dpose, task.env_num), -clip_delta, clip_delta)
    delta = control_ik(task.env.jacobian_tensor[:, task.env.hand_rigid_body_index - 1, :, :-2], task.env.device, dpose, task.env.env_num)
    task.env.pos_act[:, :-3] = dof_pos.squeeze(-1)[:, :-2] + delta
    if task.active_task_type == 'pick_and_place':
        _grasp(task, actions)
        #pass
    else:
        task.env.pos_act[:, -3:-1] = actions[:, -2:]

    task.env.pos_act_all[:, :task.env.pos_act.shape[1]] = task.env.pos_act
    task.env.vel_act_all[:, :task.env.vel_act.shape[1]] = task.env.vel_act
    task.env.eff_act_all[:, :task.env.eff_act.shape[1]] = task.env.eff_act
    task.env.gym.set_dof_position_target_tensor(task.env.sim, gymtorch.unwrap_tensor(task.env.pos_act_all.flatten()))
    task.env.gym.set_dof_velocity_target_tensor(task.env.sim, gymtorch.unwrap_tensor(task.env.vel_act_all.flatten()))
    task.env.gym.set_dof_actuation_force_tensor(task.env.sim, gymtorch.unwrap_tensor(task.env.eff_act_all.flatten()))