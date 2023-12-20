
from typing import List
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_rotate, quat_apply, quat_conjugate,\
     quat_mul, tensor_clamp
import numpy as np
import time
import torch
from .misc import _draw_line
from scipy.spatial.transform import Rotation as R
from .get_running_bbox import get_bbox_for_now, get_bbox_from_world_to_isaac, _draw_bbox, \
    get_bbox_isaac, get_bbox_for_now_tensor, get_bbox_from_world_to_isaac_tensor, \
    get_bbox_isaac_tensor, _draw_bbox_tensor

# Y apply rotation in basis vector
def quat_axis(q, axis=0):
    '''
    :func apply rotation represented by quanternion `q`
    on basis vector(along axis)
    :return vector after rotation
    '''
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)

def normalize_and_clip_in_interval(x, min_x, max_x=None):
    if max_x is None:
        min_x = -abs(min_x)
        max_x = abs(min_x)
    len_x = max_x - min_x
    # print(torch.max(x, torch.ones_like(x) * min_x))
    # print(torch.min(torch.max(x, torch.ones_like(x) * min_x), torch.ones_like(x) * max_x))
    # print(torch.min(torch.max(x, torch.ones_like(x) * min_x), torch.ones_like(x) * max_x) - min_x)
    # print(len_x)
    # print((torch.min(torch.max(x, torch.ones_like(x) * min_x), torch.ones_like(x) * max_x) - min_x) / len_x)
    return (torch.min(torch.max(x, torch.ones_like(x) * min_x), torch.ones_like(x) * max_x) - min_x) / len_x

def get_reward_done(task):
    finger_mid = task.env.get_robot_gripper_position()
    handle_mid = task.env.get_handle_position()
    dist_mid = torch.norm(finger_mid - handle_mid, dim = -1)
   
    task.rew_buf, success = task.compute_reward()

    time_out = (task.progress_buf >= task.env.max_episode_length-1)
    
    task.reset_buf = (task.reset_buf | time_out)
    task.success_buf = task.success_buf | success

    task.success = task.success_buf # & time_out
    task.success_queue.view(-1)[task.success_idx + torch.arange(0, task.env.env_num).to(task.success_queue.device)*task.success_queue.shape[1]] *= 1 - time_out.long()
    task.success_queue.view(-1)[task.success_idx + torch.arange(0, task.env.env_num).to(task.success_queue.device)*task.success_queue.shape[1]] += task.success

    task.success_rate = task.success_queue.mean(dim=-1)
    task.total_success_rate = task.success_rate.sum(dim=-1)
    task.success_entropy = - task.success_rate/(task.total_success_rate+1e-8) * torch.log(task.success_rate/(task.total_success_rate+1e-8) + 1e-8) * task.env.env_num
    
    return task.rew_buf, task.reset_buf, dist_mid
