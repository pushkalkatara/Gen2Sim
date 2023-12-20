from lib2to3.pgen2.literals import simple_escapes
import numpy as np
import torch
from isaacgym import gymtorch
from .compute import *
from .get_running_bbox import _draw_bbox_tensor, _draw_line

def _partial_reset(task, to_reset = "all") :
    '''
    Partially reset environments
    '''
    if to_reset == "all" :
        to_reset = np.ones((task.env.env_num,))
    reseted = False
    for env_id, reset in enumerate(to_reset) :
        if reset.item() :
            reseted = True
            task.progress_buf[env_id] = 0
            task.reset_buf[env_id] = 0
            task.success_buf[env_id] = 0
            task.success_grasp_buf[env_id] = 0
            task.success_idx[env_id] = (task.success_idx[env_id] + 1) % task.success_queue.shape[1]
    
    if reseted:
        task.env.gym.refresh_jacobian_tensors(task.env.sim)
        task.env.gym.set_dof_state_tensor(
            task.env.sim,
            gymtorch.unwrap_tensor(task.env.initial_dof_states)
        )
        task.env.gym.set_actor_root_state_tensor(
            task.env.sim,
            gymtorch.unwrap_tensor(task.env.initial_root_states)
        )

def reset(task, to_reset = "all") :
    #print("reset")
    
    task._partial_reset(to_reset)

    task.env.gym.simulate(task.env.sim)
    task.env.gym.fetch_results(task.env.sim, True)
    if not task.env.headless:
        task.env.render()
    if task.env.cfg["env"]["enableCameraSensors"] == True:
        task.env.gym.step_graphics(task.env.sim)

    task._refresh_observation()

    success = task.success.clone()
    reward, done, dist_tip = task._get_reward_done()
    
    task.extras["successes"] = success
    task.extras["success_rate"] = task.success_rate
    task.extras["success_entropy"] = task.success_entropy
    task.extras["dist_tip"] = dist_tip
    return task.obs_buf, task.rew_buf, task.reset_buf, None