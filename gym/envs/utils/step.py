import os
import torch
from .get_observation import _refresh_observation
from isaacgym import gymapi
import json

def step(task, actions) :
    task._perform_actions(actions)
    
    task.env.gym.simulate(task.env.sim)
    task.env.gym.fetch_results(task.env.sim, True)
    
    if not task.env.headless :
        task.env.render()
    if task.env.cfg["env"]["enableCameraSensors"] == True:
        task.env.gym.step_graphics(task.env.sim)

    task._refresh_observation()

    reward, done ,dist_tip = task._get_reward_done()

    done = task.reset_buf.clone()
    success = task.success.clone()
    
    task.extras["successes"] = success
    task.extras["success_rate"] = task.success_rate
    task.extras["success_entropy"] = task.success_entropy
    task._partial_reset(task.reset_buf)

    if task.average_reward == None:
        task.average_reward = task.rew_buf.mean()
    else :
        task.average_reward = task.rew_buf.mean() * 0.01 + task.average_reward * 0.99
    task.progress_buf += torch.tensor([1], device = task.env.device)
    
    if task.env.cfg["save_video"]:
        cam_pos = gymapi.Vec3(1.7,0,2.7)
        cam_target = gymapi.Vec3(-2.0,0., -0.5)
        task.env.gym.viewer_camera_look_at(task.env.viewer, None, cam_pos, cam_target)
        folder_path = task.env.save_video_dir + "/tmp/"
        os.makedirs(folder_path, exist_ok=True)
        #file_name = "{}_{}.png".format(task.current_eval_round, task.current_step_num)
        file_name = "{}_{}.png".format(task.__class__.__name__, str(task.current_step_num).zfill(5))
        status = task.env.gym.write_viewer_image_to_file(task.env.viewer, os.path.join(folder_path, file_name))

    return task.obs_buf, task.rew_buf, done, None
