from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate, quat_apply, quat_conjugate,\
    quat_mul, tensor_clamp
import numpy as np
from requests import head
import torch
from tqdm import tqdm
import os,sys
from os.path import join as pjoin
from random import shuffle
import json
import yaml
from isaacgym.gymtorch import wrap_tensor
from utils.gym_info import *
from rich.progress import track
from glob import glob
from pathlib import Path
from .utils.load_env import _load_static_asset, _load_static, _load_articulated_asset, _load_articulated,\
     _load_franka, _create_ground_plane, _place_agents
from .utils.compute import *
from .utils.misc import _draw_line, _draw_cross
from .utils.get_running_bbox import get_bbox_for_now, get_bbox_from_world_to_isaac, _draw_bbox, \
    get_bbox_isaac, get_bbox_for_now_tensor, get_bbox_from_world_to_isaac_tensor, \
    get_bbox_isaac_tensor, _draw_bbox_tensor, get_bbox_pt, get_bbox_isaac_tensor_nohandle, get_bbox_pt_nohandle
from isaacgym.gymtorch import wrap_tensor

# for any representation
# only focus on loadind object, franka and acquire their information


class BaseEnv():
    def __init__(
        self,
        cfg,
        sim_params,
        physics_engine,
        device_type,
        device_id,
        headless,
        log_dir = None,
    ):
        # init cfg, sim, phy, device
        print(cfg)
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.log_dir = log_dir
        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.device_type = device_type
        self.device_id = device_id
        self.headless = headless
        self.device = "cpu"
        if self.device_type == "cuda" or self.device_type == "GPU":
            self.device = "cuda" + ":" + str(self.device_id)
        self.graphics_device_id = cfg["graphics_device_id"]
        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]
        self.save_video_dir = cfg["save_video_dir"]
        self.env_num = cfg["env"]["env_num"]

        # create sim, load object, actor
        self.init_gym()
        self.init_indices()

        self.initial_dof_states = self.dof_state_tensor_all.clone()
        self.initial_root_states = self.root_tensor.clone()

        self.franka_root_tensor = self.root_tensor[:, 0, :] # N*13
        self.articulated_asset_root_tensor = self.root_tensor[:, 1, :] # N*13
        self.static_asset_root_tensor = self.root_tensor[:, 2, :] # N*13

        self.init_act_tensors()

    def init_act_tensors(self):
        # cross-check these
        self.dof_dim = self.franka_num_dofs + 1
        
        self.pos_act = torch.zeros((self.env_num, self.dof_dim), device=self.device)
        self.vel_act = torch.zeros((self.env_num, self.dof_dim), device=self.device)
        self.eff_act = torch.zeros((self.env_num, self.dof_dim), device=self.device)

        self.pos_act_all = torch.zeros_like(self.dof_state_tensor_all[:,0], device = self.device).reshape((self.env_num, -1))
        self.vel_act_all = torch.zeros_like(self.dof_state_tensor_all[:,0], device = self.device).reshape((self.env_num, -1))
        self.eff_act_all = torch.zeros_like(self.dof_state_tensor_all[:,0], device = self.device).reshape((self.env_num, -1))

    def init_indices(self):
        # precise for slices of tensors
        env_ptr = self.env_ptr_list[0]

        self.hand_rigid_body_index = self.gym.find_actor_rigid_body_index(
            env_ptr, self.franka_actor, "panda_hand", gymapi.DOMAIN_ENV) # 10
        self.hand_lfinger_rigid_body_index = self.gym.find_actor_rigid_body_index(
            env_ptr, self.franka_actor, "panda_leftfinger", gymapi.DOMAIN_ENV) # 11
        self.hand_rfinger_rigid_body_index = self.gym.find_actor_rigid_body_index(
            env_ptr, self.franka_actor, "panda_rightfinger", gymapi.DOMAIN_ENV) # 12

    def set_sim_params_up_axis(self, sim_params):
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity.x = 0
        sim_params.gravity.y = 0
        sim_params.gravity.z = -9.81

    def get_states(self):
        return self.states_buf
    
    def render(self, sync_frame_time = False):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()
            
            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
            
            # fetch results
            if self.device != "cpu":
                self.gym.fetch_results(self.sim, True)
            
            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
            else:
                self.gym.poll_viewer_events(self.viewer)
        
    def init_gym(self):
        self.gym = gymapi.acquire_gym()
        self.dt = self.sim_params.dt
        self.set_sim_params_up_axis(self.sim_params)

        self.sim = self.gym.create_sim(self.device_id, self.graphics_device_id,
            self.physics_engine, self.sim_params)
        if self.sim is None:
            print("*** Failed to create sim")
            assert False

        self.env_ptr_list = []
        _create_ground_plane(self)
        self.franka_loaded = False # Speeds up env
        _place_agents(self, self.env_num, self.cfg["env"]["envSpacing"], use_cam=False)
        self.gym.prepare_sim(self.sim)

        print("Env: number of environments", self.env_num)
        # init viewer
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

            # set the camera position based on up axis
            sim_params = self.gym.get_sim_params(self.sim)
            cam_pos = gymapi.Vec3(1.4, -1.3, 1.7)
            cam_target = gymapi.Vec3(0, 0, 0.7)

            self.gym.viewer_camera_look_at(
                self.viewer, None, cam_pos, cam_target)

        # from simulator acquire tensors
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.root_tensor = wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
        self.root_tensor = self.root_tensor.view(self.env_num, -1, 13)

        self.jacobian_tensor = wrap_tensor(self.gym.acquire_jacobian_tensor(self.sim, "franka"))
        
        self.dof_state_tensor_all = wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim))
        self.dof_state_tensor_used = self.dof_state_tensor_all.reshape([self.env_num, self.franka_num_dofs+self.cabinet_dof_num+self.static_asset_dof_num, self.dof_state_tensor_all.shape[-1]])
        
        self.actions = torch.zeros((self.env_num, self.num_actions), device=self.device)

        self.static_asset_actor_ids = torch.tensor(self.static_asset_actor_ids, dtype=torch.int32, device=self.device)

    def update_init_states(self, active_env_state):
        self.initial_dof_states = active_env_state['dof_states']
        self.initial_root_states = active_env_state['root_states']
        self.initial_rigid_body_tensor = active_env_state['rigid_body_states']

        # check if we want to update others as well.
        # self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.initial_dof_states))
        # self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.initial_root_states))

        # self.gym.refresh_actor_root_state_tensor(self.sim)
        # self.gym.refresh_dof_state_tensor(self.sim)
        # self.gym.refresh_rigid_body_state_tensor(self.sim)

    def _load_articulated_asset(self, progress):
        return _load_articulated_asset(self, progress)

    def _load_static_asset(self, progress):
        return _load_static_asset(self, progress)

    def load(self, path, iteration):
        pass

    def intervaledRandom_(self, tensor, dist, lower=None, upper=None) :
        tensor += torch.rand(tensor.shape, device=self.device)*dist*2 - dist
        if lower is not None and upper is not None :
            torch.clamp_(tensor, min=lower, max=upper)

    def get_handle_position(self):
        if self.target_part != 'obj':
            if self.target_part in ["door", "handle"]:
                self.part_bbox_tensor, self.handle_bbox_tensor = get_bbox_isaac_tensor(self, self.cabinet_dof_tensor[:,0], 0)
            elif self.target_part == "drawer":
                self.part_bbox_tensor, self.handle_bbox_tensor = get_bbox_isaac_tensor(self, self.cabinet_dof_tensor[:,0], 1)

            # print("open branch reward!")
            handle_out = self.handle_bbox_tensor[:, 0] - self.handle_bbox_tensor[:, 4]
            handle_long = self.handle_bbox_tensor[:, 1] - self.handle_bbox_tensor[:, 0]
            handle_short = self.handle_bbox_tensor[:, 3] - self.handle_bbox_tensor[:, 0]
            handle_mid = (self.handle_bbox_tensor[:, 0] + self.handle_bbox_tensor[:, 6]) / 2
            handle_out_length = torch.norm(handle_out, dim = -1)
            handle_long_length = torch.norm(handle_long, dim = -1)
            handle_short_length = torch.norm(handle_short, dim = -1)
            handle_shortest = torch.min(torch.min(handle_out_length, handle_long_length), handle_short_length)
        else:
            handle_mid = torch.zeros((self.env_num, 3), device=self.device)
            assert False
        return handle_mid

    def get_position_by_link_name(self):
        '''
        Modify API later to query by Link Name
        '''
        if self.target_part in ["door", "handle"]:
            self.part_bbox_tensor, self.handle_bbox_tensor = get_bbox_isaac_tensor(self, self.cabinet_dof_tensor[:,0], 0)
        elif self.target_part == "drawer":
            self.part_bbox_tensor, self.handle_bbox_tensor = get_bbox_isaac_tensor(self, self.cabinet_dof_tensor[:,0], 1)
        handle_out = self.handle_bbox_tensor[:, 0] - self.handle_bbox_tensor[:, 4]
        handle_long = self.handle_bbox_tensor[:, 1] - self.handle_bbox_tensor[:, 0]
        handle_short = self.handle_bbox_tensor[:, 3] - self.handle_bbox_tensor[:, 0]
        handle_mid = (self.handle_bbox_tensor[:, 0] + self.handle_bbox_tensor[:, 6]) / 2
        return handle_mid

    def get_position_static_asset(self):
        return self.static_asset_rigid_body_tensor[:, :3]

    def get_robot_gripper_pose(self):
        def quat_axis(q, axis=0):
            from isaacgym.torch_utils import quat_rotate
            '''
            :func apply rotation represented by quanternion `q`
            on basis vector(along axis)
            :return vector after rotation
            '''
            basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
            basis_vec[:, axis] = 1
            return quat_rotate(q, basis_vec)
        hand_rot = self.hand_rigid_body_tensor[..., 3:7]
        hand_grip_dir = quat_axis(hand_rot, 2)
        franka_lfinger_pos = self.rigid_body_tensor_used[:, self.hand_lfinger_rigid_body_index][:, 0:3]\
            + hand_grip_dir*0.1
        franka_rfinger_pos = self.rigid_body_tensor_used[:, self.hand_rfinger_rigid_body_index][:, 0:3]\
            + hand_grip_dir*0.1
        return franka_lfinger_pos, franka_rfinger_pos

    def get_robot_gripper_position(self):
        franka_lfinger_pos, franka_rfinger_pos = self.get_robot_gripper_pose()
        finger_mid = (franka_lfinger_pos + franka_rfinger_pos) / 2
        return finger_mid
    
    def get_robot_gripper_distance_tips(self):
        franka_lfinger_pos, franka_rfinger_pos = self.get_robot_gripper_pose()
        distance_gripper_tips = torch.norm(franka_lfinger_pos - franka_rfinger_pos, dim = -1)
        return distance_gripper_tips

    def get_state_by_link_name(self):
        return self.cabinet_dof_tensor[:, 0]

    def get_limits_by_joint_name(self):
        joint_limit_tensor = {
            "lower": self.cabinet_target_joint_limits_tensor,
            "upper": self.cabinet_target_joint_upp_limits_tensor
        }
        return joint_limit_tensor