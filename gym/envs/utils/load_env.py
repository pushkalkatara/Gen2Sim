from random import shuffle
from utils.gym_info import *
from pathlib import Path
import torch
from os.path import join as pjoin
import numpy as np
from isaacgym import gymtorch
import isaacgym
import json
from scipy.spatial.transform import Rotation as R
from utils.gym_info import cam1_pos, cam1_rot, cam2_pos,cam2_rot,cam3_pos, cam3_rot
import os, cv2

def _place_agents(task, env_num, spacing, use_cam):
    print("Simulation: creating agents")

    lower = gymapi.Vec3(-spacing, -spacing, -spacing)
    upper = gymapi.Vec3(spacing, spacing, spacing)
    num_per_row = int(np.sqrt(env_num))

    task.static_asset_actor_ids = []
    from rich.progress import Progress
    with Progress() as progress:
        task._load_static_asset(progress)
        task._load_articulated_asset(progress)
        _load_distractor_asset(task, progress)
        _load_distractor_asset_1(task, progress)
        task1 = progress.add_task('[red]Creating envs:', total=env_num)
        for i, env_id in enumerate(range(env_num)):
            env_ptr = task.gym.create_env(task.sim, lower, upper, num_per_row)
            _load_franka(task, env_ptr, env_id)
            _load_articulated(task, env_ptr, env_id)
            _load_static(task, env_ptr, env_id)
            _load_distractor(task, env_ptr, env_id)
            _load_distractor_1(task, env_ptr, env_id)
            task.static_asset_actor_ids.append(i)
            task.env_ptr_list.append(env_ptr)
            progress.update(task1, advance=1)

def _create_ground_plane(task):
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.,0.,1.)
    plane_params.static_friction = plane_params_static_friction
    plane_params.dynamic_friction = plane_params_dynamic_friction
    task.gym.add_ground(task.sim, plane_params)

def _load_distractor_asset(task, progress):
    def remove_joints(urdf):
        import xml.etree.ElementTree as ET
        tree = ET.parse(urdf)
        root = tree.getroot()
        for link in root.findall(".//link"):
            for collision in link.findall(".//collision"):
                link.remove(collision)
        for joint in root.findall('.//joint'):
            joint.set('type', 'fixed')
        tree.write(urdf.replace('mobility.urdf', 'visual.urdf'))

    asset_root_parent = str(Path(".").parent.absolute())
    asset_root = task.cfg["env"]["asset"]["assetRoot"]
    static_asset_name = "partnet-mobility-dataset/100496"
    static_asset_path = Path(pjoin(asset_root, static_asset_name))

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.disable_gravity = True
    asset_options.collapse_fixed_joints = True

    if static_asset_path.parent.name == 'partnet-mobility-dataset':
        remove_joints(pjoin(str(static_asset_path), "mobility.urdf"))
        asset = task.gym.load_asset(task.sim, asset_root, \
            pjoin(str(static_asset_name), "visual.urdf"), asset_options)
    else:
        remove_joints(pjoin(str(static_asset_path), "mobility_new.urdf"))
        asset = task.gym.load_asset(task.sim, asset_root, \
            pjoin(str(static_asset_name), "visual.urdf"), asset_options)

    task.distractor_asset = asset

    asset_rig_names = task.gym.get_asset_rigid_body_dict(asset)
    task.distractor_rig_num = len(asset_rig_names)

def _load_distractor_asset_1(task, progress):
    def remove_joints(urdf):
        import xml.etree.ElementTree as ET
        tree = ET.parse(urdf)
        root = tree.getroot()
        for link in root.findall(".//link"):
            for collision in link.findall(".//collision"):
                link.remove(collision)
        for joint in root.findall('.//joint'):
            joint.set('type', 'fixed')
        tree.write(urdf.replace('mobility.urdf', 'visual.urdf'))

    asset_root = task.cfg["env"]["asset"]["assetRoot"]
    static_asset_name = "partnet-mobility-dataset/100466"
    static_asset_path = Path(pjoin(asset_root, static_asset_name))

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.disable_gravity = True
    asset_options.collapse_fixed_joints = True

    if static_asset_path.parent.name == 'partnet-mobility-dataset':
        remove_joints(pjoin(str(static_asset_path), "mobility.urdf"))
        asset = task.gym.load_asset(task.sim, asset_root, \
            pjoin(str(static_asset_name), "visual.urdf"), asset_options)
    else:
        remove_joints(pjoin(str(static_asset_path), "mobility_new.urdf"))
        asset = task.gym.load_asset(task.sim, asset_root, \
            pjoin(str(static_asset_name), "visual.urdf"), asset_options)

    task.distractor_asset_1 = asset

    asset_rig_names = task.gym.get_asset_rigid_body_dict(asset)
    task.distractor_1_rig_num = len(asset_rig_names)

def _load_distractor(task, env_ptr, env_id):
    subenv_id = env_id % task.env_num

    object_init_pose = gymapi.Transform()
    object_init_pose.p = gymapi.Vec3(-1.7, 1.3, 0.8)
    #object_init_pose.r = gymapi.Quat(0.0, 0.0, 0.7071, 0.7071)
    object_init_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)
    obj_actor = task.gym.create_actor(
        env_ptr,
        task.distractor_asset,
        object_init_pose,
        "distractor_asset",
        env_id,
        1, 0
    )
    # asset_dof_props = task.gym.get_asset_dof_properties(task.static_asset)
    # task.gym.set_actor_dof_properties(env_ptr, obj_actor, cabinet_dof_props)
    task.gym.set_actor_scale(env_ptr, obj_actor, 0.7)

def _load_distractor_1(task, env_ptr, env_id):
    subenv_id = env_id % task.env_num

    object_init_pose = gymapi.Transform()
    object_init_pose.p = gymapi.Vec3(-0.5, -0.5, 0.6)
    object_init_pose.r = gymapi.Quat(0.0, 0.0, -0.34, 0.7071)
    obj_actor = task.gym.create_actor(
        env_ptr,
        task.distractor_asset_1,
        object_init_pose,
        "distractor_asset_1",
        env_id,
        1, 0
    )
    # asset_dof_props = task.gym.get_asset_dof_properties(task.static_asset)
    # task.gym.set_actor_dof_properties(env_ptr, obj_actor, cabinet_dof_props)
    task.gym.set_actor_scale(env_ptr, obj_actor, 0.5)


def _load_static(task, env_ptr, env_id):
    subenv_id = env_id % task.env_num

    object_init_pose = gymapi.Transform()
    object_init_pose.p = static_object_init_pose_p
    object_init_pose.r = static_object_init_pose_r
    obj_actor = task.gym.create_actor(
        env_ptr,
        task.static_asset,
        object_init_pose,
        "static_asset",
        env_id,
        1, 0
    )
    # asset_dof_props = task.gym.get_asset_dof_properties(task.static_asset)
    # task.gym.set_actor_dof_properties(env_ptr, obj_actor, cabinet_dof_props)
    task.gym.set_actor_scale(env_ptr, obj_actor, 2)

def _load_static_asset(task, progress):
    asset_root_parent = str(Path(".").parent.absolute())
    asset_root = task.cfg["env"]["asset"]["assetRoot"]
    static_asset_name = task.cfg["env"]["asset"]["staticAsset"]
    static_asset_path = Path(pjoin(asset_root, static_asset_name))

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = static_asset_options_fix_base_link_object
    asset_options.disable_gravity = static_asset_options_disable_gravity_object
    asset_options.collapse_fixed_joints = static_asset_options_collapse_fixed_joints_object
    asset_options.vhacd_enabled = True

    if static_asset_path.parent.name == 'partnet-mobility-dataset':
        asset = task.gym.load_asset(task.sim, asset_root, \
            pjoin(str(static_asset_name), "mobility.urdf"), asset_options)
    else:
        asset = task.gym.load_asset(task.sim, asset_root, \
            pjoin(str(static_asset_name), "urdf.urdf"), asset_options)
    asset_options.convex_decomposition_from_submeshes = True

    task.static_asset = asset
    asset_dof_num = task.gym.get_asset_dof_count(asset)
    asset_rig_names = task.gym.get_asset_rigid_body_dict(asset)

    # for suction gripper: remove to test physics
    asset_rigid_props = task.gym.get_asset_rigid_shape_properties(task.static_asset)[0]
    asset_rigid_props.contact_offset = 0
    asset_rigid_props.friction = 0

    task.gym.set_asset_rigid_shape_properties(asset, [asset_rigid_props])
    task.static_asset_dof_num = asset_dof_num
    task.static_asset_rigid_num = len(asset_rig_names)

def _load_articulated(task, env_ptr, env_id):
    subenv_id = env_id % task.env_num

    object_init_pose = gymapi.Transform()
    object_init_pose.p = object_init_pose_p
    object_init_pose.r = object_init_pose_r
    obj_actor = task.gym.create_actor(
        env_ptr,
        task.articulated_asset,
        object_init_pose,
        "articulated_asset",
        env_id,
        1, 0
    )

    cabinet_dof_props = task.gym.get_asset_dof_properties(task.articulated_asset)
    cabinet_dof_props['stiffness'][0] = 20.0 ##刚性系数
    cabinet_dof_props['damping'][0] = 200    ##阻尼系数
    cabinet_dof_props['friction'][0] = 5  ##摩擦系数
    cabinet_dof_props['effort'][0] = 0

    cabinet_dof_props["driveMode"][0] = gymapi.DOF_MODE_NONE

    task.gym.set_actor_dof_properties(env_ptr, obj_actor, cabinet_dof_props)

def _load_articulated_asset(task, progress):
    asset_root = task.cfg["env"]["asset"]["assetRoot"] + "/assets"
    articulated_asset_name = task.cfg["env"]["asset"]["articulatedAsset"]
    task.articulated_asset_path = Path(pjoin(asset_root, articulated_asset_name))

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = articulated_asset_options_fix_base_link_object
    asset_options.disable_gravity = articulated_asset_options_disable_gravity_object
    asset_options.collapse_fixed_joints = articulated_asset_options_collapse_fixed_joints_object

    task.asset_linkname = task.articulated_asset_path.name.split("-")[-4]
    task.asset_handlename = task.articulated_asset_path.name.split("-")[-3]
    task.asset_jointname = task.articulated_asset_path.name.split("-")[-2]

    with open(os.path.join(task.articulated_asset_path, 'semantics_relabel.txt')) as fid:
        semantics = [tuple(sem.split()) for sem in fid.read().splitlines()]

    task.target_part = next((sem[2] for sem in semantics if sem[0] == task.asset_linkname), None) # door, drawer, etc.

    if task.target_part is None:
        asset = task.gym.load_asset(task.sim, asset_root, \
            pjoin(str(articulated_asset_name), "mobility.urdf"), asset_options)
    else:
        asset = task.gym.load_asset(task.sim, asset_root, \
            pjoin(str(articulated_asset_name), "mobility_new.urdf"), asset_options)
    asset_options.convex_decomposition_from_submeshes = True

    if task.target_part != 'obj':
        path_bbox = str(task.articulated_asset_path) + "/bbox_info.json"
        with open(path_bbox, "r") as f:
            bbox_info = json.load(f)
    
    dof_dict = task.gym.get_asset_dof_dict(asset)
    dof_props = task.gym.get_asset_dof_properties(asset)
    dof_num = task.gym.get_asset_dof_count(asset)
    rig_names = task.gym.get_asset_rigid_body_dict(asset)
    joint_dict = task.gym.get_asset_joint_dict(asset)

    task.articulated_asset = asset
    task.part_rigid_id = rig_names[task.asset_linkname]
    task.handle_rigid_id = rig_names[task.asset_handlename]
    task.part_dof_id = dof_dict[task.asset_jointname]
    task.base_rigid_id = rig_names["base"]

    if task.target_part != 'obj':
        task.part_bbox = np.array(bbox_info['bbox_world'][bbox_info['link_name'].index(task.asset_linkname)]).astype(np.float32)
        task.handle_bbox = np.array(bbox_info['bbox_world'][bbox_info['link_name'].index(task.asset_handlename)]).astype(np.float32)
        task.part_axis_xyz = np.array(bbox_info['axis_xyz_world'][bbox_info['link_name'].index(task.asset_linkname)]).astype(np.float32)
        task.part_axis_dir = np.array(bbox_info['axis_dir_world'][bbox_info['link_name'].index(task.asset_linkname)]).astype(np.float32)
    
    task.cabinet_target_joint_limits_tensor = torch.tensor(dof_props["lower"][task.part_dof_id], device=task.device).repeat(task.env_num)
    task.cabinet_target_joint_upp_limits_tensor = torch.tensor(dof_props["upper"][task.part_dof_id], device=task.device).repeat(task.env_num)

    task.cabinet_dof_num = dof_num
    task.cabinet_rigid_num = len(rig_names)

    if task.target_part != 'obj':
        task.part_bbox_tensor_init = torch.tensor(np.array([task.part_bbox]).astype(np.float32), device=task.device, dtype=torch.float32).repeat_interleave(task.env_num, dim=0)
        task.handle_bbox_tensor_init = torch.tensor(np.array([task.handle_bbox]).astype(np.float32), device=task.device, dtype=torch.float32).repeat_interleave(task.env_num, dim=0)
        task.part_axis_xyz_tensor_init = torch.tensor(np.array([task.part_axis_xyz]).astype(np.float32), device = task.device).repeat_interleave(task.env_num, dim=0)
        task.part_axis_dir_tensor_init = torch.tensor(np.array([task.part_axis_dir]).astype(np.float32), device = task.device).repeat_interleave(task.env_num, dim=0)
        task.object_init_pose_p_tensor = torch.tensor(object_init_pose_p_np.astype(np.float32), device = task.device)
        task.object_init_pose_r_matrix_tensor = torch.tensor(R.from_quat(object_init_pose_r_np).as_matrix().astype(np.float32),device = task.device)

        matrix = torch.tensor(R.from_quat(object_init_pose_r_np).as_matrix().astype(np.float32), device = task.device).reshape(1, 3,3).repeat_interleave(task.env_num, dim=0)
        task.part_axis_xyz_tensor = torch.bmm(matrix, task.part_axis_xyz_tensor_init.reshape(-1, 3, 1)).reshape(-1,3) + torch.tensor(+object_init_pose_p_np.astype(np.float32), device = task.device)
        task.part_axis_dir_tensor = torch.bmm(matrix, task.part_axis_dir_tensor_init.reshape(-1, 3, 1)).reshape(-1,3)
    
    task.env_base_rigid_id = torch.tensor([task.base_rigid_id], device = task.device).repeat_interleave(task.env_num, dim=0)
    task.env_part_rigid_id = torch.tensor([task.part_rigid_id], device = task.device).repeat_interleave(task.env_num, dim=0)
    task.env_handle_rigid_id = torch.tensor([task.handle_rigid_id], device = task.device).repeat_interleave(task.env_num, dim=0)
    task.env_part_dof_id = torch.tensor([task.part_dof_id], device = task.device).repeat_interleave(task.env_num, dim=0)
    task.cabinet_dof_num_tensor = torch.tensor(task.cabinet_dof_num, device = task.device)

def _load_franka(task, env_ptr, env_id):
    if task.franka_loaded == False:
        franka_name = task.cfg["env"]["robotName"]
        franka_file = task.cfg["env"]["asset"]["robot"][franka_name]["filePath"]
        asset_root = task.cfg["env"]["asset"]["assetRoot"]
        task.num_qpose_actions = task.cfg["env"]["asset"]["robot"][franka_name]["numActions"]
        task.num_actions = task.cfg["env"]["asset"]["robot"][franka_name]["ikNumActions"]

        asset_options = gymapi.AssetOptions()
            
        asset_options.flip_visual_attachments = asset_options_flip_visual_attachments
        asset_options.fix_base_link = asset_options_fix_base_link
        asset_options.disable_gravity = asset_options_disable_gravity
        asset_options.armature = asset_options_armature
        task.franka_asset = task.gym.load_asset(task.sim, asset_root, 
            franka_file, asset_options)
        task.franka_loaded = True
    
    ######### The problem happened here! ##############
    franka_dof_max_torque, franka_dof_lower_limits, franka_dof_upper_limits, franka_dof_vel_upper_limits_tensor\
            = _get_dof_property(task, task.franka_asset)
            
    task.franka_dof_max_torque_tensor = torch.tensor(franka_dof_max_torque, device=task.device)
    task.franka_dof_upper_limits_tensor = torch.tensor(franka_dof_upper_limits, device=task.device)
    task.franka_dof_lower_limits_tensor = torch.tensor(franka_dof_lower_limits, device=task.device)
    task.franka_dof_vel_upper_limits_tensor = torch.tensor(franka_dof_vel_upper_limits_tensor, device = task.device)

    dof_props = task.gym.get_asset_dof_properties(task.franka_asset)
    # use position drive for all dofs
    dof_props["driveMode"][:].fill(gymapi.DOF_MODE_POS)
    dof_props["stiffness"][:].fill(400.0)
    dof_props["damping"][:].fill(40.0)
    dof_props["stiffness"][-2:].fill(0)
    dof_props["damping"][-2:].fill(0)   

    # root pose
    initial_franka_pose = gymapi.Transform()
    initial_franka_pose.r = initial_franka_pose_r
    initial_franka_pose.p = initial_franka_pose_p_open_door

    # set start dof
    task.franka_num_dofs = task.gym.get_asset_dof_count(task.franka_asset)
    task.franka_rigid_num = len(task.gym.get_asset_rigid_body_dict(task.franka_asset))
    default_dof_pos = np.zeros(task.franka_num_dofs, dtype=np.float32)

    default_dof_pos[:-2] = (franka_dof_lower_limits + franka_dof_upper_limits)[:-2] * 0.3
    # grippers open
    default_dof_pos[-2:] = franka_dof_upper_limits[-2:]
    franka_dof_state = np.zeros_like(franka_dof_max_torque, gymapi.DofState.dtype)
    if task.target_part == 'obj':
        franka_dof_state["pos"] = default_dof_pos
    else:
        state = np.load(str(task.articulated_asset_path) + f"/part_pregrasp_dof_state.npy",allow_pickle=True)
        franka_dof_state["pos"] = state[:-1, 0]

    franka_actor = task.gym.create_actor(
        env_ptr,task.franka_asset, initial_franka_pose,"franka",env_id,2,0)
    
    shape_props = task.gym.get_actor_rigid_shape_properties(env_ptr, franka_actor)

    task.gym.set_actor_dof_properties(env_ptr, franka_actor, dof_props)
    task.gym.set_actor_dof_states(env_ptr, franka_actor, franka_dof_state, gymapi.STATE_ALL)
    franka_scale = task.cfg["env"]["franka_scale"]
    task.gym.set_actor_scale(env_ptr, franka_actor, franka_scale)  
    task.franka_actor = franka_actor

def _get_dof_property(task, asset):
    dof_props = task.gym.get_asset_dof_properties(asset)

    dof_num = task.gym.get_asset_dof_count(asset)
    dof_lower_limits = []
    dof_upper_limits = []
    dof_max_torque = []
    dof_max_vel = []
    for i in range(dof_num):
        dof_max_torque.append(dof_props["effort"][i])
        dof_lower_limits.append(dof_props["lower"][i])
        dof_upper_limits.append(dof_props["upper"][i])
        dof_max_vel.append(dof_props["velocity"][i])
    dof_max_torque = np.array(dof_max_torque)
    dof_lower_limits = np.array(dof_lower_limits)
    dof_upper_limits = np.array(dof_upper_limits)
    dof_max_vel = np.array(dof_max_vel)
    return dof_max_torque, dof_lower_limits, dof_upper_limits, dof_max_vel
