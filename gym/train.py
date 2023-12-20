import os
import json
import isaacgym
import numpy as np
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym import gymapi
import datetime
import wandb
from pathlib import Path
from utils.gym_info import *
import random
import torch

def get_args():
    custom_params = [
        {"name": "--headless", "action": "store_true", "default": False},
        {"name": "--data_root", "type": str, "default": '/projects/katefgroup/learning-simulations/gym_scale/data', "help": "Assets Root Path"},
        {"name": "--env_name", "type": str, "default": 'StorageFurniture-46037-link_1-handle_3-joint_1-handlejoint_3', "help": "Asset Name"},
        {"name": "--model_dir", "type": str, "default": "","help": "Choose a model dir"},
        {"name": "--group_name", "type": str, "default": "default_group"},
        {"name": "--env_num", "type": int, "default": 250},
        {"name": "--seed", "type": int, "default": 526},
        {"name": "--is_testing", "type": bool, "default": False},
        {"name": "--save_video", "type": bool, "default": False},
        {"name": "--log_dir", "type": str, "default": '/projects/katefgroup/learning-simulations/gym_scale/logs', "help": "Logging Directory for tb plots, checkpoint etc."},
        
    ]
    args = gymutil.parse_arguments(
        custom_parameters=custom_params
    )

    args.device_id = args.compute_device_id
    args.device = args.sim_device_type if args.use_gpu_pipeline else "cpu"
    
    return args

def set_seed(seed, torch_deterministic = False):
    if seed == -1 and torch_deterministic:
        seed = 2333
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed) # random seed for random module
    np.random.seed(seed) # for np module
    torch.manual_seed(seed) # for pytorch module
    os.environ['PYTHONHASHSEED'] = str(seed) # for os env Python hash seed
    torch.cuda.manual_seed(seed) # cuda manual seed
    torch.cuda.manual_seed_all(seed) # cuda manual seed all

    if torch_deterministic: # torch deterministic
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_deterministic(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    return seed

def parse_sim_params():
    sim_params = gymapi.SimParams()
    sim_params.dt = sim_params_dt
    sim_params.num_client_threads = 0

    sim_params.physx.solver_type = sim_params_physx_solver_type
    sim_params.physx.num_position_iterations = sim_params_physx_num_position_iterations
    sim_params.physx.num_velocity_iterations = sim_params_physx_num_velocity_iterations
    sim_params.physx.num_threads = sim_params_physx_num_threads
    sim_params.physx.use_gpu = True
    sim_params.physx.num_subscenes = 0
    sim_params.physx.max_gpu_contact_pairs = sim_params_physx_max_gpu_contact_pairs
    sim_params.physx.rest_offset = sim_params_physx_rest_offset
    sim_params.physx.bounce_threshold_velocity = sim_params_physx_bounce_threshold_velocity
    sim_params.physx.max_depenetration_velocity = sim_params_physx_max_depenetration_velocity
    sim_params.physx.default_buffer_size_multiplier = sim_params_physx_default_buffer_size_multiplier
    sim_params.physx.contact_offset = sim_params_physx_contact_offset

    sim_params.use_gpu_pipeline = True
    sim_params.physx.use_gpu = True

    return sim_params

def get_ppo(args, env, cfg_train, logdir, wandb_writer):
    from algorithms.ppo import PPO, ActorCriticPC
    learn_cfg = cfg_train["learn"]
    is_testing = args.is_testing
    ckpt_paths = []
    ckpt_path = ''
    
    curr_task = env.task.__class__.__name__
    print("Running Current Task:", curr_task)
    if args.model_dir != "":
        ckpt_paths = args.model_dir.split('_sep_')
        ckpt_path = ckpt_paths[0]
        '''
        ckpt_tasks = [ckpt_path.split('/')[-1].split('_')[0] for ckpt_path in ckpt_paths]
        idx = ckpt_tasks.index(curr_task) if curr_task in ckpt_tasks else None
        if idx is not None:
            ckpt_path = ckpt_paths[idx]
            is_testing = True
        '''
    
    logdir = logdir + "_algo-seed{}".format(cfg_train["seed"])
    
    ppo = PPO(vec_env=env,
              actor_critic_class=ActorCriticPC,
              num_transitions_per_env=learn_cfg["nsteps"],
              num_learning_epochs=learn_cfg["noptepochs"],
              num_mini_batches=learn_cfg["nminibatches"],
              clip_param=learn_cfg["cliprange"],
              gamma=learn_cfg["gamma"],
              lam=learn_cfg["lam"],
              init_noise_std=learn_cfg.get("init_noise_std", 0.3),
              value_loss_coef=learn_cfg.get("value_loss_coef", 2.0),
              entropy_coef=learn_cfg["ent_coef"],
              learning_rate=learn_cfg["optim_stepsize"],
              max_grad_norm=learn_cfg.get("max_grad_norm", 2.0),
              use_clipped_value_loss=learn_cfg.get("use_clipped_value_loss", False),
              schedule=learn_cfg.get("schedule", "fixed"),
              desired_kl=learn_cfg.get("desired_kl", None),
              max_lr=learn_cfg.get("lr_upper", 1e-3),
              min_lr=learn_cfg.get("lr_lower", 1e-7),
              model_cfg=cfg_train["policy"],
              device=env.rl_device,
              sampler=learn_cfg.get("sampler", 'sequential'),

              log_dir=logdir,
              is_testing=is_testing,
              eval_freq=learn_cfg["eval_freq"],
              eval_round=learn_cfg["eval_round"],
              print_log=learn_cfg["print_log"],
              max_iterations =learn_cfg["max_iterations"],
              wandb_writer = wandb_writer, 
              checkpoint_path= ckpt_path,

              use_adv_norm = learn_cfg["use_adv_norm"],           ### trick 1
              adv_norm_epsilon= learn_cfg["adv_norm_epsilon"],
              learning_rate_decay = learn_cfg["learning_rate_decay"],   ### trick 6
              use_grad_clip = learn_cfg["use_grad_clip"], 
              adam_epsilon = learn_cfg["adam_epsilon"],           ### trick 9
            
              )
    print("checkpoint paths:", ckpt_path)
    if ckpt_path != "":
        print("Loading model from {}".format(ckpt_path))
        ppo.load(ckpt_path)

    return ppo

from envs.base.vec_task import VecTaskPythonArm
from envs.gpt_task import *
from envs.base_env import BaseEnv

def load_env(args, cfg, cfg_algo, sim_params, log_dir):
    print(args.device_id, args.sim_device)
    # seed
    cfg["seed"] = cfg_algo["seed"]
    
    cfg["save_video_dir"] = "video/"
    cfg_env = cfg["env"]
    cfg_env["seed"] = cfg["seed"]

    log_dir = log_dir + "_seed{}".format(cfg["seed"])

    env = BaseEnv(
        cfg=cfg,
        sim_params=sim_params,
        physics_engine=args.physics_engine,
        device_type=args.device,
        device_id=args.device_id,
        headless=args.headless,
        log_dir=log_dir
    )
    return env

def main():
    args = get_args()

    sim_params = parse_sim_params()

    cfg = {
        'save_video': args.save_video,
        'env':{
            'maxEpisodeLength': 200,
            'env_num': 250,   
            'action_normalization': 'clip',
            'asset': {
                'assetRoot': args.data_root,
                'envName': args.env_name,
                'articulatedAsset': "",
                'staticAsset': "",
                'robot': {
                    'FrankaSlider_newtips': {
                        'filePath': 'franka_description/robots/franka_panda_slider_new.urdf',
                        'numActions': 11,
                        'ikNumActions': 8
                    },
                },
            },
            'envSpacing': 3.0,
            'robotName': 'FrankaSlider_newtips',
            'franka_scale': 2,
            'usePartRecoverForce': False,
            'enableCameraSensors': False,
        },
        'log':{
            'root_dir': args.log_dir,
            'group_name': args.group_name,
            'exp_name': "exp",
        },

        'obs': {
            'pose_baseline': False,
            'state': {
                'use_robot_qpose': True,
                'use_cabinet_qpose': True,
                'use_cabinet_bbox': True,
                'use_bbox_type': 'gt',
                'use_bbox_form': 'edges',
                'use_hand_root_pose': True,
            },
        },
    }

    cfg_algo = {
        'learn': {
            'agent_name': 'franka',
            'test': False,
            'resume': 0,
            'save_interval': 10,
            'checkpoint': -1,
            'eval_round': 1, # note check if we need more eval round, if yes update video saver
            'eval_freq': 50,
            'print_log': True,
            'max_iterations': 20000,
            'cliprange': 0.1,
            'ent_coef': 0.01,
            'nsteps': 5,
            'noptepochs': 8,
            'nminibatches': 2,
            'optim_stepsize': 0.0003,
            'schedule': 'adaptive',
            'desired_kl': 0.01,
            'lr_upper': '1e-3',
            'lr_lower': '1e-7',
            'gamma': 0.99,
            'lam': 0.95,
            'init_noise_std': 1,
            'log_interval': 10,
            'asymmetric': False,
            'use_adv_norm': True,
            'adv_norm_epsilon': 1e-08,
            'learning_rate_decay': True,
            'use_grad_clip': True,
            'max_grad_norm': 0.5,
            'adam_epsilon': 1e-05
        },
        'policy': {
            'save_obs_path': None,
            'pi_hid_sizes': [512, 512, 64],
            'vf_hid_sizes': [512, 512, 64],
            'activation': 'elu',
            'action_normalization': None,
            'actor_freeze': False,
            'feature_dim': 112,
            'backbone_type': 'None',
            'ckpt': 'None',
            'buffer_length': 1,
            'bc_epochs': 20,
            'use_expert': True,
            'max_batch': 100,
            'demo_num': 5,
            'use_seg': False,
            'freeze': False,
            'action_clip': True
        },
    }

    # Parse gpt-generated tasks based on asset 
    asset_category = args.env_name

    with open(os.path.join(args.data_root, 'category_to_assets.json')) as f:
        category_to_assets = json.load(f)
    # select any asset, use all while scaling demonstrations.
    asset = category_to_assets[asset_category][0]
    cfg["env"]["asset"]["articulatedAsset"] = asset

    # later update this for complex environments with multiple assets.
    cfg["env"]["asset"]["staticAsset"] = 'green_pepper'

    with open(os.path.join(args.data_root, 'asset_to_tasks.json')) as f:
        asset_to_tasks = json.load(f)
    # select any task, use all while scaling demonstrations.
    args.tasks = asset_to_tasks[asset][0]

    cfg["log"]["exp_name"] = f"{args.tasks}_{args.seed}"
    cfg["graphics_device_id"] = args.graphics_device_id
    cfg["env"]["env_num"] = args.env_num
    cfg_algo["seed"] = args.seed

    if args.headless:
        cfg["graphics_device_id"] = -1

    logdir = f'{cfg["log"]["root_dir"]}/{cfg["log"]["group_name"]}/{cfg["log"]["exp_name"]}'

    set_seed(args.seed)

    print("------------------ log saving information -------------------")
    print(cfg["log"]["group_name"])
    print("------------------------------------------------------------")

    # We use tensorboard for logging, but wandb can also be enabled.
    wandb.init(
        project="gen2sim",
        config=cfg_algo,
        name=cfg["log"]["exp_name"],
        mode="disabled"
    )

    env = load_env(args, cfg, cfg_algo, sim_params, logdir)
    tasks = args.tasks.split(',')

    active_env_state = None
    for task in tasks:
        task = eval(task)(env, active_env_state)
        task = VecTaskPythonArm(task, args.sim_device)

        # cam_pos = gymapi.Vec3(3, -2, 3)
        # cam_lookat = gymapi.Vec3(-2.0, 0, -0.0)
        # task.gym.viewer_camera_look_at(task.viewer, None, cam_pos, cam_lookat)

        PPO = get_ppo(args, task, cfg_algo, logdir, wandb)
        PPO.run(num_learning_iterations=cfg_algo["learn"]["max_iterations"], 
                    log_interval=cfg_algo["learn"]["save_interval"])
        
        active_env_state = task.task.end_task_dict
    
    # # convert all images to video
    if args.save_video:
        import subprocess
        import glob
        import cv2

        def convert_images_to_video(image_folder, video_name):
            # Get the list of image filenames in the folder
            image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
            image_files.sort()  # Sort the files in ascending order if needed
            
            # Read the first image to get dimensions
            first_image = cv2.imread(os.path.join(image_folder, image_files[0]))
            height, width, _ = first_image.shape
            
            # Define the video codec and create a VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use appropriate codec (e.g., "XVID" for AVI format)
            video = cv2.VideoWriter(video_name, fourcc, 30, (width, height))  # Adjust frame rate if needed
            
            # Iterate through the image files and write them to the video
            for image_file in image_files:
                image_path = os.path.join(image_folder, image_file)
                image = cv2.imread(image_path)
                video.write(image)
                #os.remove(image_path)
                
            
            # Release the VideoWriter and destroy any remaining windows
            video.release()
            cv2.destroyAllWindows()
        
        video_path = os.path.join(task.task.env.save_video_dir, "video.mp4")
        print(video_path)
        convert_images_to_video(task.task.env.save_video_dir + "/tmp/", video_path)

        # Save status of policy to a file
        def save_dict(dictionary, filename):
            with open(filename, 'w') as file:
                json.dump(dictionary, file)

        json_file = os.path.join(task.task.env.save_video_dir, "status.json")
        save_dict(task.env.status_dict, json_file)

if __name__ == "__main__":
    main()