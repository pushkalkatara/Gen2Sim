# Gen2Sim: Scaling up Robot Learning in Simulation with Generative Models

<div align="center">

[[Website]](https://gen2sim.github.io/)
[[arXiv]](https://arxiv.org/abs/2310.18308)
______________________________________________________________________

![](gifs/articulated_policy.gif)

</div>

Generalist robot manipulators need to learn a wide variety of manipulation skills across diverse environments. Current robot training pipelines rely on humans to provide kinesthetic demonstrations or to program simulation environments and to code up reward functions for reinforcement learning. Such human involvement is an important bottleneck towards scaling up robot learning across diverse tasks and environments. We propose Generation to Simulation (Gen2Sim), a method for scaling up robot skill learning in simulation by automating generation of 3D assets, task descriptions, task decompositions and reward functions using large pre-trained generative models of language and vision. We generate 3D assets for simulation by lifting open-world 2D object-centric images to 3D using image diffusion models and querying LLMs to determine plausible physics parameters. Given URDF files of generated and human-developed assets, we chain-of-thought prompt LLMs to map these to relevant task descriptions, temporal decompositions, and corresponding python reward functions for reinforcement learning. We show Gen2Sim succeeds in learning policies for diverse long horizon tasks, where reinforcement learning with non temporally decomposed reward functions fails. Gen2Sim provides a viable path for scaling up reinforcement learning for robot manipulators in simulation, both by diversifying and expanding task and environment development, and by facilitating the discovery of reinforcement-learned behaviors through temporal task decomposition in RL. Our work contributes hundreds of simulated assets, tasks and demonstrations, taking a step towards fully autonomous robotic manipulation skill acquisition in simulation.

# Installation
Gen2Sim requires Python ≥ 3.8. We have tested on Ubuntu 20.04.

1. Create a new conda environment with:
    ```
    conda create -n gen2sim python=3.8
    conda activate gen2sim
    ```

2. Install IsaacGym. Follow the [instruction](https://developer.nvidia.com/isaac-gym) to download the package.
    ```	
    cd isaacgym/python
    pip install -e .
    (test installation) python examples/joint_monkey.py
    ```

3. Install Gen2Sim dependencies. We also provide `req.txt` file with all the listed requirements.
    ```	
    pip install openai
    ```

4. Gen2Sim currently uses OpenAI API to query LLMs (GPT-4). You need to have an OpenAI API key to use Gen2Sim [here](https://platform.openai.com/account/api-keys)/. Then, set the environment variable in your terminal
    ```
    export OPENAI_API_KEY= "YOUR_API_KEY"
    ```

# Getting Started

## Asset Generation
Navigate to the `asset-gen` directory and and follow the `README.md` to generate asset URDFs.

## Task and Reward Generation
We provide scripts to prepare the prompt using asset description (URDF):
```
python task-gen/prompt.py <assets_path> <log_dir> <task_export_class_path>
```
- `<assets_path>` path to all the assets/environment for which tasks need to be generated.
- `<log_dir>` path to log LLM outputs for debugging.
- `<task_export_class_path>` path to export all generated classes.

## Skill Training in Simulation
We provide scripts to train RL policy using the generated task and reward function in IsaacGym. We use a modified version of FrankaPanda arm with mobile base as default robot.

```
python gym/train.py --group_name <group_name> --seed <seed> --env_num <num_envs> --env_name <env_name> --log_dir <log_dir>
```
- `<group_name>` logging name for the run.
- `<seed>` seed for RL policy.
- `<num_envs>` nums of parallel environments for exploration while RL.
- `<env_name>` environment name.
- `<log_dir>` Path to log tb plots, checkpoints etc.

Examples:

1. Tasks generated for asset Microwave (Simple Tasks like OpenMicrowaveDoor, CloseMicrowaveDoor)
```
python gym/train.py --group_name Microwave --seed 0 --env_num 250 --env_name Microwave --headless
```

2. Tasks generated for environment with Tennis Ball and Storage Furniture (Long-horizon tasks which require decomposition)
```
python gym/train.py --group_name Microwave --seed 0 --env_num 250 --env_name Microwave --headless
```

To collect demonstrations, evaluate trained policies using the following eval script:
```
python gym/train.py --group_name <group_name> --seed <seed> --env_num 1 --env_name <env_name> --log_dir <log_dir> --save_video <save_video>
```
- `<group_name>` logging name for the run.
- `<seed>` seed for RL policy.
- `<num_envs>` use 1 env in eval.
- `<env_name>` environment name.
- `<model_dir>` Path to checkpoint.
- `<save_video>` Boolean to save video or not.


We have released policies for task generated for microwave asset in `data/checkpoints`. Try visualizing using the following command:
```
python gym/train.py --group_name Microwave --seed 0 --env_num 1 --is_testing True --asset Microwave-7167-link_0-handle_0-joint_0-handlejoint_0 --model_dir data/checkpoints/06_30/06_30_open_door_Microwave-7167-link_0-handle_0-joint_0-handlejoint_0_0_algo-seed0/model_3000.tar --save_video True
```

In order to scale demonstrations, we suggest using a singularity image of IsaacGym.

## ToDos:
1. Support for USD to better structure environments.
2. Adding GPT-4V to automate part, affordance prediction.
3. Add support for more tasks in gym.

# Acknowledgement
We thank the following open-sourced projects:
- Our environments are from [IsaacGym](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)
- Our RL training code is based on [PartManip](https://github.com/PKU-EPIC/PartManip)


# Citation
If you find our work useful, please consider citing us!

```bibtex
@misc{katara2023gen2sim,
            title={Gen2Sim: Scaling up Robot Learning in Simulation with Generative Models}, 
            author={Pushkal Katara and Zhou Xian and Katerina Fragkiadaki},
            year={2023},
            eprint={2310.18308},
            archivePrefix={arXiv},
            primaryClass={cs.RO}
      }
```