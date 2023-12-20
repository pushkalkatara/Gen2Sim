import json
import os
from tqdm import tqdm
import sys

import parser
import llm_utils
from template_asset import example, question

PARTNET_TO_ISAAC_GYM_CONVENTION = {
    'hinge': 'revolute',
    'static': 'fixed',
    'slider': 'prismatic'
}


def prepare_prompt(assets_path, name):
    # This function can be off-loaded to GPT-4V later to automatically parse parts and their affordances.
    # For now, we will manually parse the semantics for each asset.

    assets_path = os.path.join(assets_path, name)
    cat, id, part_link, affordance_link, part_joint, affordance_joint = name.split('-')

    print(f"Asset Category: {cat}")
    print(f"Asset Part Link: {part_link}")
    print(f"Asset Affordance Link: {affordance_link}")
    print(f"Asset Part Joint: {part_joint}")
    print(f"Asset Affordance Joint: {affordance_joint}")

    with open(os.path.join(assets_path, 'semantics_relabel.txt')) as fid:
        semantics = [tuple(sem.split()) for sem in fid.read().splitlines()]

    target_part = next((sem[2] for sem in semantics if sem[0] == part_link), None) # door, drawer, etc.

    part_joint_type = next((sem[1] for sem in semantics if sem[0] == part_link), None)
    part_joint_type = PARTNET_TO_ISAAC_GYM_CONVENTION[part_joint_type] # revolute, prismatic, etc.

    target_affordance = next((sem[2] for sem in semantics if sem[0] == affordance_link), None) # handle, knob, etc.
    affordance_joint_type = 'fixed' # later add support for other affordance types based on dataset!

    prompt = example + question.format(
        cat,
        target_part,
        part_link,
        part_joint,
        part_joint_type,
        target_affordance,
        affordance_link,
        affordance_joint,
        affordance_joint_type
    )

    return prompt, target_part, target_affordance


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python prompt.py <assets_path> <log_dir> <task_export_class_path>, Using default paths")
        assets_path = '/projects/katefgroup/learning-simulations/gym_scale/assets/release_assets'
        log_dir = '/projects/katefgroup/learning-simulations/log_dir/task-gen-log/'
        task_export_class_path = '/home/pkatara/xian/learning-simulations/gym/envs/gpt_task_generated.py'
    else:
        assets_path = sys.argv[1]
        log_dir = sys.argv[2]
        task_export_class_path = sys.argv[3]

    asset_to_task = {}
    for name in tqdm(sorted(os.listdir(assets_path))):
        print("Processing asset:", name)
        
        # Later offload part and affordance prediction for each task to GPT-4V!
        # GPT-4V should reason about which part and affordance to use for each task
        # based on the task description.

        prompt, target_part, target_affordance = prepare_prompt(assets_path, name)
        response = llm_utils.query(prompt, log_dir)

        if response is not None:
            print("GPT Response Successful, Exporting Tasks and Reward Functions!")
            tasks = parser.parse_tasks(response, task_export_class_path)
            asset_to_task[name] = tasks
        else:
            print("Asset Failed")
    
    with open(os.path.join(log_dir, 'asset_to_tasks.json'), 'w') as fp:
        json.dump(asset_to_task, fp)