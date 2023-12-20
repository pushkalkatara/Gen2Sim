example = '''
Q: The environment contains the following assets:

1. asset_name: "microwave"
   part_config:
        part_1: "door"
            - link_name: "link_0"
            - joint_name: "joint_0"
            - joint_type: "revolute"
        part_2: "handle"
            - link_name: "handle_0"
            - joint_name: "handlejoint_0"
            - joint_type: "fixed"

1. asset_name: "cup"
   part_config:
        part_1: "body"
            - link_name: "base"
            - joint_name: "base_joint"
            - joint_type: "fixed"

Available API from simulator are:
1. get_pose_by_link_name(asset_name, link_name) # returns the x, y, z position of the link
2. get_robot_gripper_pose(asset_name, link_name) # returns the x, y, z position of the gripper
3. get_state_by_joint_name(asset_name, joint_name) # returns the state of the link
4. get_limits_by_joint_name(asset_name, joint_name) # returns the limit of the joint

Remember:
1. Only use the available APIs from simulator.
2. Generate the reward function code snippet in Python.

List meaningful manipulation tasks that can be performed in this environment. Give sub-task decomposition and the order
of execution to solve the task. Also, provide reward function for each subtask.

A: The following tasks can be performed in this environment:
1. OpenMicrowaveDoor
2. CloseMicrowaveDoor
3. PickCup
4. PlaceCup
5. PutCupInMicrowave
    Sub-tasks:
    5.1 OpenMicrowaveDoor
    5.2 PickCup
    5.3 Place the cup in the microwave
    5.4 Close the microwave door

The reward function for each task is as follows:

Task: OpenMicrowaveDoor
Task Description: open the door of the microwave
```
def compute_reward(env):
    # reward function
    door_handle_pose = env.get_pose_by_link_name("microwave", "handle_0")
    gripper_pose = env.get_robot_gripper_pose()
    distance_gripper_to_handle = torch.norm(door_handle_pose - door_handle_pose, dim=-1)

    door_state = env.get_state_by_joint_name("microwave", "joint_0")

    cost = distance_gripper_to_handle - door_state
    reward = - cost

    # success condition
    target_door_state = env.get_limits_by_joint_name("microwave", "joint_0")["upper"]
    success = torch.abs(door_state - target_door_state) < 0.1

    return reward, success
```

Task: CloseMicrowaveDoor
Task Description: close the door of the microwave
```
def compute_reward(env):
    # reward function
    door_handle_pose = env.get_pose_by_link_name("microwave", "handle_0")
    gripper_pose = env.get_robot_gripper_position()
    distance_gripper_to_handle = torch.norm(gripper_pose - door_handle_pose, dim=-1)

    door_state = self.env.get_state_by_joint_name("microwave", "joint_0")

    cost = distance_gripper_to_handle + door_state
    reward = - cost

    # success condition
    target_door_state = env.get_limits_by_joint_name("microwave", "joint_0")["lower"]
    success = torch.abs(door_state - target_door_state) < 0.1
    return reward, success
```

Task: PickCup
```
def compute_reward(env):
    # reward function
    cup_pose = env.get_pose_by_link_name("cup", "base")
    gripper_pose = env.get_robot_gripper_position()
    distance_gripper_to_asset = torch.norm(gripper_pose - asset_pose, dim=-1)
    
    cost = distance_gripper_to_asset - door_state
    reward = - cost

    # success condition
    success = distance_gripper_to_asset < 0.1
    return reward, success
```

Task: PlaceCupInMicrowave
```
def compute_reward(self):
    # reward function
    asset_pose = self.env.get_position_by_asset_link_name("cup", "cup")
    gripper_pose = self.env.get_robot_gripper_position()
    distance_gripper_to_asset = torch.norm(gripper_pose - asset_pose, dim=-1)
    center = self.env.get_center_by_asset_link_name("microwave", "link_0")
    distance_goal_to_asset = torch.norm(asset_pose - center, dim=-1)
    door_state = self.env.get_state_by_asset_joint_name("microwave", "joint_0")
    cost = distance_goal_to_asset + door_state
    reward = - cost

    # success condition
    success = torch.logical_and(distance_goal_to_asset < 0.1)
    return reward, success
```

'''

question = '''
Q: The simulation environment has the following assets:

1. {} is an articulated asset with following parts:
    Part 1: door
        - link_name: {}
        - joint_name: {} 
        - joint_type: {}
    Part 2: handle
        - link_name: {}
        - joint_name: {}
        - joint_type: {}
'''