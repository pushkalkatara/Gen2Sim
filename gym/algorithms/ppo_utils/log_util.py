import torch

import statistics

def log(algo, locs, width=80, pad=35):
    algo.tot_timesteps += algo.num_transitions_per_env * algo.vec_env.num_envs
    algo.tot_time += locs['collection_time'] + locs['learn_time']
    iteration_time = locs['collection_time'] + locs['learn_time']

    ep_string = f''
    if_val = algo.env_num<algo.vec_env.num_envs
    if locs['ep_infos']:

        for key in locs['ep_infos'][0]:
            
            if key == "successes" or key == "success_entropy":
                continue
                infotensor_train = torch.zeros((algo.env_num), device= algo.device)
                infotensor_val = torch.zeros((algo.val_env_num), device=algo.device)
                for ep_info in locs['ep_infos']:
                    infotensor_train = torch.logical_or(infotensor_train, ep_info[key].to(algo.device)[:algo.env_num])
                    infotensor_val = torch.logical_or(infotensor_val, ep_info[key].to(algo.device)[algo.env_num:])
            elif key == "part_pos" or key == "grasp_success_rate" or 'grasp_success_rate' in key:
                infotensor_train = torch.zeros((algo.env_num), device= algo.device)
                for ep_info in locs['ep_infos']:
                    try:
                        infotensor_train = torch.max(infotensor_train, ep_info[key].to(algo.device)[:algo.env_num])
                    except:
                        pass
            else: 
                infotensor_train = torch.tensor([], device=algo.device)
                infotensor_valIntra = torch.tensor([], device=algo.device)
                infotensor_valInter = torch.tensor([], device=algo.device)
                for ep_info in locs['ep_infos']:

                    infotensor_train = torch.cat((infotensor_train, ep_info[key].to(algo.device)[:algo.env_num].float()))
            if key=="success_rate":
                sorted_success_rate_train, _ = torch.sort(infotensor_train)
                worst_rate = 0.5
                num_worst_train = int(infotensor_train.shape[0]*worst_rate)
                worst_success_rate_train = sorted_success_rate_train[:num_worst_train]
                worst_mean_train = worst_success_rate_train.mean()

                algo.writer.add_scalar(f"""Episode/worst_{worst_rate*100}%_success_rate_train""", worst_mean_train, locs['it'])
                if algo.wandb_writer is not None : algo.wandb_writer.log({f"""Episode/worst_{worst_rate*100}%_success_rate_train""": worst_mean_train})
                ep_string += f"""{f'Mean episode worst {worst_rate*100}% success rate train:':>{pad}} {worst_mean_train:.4f} \n"""

            value_train = torch.mean(infotensor_train.float())

            if key == "dist_tip" or key == "part_pos" :
                algo.writer.add_scalar('Train/' + key + '_train', value_train, locs['it'])
                if algo.wandb_writer is not None : algo.wandb_writer.log({'Train/' + key + '_train': value_train})
                ep_string += f"""{f'{key} train:':>{pad}} {value_train:.4f}\n"""
            else :
                algo.writer.add_scalar('Episode/' + key + '_train', value_train, locs['it'])
                if algo.wandb_writer is not None : algo.wandb_writer.log({'Episode/' + key + '_train': value_train})
                ep_string += f"""{f'Mean episode {key} train:':>{pad}} {value_train:.4f}\n"""
                        

    if locs['task_info']:
        for key in locs['task_info']:
            value = locs['task_info'][key]
            algo.writer.add_scalar('Episode/' + key, value, locs['it'])
            ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f} \n"""

    mean_std = algo.actor_critic.log_std.exp().mean()
    if algo.vec_env.task.env.cfg["obs"]["state"]["use_cabinet_qpose"]:
        mean_part_position_train, _ = torch.max(algo.storage.states[:, :algo.env_num, algo.order_part_pos], dim=0)
        mean_part_position_train = mean_part_position_train.mean()
    else:
        mean_part_position_train = 0
    
    ###wandb###
    if algo.wandb_writer is not None :
        algo.wandb_writer.log({
                "total_steps": locs['it'], 
                'Loss/value_function': locs['mean_value_loss'],
                'Loss/surrogate': locs['mean_surrogate_loss'],
                'Policy/mean_noise_std': mean_std.item(),
                'Policy/lr': algo.step_size,
                'Train/mean_reward/step': locs['mean_reward'],
                #'Train_/mean_episode_length/episode': locs['mean_trajectory_length'],
                'Train/part_pos_train' : mean_part_position_train,
                    })
    
    algo.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
    algo.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
    algo.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
    algo.writer.add_scalar('Policy/lr', algo.step_size, locs['it'])
    if len(locs['rewbuffer']) > 0:
        algo.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
        algo.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), algo.tot_time)

    algo.writer.add_scalar('Train/mean_reward/step', locs['mean_reward'], locs['it'])

    fps = int(algo.num_transitions_per_env * algo.vec_env.num_envs / (locs['collection_time'] + locs['learn_time']))

    str = f" \033[1m Learning iteration {locs['it']}/{locs['num_learning_iterations']} \033[0m "

    if len(locs['rewbuffer']) > 0:
        log_string = (f"""{'#' * width}\n"""
                        f"""{str.center(width, ' ')}\n\n"""
                        f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                        f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                        f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                        f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                        f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                        #f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                        f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n"""
                        f"""{'Learning Rate:':>{pad}} {algo.step_size}\n"""
                        f"""{'Mean_part_position_train:':>{pad}} {mean_part_position_train:.5f}\n"""
                        )
    else:
        log_string = (f"""{'#' * width}\n"""
                        f"""{str.center(width, ' ')}\n\n"""
                        f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                        'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                        f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                        f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                        f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                        f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n"""
                        f"""{'Learning Rate:':>{pad}} {algo.step_size}\n"""
                        f"""{'Mean_part_position_train:':>{pad}} {mean_part_position_train:.5f}\n"""
                        )

    log_string += ep_string
    log_string += (f"""{'-' * width}\n"""
                    f"""{'Total timesteps:':>{pad}} {algo.tot_timesteps}\n"""
                    f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                    f"""{'Total time:':>{pad}} {algo.tot_time:.2f}s\n"""
                    f"""{'ETA:':>{pad}} {algo.tot_time / (locs['it'] + 1) * (
                            locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
    print(log_string)

def log_test(algo, locs, width=80, pad=35) :
    ##debug

    algo.tot_timesteps += algo.num_transitions_per_env * algo.vec_env.num_envs
    algo.tot_time += locs['collection_time'] + locs['learn_time']
    iteration_time = locs['collection_time'] + locs['learn_time']

    ep_string = f''
    if locs['ep_infos']:
        for key in locs['ep_infos'][0]:
            infotensor = torch.tensor([], device=algo.device)
            for ep_info in locs['ep_infos']:
                infotensor = torch.cat((infotensor, ep_info[key].to(algo.device)))
            value = torch.mean(infotensor)
            algo.writer.add_scalar('Episode/' + key, value, locs['it'])
            ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
    mean_std = algo.actor_critic.log_std.exp().mean()

    fps = int(algo.num_transitions_per_env * algo.vec_env.num_envs / (locs['collection_time'] + locs['learn_time']))

    str = f" \033[1m Learning iteration {locs['it']}/{locs['num_learning_iterations']} \033[0m "

    # if locs['task_info']:
    #     for key in locs['task_info']:
    #         value = locs['task_info'][key]
    #         algo.writer.add_scalar('Episode/' + key, value)
    #         ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f} \n"""
    if len(locs['rewbuffer']) > 0:
        log_string = (f"""{'#' * width}\n"""
                        f"""{str.center(width, ' ')}\n\n"""
                        f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                        f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                        f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                        f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
    else:
        log_string = (f"""{'#' * width}\n"""
                        f"""{str.center(width, ' ')}\n\n"""
                        f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                        'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                        f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

    log_string += ep_string
    log_string += (f"""{'-' * width}\n"""
                    f"""{'Total timesteps:':>{pad}} {algo.tot_timesteps}\n"""
                    f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                    f"""{'Total time:':>{pad}} {algo.tot_time:.2f}s\n"""
                    f"""{'ETA:':>{pad}} {algo.tot_time / (locs['it'] + 1) * (
                            locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
    print(log_string)
