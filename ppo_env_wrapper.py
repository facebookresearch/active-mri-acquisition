import options.rl_options

from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr import algo, utils
from options.train_options import TrainOptions
from rl_env import ReconstructionEnv, generate_initial_mask

import gym
import numpy as np
import os
import time
import torch
from collections import deque
from gym.envs.registration import register


def make_env(opts):
    env_id = 'ReconstructionEnv-v0'
    register(id=env_id, entry_point='rl_env:ReconstructionEnv',
             kwargs={'initial_mask': generate_initial_mask(num_lines=opts.initial_num_lines),
                     'options': opts})
    env = gym.make(env_id)
    return env


def main(rl_opts):
    env = make_env(rl_opts)
    print(env.observation_space)

    # envs = env

    actor_critic = Policy(
        env.observation_space.shape,
        env.action_space)
    actor_critic.to(rl_opts.device)

    agent = algo.PPO(
        actor_critic,
        rl_opts.clip_param,
        rl_opts.ppo_epoch,
        rl_opts.num_mini_batch,
        rl_opts.value_loss_coef,
        rl_opts.entropy_coef,
        lr=rl_opts.lr_ppo_actor_critic,
        eps=rl_opts.eps,
        max_grad_norm=rl_opts.max_grad_norm)

    rollouts = RolloutStorage(rl_opts.budget, rl_opts.num_processes,
                              env.observation_space.shape, env.action_space)

    obs, _ = env.reset()
    rollouts.obs[0].copy_(torch.from_numpy(obs))
    rollouts.to(rl_opts.device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        rl_opts.num_env_steps) // rl_opts.num_steps // rl_opts.num_processes
    for j in range(num_updates):

        if rl_opts.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                rl_opts.lr_ppo_actor_critic)

        for step in range(rl_opts.budget):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob = actor_critic.act(
                    rollouts.obs[step])

            # Observe reward and next obs
            obs, reward, done, infos = env.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([0.0] if done else [1.0])

            rollouts.insert(obs, action,
                            action_log_prob, value, reward, masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1]).detach()

        rollouts.compute_returns(next_value, rl_opts.use_gae, rl_opts.gamma,
                                 rl_opts.gae_lambda, rl_opts.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % rl_opts.save_interval == 0
            or j == num_updates - 1) and rl_opts.checkpoints_dir != "":

            torch.save([
                actor_critic
            ], os.path.join(rl_opts.checkpoints_dir + "ppo.pt"))

        if j % rl_opts.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * rl_opts.num_processes * rl_opts.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                    .format(j, total_num_steps,
                            int(total_num_steps / (end - start)),
                            len(episode_rewards), np.mean(episode_rewards),
                            np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards), dist_entropy, value_loss,
                            action_loss))


if __name__ == '__main__':
    # Reading options
    opts = options.rl_options.RLOptions().parse()

    opts.batchSize = 1
    opts.mask_type = 'grid'  # This is ignored, only here for compatibility with loader

    opts.checkpoints_dir = '/checkpoint/sumanab/active_acq'
    opts.reconstructor_dir = '/checkpoint/sumanab/active_acq'
    opts.evaluator_dir = '/checkpoint/sumanab/active_acq/evaluator'

    opts.number_of_evaluator_filters = 128
    opts.device = torch.device('cuda:{}'.format(opts.gpu_ids[0])) if opts.gpu_ids else torch.device('cpu')

    if not os.path.isdir(opts.checkpoints_dir):
        os.makedirs(opts.checkpoints_dir)

    main(opts)



    # rl_opts = options.rl_options.RLOptions().parse()
    # # if rl_opts.results_dir is None:
    # #     rl_opts.results_dir = rl_opts.checkpoints_dir
    #
    # train_opts = TrainOptions().parse()
    # train_opts.device = torch.device('cuda:{}'.format(
    #     train_opts.gpu_ids[0])) if train_opts.gpu_ids else torch.device('cpu')
    #
    # main(rl_opts, train_opts)