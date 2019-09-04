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


def make_env(*opts):
    env_id = 'ReconstructionEnv-v0'
    register(id=env_id, entry_point='rl_env:ReconstructionEnv',
             kwargs={'initial_mask': generate_initial_mask(num_lines=opts[1].low_freq_count),
                     'options': opts[0]})
    env = gym.make(env_id)
    return env


def main(rl_opts, train_opts):
    env = make_env(rl_opts, train_opts)
    print(env.observation_space)

    # envs = env

    actor_critic = Policy(
        env.observation_space.shape,
        env.action_space)
    actor_critic.to(train_opts.device)

    agent = algo.PPO(
        actor_critic,
        train_opts.clip_param,
        train_opts.ppo_epoch,
        train_opts.num_mini_batch,
        train_opts.value_loss_coef,
        train_opts.entropy_coef,
        lr=train_opts.lr,
        eps=train_opts.eps,
        max_grad_norm=train_opts.max_grad_norm)

    rollouts = RolloutStorage(train_opts.num_steps, train_opts.num_processes,
                              env.observation_space.shape, env.action_space)

    obs, _ = env.reset()
    rollouts.obs[0].copy_(torch.from_numpy(obs))
    rollouts.to(train_opts.device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        train_opts.num_env_steps) // train_opts.num_steps // train_opts.num_processes
    for j in range(num_updates):

        if train_opts.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                train_opts.lr)

        for step in range(train_opts.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob = actor_critic.act(
                    rollouts.obs[step])

            # Obser reward and next obs
            obs, reward, done, infos = env.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, train_opts.use_gae, train_opts.gamma,
                                 train_opts.gae_lambda, train_opts.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
            or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(env), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * train_opts.num_processes * train_opts.num_steps
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
    rl_opts = options.rl_options.RLOptions().parse()
    if rl_opts.results_dir is None:
        rl_opts.results_dir = rl_opts.checkpoints_dir

    train_opts = TrainOptions().parse()
    train_opts.device = torch.device('cuda:{}'.format(
        train_opts.gpu_ids[0])) if train_opts.gpu_ids else torch.device('cpu')

    main(rl_opts, train_opts)