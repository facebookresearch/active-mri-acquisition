import options.rl_options

from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr import algo, utils
from rl_env import ReconstructionEnv, generate_initial_mask
from tensorboardX import SummaryWriter

import gym
import os
import torch
from gym.envs.registration import register


def make_env(opts):
    """
    Registers environment in openAI gym
    Args:
        opts: options required to instantiate the environment

    Returns: Wrapped environment

    """
    env_id = 'ReconstructionEnv-v0'
    register(id=env_id, entry_point='rl_env:ReconstructionEnv',
             kwargs={'initial_mask': generate_initial_mask(num_lines=opts.initial_num_lines),
                     'options': opts})
    env = gym.make(env_id)
    return env


def populate_experience_replay(rl_opts, env, actor_critic, rollouts):
    """
    Populates the experience replay buffer
    Args:
        rl_opts: RL Environment and PPO specific options
        env: gym environment
        actor_critic: object of the Policy class
        rollouts: object of the experience replay buffer (storage class)

    Returns: cumulative reward till budget reaches

    """
    episode_reward = 0
    for step in range(rl_opts.budget):
        with torch.no_grad():
            value, action, action_log_prob = actor_critic.act(
                rollouts.obs[step])

        # Observe reward and next obs
        obs, reward, done, _ = env.step(action)

        # If done then clean the history of observations.
        masks = torch.FloatTensor([0.0] if done else [1.0])

        rollouts.insert(obs, action,
                        action_log_prob, value, reward, masks)

        episode_reward += reward

    return episode_reward


def train(rl_opts):
    """
    PPO trainer class
    Args:
        rl_opts: RL environment and PPO specific options

    Returns:

    """
    env = make_env(rl_opts)
    writer = SummaryWriter(rl_opts.checkpoints_dir)

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

    total_reward = 0

    #     rl_opts.num_env_steps) // rl_opts.num_steps // rl_opts.num_processes
    num_updates = rl_opts.num_steps
    for update_step in range(num_updates):
        obs, _ = env.reset()
        if rl_opts.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, update_step, num_updates,
                rl_opts.lr_ppo_actor_critic)

        episode_reward = populate_experience_replay(rl_opts, env, actor_critic, rollouts)
        total_reward += episode_reward

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1]).detach()

        rollouts.compute_returns(next_value, rl_opts.use_gae, rl_opts.gamma,
                                 rl_opts.gae_lambda, rl_opts.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # Tensorboard Plots
        # print(value_loss, update_step)
        writer.add_scalar('cumulative_reward', total_reward, update_step)
        writer.add_scalar('value_loss', value_loss, update_step)
        writer.add_scalar('action_loss', action_loss, update_step)

        # save for every interval-th episode or for the last epoch
        if (update_step % rl_opts.save_interval == 0
            or update_step == num_updates - 1) and rl_opts.checkpoints_dir != "":

            torch.save([
                actor_critic
            ], os.path.join(rl_opts.checkpoints_dir + "ppo.pt"))

        if update_step % rl_opts.log_interval == 0:
            print(
                "Update step: {}/{}, total reward; {}, distribution entropy: {} \n Value Loss: {:.1f} Action Loss: {:.1f}\n"
                    .format(update_step, num_updates,
                            total_reward, dist_entropy, value_loss,
                            action_loss))


if __name__ == '__main__':
    # Reading options
    opts = options.rl_options.RLOptions().parse()

    opts.batchSize = 1
    opts.mask_type = 'grid'

    opts.checkpoints_dir = '/checkpoint/sumanab/ppo'
    opts.reconstructor_dir = '/checkpoint/sumanab/active_acq'
    opts.evaluator_dir = '/checkpoint/sumanab/active_acq/evaluator'

    opts.number_of_evaluator_filters = 128
    opts.device = torch.device('cuda:{}'.format(opts.gpu_ids[0])) if opts.gpu_ids else torch.device('cpu')

    if not os.path.isdir(opts.checkpoints_dir):
        os.makedirs(opts.checkpoints_dir)

    train(opts)