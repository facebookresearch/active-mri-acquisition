from rl_env import ReconstructionEnv, generate_initial_mask
import options.rl_options


def main(opts):
    mask = generate_initial_mask(num_lines=opts.initial_num_lines)
    # print(mask.shape)
    env = ReconstructionEnv(mask, opts)
    print(env.action_space)
    print(env.observation_space.shape)
    # obs, info = env.reset()
    # print(obs.shape)


if __name__ == '__main__':
    opts = options.rl_options.RLOptions().parse()
    if opts.results_dir is None:
        opts.results_dir = opts.checkpoints_dir

    main(opts)
