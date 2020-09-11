from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import activemri.envs as envs
import activemri.baselines as baselines


def evaluate(
    env: envs.envs.ActiveMRIEnv,
    policy: baselines.Policy,
    num_episodes: int,
    seed: int,
    split: str,
    verbose: Optional[bool] = False,
) -> Tuple[Dict[str, np.ndarray], List[Tuple[Any, Any]]]:
    env.seed(seed)
    if split == "test":
        env.set_test()
    elif split == "val":
        env.set_val()
    else:
        raise ValueError(f"Invalid evaluation split: {split}.")

    score_keys = env.score_keys()
    all_scores = dict(
        (k, np.zeros((num_episodes * env.batch_size, env.budget + 1)))
        for k in score_keys
    )
    all_img_ids = []
    trajectories_written = 0
    for episode in range(num_episodes):
        step = 0
        obs, meta = env.reset()
        if not obs:
            break  # no more images
        # in case the last batch is smaller
        actual_batch_size = len(obs["reconstruction"])
        if verbose:
            msg = ", ".join(
                [
                    f"({meta['fname'][i]}, {meta['slice_id'][i]})"
                    for i in range(actual_batch_size)
                ]
            )
            print(f"Read images: {msg}")
        for i in range(actual_batch_size):
            all_img_ids.append((meta["fname"][i], meta["slice_id"][i]))
        batch_idx = slice(
            trajectories_written, trajectories_written + actual_batch_size
        )
        for k in score_keys:
            all_scores[k][batch_idx, step] = meta["current_score"][k]
        trajectories_written += actual_batch_size
        all_done = False
        while not all_done:
            step += 1
            action = policy.get_action(obs)
            obs, reward, done, meta = env.step(action)
            for k in score_keys:
                all_scores[k][batch_idx, step] = meta["current_score"][k]
            all_done = all(done)

    for k in score_keys:
        all_scores[k] = all_scores[k][: len(all_img_ids), :]
    return all_scores, all_img_ids
