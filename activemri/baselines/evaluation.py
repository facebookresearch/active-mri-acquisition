from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import activemri.envs as envs
import activemri.baselines as baselines


def evaluate(
    policy: baselines.Policy,
    budget: int,
    batch_size: int,
    num_episodes: int,
    seed: int,
    verbose: Optional[bool] = False,
) -> Tuple[Dict[str, np.ndarray], List[Tuple[Any, Any]]]:
    env = envs.SingleCoilKneeRAWEnv(budget=budget, batch_size=batch_size)
    env.seed(seed)

    score_keys = env.score_keys()
    all_scores = dict(
        (k, np.zeros((num_episodes * batch_size, budget + 1))) for k in score_keys
    )
    all_img_ids = []
    trajectories_written = 0
    for episode in range(num_episodes):
        step = 0
        obs, meta = env.reset()
        # in case the last batch is smaller
        actual_batch_size = len(obs["reconstruction"])
        if verbose:
            msg = ", ".join(
                [
                    f"({meta['fname'][i].name}, {meta['slice_id'][i]})"
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
    return all_scores, all_img_ids
