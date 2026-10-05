import random

import numpy as np
import pytest

from dt_skat_environment.environment import Env
from online.agents import RandomAgent
from online.self_play import play_game, compute_target_returns
from trainer.data_collator import DecisionTransformerSkatDataCollator
from trainer.return_model import ReturnModel, train_return_model, rtg_targets, discounted_returns


def random_trajectories(n_games=6):
    trajectories = []
    for seed in range(n_games):
        random.seed(seed)
        env = Env("one-hot")
        states = env.online_reset()
        agents = [RandomAgent(np.random.default_rng(seed + pl)) for pl in range(3)]
        trajectories += play_game(env, agents, states, compute_target_returns(env, True))["trajectories"]
    return trajectories


@pytest.mark.parametrize("mode", ["expected", "deal_margin"])
def test_return_model_relabelling(mode, tmp_path):
    trajectories = random_trajectories()
    model = train_return_model(trajectories, mode, epochs=2, batch_size=16, log=lambda _: None)

    model.save(str(tmp_path))
    loaded = ReturnModel.load(str(tmp_path))
    trajectory = trajectories[0]
    np.testing.assert_allclose(loaded.relabel(trajectory), model.relabel(trajectory), rtol=1e-6)

    rtg = model.relabel(trajectory)
    assert rtg.shape == (12,) and np.isfinite(rtg).all()
    if mode == "deal_margin":
        # the achieved returns-to-go shifted by the value of the deal
        shift = discounted_returns(trajectory["rewards"]) - rtg
        np.testing.assert_allclose(shift, shift[0], rtol=1e-5)

    targets = rtg_targets(trajectories, model.relabel)
    assert set(targets) == {"declarer", "defender"}

    batch = DecisionTransformerSkatDataCollator(trajectories, batch_size=8, rtg_relabeler=model.relabel)(trajectories)
    assert batch["returns_to_go"].shape == (8, 12, 1)
