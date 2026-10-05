import random

import numpy as np
import pytest

from dt_skat_environment.environment import Env
from online.agents import RandomAgent, RuleAgent
from online.self_play import play_game, compute_target_returns
from online_training import OpponentPool, ReturnFilter


@pytest.mark.parametrize("game", [None, 24, 23])
def test_rule_agent_plays_legal_games(game):
    for seed in range(10):
        random.seed(seed)
        env = Env("one-hot")
        declaration = None if game is None else \
            (lambda e: {"declarer": seed % 3, "game": game, "hand": False, "bid": 18})
        states = env.online_reset(declaration=declaration)
        result = play_game(env, [RuleAgent()] * 3, states, compute_target_returns(env, True))
        assert result["mask_mismatches"] == 0
        skat_points = sum(card.get_value() for card in env.game.skat)
        assert sum(p.sum_trick_values() for p in env.game.players) + skat_points == 120


def test_rule_agent_beats_random_players():
    wins, games = 0, 40
    for seed in range(games):
        random.seed(seed)
        env = Env("one-hot")
        states = env.online_reset()
        declarer = env.game.get_declarer().get_id()
        agents = [RandomAgent(np.random.default_rng(seed + pl)) for pl in range(3)]
        agents[declarer] = RuleAgent()
        wins += play_game(env, agents, states, compute_target_returns(env, True))["declarer_won"]
    assert wins / games > 0.6


def test_pfsp_prefers_strong_opponents():
    pool = OpponentPool(["weak", "strong"], sampling="pfsp")
    for _ in range(100):
        pool.update(0, learner_won=True)
        pool.update(1, learner_won=False)
    rng = np.random.default_rng(0)
    samples = [pool.sample(rng) for _ in range(1000)]
    assert np.mean(samples) > 0.9

    pool.add_snapshot("snapshot")
    assert len(pool) == 3 and pool.win_rates[-1] == 0.5


def test_return_filter_keeps_good_trajectories():
    return_filter = ReturnFilter(quantile=0.5)
    for value in range(100):
        return_filter.keep({"rewards": np.array([value])}, "declarer")
    assert return_filter.keep({"rewards": np.array([90])}, "declarer")
    assert not return_filter.keep({"rewards": np.array([10])}, "declarer")
