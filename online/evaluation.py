import random
from typing import Dict

import numpy as np

from dt_skat_environment.environment import Env
from online.self_play import play_game, compute_target_returns


def _play_deal(seed, seat_models_fn, point_rewards, rtg_scale, card_enc, reward_convention, declaration):
    # the same seed gives the same deal and declarer (Kinback scheme and greedy bidding are deterministic)
    random.seed(seed)
    env = Env(card_enc)
    first_states = env.online_reset(point_rewards=point_rewards, declaration=declaration)
    seat_models = seat_models_fn(env.game.get_declarer().get_id())
    return play_game(env, seat_models, first_states,
                     compute_target_returns(env, point_rewards, rtg_scale, seat_models), greedy=True,
                     reward_convention=reward_convention)


def duplicate_evaluation(learner, baseline, n_games: int, seed: int = 1_000_000_007, point_rewards: bool = True,
                         rtg_scale: float = 0.7, card_enc: str = "one-hot",
                         reward_convention: str = "default", declaration=None) -> Dict[str, float]:
    """
    Compares the learner to the baseline (e.g. the pre-trained DT) on the same deals (duplicate evaluation).

    Every deal is played three times with greedy actions:
    - baseline on all seats (reference),
    - learner as declarer against two baseline defenders,
    - baseline declarer against two learner defenders.
    Playing the same cards in every configuration removes most of the luck of the deal from the comparison.

    :param declaration: optional bidding policy (see trainer/declaration.py), the same for all configurations.
    :return: average Seeger-Fabian points of the declarer and win rates of declarer/defenders per configuration,
        and the differences of the learner to the baseline.
    """
    declarer_points = {"baseline": [], "learner_declarer": [], "learner_defenders": []}
    declarer_won = {"baseline": [], "learner_declarer": [], "learner_defenders": []}

    configurations = {
        "baseline": lambda declarer: [baseline] * 3,
        "learner_declarer": lambda declarer: [learner if pl == declarer else baseline for pl in range(3)],
        "learner_defenders": lambda declarer: [baseline if pl == declarer else learner for pl in range(3)],
    }

    for i in range(n_games):
        for name, seat_models_fn in configurations.items():
            result = _play_deal(seed + i, seat_models_fn, point_rewards, rtg_scale, card_enc, reward_convention,
                                declaration)
            declarer_points[name].append(result["game_points"][result["declarer_id"]])
            declarer_won[name].append(result["declarer_won"])

    metrics = {}
    for name in configurations:
        metrics[f"{name}/declarer_points"] = float(np.mean(declarer_points[name]))
        metrics[f"{name}/declarer_win_rate"] = float(np.mean(declarer_won[name]))

    # positive values: the learner is better than the baseline
    metrics["delta/declarer_points"] = metrics["learner_declarer/declarer_points"] - \
        metrics["baseline/declarer_points"]
    metrics["delta/declarer_win_rate"] = metrics["learner_declarer/declarer_win_rate"] - \
        metrics["baseline/declarer_win_rate"]
    metrics["delta/defender_win_rate"] = metrics["baseline/declarer_win_rate"] - \
        metrics["learner_defenders/declarer_win_rate"]
    # Seeger-Fabian points the learner gains per deal: as declarer + as defenders (40 per defender for a win)
    metrics["delta/points_per_deal"] = metrics["delta/declarer_points"] + 2 * 40 * metrics["delta/defender_win_rate"]

    return metrics
