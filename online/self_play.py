"""
Self-play rollouts in the online environment with one (possibly different) model per seat.

The game loop follows experiments.eval_three_agents, but
- every seat can be controlled by a different model (learner or frozen opponent),
- actions can be sampled from the policy (exploration) instead of taking the argmax,
- the model's prediction is intersected with the legal actions of the game engine, so sampling never fails,
- the rewards are recorded in the same convention as in the offline data (data_pipeline.get_states_actions_rewards),
  such that collected trajectories can be mixed with offline trajectories in the replay buffer.
"""

from typing import List, Optional

import numpy as np
import torch

from dt_skat_environment.environment import Env, ACT_DIM, MAX_EPISODE_LENGTH
from experiments import get_action


def compute_target_returns(env: Env, point_reward: bool, scale: float = 1.0) -> List[float]:
    """
    Initial return-to-go of each seat, as used in the self-play evaluation of the thesis.

    The maximum reward differs for defenders and declarer, so it is set to the double of the maximum possible.
    """
    target_return = [0.0] * 3

    for pl in range(3):
        if point_reward:
            if sum(env.trump_enc) == 0:
                game_points = env.game.game_variant.get_level()
                target_return[pl] = game_points * 2
            else:
                if sum(env.trump_enc) == 4:
                    base_value = 24
                else:
                    base_value = 9 + env.trump_enc.index(1)

                level = 1 + env.game.game_variant.get_level()
                target_return[pl] = level * base_value

            if env.game.players[pl] == env.game.get_declarer():
                # Seeger-Fabian score gives declarer points for winning.
                # combination of dense reward (card points ~ 120) and sparse (game points)
                # the discount is applied to prevent reward hacking
                # double it for possibly improved behaviour as in DT paper
                target_return[pl] += (120 * 0.1 + target_return[pl] * 0.9) * 2
            else:
                # Seeger-Fabian score gives defenders points for winning.
                # We only play as 3 players -> 40 points if won as defender
                # also, the discount is applied to prevent reward hacking
                target_return[pl] += (120 * 0.1 + 40 * 0.9) * 2
        else:
            # if the reward are only the card points, set it
            target_return[pl] = 2 * 120

        target_return[pl] = target_return[pl] * scale

    return target_return


def select_card(action_pred: torch.Tensor,
                legal: np.ndarray,
                greedy: bool,
                temperature: float = 1.0,
                rng: Optional[np.random.Generator] = None):
    """
    Selects a hand position from the predicted action probabilities, restricted to the legal actions.

    :return: (card index or -1 if no card can be played, whether the model put no probability on a legal card)
    """
    if not legal.any():
        return -1, False

    probs = action_pred.detach().cpu().numpy().astype(np.float64) * legal
    mismatch = not probs.sum() > 0
    if mismatch:
        # the model's rule mask disagrees with the game engine -> fall back to a uniform choice of legal cards
        probs = legal.astype(np.float64)

    if greedy:
        return int(np.argmax(probs)), mismatch

    if temperature != 1.0:
        probs = np.power(probs, 1.0 / temperature)
    probs = probs / probs.sum()

    rng = rng if rng is not None else np.random.default_rng()
    return int(rng.choice(ACT_DIM, p=probs)), mismatch


def _predict(model, states, actions, rewards, rtg, t):
    return get_action(model,
                      torch.from_numpy(np.asarray(states[:t + 1], dtype=np.float32)),
                      torch.from_numpy(np.asarray(actions[:t + 1], dtype=np.float32)),
                      torch.from_numpy(np.asarray(rewards[:t + 1], dtype=np.float32)),
                      torch.tensor(rtg[:t + 1], dtype=torch.float32),
                      torch.arange(t + 1).reshape(1, -1))


def play_game(env: Env,
              seat_models: list,
              first_states: np.ndarray,
              target_returns: List[float],
              greedy: bool = False,
              temperature: float = 1.0,
              rng: Optional[np.random.Generator] = None) -> dict:
    """
    Plays one game in an environment that was reset with env.online_reset.

    :param env: the environment after online_reset.
    :param seat_models: one TrainableDT per seat (the same model can sit on several seats).
    :param first_states: the states returned by online_reset.
    :param target_returns: the initial return-to-go of each seat.
    :param greedy: whether to take the most probable card (evaluation) or sample (exploration).
    :param temperature: sampling temperature if not greedy.
    :param rng: numpy random generator for sampling.
    :return: dict with the trajectory of each seat (states, actions, rewards in the convention of the offline data),
        the game points of each seat, the id of the declarer and the number of decisions where the rule mask of the
        model disagreed with the game engine.
    """
    state_dim = env.state_dim
    declarer_id = env.game.get_declarer().get_id()

    states = [[np.asarray(first_states[pl], dtype=np.float32)] for pl in range(3)]
    actions = [[] for _ in range(3)]
    rewards = [[] for _ in range(3)]
    rtg = [[float(target_returns[pl])] for pl in range(3)]

    mismatches, decisions = 0, 0
    put_card_points = 0

    for t in range(MAX_EPISODE_LENGTH):
        for pl in range(3):
            actions[pl].append(np.zeros(ACT_DIM, dtype=np.float32))
            rewards[pl].append(0.0)

        next_states = [None, None, None]

        if t < 2:
            # Skat putting: only the declarer acts, defenders wait
            for pl in range(3):
                if pl != declarer_id:
                    state = states[pl][-1].copy()
                    # allow to play a card after Skat putting
                    if t == 1:
                        state[3] = 1
                    next_states[pl] = state

            legal = env.legal_actions(declarer_id)
            card_index = -1
            if legal.any():
                action_pred = _predict(seat_models[declarer_id], states[declarer_id], actions[declarer_id],
                                       rewards[declarer_id], rtg[declarer_id], t)
                card_index, mismatch = select_card(action_pred, legal, greedy, temperature, rng)
                mismatches += mismatch
                decisions += 1

                # in the offline data, the declarer receives the card points of the put card
                put_value = env.game.players[declarer_id].cards[card_index].get_value()
                rewards[declarer_id][-1] = put_value
                put_card_points += put_value
                actions[declarer_id][-1][card_index] = 1

            state, _, _, _ = env.online_step(card_index, declarer_id)
            next_states[declarer_id] = np.asarray(state, dtype=np.float32)
        else:
            current_player = env.game.trick.get_current_player().id

            for _ in range(3):
                # update the state of the current player to see the open cards of this trick
                states[current_player][-1] = env.update_state(
                    torch.from_numpy(states[current_player][-1])).numpy()

                legal = env.legal_actions(current_player)
                action_pred = _predict(seat_models[current_player], states[current_player], actions[current_player],
                                       rewards[current_player], rtg[current_player], t)
                card_index, mismatch = select_card(action_pred, legal, greedy, temperature, rng)
                mismatches += mismatch
                decisions += 1

                actions[current_player][-1][card_index] = 1

                acting_player = current_player
                state, _, _, current_player = env.online_step(card_index, current_player)
                next_states[acting_player] = np.asarray(state, dtype=np.float32)

            # the party winning the trick receives its card points (see Game.finish_trick)
            for pl in range(3):
                rewards[pl][-1] = env.game.players[pl].current_trick_points

            # as in the offline data, the declarer additionally receives the points of the put Skat in the 1st trick
            if t == 2 and not env.hand:
                rewards[declarer_id][-1] += put_card_points

        for pl in range(3):
            rtg[pl].append(rtg[pl][-1] - rewards[pl][-1])
            states[pl].append(next_states[pl])

    # final rewards (Seeger-Fabian score) as in data_pipeline.get_states_actions_rewards
    game_points = [env.finish_online_game(pl)[2] for pl in range(3)]
    declarer_won = game_points[declarer_id] > 0

    if env.hand:
        skat_points = sum(card.get_value() for card in env.game.skat)
        for pl in range(3):
            rewards[pl][-1] += skat_points if pl == declarer_id else -skat_points

    for pl in range(3):
        if pl == declarer_id:
            rewards[pl][-1] = 0.9 * game_points[pl] + rewards[pl][-1] * 0.1
        elif not declarer_won:
            rewards[pl][-1] = 0.9 * 40 + rewards[pl][-1] * 0.1

    trajectories = [{
        "states": np.stack(states[pl][:MAX_EPISODE_LENGTH]).reshape(MAX_EPISODE_LENGTH, state_dim),
        "actions": np.stack(actions[pl]),
        "rewards": np.asarray(rewards[pl], dtype=np.float32),
    } for pl in range(3)]

    return {
        "trajectories": trajectories,
        "game_points": game_points,
        "declarer_id": declarer_id,
        "declarer_won": declarer_won,
        "mask_mismatches": mismatches,
        "decisions": decisions,
    }
