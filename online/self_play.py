"""
Games in the online environment with one (possibly different) agent per seat.

- every seat can be controlled by a different agent (learner, frozen opponent, random or human player),
- actions can be sampled from the policy (exploration) instead of taking the argmax,
- the agent's prediction is intersected with the legal actions of the game engine, so sampling never fails,
- the rewards are recorded in the same convention as in the offline data (data_pipeline.get_states_actions_rewards),
  such that collected trajectories can be mixed with offline trajectories and the returns-to-go during play match
  the ones seen in training.
"""

from collections import defaultdict
from typing import List, Optional

import numpy as np
import torch

from dt_skat_environment.environment import Env, ACT_DIM, MAX_EPISODE_LENGTH, hidden_card_labels
from online.agents import as_agent, DTAgent
from trainer.inference import get_action_batch

REWARD_CONVENTIONS = ("default", "thesis")


def compute_target_returns(env: Env, point_reward: bool, scale: float = 1.0, seat_agents=None) -> List[float]:
    """
    Initial return-to-go of each seat, as used in the self-play evaluation of the thesis.

    The maximum reward differs for defenders and declarer, so it is set to the double of the maximum possible.
    Models trained on relabelled returns-to-go (config.rtg_mode) use the targets stored in their config instead.
    """
    target_return = [0.0] * 3

    for pl in range(3):
        config = getattr(getattr(as_agent(seat_agents[pl]), "model", None), "config", None) \
            if seat_agents is not None else None
        if config is not None and getattr(config, "rtg_mode", "raw") != "raw":
            role = "declarer" if env.game.players[pl] == env.game.get_declarer() else "defender"
            target_return[pl] = config.rtg_targets[role]
            continue

        if point_reward:
            target_return[pl] = env.get_game_value()

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


def select_card(action_probs,
                legal: np.ndarray,
                greedy: bool,
                temperature: float = 1.0,
                rng: Optional[np.random.Generator] = None):
    """
    Selects a hand position from the predicted action probabilities, restricted to the legal actions.

    :return: (card index or -1 if no card can be played, whether the agent put no probability on a legal card)
    """
    if not legal.any():
        return -1, False

    if isinstance(action_probs, torch.Tensor):
        action_probs = action_probs.detach().cpu().numpy()
    probs = np.asarray(action_probs, dtype=np.float64) * legal
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


def _game(env: Env, seat_agents: list, first_states: np.ndarray, target_returns: List[float],
          reward_convention: str):
    """
    The course of one game as a coroutine: it yields a request for every decision (see play_games) and receives the
    index of the selected card.
    """
    declarer_id = env.game.get_declarer().get_id()

    states = [[np.asarray(first_states[pl], dtype=np.float32)] for pl in range(3)]
    actions = [[] for _ in range(3)]
    rewards = [[] for _ in range(3)]
    rtg = [[float(target_returns[pl])] for pl in range(3)]
    legal_masks = [[] for _ in range(3)]
    hidden_labels = [[] for _ in range(3)]

    put_card_points = 0

    def request(player_id, t):
        legal = env.legal_actions(player_id)
        legal_masks[player_id][-1] = legal
        hidden_labels[player_id][-1] = hidden_card_labels(env.game, env.game.players[player_id], env.hand)
        trajectory = {"states": states[player_id], "actions": actions[player_id], "rewards": rewards[player_id],
                      "rtg": rtg[player_id], "legal": legal_masks[player_id]}
        return {"env": env, "agent": seat_agents[player_id], "player_id": player_id, "trajectory": trajectory, "t": t,
                "legal": legal}

    for t in range(MAX_EPISODE_LENGTH):
        for pl in range(3):
            actions[pl].append(np.zeros(ACT_DIM, dtype=np.float32))
            rewards[pl].append(0.0)
            legal_masks[pl].append(np.zeros(ACT_DIM, dtype=np.int8))
            hidden_labels[pl].append(hidden_card_labels(env.game, env.game.players[pl], env.hand))

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

            card_index = -1
            if env.legal_actions(declarer_id).any():
                card_index = yield request(declarer_id, t)

                # the declarer receives the card points of the put card
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

                card_index = yield request(current_player, t)
                actions[current_player][-1][card_index] = 1

                acting_player = current_player
                state, _, _, current_player = env.online_step(card_index, current_player)
                next_states[acting_player] = np.asarray(state, dtype=np.float32)

            # the party winning the trick receives its card points (see Game.finish_trick)
            for pl in range(3):
                rewards[pl][-1] = env.game.players[pl].current_trick_points

            # the data of the thesis additionally rewards the declarer with the put Skat in the first trick
            if reward_convention == "thesis" and t == 2 and not env.hand:
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
        "states": np.stack(states[pl][:MAX_EPISODE_LENGTH]).reshape(MAX_EPISODE_LENGTH, env.state_dim),
        "actions": np.stack(actions[pl]),
        "rewards": np.asarray(rewards[pl], dtype=np.float32),
        "legal": np.stack(legal_masks[pl]),
        "hidden": np.stack(hidden_labels[pl]),
    } for pl in range(3)]

    return {
        "trajectories": trajectories,
        "game_points": game_points,
        "declarer_id": declarer_id,
        "declarer_won": declarer_won,
    }


def play_games(envs: List[Env],
               seat_agents: list,
               first_states: list,
               target_returns: list,
               greedy: bool = False,
               temperature: float = 1.0,
               rng: Optional[np.random.Generator] = None,
               reward_convention: str = "default") -> List[dict]:
    """
    Plays several games in lockstep. In every step, the decisions of all games that are made by the same Decision
    Transformer are predicted in one batch, which is much faster than playing the games one after another.

    :param envs: environments after online_reset.
    :param seat_agents: for every game, one agent or TrainableDT per seat.
    :param first_states: for every game, the states returned by online_reset.
    :param target_returns: for every game, the initial return-to-go of each seat.
    :param greedy: whether to take the most probable card (evaluation) or sample (exploration).
    :param temperature: sampling temperature if not greedy.
    :param rng: numpy random generator for sampling.
    :param reward_convention: "default", or "thesis" to additionally reward the put Skat in the first trick.
    :return: for every game a dict with the trajectory of each seat (states, actions, rewards in the convention of the
        offline data, legal actions and locations of the unseen cards), the game points of each seat, the id of the
        declarer and the number of decisions where the rule mask of the model disagreed with the game engine.
    """
    if reward_convention not in REWARD_CONVENTIONS:
        raise ValueError(f"Unknown reward convention {reward_convention}")

    games = [_game(env, [as_agent(agent) for agent in agents], states, targets, reward_convention)
             for env, agents, states, targets in zip(envs, seat_agents, first_states, target_returns)]
    results = [None] * len(games)
    mismatches, decisions = [0] * len(games), [0] * len(games)

    active = {}
    for i, game in enumerate(games):
        try:
            active[i] = next(game)
        except StopIteration as stop:
            results[i] = stop.value

    while active:
        # batch the requests of the same Decision Transformer
        probs = {}
        batches = defaultdict(list)
        for i, req in active.items():
            if isinstance(req["agent"], DTAgent):
                batches[id(req["agent"].model)].append(i)
            else:
                probs[i] = req["agent"].action_probs(req["env"], req["player_id"], req["trajectory"], req["t"])
        for indices in batches.values():
            model = active[indices[0]]["agent"].model
            batch_probs = get_action_batch(model, [active[i]["trajectory"] for i in indices],
                                           [active[i]["t"] for i in indices])
            for i, p in zip(indices, batch_probs):
                probs[i] = p

        for i in list(active):
            req = active[i]
            card_index, mismatch = select_card(probs[i], req["legal"], greedy or req["agent"].deterministic,
                                               temperature, rng)
            mismatches[i] += mismatch
            decisions[i] += 1
            try:
                active[i] = games[i].send(card_index)
            except StopIteration as stop:
                results[i] = stop.value
                del active[i]

    for i, result in enumerate(results):
        result["mask_mismatches"] = mismatches[i]
        result["decisions"] = decisions[i]
    return results


def play_game(env: Env,
              seat_agents: list,
              first_states: np.ndarray,
              target_returns: List[float],
              greedy: bool = False,
              temperature: float = 1.0,
              rng: Optional[np.random.Generator] = None,
              reward_convention: str = "default") -> dict:
    """
    Plays one game in an environment that was reset with env.online_reset (see play_games).

    :param seat_agents: one agent or TrainableDT per seat (the same one can sit on several seats).
    """
    return play_games([env], [seat_agents], [first_states], [target_returns], greedy, temperature, rng,
                      reward_convention)[0]
