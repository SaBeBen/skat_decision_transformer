"""
Players for the online environment. Every agent returns a score per hand position, which is restricted to the legal
actions of the game engine before a card is selected (see online/self_play.py).
"""

import numpy as np
import torch

from dt_skat_environment.environment import ACT_DIM
from trainer.inference import get_action


class Agent:
    # whether the agent's scores already determine the action (no sampling from them)
    deterministic = False

    def action_probs(self, env, player_id, states, actions, rewards, rtg, t) -> np.ndarray:
        """
        :param env: the online environment.
        :param player_id: the acting player.
        :param states, actions, rewards, rtg: the trajectory of the player up to the current time step t.
        :return: a non-negative score for each of the 12 hand positions.
        """
        raise NotImplementedError


class DTAgent(Agent):
    def __init__(self, model):
        self.model = model

    def action_probs(self, env, player_id, states, actions, rewards, rtg, t):
        return get_action(self.model,
                          torch.from_numpy(np.asarray(states[:t + 1], dtype=np.float32)),
                          torch.from_numpy(np.asarray(actions[:t + 1], dtype=np.float32)),
                          torch.from_numpy(np.asarray(rewards[:t + 1], dtype=np.float32)),
                          torch.tensor(rtg[:t + 1], dtype=torch.float32),
                          torch.arange(t + 1).reshape(1, -1)).numpy()


class RandomAgent(Agent):
    """Plays a uniformly random legal card."""
    deterministic = True

    def __init__(self, rng=None):
        self.rng = rng if rng is not None else np.random.default_rng()

    def action_probs(self, env, player_id, states, actions, rewards, rtg, t):
        legal = np.flatnonzero(env.legal_actions(player_id))
        probs = np.zeros(ACT_DIM)
        probs[self.rng.choice(legal)] = 1
        return probs


class HumanAgent(Agent):
    """Asks for the card to play on the command line."""
    deterministic = True

    def action_probs(self, env, player_id, states, actions, rewards, rtg, t):
        player = env.game.players[player_id]
        legal = np.flatnonzero(env.legal_actions(player_id))

        if t < 2:
            print(f"\nYou play {env.game.game_variant.get_variant_name()} as the declarer (player {player_id}).\n"
                  f"Your cards are {player.cards}")
            prompt = f"Which card do you want to put in the Skat? Possible indices are {legal}: "
        else:
            if t == 2 and len(env.game.trick.stack) == env.game.trick.players.index(player):
                print(f"\nYou play as player {player_id} ({player.get_role()}). "
                      f"{env.game.game_variant.get_variant_name()} is played by player "
                      f"{env.game.get_declarer().get_id()}.")
            if env.game.last_trick_cards is not None:
                print(f"The last trick was {env.game.last_trick_cards}")
            print(f"\nYour cards are {player.cards}\nThe open cards are {env.game.trick.get_open_cards()}")
            prompt = f"Which card do you want to select? Possible indices are {legal}: "

        card_index = None
        while card_index not in legal:
            try:
                card_index = int(input(prompt))
            except ValueError:
                print("Invalid input. Please enter a valid number.")

        probs = np.zeros(ACT_DIM)
        probs[card_index] = 1
        return probs


def as_agent(player) -> Agent:
    """Wraps a Decision Transformer into an agent, agents are returned unchanged."""
    return player if isinstance(player, Agent) else DTAgent(player)
