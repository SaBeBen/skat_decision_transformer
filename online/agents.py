"""
Players for the online environment. Every agent returns a score per hand position, which is restricted to the legal
actions of the game engine before a card is selected (see online/self_play.py).
"""

import numpy as np
import torch

from dt_skat_environment.environment import ACT_DIM
from dt_skat_environment.game_engine.game.game_variant import GameVariantNull
from dt_skat_environment.game_engine.model.card import Card
from trainer.inference import get_action


class Agent:
    # whether the agent's scores already determine the action (no sampling from them)
    deterministic = False

    def action_probs(self, env, player_id, trajectory, t) -> np.ndarray:
        """
        :param env: the online environment.
        :param player_id: the acting player.
        :param trajectory: dict with the lists "states", "actions", "rewards", "rtg" and "legal" of the player up to
            the current time step t.
        :return: a non-negative score for each of the 12 hand positions.
        """
        raise NotImplementedError


class DTAgent(Agent):
    def __init__(self, model):
        self.model = model

    def action_probs(self, env, player_id, trajectory, t):
        return get_action(self.model,
                          torch.from_numpy(np.asarray(trajectory["states"][:t + 1], dtype=np.float32)),
                          torch.from_numpy(np.asarray(trajectory["actions"][:t + 1], dtype=np.float32)),
                          torch.from_numpy(np.asarray(trajectory["rewards"][:t + 1], dtype=np.float32)),
                          torch.tensor(trajectory["rtg"][:t + 1], dtype=torch.float32),
                          torch.arange(t + 1).reshape(1, -1),
                          action_mask=torch.from_numpy(np.asarray(trajectory["legal"][:t + 1], dtype=np.float32))
                          ).numpy()


class RandomAgent(Agent):
    """Plays a uniformly random legal card."""
    deterministic = True

    def __init__(self, rng=None):
        self.rng = rng if rng is not None else np.random.default_rng()

    def action_probs(self, env, player_id, trajectory, t):
        legal = np.flatnonzero(env.legal_actions(player_id))
        probs = np.zeros(ACT_DIM)
        probs[self.rng.choice(legal)] = 1
        return probs


class RuleAgent(Agent):
    """
    A simple rule-based player as a fixed, diverse opponent:
    - puts the most valuable side cards (no aces) into the Skat,
    - wins a trick as cheaply as possible, gives points to a winning partner and otherwise plays the cheapest card,
    - leads trumps as declarer and aces as defender,
    - avoids winning tricks as declarer in null games.
    """
    deterministic = True

    @staticmethod
    def _cost(card, variant):
        # card points first, trumps are more valuable than side cards
        return card.get_value() + (20 if variant.is_trump(card) else 0)

    def action_probs(self, env, player_id, trajectory, t):
        game = env.game
        variant = game.game_variant
        player = game.players[player_id]
        legal = np.flatnonzero(env.legal_actions(player_id))
        cards = {index: player.cards[index] for index in legal}
        is_declarer = player == game.get_declarer()
        null_game = isinstance(variant, GameVariantNull)

        if t < 2:
            # Skat putting
            side = [i for i, card in cards.items() if not variant.is_trump(card) and card.face is not Card.Face.ACE]
            candidates = side if side else list(cards)
            choice = max(candidates, key=lambda i: cards[i].get_value())
        elif not game.trick.stack:
            # leading a trick
            if null_game:
                choice = min(cards, key=lambda i: GameVariantNull.NULL_ORDER.index(cards[i].face))
            elif is_declarer and any(variant.is_trump(card) for card in cards.values()):
                trumps = [i for i, card in cards.items() if variant.is_trump(card)]
                choice = max(trumps, key=lambda i: (cards[i].face is Card.Face.JACK, cards[i].face.value,
                                                    cards[i].suit.value))
            else:
                aces = [i for i, card in cards.items() if card.face is Card.Face.ACE and not variant.is_trump(card)]
                choice = aces[0] if aces else min(cards, key=lambda i: self._cost(cards[i], variant))
        else:
            open_cards = game.trick.get_open_cards()
            highest = variant.get_highest_card(open_cards)
            winner = next(p for p, card in game.trick.stack if card == highest)
            partner_wins = not is_declarer and winner != game.get_declarer()
            winning = [i for i, card in cards.items() if variant.get_highest_card(open_cards + [card]) == card]

            if null_game:
                losing = [i for i in cards if i not in winning]
                pool = losing if (is_declarer and losing) else list(cards)
                choice = max(pool, key=lambda i: GameVariantNull.NULL_ORDER.index(cards[i].face)) if is_declarer \
                    else min(pool, key=lambda i: GameVariantNull.NULL_ORDER.index(cards[i].face))
            elif partner_wins:
                # give points to the partner
                choice = max(cards, key=lambda i: cards[i].get_value() - (20 if variant.is_trump(cards[i]) else 0))
            elif winning:
                choice = min(winning, key=lambda i: self._cost(cards[i], variant))
            else:
                choice = min(cards, key=lambda i: self._cost(cards[i], variant))

        probs = np.zeros(ACT_DIM)
        probs[choice] = 1
        return probs


class HumanAgent(Agent):
    """Asks for the card to play on the command line."""
    deterministic = True

    def action_probs(self, env, player_id, trajectory, t):
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
