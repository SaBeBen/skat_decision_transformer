"""
Data augmentation by permuting the suits of a game (one-hot encoding).

Permuting suits preserves the course and outcome of a game if the game value is unchanged:
- suit games: the three non-trump suits are permuted (the trump suit determines the game value),
- grand: all four suits are permuted, the jacks keep their suits (they rank by suit),
- null: all four suits are permuted including the jacks.
The hand cards are sorted by suit, so the hand positions (and thereby actions and legal masks) are remapped.
"""

from typing import Dict, Optional

import numpy as np

from dt_skat_environment.card_representation_conversion import CARD_INDEX
from dt_skat_environment.environment import get_dims_in_enc, ACT_DIM
from dt_skat_environment.game_engine.model.card import Card

CARD_DIM = 12
JACK = 11  # index of the jack in the face part of the one-hot card encoding
TRUMP_SLICE = slice(6, 10)
LAST_TRICK_START, OPEN_CARDS_START, HAND_START = 10, 46, 70
STATE_DIM = get_dims_in_enc("one-hot")[2]

# index of a card (suit, face index in the one-hot encoding) in the card numbering of the data
_FACES = [Card.Face.SEVEN, Card.Face.EIGHT, Card.Face.NINE, Card.Face.QUEEN, Card.Face.KING, Card.Face.TEN,
          Card.Face.ACE, Card.Face.JACK]
_DATA_INDEX = np.array([[CARD_INDEX[Card(Card.Suit(suit), face)] for face in _FACES] for suit in range(4)])


def sample_suit_permutation(trump_enc, rng: np.random.Generator) -> np.ndarray:
    """:return: array mapping every suit to its new suit."""
    trump_enc = np.asarray(trump_enc)
    sigma = np.arange(4)
    if trump_enc.sum() == 1:
        trump = int(np.argmax(trump_enc))
        others = [suit for suit in range(4) if suit != trump]
        sigma[others] = rng.permutation(others)
    else:
        sigma = rng.permutation(4)
    return sigma


def _permute_card(card: np.ndarray, sigma: np.ndarray, permute_jacks: bool) -> np.ndarray:
    if not card.any():
        return card
    suit, face = int(np.argmax(card[:4])), int(np.argmax(card[4:]))
    if face + 4 == JACK and not permute_jacks:
        return card
    permuted = card.copy()
    permuted[:4] = 0
    permuted[sigma[suit]] = 1
    return permuted


def _card_mapping(sigma: np.ndarray, permute_jacks: bool) -> np.ndarray:
    """Maps the data index of every card to the data index of the permuted card."""
    mapping = np.zeros(32, dtype=np.int64)
    for suit in range(4):
        for face in range(8):
            new_suit = suit if (face + 4 == JACK and not permute_jacks) else sigma[suit]
            mapping[_DATA_INDEX[suit, face]] = _DATA_INDEX[new_suit, face]
    return mapping


def permute_suits(trajectory: Dict[str, np.ndarray], sigma: Optional[np.ndarray] = None,
                  rng: Optional[np.random.Generator] = None) -> Dict[str, np.ndarray]:
    """
    Applies one suit permutation to all time steps of a trajectory in the one-hot encoding.

    :param trajectory: dict with "states" (12, 214), "actions" (12, 12) and optionally "legal" (12, 12) and
        "hidden" (12, 32). Other entries (e.g. rewards) are copied.
    :param sigma: the permutation of the suits, sampled if None.
    :return: the permuted trajectory (a new dict).
    """
    states = np.array(trajectory["states"], dtype=np.float32).reshape(-1, STATE_DIM)
    trump_enc = states[0, TRUMP_SLICE]
    if sigma is None:
        sigma = sample_suit_permutation(trump_enc, rng if rng is not None else np.random.default_rng())
    permute_jacks = trump_enc.sum() == 0  # null game

    result = {key: np.array(value) for key, value in trajectory.items()}
    actions = np.array(trajectory["actions"], dtype=np.float32)
    legal = np.array(trajectory["legal"]) if "legal" in trajectory else None

    for t in range(states.shape[0]):
        state = states[t]
        for start in list(range(LAST_TRICK_START, HAND_START, CARD_DIM)):
            state[start:start + CARD_DIM] = _permute_card(state[start:start + CARD_DIM], sigma, permute_jacks)

        # permute the hand cards and sort them again (by suit, then face as in Card.__lt__)
        hand = state[HAND_START:].reshape(ACT_DIM, CARD_DIM)
        hand = np.stack([_permute_card(card, sigma, permute_jacks) for card in hand])
        occupied = [slot for slot in range(ACT_DIM) if hand[slot].any()]
        order = sorted(occupied, key=lambda slot: (int(np.argmax(hand[slot, :4])), int(np.argmax(hand[slot, 4:]))))
        order += [slot for slot in range(ACT_DIM) if slot not in occupied]
        state[HAND_START:] = hand[order].reshape(-1)

        # new position j holds the card of the old position order[j]
        actions[t] = actions[t][order]
        if legal is not None:
            legal[t] = legal[t][order]

    result["states"] = states
    result["actions"] = actions
    if legal is not None:
        result["legal"] = legal
    if "hidden" in trajectory:
        hidden = np.array(trajectory["hidden"])
        permuted = np.full_like(hidden, -100)
        permuted[:, _card_mapping(sigma, permute_jacks)] = hidden
        result["hidden"] = permuted
    return result
