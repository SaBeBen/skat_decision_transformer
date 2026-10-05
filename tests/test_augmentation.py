import random

import numpy as np
import pytest

from dt_skat_environment.environment import Env
from online.agents import RandomAgent
from online.self_play import play_game, compute_target_returns
from trainer.augmentation import permute_suits, sample_suit_permutation, HAND_START, CARD_DIM
from trainer.data_collator import DecisionTransformerSkatDataCollator

CARD_VALUES = np.array([0, 0, 0, 3, 4, 10, 11, 2])  # 7, 8, 9, Q, K, 10, A, J


def random_game(game, seed):
    random.seed(seed)
    env = Env("one-hot")
    states = env.online_reset(declaration=lambda e: {"declarer": seed % 3, "game": game, "hand": False, "bid": 18})
    agents = [RandomAgent(np.random.default_rng(seed + pl)) for pl in range(3)]
    return play_game(env, agents, states, compute_target_returns(env, True))["trajectories"]


def card_at(state, slot):
    card = state[HAND_START + slot * CARD_DIM:HAND_START + (slot + 1) * CARD_DIM]
    return int(np.argmax(card[:4])), int(np.argmax(card[4:]))


def hand_points(state):
    hand = state[HAND_START:].reshape(12, CARD_DIM)
    return sum(CARD_VALUES[np.argmax(card[4:])] for card in hand if card.any())


@pytest.mark.parametrize("game", [10, 24, 23])
def test_permutation_keeps_the_game(game):
    rng = np.random.default_rng(0)
    for seed in range(3):
        for trajectory in random_game(game, seed):
            sigma = sample_suit_permutation(trajectory["states"][0, 6:10], rng)
            if game == 10:
                assert sigma[1] == 1  # the trump suit stays
            permuted = permute_suits(trajectory, sigma=sigma)

            np.testing.assert_array_equal(permuted["states"][:, :10], trajectory["states"][:, :10])
            np.testing.assert_array_equal(permuted["rewards"], trajectory["rewards"])
            assert (permuted["hidden"] != -100).sum() == (trajectory["hidden"] != -100).sum()

            for t in range(12):
                state, new_state = trajectory["states"][t], permuted["states"][t]
                assert hand_points(state) == hand_points(new_state)
                if not trajectory["actions"][t].any():
                    continue
                slot, new_slot = trajectory["actions"][t].argmax(), permuted["actions"][t].argmax()
                suit, face = card_at(state, slot)
                new_suit, new_face = card_at(new_state, new_slot)
                # the action still points to the (permuted) card that was played
                assert face == new_face
                assert new_suit == (suit if face == 7 and game != 23 else sigma[suit])
                assert permuted["legal"][t][new_slot] == 1
                assert permuted["legal"][t].sum() == trajectory["legal"][t].sum()

            # the inverse permutation restores the game
            restored = permute_suits(permuted, sigma=np.argsort(sigma))
            for key in ("states", "actions", "legal", "hidden"):
                np.testing.assert_array_equal(restored[key], trajectory[key])


def test_hands_stay_sorted():
    rng = np.random.default_rng(1)
    for trajectory in random_game(24, 0):
        permuted = permute_suits(trajectory, rng=rng)
        for state in permuted["states"]:
            cards = [card_at(state, slot) for slot in range(12) if state[HAND_START + slot * CARD_DIM:
                                                                          HAND_START + (slot + 1) * CARD_DIM].any()]
            assert cards == sorted(cards)


def test_collator_augments():
    trajectories = random_game(11, 0)
    collator = DecisionTransformerSkatDataCollator(trajectories, batch_size=16, augment_prob=1.0, seed=0)
    batch = collator(trajectories)
    assert batch["states"].shape == (16, 12, 214)
    assert batch["action_mask"].shape == (16, 12, 12)
