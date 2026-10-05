import random

import numpy as np
import pytest
import torch

from dt_skat_environment.environment import Env
from dt_skat_environment.game_engine.model.card import Card
from online.agents import RandomAgent
from online.self_play import play_game, compute_target_returns
from trainer.declaration import DECLARATIONS, DeclarationDT, DeclarationPolicy, declaration_value, run_auction, \
    encode_hand, declarations_from_data, train_declaration_model

S, F = Card.Suit, Card.Face


@pytest.mark.parametrize("limits,expected", [
    ([0, 0, 0], None),
    ([18, 0, 0], (0, 18)),  # fore-hand plays at 18 after two passes
    ([0, 0, 23], (2, 18)),
    ([24, 30, 0], (1, 27)),  # middle-hand outbids fore-hand
    ([36, 30, 0], (0, 30)),  # fore-hand holds
    ([36, 30, 48], (2, 40)),  # rear-hand outbids the winner of the first duel
    ([48, 30, 46], (0, 46)),
])
def test_auction(limits, expected):
    assert run_auction(limits) == expected


def test_declaration_values():
    cards = [Card(S.CLUB, F.JACK), Card(S.SPADE, F.JACK), Card(S.HEARTS, F.SEVEN)]
    # hearts with 2: (1 + 2) * 10
    assert declaration_value(cards, (10, False)) == 30
    # hearts hand with 2: (1 + 2 + 1) * 10
    assert declaration_value(cards, (10, True)) == 40
    assert declaration_value(cards, (24, False)) == 72
    assert declaration_value(cards, (23, True)) == 35
    assert declaration_value(cards, None) == 0


def test_declarations_from_data_and_training():
    # two synthetic games: seat 0 declares clubs and wins, seat 1 declares null and loses
    deck = np.random.default_rng(0).permutation(32)
    games = []
    for declarer, game, points in ((0, 12, 72), (1, 23, -96)):
        row = np.zeros(4 + 32 + 13, dtype=np.int64)
        row[1:4] = [100, 200, 300]
        row[4:36] = deck
        row[-13], row[-12], row[-11], row[-5] = row[1 + declarer], game, 0, points
        games.append(row)
    states, declarations, returns = declarations_from_data(np.stack(games))

    assert states.shape == (6, 35)
    assert list(declarations) == [DECLARATIONS.index((12, False)), 0, 0, 0, DECLARATIONS.index((23, False)), 0]
    assert list(returns) == [72, 0, 0, 40, -96, 40]

    model = DeclarationDT.create(hidden_size=16)
    train_declaration_model(model, states, declarations, returns, epochs=2, batch_size=4, log=lambda _: None)
    logits = model.declaration_logits(torch.from_numpy(states), torch.from_numpy(returns))
    assert logits.shape == (6, len(DECLARATIONS))


def test_declaration_policy_starts_games():
    torch.manual_seed(0)
    model = DeclarationDT.create(hidden_size=16).eval()
    # bias the untrained model towards declaring a game
    with torch.no_grad():
        model.predict_action[0].bias[0] = -10.0
    policy = DeclarationPolicy(model, greedy=True)

    for seed in range(5):
        random.seed(seed)
        env = Env("one-hot")
        states = env.online_reset(declaration=policy)
        declarer = env.game.get_declarer().get_id()
        assert env.game.bid_value >= 18
        assert [index for _, index in policy.last_choices].count(0) == 2
        assert policy.last_choices[declarer][1] != 0
        np.testing.assert_array_equal(policy.last_choices[declarer][0][32:], np.eye(3)[declarer])

        agents = [RandomAgent(np.random.default_rng(pl)) for pl in range(3)]
        result = play_game(env, agents, states, compute_target_returns(env, True))
        assert result["game_points"][declarer] != 0


def test_encode_hand():
    state = encode_hand([Card(S.CLUB, F.ACE), Card(S.DIAMOND, F.SEVEN)], seat=2)
    assert state.sum() == 3 and state[34] == 1
