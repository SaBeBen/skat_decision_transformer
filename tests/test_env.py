import random

import numpy as np
import pytest

from dt_skat_environment.environment import Env, ACT_DIM, get_dims_in_enc


def play_random_game(seed, enc="one-hot"):
    """Plays one game on a random deal with uniformly random legal actions in the order used by the self-play."""
    random.seed(seed)
    env = Env(enc)
    states = env.online_reset()
    declarer_id = env.game.get_declarer().get_id()
    n_steps = [0, 0, 0]

    # Skat putting: only the declarer acts
    for _ in range(2):
        legal = env.legal_actions(declarer_id)
        card_index = int(np.random.choice(np.flatnonzero(legal))) if legal.any() else -1
        state, _, done, _ = env.online_step(card_index, declarer_id)
        assert len(state) == env.state_dim
        assert not done

    # 10 tricks
    current_player = env.game.trick.get_current_player().id
    done = False
    for _ in range(30):
        legal = env.legal_actions(current_player)
        assert legal.any(), "a player on the move must always have a legal card"
        # nobody else can act
        for other in range(3):
            if other != current_player:
                assert not env.legal_actions(other).any()
        card_index = int(np.random.choice(np.flatnonzero(legal)))
        n_steps[current_player] += 1
        state, reward, done, current_player = env.online_step(card_index, current_player)
        assert len(state) == env.state_dim

    return env, states, declarer_id, n_steps


@pytest.mark.parametrize("seed", range(50))
def test_random_legal_games_complete(seed):
    env, _, declarer_id, n_steps = play_random_game(seed)

    assert env.game.round == 10
    assert n_steps == [10, 10, 10]
    for player in env.game.players:
        assert len(player.cards) == 0

    skat_points = sum(card.get_value() for card in env.game.skat)
    assert sum(p.sum_trick_values() for p in env.game.players) + skat_points == 120

    for pl in range(3):
        _, _, game_points = env.finish_online_game(pl)
        if pl == declarer_id:
            assert game_points != 0
        else:
            assert game_points in (0, 40)


@pytest.mark.parametrize("seed", range(50))
def test_seat_positions_are_consistent(seed):
    # pos_p[i][j] is -1 for an opponent and 1 for the team mate of player i, 0 for the player herself
    random.seed(seed)
    env = Env("one-hot")
    states = env.online_reset()
    declarer_id = env.game.get_declarer().get_id()

    for pl in range(3):
        assert env.pos_p[pl][pl] == 0
        for other in range(3):
            if other == pl:
                continue
            same_team = pl != declarer_id and other != declarer_id
            assert env.pos_p[pl][other] == (1 if same_team else -1)
        # the position is also part of the state
        assert list(states[pl][:3]) == env.pos_p[pl]

    # only the declarer puts the Skat
    assert states[declarer_id][3] == 1
    assert all(states[pl][3] == 0 for pl in range(3) if pl != declarer_id)


def test_seeded_deals_are_reproducible():
    def deal(seed):
        random.seed(seed)
        env = Env("one-hot")
        env.online_reset()
        return [str(c) for p in env.game.players for c in p.cards] + [str(c) for c in env.game.skat]

    assert deal(7) == deal(7)
    assert deal(7) != deal(8)


def test_legal_actions_during_skat_putting():
    random.seed(0)
    env = Env("one-hot")
    env.online_reset()
    declarer_id = env.game.get_declarer().get_id()

    for pl in range(3):
        legal = env.legal_actions(pl)
        assert legal.shape == (ACT_DIM,)
        if pl == declarer_id:
            assert legal.sum() == 12
        else:
            assert legal.sum() == 0


def test_state_dims():
    assert get_dims_in_enc("one-hot")[2] == 214


def test_reward_conventions():
    from online.agents import RandomAgent
    from online.self_play import play_game, compute_target_returns

    results = {}
    for convention in ("default", "thesis"):
        random.seed(3)
        env = Env("one-hot")
        states = env.online_reset()
        agents = [RandomAgent(np.random.default_rng(pl)) for pl in range(3)]
        results[convention] = play_game(env, agents, states, compute_target_returns(env, True),
                                        reward_convention=convention)

    declarer = results["default"]["declarer_id"]
    default, thesis = (results[c]["trajectories"][declarer]["rewards"] for c in ("default", "thesis"))
    put_points = default[0] + default[1]
    # the thesis rewards the put Skat a second time in the first trick
    assert thesis[2] - default[2] == put_points
    assert np.array_equal(default[3:11], thesis[3:11])


@pytest.mark.parametrize("game,hand", [(24, False), (23, False), (35, True), (12, True), (59, True)])
def test_declared_variants(game, hand):
    from online.agents import RandomAgent
    from online.self_play import play_game, compute_target_returns
    from dt_skat_environment.game_engine.game.game_variant import GameVariantNull, GameVariantGrand

    for seed in range(10):
        random.seed(seed)
        env = Env("one-hot")
        states = env.online_reset(declaration=lambda e: {"declarer": seed % 3, "game": game, "hand": hand, "bid": 18})
        declarer = seed % 3
        assert env.game.get_declarer().get_id() == declarer
        assert env.hand == hand
        assert states[declarer][3] == (0 if hand else 1)
        if game in (23, 35, 59):
            assert isinstance(env.game.game_variant, GameVariantNull) and sum(env.trump_enc) == 0
        elif game == 24:
            assert type(env.game.game_variant) is GameVariantGrand and sum(env.trump_enc) == 4

        agents = [RandomAgent(np.random.default_rng(seed + pl)) for pl in range(3)]
        result = play_game(env, agents, states, compute_target_returns(env, True))
        skat_points = sum(card.get_value() for card in env.game.skat)
        assert sum(p.sum_trick_values() for p in env.game.players) + skat_points == 120
        assert result["game_points"][declarer] != 0
