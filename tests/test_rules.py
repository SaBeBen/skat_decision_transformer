import pytest

from dt_skat_environment.card_representation_conversion import convert_numerical_to_card, convert_tuple_to_card, \
    all_cards
from dt_skat_environment.environment import Env, get_peaks, get_game_variant
from dt_skat_environment.game_engine.game.game import Trick
from dt_skat_environment.game_engine.game.game_variant import GameVariantSuit, GameVariantGrand, GameVariantNull
from dt_skat_environment.game_engine.model.card import Card
from dt_skat_environment.game_engine.model.player import Player

S, F = Card.Suit, Card.Face


def c(face, suit):
    return Card(suit, face)


J_CLUB, J_SPADE, J_HEARTS, J_DIAMOND = (c(F.JACK, s) for s in (S.CLUB, S.SPADE, S.HEARTS, S.DIAMOND))


@pytest.mark.parametrize("cards,trump,expected", [
    # with 2: J♣ J♠, without J♥
    ([J_CLUB, J_SPADE, c(F.SEVEN, S.HEARTS)], S.HEARTS, 2),
    # with 4 jacks and A, 10 of trump
    ([J_CLUB, J_SPADE, J_HEARTS, J_DIAMOND, c(F.ACE, S.CLUB), c(F.TEN, S.CLUB)], S.CLUB, 6),
    # without 1: J♣ missing, J♠ present
    ([J_SPADE, c(F.SEVEN, S.HEARTS)], S.HEARTS, 1),
    # without 3: no jacks, but the ace of trump
    ([c(F.ACE, S.DIAMOND)], S.DIAMOND, 4),
    # grand: only jacks count
    ([J_CLUB, J_SPADE, J_HEARTS, J_DIAMOND, c(F.ACE, S.CLUB)], None, 4),
    ([J_DIAMOND], None, 3),
])
def test_peaks(cards, trump, expected):
    assert get_peaks(cards, trump) == expected


def test_game_variant_levels():
    cards = [J_CLUB, J_SPADE, c(F.SEVEN, S.HEARTS)]
    suit = get_game_variant([0, 1, 0, 0], cards, hand=True)
    assert isinstance(suit, GameVariantSuit) and suit.peaks == 2 and suit.get_level() == 3
    grand = get_game_variant([1, 1, 1, 1], cards)
    assert type(grand) is GameVariantGrand and grand.get_level() == 2
    null = get_game_variant([0, 0, 0, 0], cards, ouvert=True)
    assert isinstance(null, GameVariantNull) and null.get_value() == 46


def make_player(pid, cards):
    player = Player(pid, str(pid))
    player.cards = list(cards)
    return player


def trick_with(lead_player, lead_card, players):
    trick = Trick(players)
    trick.leader = lead_player
    trick.add(lead_player, lead_card)
    return trick


def test_jack_does_not_follow_its_suit_in_suit_game():
    variant = GameVariantSuit(S.CLUB)
    leader = make_player(0, [])
    follower = make_player(1, [J_HEARTS, c(F.SEVEN, S.SPADE)])
    trick = trick_with(leader, c(F.ACE, S.HEARTS), [leader, follower, make_player(2, [])])

    # J♥ is a trump, the player has no hearts and may play anything
    assert trick.is_valid_card_move(variant, follower, J_HEARTS)
    assert trick.is_valid_card_move(variant, follower, c(F.SEVEN, S.SPADE))

    follower.cards.append(c(F.KING, S.HEARTS))
    assert trick.is_valid_card_move(variant, follower, c(F.KING, S.HEARTS))
    assert not trick.is_valid_card_move(variant, follower, J_HEARTS)


def test_trump_must_be_served_with_jack():
    variant = GameVariantSuit(S.CLUB)
    leader = make_player(0, [])
    follower = make_player(1, [J_DIAMOND, c(F.ACE, S.HEARTS)])
    trick = trick_with(leader, c(F.SEVEN, S.CLUB), [leader, follower, make_player(2, [])])

    assert trick.is_valid_card_move(variant, follower, J_DIAMOND)
    assert not trick.is_valid_card_move(variant, follower, c(F.ACE, S.HEARTS))


def test_null_rules():
    variant = GameVariantNull()
    leader = make_player(0, [])
    follower = make_player(1, [J_HEARTS, c(F.SEVEN, S.SPADE)])
    trick = trick_with(leader, c(F.ACE, S.HEARTS), [leader, follower, make_player(2, [])])

    # jacks belong to their suit in null games
    assert trick.is_valid_card_move(variant, follower, J_HEARTS)
    assert not trick.is_valid_card_move(variant, follower, c(F.SEVEN, S.SPADE))

    # order 7 8 9 10 J Q K A, only the led suit wins
    assert variant.get_highest_card([c(F.TEN, S.HEARTS), J_HEARTS, c(F.NINE, S.HEARTS)]) == J_HEARTS
    assert variant.get_highest_card([c(F.TEN, S.HEARTS), c(F.KING, S.HEARTS)]) == c(F.KING, S.HEARTS)
    assert variant.get_highest_card([c(F.SEVEN, S.HEARTS), c(F.ACE, S.SPADE)]) == c(F.SEVEN, S.HEARTS)


def play_out(env, declarer_cards, defender_cards, skat, variant, bid=18, declarer_tricks=None):
    """Assigns cards and tricks directly to evaluate the scoring."""
    declarer, d1, d2 = env.game.players
    declarer.type, d1.type, d2.type = Player.Type.DECLARER, Player.Type.DEFENDER, Player.Type.DEFENDER
    env.game.game_variant = variant
    env.game.skat = skat
    env.game.bid_value = bid
    for rnd, cards in enumerate(declarer_tricks or []):
        declarer.trick_stack[rnd] = [(declarer, card) for card in cards]
    return declarer, d1


def test_game_points_suit_and_overbid():
    env = Env("one-hot")
    # declarer won 61 points with diamonds, with 1 -> (1 + 1) * 9 = 18
    tricks = [[c(F.ACE, S.CLUB), c(F.ACE, S.SPADE), c(F.ACE, S.HEARTS)],
              [c(F.TEN, S.CLUB), c(F.TEN, S.SPADE), c(F.SEVEN, S.HEARTS)]]
    declarer, defender = play_out(env, [], [], [c(F.KING, S.CLUB), c(F.SEVEN, S.CLUB)],
                                  GameVariantSuit(S.DIAMOND, peaks=1), declarer_tricks=tricks)
    assert declarer.sum_trick_values() + 4 == 57
    # 57 points -> lost: -2 * 18 - 50
    assert env.get_game_points(declarer) == -2 * 18 - 50
    assert env.get_game_points(defender) == 40

    env.game.skat = [c(F.KING, S.CLUB), c(F.TEN, S.HEARTS)]  # 67 points -> won
    assert env.get_game_points(declarer) == 18 + 50
    assert env.get_game_points(defender) == 0

    # overbid: bid 20 > 18 -> lost with the next multiple of 9 above the bid (27)
    env.game.bid_value = 20
    assert env.get_game_points(declarer) == -2 * 27 - 50


def test_game_points_null():
    env = Env("one-hot")
    declarer, defender = play_out(env, [], [], [c(F.SEVEN, S.CLUB), c(F.EIGHT, S.CLUB)], GameVariantNull())
    assert env.get_game_points(declarer) == 23 + 50
    # a trick without card points loses a null game
    declarer.trick_stack[0] = [(declarer, c(F.SEVEN, S.HEARTS))]
    assert env.get_game_points(declarer) == -2 * 23 - 50
    assert env.get_game_points(defender) == 40


def test_card_conversions_are_consistent():
    cards = [convert_numerical_to_card(i) for i in range(32)]
    assert len(set(cards)) == 32
    tuples = [tuple(t) for t in all_cards]
    assert len(set(tuples)) == 32
    assert len({convert_tuple_to_card(t) for t in tuples}) == 32
