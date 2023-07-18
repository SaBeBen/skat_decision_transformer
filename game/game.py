from model import card
from model.card import Card
from model.player import Player


class Game:
    MAX_ROUNDS = 10

    def __init__(self, players):
        self.players = players
        self.card_deck = list()
        self.skat = list()
        self.dealer = -1
        self.bid_value = -1
        self.game_variant = None
        self.passed_bid_players = list()
        self.round = -1

        self.trick = Trick([self.get_first_seat(), self.get_second_seat(), self.get_third_seat()])
        self.trick.leader = self.get_first_seat()

        # self.create_deck()

    def finish_trick(self):
        trick_winner, current_trick_points = self.trick.get_winner(self.game_variant)

        # TODO: play with the reward settings
        # update current trick points: the party winning this trick get a reward of the points
        if trick_winner is self.get_declarer():
            trick_winner.current_trick_points = current_trick_points
            for player in self.players:
                if player is not trick_winner:
                    # both players of the counter party receive a negative reward
                    player.current_trick_points = -current_trick_points
        else:
            for player in self.players:
                if player is not self.get_declarer():
                    # both players of the counter party receive the reward
                    player.current_trick_points = current_trick_points
                else:
                    # the declarer receives a negative reward
                    player.current_trick_points = -current_trick_points

        # add trick to players trick_stack
        trick_winner.trick_stack[self.round] = self.trick.stack
        # trick_winner.cardPoints += self.trick.stack

        # new trick
        idx_leader = self.players.index(trick_winner)
        second_player = self.players[(idx_leader + 1) % len(self.players)]
        third_player = self.players[(idx_leader + 2) % len(self.players)]
        self.trick = Trick([trick_winner, second_player, third_player])
        # set trick leader for next round
        self.trick.leader = trick_winner

    # TODO: surrender

    def get_dealer(self):
        return self.players[self.dealer]

    def get_first_seat(self):
        return self.players[(self.dealer + 1) % len(self.players)]

    def get_second_seat(self):
        return self.players[(self.dealer + 2) % len(self.players)]

    def get_third_seat(self):
        return self.players[(self.dealer + 3) % len(self.players)]

    def get_declarer(self):
        for player in self.players:
            if player.type is Player.Type.DECLARER:
                return player

    def has_declarer_won(self):
        # TODO check overbid and bid variants like schwarz
        return self.get_declarer().sum_trick_values() > 60

    def create_deck(self):
        for suit in Card.Suit:
            for face in Card.Face:
                self.card_deck.append(Card(suit, face))

    def clear_cards(self):
        self.skat.clear()
        for player in self.players:
            player.cards.clear()
            player.trick_stack.clear()

    def reset(self, with_dealer=False):
        self.clear_cards()
        self.bid_value = -1
        self.game_variant = None
        self.passed_bid_players.clear()

        if with_dealer:
            self.dealer = -1


class Trick:
    def __init__(self, players):
        self.stack = list()  # list of tuples (player, card)
        self.players = players
        self.leader = None

    def add(self, player, card):
        self.stack.append((player, card))

    def get_next_player(self, player, skip=0):
        idx_player = self.players.index(player)
        return self.players[(idx_player + 1 + skip) % len(self.players)]

    def get_current_player(self):
        if len(self.stack) == 0:
            return self.leader
        elif len(self.stack) == 1:
            return self.get_next_player(self.leader)
        else:
            return self.get_next_player(self.leader, 1)

    def has_already_played_card(self, player):
        if self.is_empty():
            return False

        for entry in self.stack:
            if player is entry[0]:
                return True

        return False

    def is_valid_card_move(self, game_variant, player, card):
        # first played card
        if self.is_empty():
            return True

        first_card = self.stack[0][1]
        # check if player can follow by trump
        if game_variant.is_trump(first_card) and game_variant.has_trump(player):
            return game_variant.is_trump(card)
        # check if player can follow by suit if no trump was played
        elif not game_variant.is_trump(first_card) and player.has_suit(first_card.suit):
            return card.suit is first_card.suit
        else:
            return True

    def is_empty(self):
        return len(self.stack) == 0

    def can_move(self, player):
        return player is self.get_current_player()

    def get_winner(self, game_variant):
        trick_map = dict()

        # map all cards and players to dict {card: player}
        for entry in self.stack:
            trick_map[entry[1]] = entry[0]

        highest_card = game_variant.get_highest_card(list(trick_map))

        current_trick_points = sum(map(Card.get_value, list(trick_map)))

        # get winner and the points for this trick
        return trick_map[highest_card], current_trick_points

    def is_complete(self):
        return len(self.stack) == 3
