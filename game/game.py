from game.game_variant import GameVariantNull
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
        self.last_trick_cards = None

        self.create_deck()

    def finish_trick(self):
        trick_winner, current_trick_points = self.trick.get_winner(self.game_variant)
        self.last_trick_cards = self.trick.get_open_cards()

        # update current trick points: the party winning this trick get a reward of the points
        if trick_winner is self.get_declarer():
            trick_winner.current_trick_points = current_trick_points
            for player in self.players:
                if player is not trick_winner:
                    player.current_trick_points = 0
        else:
            for player in self.players:
                if player is not self.get_declarer():
                    # both players of the counterparty receive the reward
                    player.current_trick_points = current_trick_points
                else:
                    player.current_trick_points = 0

        # add trick to players trick_stack
        trick_winner.trick_stack[self.round] = self.trick.stack

        # new trick
        idx_leader = self.players.index(trick_winner)
        second_player = self.players[(idx_leader + 1) % len(self.players)]
        third_player = self.players[(idx_leader + 2) % len(self.players)]
        self.trick = Trick([trick_winner, second_player, third_player])
        # set trick leader for next round
        self.trick.leader = trick_winner

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
        # check overbid and bid variants
        if isinstance(self.game_variant, GameVariantNull):
            return self.get_declarer().sum_trick_values() == 0
        if self.game_variant.get_level() > 2:
            return self.get_declarer().sum_trick_values() == 120
        elif self.game_variant.get_level() == 1:
            return self.get_declarer().sum_trick_values() >= 30
        else:
            return self.get_declarer().sum_trick_values() > 60

    def create_deck(self):
        for suit in Card.Suit:
            for face in Card.Face:
                self.card_deck.append(Card(suit, face))

    def clear_cards(self):
        self.skat.clear()
        for player in self.players:
            player.current_trick_points = 0
            player.cards.clear()
            player.trick_stack.clear()

    def reset(self, with_dealer=False):
        self.clear_cards()
        self.bid_value = -1
        self.game_variant = None
        self.passed_bid_players.clear()

        if with_dealer:
            self.dealer = -1

    def get_last_trick_cards(self):
        if self.last_trick_cards is None:
            raise ValueError("There is no last trick.")
        else:
            return self.last_trick_cards

    def get_last_trick_points(self):
        if self.round == 1:
            return 0
        elif self.last_trick_cards is None:
            return 0
            # raise ValueError("There is no last trick.")
        else:
            return sum([card.get_value() for card in self.last_trick_cards])

    def get_round(self):
        return self.round

    def surrender(self, current_player):
        # calculate the remaining card points of the game
        remaining_points = (120 - sum([player.sum_trick_values() for player in self.players]))

        if current_player is Player.Type.DECLARER:
            # if the declarer surrenders...
            # ...the declarer receives no reward...
            current_player.current_trick_points = 0

            # ...and the points are given to the defenders
            for player in self.players:
                if player is not current_player:
                    # both players of the counterparty receive the reward
                    player.current_trick_points += remaining_points
        else:
            # if the defenders surrender...
            # ...the declarer receives the rest of the card points as a reward
            current_player.current_trick_points += remaining_points

            # ...and the defenders receive no reward
            for player in self.players:
                if player is not current_player:
                    # both players of the counterparty receive the reward
                    player.current_trick_points = 0


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

    def get_open_cards(self):
        return [entry[1] for entry in self.stack]
