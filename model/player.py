from enum import Enum


class Player:
    class Type(Enum):
        DECLARER = 0
        DEFENDER = 1

    def __init__(self, id, name):
        self.id = id
        self.name = name
        self.type = None
        self.cards = list()
        self.trick_stack = dict()  # {game_round: [(player, card), ...]}
        self.current_trick_points = 0

    def sum_trick_values(self):
        sum_tricks = 0
        for trick in self.trick_stack.values():
            for entry in trick:
                sum_tricks += entry[1].get_value()

        return sum_tricks

    def has_card(self, card):
        return card in self.cards

    def has_suit(self, suit):
        for card in self.cards:
            if card.face is not card.face.JACK and card.suit is suit:
                return True

        return False

    def has_face(self, face):
        for card in self.cards:
            if card.face is face:
                return True

        return False

    def set_cards(self, cards):
        self.cards = cards

    def get_id(self):
        return self.id

    def __repr__(self):
        return "id=" + str(id) + "name=" + self.name + " cards=" + str(self.cards)

    def __eq__(self, other):
        return other is not None and self.id is other.id

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.id, self.name))
