from enum import Enum


class Card:
    class Face(Enum):
        SEVEN = 1
        EIGHT = 2
        NINE = 3
        QUEEN = 4
        KING = 5
        TEN = 6
        ACE = 7
        JACK = 8

        @staticmethod
        def to_display(suit):
            if suit is Card.Face.SEVEN:
                return str(7)
            elif suit is Card.Face.EIGHT:
                return str(8)
            elif suit is Card.Face.NINE:
                return str(9)
            elif suit is Card.Face.TEN:
                return str(10)
            elif suit is Card.Face.JACK:
                return "J"
            elif suit is Card.Face.QUEEN:
                return "Q"
            elif suit is Card.Face.KING:
                return "K"
            elif suit is Card.Face.ACE:
                return "A"

    class FaceNull(Enum):
        SEVEN = 1
        EIGHT = 2
        NINE = 3
        TEN = 4
        JACK = 5
        QUEEN = 6
        KING = 7
        ACE = 8

        @staticmethod
        def to_display(suit):
            if suit is Card.Face.SEVEN:
                return str(7)
            elif suit is Card.Face.EIGHT:
                return str(8)
            elif suit is Card.Face.NINE:
                return str(9)
            elif suit is Card.Face.TEN:
                return str(10)
            elif suit is Card.Face.JACK:
                return "J"
            elif suit is Card.Face.QUEEN:
                return "Q"
            elif suit is Card.Face.KING:
                return "K"
            elif suit is Card.Face.ACE:
                return "A"

    class Suit(Enum):
        DIAMOND = 0
        HEARTS = 1
        SPADE = 2
        CLUB = 3

        @staticmethod
        def to_display(suit):
            if suit is Card.Suit.DIAMOND:
                return "\u2666"
            elif suit is Card.Suit.HEARTS:
                return "\u2665"
            elif suit is Card.Suit.SPADE:
                return "\u2660"
            elif suit is Card.Suit.CLUB:
                return "\u2663"

    def get_value(self):
        if self.face is Card.Face.JACK:
            return 2
        elif self.face is Card.Face.ACE:
            return 11
        elif self.face is Card.Face.TEN:
            return 10
        elif self.face is Card.Face.KING:
            return 4
        elif self.face is Card.Face.QUEEN:
            return 3
        else:
            return 0

    def __init__(self, suit, face):
        self.suit = suit
        self.face = face

    def __repr__(self):
        return Card.Face.to_display(self.face) + Card.Suit.to_display(self.suit)

    def __eq__(self, other):
        return other is not None and self.suit is other.suit and self.face is other.face

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.suit, self.face))

    def __lt__(self, other):
        if self.suit is other.suit:
            return self.face.value < other.face.value
        else:
            return self.suit.value < other.suit.value

    def gt_for_peaks(self, other, trump_suit):
        # comparison to sort the Js to the left of the hand followed by the trump suit
        if self.face.value == 8:
            if other.face.value == 8:
                return self.suit.value > other.suit.value
            return True
        if other.face.value == 8:
            return False
        if self.suit.value == trump_suit:
            if other.suit.value == trump_suit:
                return self.face.value > other.face.value
            return True
        if other.suit == trump_suit:
            return False
        if self.suit is other.suit:
            return self.face.value > other.face.value
        else:
            return self.suit.value > other.suit.value
