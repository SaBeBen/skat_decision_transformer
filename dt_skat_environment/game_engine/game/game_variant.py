from abc import ABC, abstractmethod
from dt_skat_environment.game_engine.model.card import Card


# ------------------------------------------------------------
# Abstract game variant class
# ------------------------------------------------------------
class GameVariant(ABC):

    def __init__(self, hand=False, schneider_called=False, schwarz_called=False, ouvert=False):
        self.hand = hand
        self.scheider_called = schneider_called | schwarz_called | ouvert
        self.schwarz_called = schwarz_called | ouvert
        self.ouvert = ouvert

    @abstractmethod
    def compare_jacks(self, jack_higher, jack_lower):
        raise NotImplementedError()

    @abstractmethod
    def compare_cards(self, card_higher, card_lower):
        raise NotImplementedError()

    @abstractmethod
    def is_trump(self, card):
        raise NotImplementedError()

    @abstractmethod
    def has_trump(self, player):
        raise NotImplementedError()

    @abstractmethod
    def get_trump(self):
        raise NotImplementedError()

    @abstractmethod
    def get_level(self):
        raise NotImplementedError()

    @abstractmethod
    def get_variant_name(self):
        raise NotImplementedError()

    def effective_suit(self, card):
        # trumps form an own suit, all other cards belong to their printed suit
        return "trump" if self.is_trump(card) else card.suit

    def follows_suit(self, card, led_card):
        return self.effective_suit(card) == self.effective_suit(led_card)

    def get_highest_card(self, cards):
        highest_card = None
        for card in cards:
            if highest_card is None or self.compare_cards(card, highest_card) == 1:
                highest_card = card

        return highest_card


# ------------------------------------------------------------
# Concrete game variant class for grand game
# ------------------------------------------------------------
class GameVariantGrand(GameVariant):

    def __init__(self, peaks=1, hand=False, schneider_called=False, schwarz_called=False, ouvert=False):
        super().__init__(hand=hand, schneider_called=schneider_called, schwarz_called=schwarz_called, ouvert=ouvert)
        # number of matadors "with" or "without"
        self.peaks = peaks

    def compare_jacks(self, jack_higher, jack_lower):
        if jack_higher.face is not Card.Face.JACK:
            raise TypeError(jack_higher + " is no jack")
        elif jack_lower.face is not Card.Face.JACK:
            raise TypeError(jack_lower + " is no jack")

        if jack_higher.suit.value > jack_lower.suit.value:
            return 1
        elif jack_higher.suit.value < jack_lower.suit.value:
            return -1
        else:
            return 0

    def compare_cards(self, card_higher, card_lower):
        if card_higher.face is Card.Face.JACK and card_lower.face is not Card.Face.JACK:
            return 1
        elif card_higher.face is not Card.Face.JACK and card_lower.face is Card.Face.JACK:
            return -1
        elif card_higher.face is Card.Face.JACK and card_lower.face is Card.Face.JACK:
            return self.compare_jacks(card_higher, card_lower)
        elif card_higher.suit is not card_lower.suit:
            return 0
        # elif card_higher.face is Card.Face.TEN and card_lower.face is not Card.Face.TEN:
        #     return 1 if card_lower.face is not Card.Face.ACE else -1
        # elif card_higher.face is not Card.Face.TEN and card_lower.face is Card.Face.TEN:
        #     return 1 if card_higher.face is Card.Face.ACE else -1
        elif card_higher.face.value > card_lower.face.value:
            return 1
        elif card_higher.face.value < card_lower.face.value:
            return -1
        else:
            return 0

    def is_trump(self, card):
        return card.face is Card.Face.JACK

    def has_trump(self, player):
        return player.has_face(Card.Face.JACK)

    def get_trump(self):
        return 24

    def get_level(self):
        return self.peaks + self.ouvert + self.schwarz_called + self.scheider_called + self.hand

    def get_variant_name(self):
        return "Grand"

    def has_called_schneider(self):
        return self.scheider_called

    def has_called_schwarz(self):
        return self.schwarz_called


# ------------------------------------------------------------
# Concrete game variant class for suit game
# ------------------------------------------------------------
class GameVariantSuit(GameVariantGrand):
    def __init__(self, trump_suit, peaks=1, hand=False, schneider_called=False, schwarz_called=False, ouvert=False):
        # expects trump suit
        super().__init__(peaks=peaks, hand=hand, schneider_called=schneider_called, schwarz_called=schwarz_called,
                         ouvert=ouvert)
        self.trump_suit = trump_suit

    def compare_cards(self, card_higher, card_lower):
        if self.is_trump(card_higher) and not self.is_trump(card_lower):
            return 1
        elif not self.is_trump(card_higher) and self.is_trump(card_lower):
            return -1
        else:
            return GameVariantGrand.compare_cards(self, card_higher, card_lower)

    def is_trump(self, card):
        return card.suit is self.trump_suit or card.face is Card.Face.JACK

    def has_trump(self, player):
        return player.has_face(Card.Face.JACK) or player.has_suit(self.trump_suit)

    def get_trump(self):
        return self.trump_suit

    def get_variant_name(self):
        return self.trump_suit.name



# ------------------------------------------------------------
# Concrete game variant class for null game
# ------------------------------------------------------------
class GameVariantNull(GameVariant):
    # order of the faces in a null game: 7, 8, 9, 10, J, Q, K, A
    NULL_ORDER = [Card.Face.SEVEN, Card.Face.EIGHT, Card.Face.NINE, Card.Face.TEN, Card.Face.JACK, Card.Face.QUEEN,
                  Card.Face.KING, Card.Face.ACE]

    def __init__(self, hand=False, ouvert=False):
        GameVariant.__init__(self, hand=hand, ouvert=ouvert)
        # schneider and schwarz do not exist in null games
        self.scheider_called = False
        self.schwarz_called = False

    def compare_jacks(self, jack_higher, jack_lower):
        if jack_higher.face is not Card.Face.JACK:
            raise TypeError(jack_higher + " is no jack")
        elif jack_lower.face is not Card.Face.JACK:
            raise TypeError(jack_lower + " is no jack")

        return 0

    def compare_cards(self, card_higher, card_lower):
        # there are no trumps, only cards of the same suit can be compared
        if card_higher.suit is not card_lower.suit:
            return 0
        rank_higher = self.NULL_ORDER.index(card_higher.face)
        rank_lower = self.NULL_ORDER.index(card_lower.face)
        if rank_higher > rank_lower:
            return 1
        elif rank_higher < rank_lower:
            return -1
        else:
            return 0

    def get_value(self):
        if self.hand:
            return 59 if self.ouvert else 35
        return 46 if self.ouvert else 23

    def has_called_schneider(self):
        return False

    def has_called_schwarz(self):
        return False

    def is_trump(self, card):
        return False

    def has_trump(self, player):
        return False

    def get_trump(self):
        return 0

    def get_level(self):
        return self.ouvert + self.hand

    def get_variant_name(self):
        return "Null"
