from model.card import Card


def convert_card_to_enc(card, encoding):
    if encoding == "one-hot" or encoding == "one-hot_comp":
        return convert_card_to_one_hot(card)
    if encoding == "mixed" or encoding == "mixed_comp":
        return convert_card_to_vector(card)
    else:
        raise NotImplementedError(f"The encoding {encoding} is not supported.")


def convert_numerical_to_enc(card, encoding):
    if encoding == "one-hot" or encoding == "one-hot_comp":
        return convert_numerical_to_one_hot(card)
    if encoding == "mixed" or encoding == "mixed_comp":
        return convert_numerical_to_vector(card)
    else:
        raise NotImplementedError(f"The encoding {encoding} is not supported.")


def convert_tuple_to_card(card):
    tuple_rep = {
        (0, 0, 0, 1, 7): Card(Card.Suit.CLUB, Card.Face.ACE),  # A♣
        (0, 0, 0, 1, 5): Card(Card.Suit.CLUB, Card.Face.KING),  # K♣
        (0, 0, 0, 1, 4): Card(Card.Suit.CLUB, Card.Face.QUEEN),  # Q♣
        (0, 0, 0, 1, 8): Card(Card.Suit.CLUB, Card.Face.JACK),  # J♣
        (0, 0, 0, 1, 6): Card(Card.Suit.CLUB, Card.Face.TEN),  # 10♣
        (0, 0, 0, 1, 3): Card(Card.Suit.CLUB, Card.Face.NINE),  # 9♣
        (0, 0, 0, 1, 2): Card(Card.Suit.CLUB, Card.Face.EIGHT),  # 8♣
        (0, 0, 0, 1, 1): Card(Card.Suit.CLUB, Card.Face.SEVEN),  # 7♣
        (0, 0, 0, 1, 7): Card(Card.Suit.SPADE, Card.Face.ACE),  # A♠
        (0, 0, 0, 1, 5): Card(Card.Suit.SPADE, Card.Face.KING),  # K♠
        (0, 0, 1, 0, 4): Card(Card.Suit.SPADE, Card.Face.QUEEN),  # Q♠
        (0, 0, 1, 0, 8): Card(Card.Suit.SPADE, Card.Face.JACK),  # J♠
        (0, 0, 1, 0, 6): Card(Card.Suit.SPADE, Card.Face.TEN),  # 10♠
        (0, 0, 1, 0, 3): Card(Card.Suit.SPADE, Card.Face.NINE),  # 9♠
        (0, 0, 1, 0, 2): Card(Card.Suit.SPADE, Card.Face.EIGHT),  # 8♠
        (0, 0, 1, 0, 1): Card(Card.Suit.SPADE, Card.Face.SEVEN),  # 7♠
        (0, 1, 0, 0, 7): Card(Card.Suit.HEARTS, Card.Face.ACE),  # A♥
        (0, 1, 0, 0, 5): Card(Card.Suit.HEARTS, Card.Face.KING),  # K♥
        (0, 1, 0, 0, 4): Card(Card.Suit.HEARTS, Card.Face.QUEEN),  # Q♥
        (0, 1, 0, 0, 8): Card(Card.Suit.HEARTS, Card.Face.JACK),  # J♥
        (0, 1, 0, 0, 6): Card(Card.Suit.HEARTS, Card.Face.TEN),  # 10♥
        (0, 1, 0, 0, 3): Card(Card.Suit.HEARTS, Card.Face.NINE),  # 9♥
        (0, 1, 0, 0, 2): Card(Card.Suit.HEARTS, Card.Face.EIGHT),  # 8♥
        (0, 1, 0, 0, 1): Card(Card.Suit.HEARTS, Card.Face.SEVEN),  # 7♥
        (1, 0, 0, 0, 7): Card(Card.Suit.DIAMOND, Card.Face.ACE),  # A♦
        (1, 0, 0, 0, 5): Card(Card.Suit.DIAMOND, Card.Face.KING),  # K♦
        (1, 0, 0, 0, 4): Card(Card.Suit.DIAMOND, Card.Face.QUEEN),  # Q♦
        (1, 0, 0, 0, 8): Card(Card.Suit.DIAMOND, Card.Face.JACK),  # J♦
        (1, 0, 0, 0, 6): Card(Card.Suit.DIAMOND, Card.Face.TEN),  # 10♦
        (1, 0, 0, 0, 3): Card(Card.Suit.DIAMOND, Card.Face.NINE),  # 9♦
        (1, 0, 0, 0, 2): Card(Card.Suit.DIAMOND, Card.Face.EIGHT),  # 8♦
        (1, 0, 0, 0, 1): Card(Card.Suit.DIAMOND, Card.Face.SEVEN)  # 7♦
    }
    converted_card = tuple_rep[card]

    return converted_card


def convert_card_to_vector(card):
    vector_rep = {
        Card(Card.Suit.CLUB, Card.Face.ACE): [0, 0, 0, 1, 7],  # A♣
        Card(Card.Suit.CLUB, Card.Face.KING): [0, 0, 0, 1, 5],  # K♣
        Card(Card.Suit.CLUB, Card.Face.QUEEN): [0, 0, 0, 1, 4],  # Q♣
        Card(Card.Suit.CLUB, Card.Face.JACK): [0, 0, 0, 1, 8],  # J♣
        Card(Card.Suit.CLUB, Card.Face.TEN): [0, 0, 0, 1, 6],  # 10♣
        Card(Card.Suit.CLUB, Card.Face.NINE): [0, 0, 0, 1, 3],  # 9♣
        Card(Card.Suit.CLUB, Card.Face.EIGHT): [0, 0, 0, 1, 2],  # 8♣
        Card(Card.Suit.CLUB, Card.Face.SEVEN): [0, 0, 0, 1, 1],  # 7♣
        Card(Card.Suit.SPADE, Card.Face.ACE): [0, 0, 1, 0, 7],  # A♠
        Card(Card.Suit.SPADE, Card.Face.KING): [0, 0, 1, 0, 5],  # K♠
        Card(Card.Suit.SPADE, Card.Face.QUEEN): [0, 0, 1, 0, 4],  # Q♠
        Card(Card.Suit.SPADE, Card.Face.JACK): [0, 0, 1, 0, 8],  # J♠
        Card(Card.Suit.SPADE, Card.Face.TEN): [0, 0, 1, 0, 6],  # 10♠
        Card(Card.Suit.SPADE, Card.Face.NINE): [0, 0, 1, 0, 3],  # 9♠
        Card(Card.Suit.SPADE, Card.Face.EIGHT): [0, 0, 1, 0, 2],  # 8♠
        Card(Card.Suit.SPADE, Card.Face.SEVEN): [0, 0, 1, 0, 1],  # 7♠
        Card(Card.Suit.HEARTS, Card.Face.ACE): [0, 1, 0, 0, 7],  # A♥
        Card(Card.Suit.HEARTS, Card.Face.KING): [0, 1, 0, 0, 5],  # K♥
        Card(Card.Suit.HEARTS, Card.Face.QUEEN): [0, 1, 0, 0, 4],  # Q♥
        Card(Card.Suit.HEARTS, Card.Face.JACK): [0, 1, 0, 0, 8],  # J♥
        Card(Card.Suit.HEARTS, Card.Face.TEN): [0, 1, 0, 0, 6],  # 10♥
        Card(Card.Suit.HEARTS, Card.Face.NINE): [0, 1, 0, 0, 3],  # 9♥
        Card(Card.Suit.HEARTS, Card.Face.EIGHT): [0, 1, 0, 0, 2],  # 8♥
        Card(Card.Suit.HEARTS, Card.Face.SEVEN): [0, 1, 0, 0, 1],  # 7♥
        Card(Card.Suit.DIAMOND, Card.Face.ACE): [1, 0, 0, 0, 7],  # A♦
        Card(Card.Suit.DIAMOND, Card.Face.KING): [1, 0, 0, 0, 5],  # K♦
        Card(Card.Suit.DIAMOND, Card.Face.QUEEN): [1, 0, 0, 0, 4],  # Q♦
        Card(Card.Suit.DIAMOND, Card.Face.JACK): [1, 0, 0, 0, 8],  # J♦
        Card(Card.Suit.DIAMOND, Card.Face.TEN): [1, 0, 0, 0, 6],  # 10♦
        Card(Card.Suit.DIAMOND, Card.Face.NINE): [1, 0, 0, 0, 3],  # 9♦
        Card(Card.Suit.DIAMOND, Card.Face.EIGHT): [1, 0, 0, 0, 2],  # 8♦
        Card(Card.Suit.DIAMOND, Card.Face.SEVEN): [1, 0, 0, 0, 1]  # 7♦
    }
    converted_card = vector_rep[card]

    return converted_card


def convert_card_to_trump_vector(card, trump_enc):
    vector_rep = {
        Card(Card.Suit.CLUB, Card.Face.ACE): [0, 0, 0, 1, 7, 0],  # A♣
        Card(Card.Suit.CLUB, Card.Face.KING): [0, 0, 0, 1, 5, 0],  # K♣
        Card(Card.Suit.CLUB, Card.Face.QUEEN): [0, 0, 0, 1, 4, 0],  # Q♣
        Card(Card.Suit.CLUB, Card.Face.JACK): [0, 0, 0, 1, 8, 1],  # J♣
        Card(Card.Suit.CLUB, Card.Face.TEN): [0, 0, 0, 1, 6, 0],  # 10♣
        Card(Card.Suit.CLUB, Card.Face.NINE): [0, 0, 0, 1, 3, 0],  # 9♣
        Card(Card.Suit.CLUB, Card.Face.EIGHT): [0, 0, 0, 1, 2, 0],  # 8♣
        Card(Card.Suit.CLUB, Card.Face.SEVEN): [0, 0, 0, 1, 1, 0],  # 7♣
        Card(Card.Suit.SPADE, Card.Face.ACE): [0, 0, 1, 0, 7, 0],  # A♠
        Card(Card.Suit.SPADE, Card.Face.KING): [0, 0, 1, 0, 5, 0],  # K♠
        Card(Card.Suit.SPADE, Card.Face.QUEEN): [0, 0, 1, 0, 4, 0],  # Q♠
        Card(Card.Suit.SPADE, Card.Face.JACK): [0, 0, 1, 0, 8, 1],  # J♠
        Card(Card.Suit.SPADE, Card.Face.TEN): [0, 0, 1, 0, 6, 0],  # 10♠
        Card(Card.Suit.SPADE, Card.Face.NINE): [0, 0, 1, 0, 3, 0],  # 9♠
        Card(Card.Suit.SPADE, Card.Face.EIGHT): [0, 0, 1, 0, 2, 0],  # 8♠
        Card(Card.Suit.SPADE, Card.Face.SEVEN): [0, 0, 1, 0, 1, 0],  # 7♠
        Card(Card.Suit.HEARTS, Card.Face.ACE): [0, 1, 0, 0, 7, 0],  # A♥
        Card(Card.Suit.HEARTS, Card.Face.KING): [0, 1, 0, 0, 5, 0],  # K♥
        Card(Card.Suit.HEARTS, Card.Face.QUEEN): [0, 1, 0, 0, 4, 0],  # Q♥
        Card(Card.Suit.HEARTS, Card.Face.JACK): [0, 1, 0, 0, 8, 1],  # J♥
        Card(Card.Suit.HEARTS, Card.Face.TEN): [0, 1, 0, 0, 6, 0],  # 10♥
        Card(Card.Suit.HEARTS, Card.Face.NINE): [0, 1, 0, 0, 3, 0],  # 9♥
        Card(Card.Suit.HEARTS, Card.Face.EIGHT): [0, 1, 0, 0, 2, 0],  # 8♥
        Card(Card.Suit.HEARTS, Card.Face.SEVEN): [0, 1, 0, 0, 1, 0],  # 7♥
        Card(Card.Suit.DIAMOND, Card.Face.ACE): [1, 0, 0, 0, 7, 0],  # A♦
        Card(Card.Suit.DIAMOND, Card.Face.KING): [1, 0, 0, 0, 5, 0],  # K♦
        Card(Card.Suit.DIAMOND, Card.Face.QUEEN): [1, 0, 0, 0, 4, 0],  # Q♦
        Card(Card.Suit.DIAMOND, Card.Face.JACK): [1, 0, 0, 0, 8, 1],  # J♦
        Card(Card.Suit.DIAMOND, Card.Face.TEN): [1, 0, 0, 0, 6, 0],  # 10♦
        Card(Card.Suit.DIAMOND, Card.Face.NINE): [1, 0, 0, 0, 3, 0],  # 9♦
        Card(Card.Suit.DIAMOND, Card.Face.EIGHT): [1, 0, 0, 0, 2, 0],  # 8♦
        Card(Card.Suit.DIAMOND, Card.Face.SEVEN): [1, 0, 0, 0, 1, 0]  # 7♦
    }
    converted_card = vector_rep[card]

    if converted_card[:4] == trump_enc:
        converted_card[-1] = 1

    return converted_card


def convert_card_to_one_hot(card):
    vector_rep = {
        Card(Card.Suit.CLUB, Card.Face.ACE): [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],  # A♣
        Card(Card.Suit.CLUB, Card.Face.KING): [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],  # K♣
        Card(Card.Suit.CLUB, Card.Face.QUEEN): [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # Q♣
        Card(Card.Suit.CLUB, Card.Face.JACK): [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],  # J♣
        Card(Card.Suit.CLUB, Card.Face.TEN): [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # 10♣
        Card(Card.Suit.CLUB, Card.Face.NINE): [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],  # 9♣
        Card(Card.Suit.CLUB, Card.Face.EIGHT): [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # 8♣
        Card(Card.Suit.CLUB, Card.Face.SEVEN): [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # 7♣
        Card(Card.Suit.SPADE, Card.Face.ACE): [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # A♠
        Card(Card.Suit.SPADE, Card.Face.KING): [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # K♠
        Card(Card.Suit.SPADE, Card.Face.QUEEN): [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Q♠
        Card(Card.Suit.SPADE, Card.Face.JACK): [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # J♠
        Card(Card.Suit.SPADE, Card.Face.TEN): [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 10♠
        Card(Card.Suit.SPADE, Card.Face.NINE): [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 9♠
        Card(Card.Suit.SPADE, Card.Face.EIGHT): [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 8♠
        Card(Card.Suit.SPADE, Card.Face.SEVEN): [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 7♠
        Card(Card.Suit.HEARTS, Card.Face.ACE): [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # A♥
        Card(Card.Suit.HEARTS, Card.Face.KING): [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # K♥
        Card(Card.Suit.HEARTS, Card.Face.QUEEN): [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Q♥
        Card(Card.Suit.HEARTS, Card.Face.JACK): [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # J♥
        Card(Card.Suit.HEARTS, Card.Face.TEN): [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 10♥
        Card(Card.Suit.HEARTS, Card.Face.NINE): [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 9♥
        Card(Card.Suit.HEARTS, Card.Face.EIGHT): [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 8♥
        Card(Card.Suit.HEARTS, Card.Face.SEVEN): [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 7♥
        Card(Card.Suit.DIAMOND, Card.Face.ACE): [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # A♦
        Card(Card.Suit.DIAMOND, Card.Face.KING): [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # K♦
        Card(Card.Suit.DIAMOND, Card.Face.QUEEN): [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Q♦
        Card(Card.Suit.DIAMOND, Card.Face.JACK): [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # J♦
        Card(Card.Suit.DIAMOND, Card.Face.TEN): [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 10♦
        Card(Card.Suit.DIAMOND, Card.Face.NINE): [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 9♦
        Card(Card.Suit.DIAMOND, Card.Face.EIGHT): [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 8♦
        Card(Card.Suit.DIAMOND, Card.Face.SEVEN): [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  # 7♦
    }
    converted_card = vector_rep[card]

    return converted_card


def convert_card_to_numerical(card):
    card_rep = {
        Card(Card.Suit.CLUB, Card.Face.ACE): 1,  # A♣
        Card(Card.Suit.CLUB, Card.Face.KING): 2,  # K♣
        Card(Card.Suit.CLUB, Card.Face.QUEEN): 3,  # Q♣
        Card(Card.Suit.CLUB, Card.Face.JACK): 4,  # J♣
        Card(Card.Suit.CLUB, Card.Face.TEN): 5,  # 10♣
        Card(Card.Suit.CLUB, Card.Face.NINE): 6,  # 9♣
        Card(Card.Suit.CLUB, Card.Face.EIGHT): 7,  # 8♣
        Card(Card.Suit.CLUB, Card.Face.SEVEN): 8,  # 7♣
        Card(Card.Suit.SPADE, Card.Face.ACE): 9,  # A♠
        Card(Card.Suit.SPADE, Card.Face.KING): 10,  # K♠
        Card(Card.Suit.SPADE, Card.Face.QUEEN): 11,  # Q♠
        Card(Card.Suit.SPADE, Card.Face.JACK): 12,  # J♠
        Card(Card.Suit.SPADE, Card.Face.TEN): 13,  # 10♠
        Card(Card.Suit.SPADE, Card.Face.NINE): 14,  # 9♠
        Card(Card.Suit.SPADE, Card.Face.EIGHT): 15,  # 8♠
        Card(Card.Suit.SPADE, Card.Face.SEVEN): 16,  # 7♠
        Card(Card.Suit.HEARTS, Card.Face.ACE): 17,  # A♥
        Card(Card.Suit.HEARTS, Card.Face.KING): 18,  # K♥
        Card(Card.Suit.HEARTS, Card.Face.QUEEN): 19,  # Q♥
        Card(Card.Suit.HEARTS, Card.Face.JACK): 20,  # J♥
        Card(Card.Suit.HEARTS, Card.Face.TEN): 21,  # 10♥
        Card(Card.Suit.HEARTS, Card.Face.NINE): 22,  # 9♥
        Card(Card.Suit.HEARTS, Card.Face.EIGHT): 23,  # 8♥
        Card(Card.Suit.HEARTS, Card.Face.SEVEN): 24,  # 7♥
        Card(Card.Suit.DIAMOND, Card.Face.ACE): 25,  # A♦
        Card(Card.Suit.DIAMOND, Card.Face.KING): 26,  # K♦
        Card(Card.Suit.DIAMOND, Card.Face.QUEEN): 27,  # Q♦
        Card(Card.Suit.DIAMOND, Card.Face.JACK): 28,  # J♦
        Card(Card.Suit.DIAMOND, Card.Face.TEN): 29,  # 10♦
        Card(Card.Suit.DIAMOND, Card.Face.NINE): 30,  # 9♦
        Card(Card.Suit.DIAMOND, Card.Face.EIGHT): 31,  # 8♦
        Card(Card.Suit.DIAMOND, Card.Face.SEVEN): 32  # 7♦
    }
    converted_card = card_rep[card]

    return converted_card


def convert_numerical_to_vector(card):
    # in the beginning, the card values start at 0, but 0s are used to pad the states -> need for other representation
    vector_rep = {
        0: [0, 0, 0, 1, 7],  # A♣
        1: [0, 0, 0, 1, 5],  # K♣
        2: [0, 0, 0, 1, 4],  # Q♣
        3: [0, 0, 0, 1, 8],  # J♣
        4: [0, 0, 0, 1, 6],  # 10♣
        5: [0, 0, 0, 1, 3],  # 9♣
        6: [0, 0, 0, 1, 2],  # 8♣
        7: [0, 0, 0, 1, 1],  # 7♣
        8: [0, 0, 1, 0, 7],  # A♠
        9: [0, 0, 1, 0, 5],  # K♠
        10: [0, 0, 1, 0, 4],  # Q♠
        11: [0, 0, 1, 0, 8],  # J♠
        12: [0, 0, 1, 0, 6],  # 10♠
        13: [0, 0, 1, 0, 3],  # 9♠
        14: [0, 0, 1, 0, 2],  # 8♠
        15: [0, 0, 1, 0, 1],  # 7♠
        16: [0, 1, 0, 0, 7],  # A♥
        17: [0, 1, 0, 0, 5],  # K♥
        18: [0, 1, 0, 0, 4],  # Q♥
        19: [0, 1, 0, 0, 8],  # J♥
        20: [0, 1, 0, 0, 6],  # 10♥
        21: [0, 1, 0, 0, 3],  # 9♥
        22: [0, 1, 0, 0, 2],  # 8♥
        23: [0, 1, 0, 0, 1],  # 7♥
        24: [1, 0, 0, 0, 7],  # A♦
        25: [1, 0, 0, 0, 5],  # K♦
        26: [1, 0, 0, 0, 4],  # Q♦
        27: [1, 0, 0, 0, 8],  # J♦
        28: [1, 0, 0, 0, 6],  # 10♦
        29: [1, 0, 0, 0, 3],  # 9♦
        30: [1, 0, 0, 0, 2],  # 8♦
        31: [1, 0, 0, 0, 1]  # 7♦
    }
    converted_card = vector_rep[card]

    return converted_card


def convert_numerical_to_trump_vector(card, trump_enc):
    # in the beginning, the card values start at 0, but 0s are used to pad the states -> need for other representation
    vector_rep = {
        0: [0, 0, 0, 1, 7, 0],  # A♣
        1: [0, 0, 0, 1, 5, 0],  # K♣
        2: [0, 0, 0, 1, 4, 0],  # Q♣
        3: [0, 0, 0, 1, 8, 1],  # J♣
        4: [0, 0, 0, 1, 6, 0],  # 10♣
        5: [0, 0, 0, 1, 3, 0],  # 9♣
        6: [0, 0, 0, 1, 2, 0],  # 8♣
        7: [0, 0, 0, 1, 1, 0],  # 7♣
        8: [0, 0, 1, 0, 7, 0],  # A♠
        9: [0, 0, 1, 0, 5, 0],  # K♠
        10: [0, 0, 1, 0, 4, 0],  # Q♠
        11: [0, 0, 1, 0, 8, 1],  # J♠
        12: [0, 0, 1, 0, 6, 0],  # 10♠
        13: [0, 0, 1, 0, 3, 0],  # 9♠
        14: [0, 0, 1, 0, 2, 0],  # 8♠
        15: [0, 0, 1, 0, 1, 0],  # 7♠
        16: [0, 1, 0, 0, 7, 0],  # A♥
        17: [0, 1, 0, 0, 5, 0],  # K♥
        18: [0, 1, 0, 0, 4, 0],  # Q♥
        19: [0, 1, 0, 0, 8, 1],  # J♥
        20: [0, 1, 0, 0, 6, 0],  # 10♥
        21: [0, 1, 0, 0, 3, 0],  # 9♥
        22: [0, 1, 0, 0, 2, 0],  # 8♥
        23: [0, 1, 0, 0, 1, 0],  # 7♥
        24: [1, 0, 0, 0, 7, 0],  # A♦
        25: [1, 0, 0, 0, 5, 0],  # K♦
        26: [1, 0, 0, 0, 4, 0],  # Q♦
        27: [1, 0, 0, 0, 8, 1],  # J♦
        28: [1, 0, 0, 0, 6, 0],  # 10♦
        29: [1, 0, 0, 0, 3, 0],  # 9♦
        30: [1, 0, 0, 0, 2, 0],  # 8♦
        31: [1, 0, 0, 0, 1, 0]  # 7♦
    }
    converted_card = vector_rep[card]

    if converted_card[:4] == trump_enc:
        converted_card[-1] = 1

    return converted_card


def convert_numerical_to_one_hot(card):
    # first 4 cat enc are used for suit, the other 8 for the face
    vector_rep = {
        0: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],  # A♣
        1: [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],  # K♣
        2: [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # Q♣
        3: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],  # J♣
        4: [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # 10♣
        5: [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],  # 9♣
        6: [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # 8♣
        7: [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # 7♣
        8: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # A♠
        9: [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # K♠
        10: [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Q♠
        11: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # J♠
        12: [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 10♠
        13: [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 9♠
        14: [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 8♠
        15: [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 7♠
        16: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # A♥
        17: [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # K♥
        18: [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Q♥
        19: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # J♥
        20: [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 10♥
        21: [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 9♥
        22: [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 8♥
        23: [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 7♥
        24: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # A♦
        25: [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # K♦
        26: [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Q♦
        27: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # J♦
        28: [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 10♦
        29: [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 9♦
        30: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 8♦
        31: [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  # 7♦
    }
    converted_card = vector_rep[card]

    return converted_card


def convert_numerical_to_tuple(card):
    # in the beginning, the card values start at 0, but 0s are used to pad the states -> need for other representation
    tuple_rep = {
        0: (0, 0, 0, 1, 7),  # A♣
        1: (0, 0, 0, 1, 5),  # K♣
        2: (0, 0, 0, 1, 4),  # Q♣
        3: (0, 0, 0, 1, 8),  # J♣
        4: (0, 0, 0, 1, 6),  # 10♣
        5: (0, 0, 0, 1, 3),  # 9♣
        6: (0, 0, 0, 1, 2),  # 8♣
        7: (0, 0, 0, 1, 1),  # 7♣
        8: (0, 0, 1, 0, 7),  # A♠
        9: (0, 0, 1, 0, 5),  # K♠
        10: (0, 0, 1, 0, 4),  # Q♠
        11: (0, 0, 1, 0, 8),  # J♠
        12: (0, 0, 1, 0, 6),  # 10♠
        13: (0, 0, 1, 0, 3),  # 9♠
        14: (0, 0, 1, 0, 2),  # 8♠
        15: (0, 0, 1, 0, 1),  # 7♠
        16: (0, 1, 0, 0, 7),  # A♥
        17: (0, 1, 0, 0, 5),  # K♥
        18: (0, 1, 0, 0, 4),  # Q♥
        19: (0, 1, 0, 0, 8),  # J♥
        20: (0, 1, 0, 0, 6),  # 10♥
        21: (0, 1, 0, 0, 3),  # 9♥
        22: (0, 1, 0, 0, 2),  # 8♥
        23: (0, 1, 0, 0, 1),  # 7♥
        24: (1, 0, 0, 0, 7),  # A♦
        25: (1, 0, 0, 0, 5),  # K♦
        26: (1, 0, 0, 0, 4),  # Q♦
        27: (1, 0, 0, 0, 8),  # J♦
        28: (1, 0, 0, 0, 6),  # 10♦
        29: (1, 0, 0, 0, 3),  # 9♦
        30: (1, 0, 0, 0, 2),  # 8♦
        31: (1, 0, 0, 0, 1)  # 7♦
    }
    converted_card = tuple_rep[card]

    return converted_card


def convert_card_to_tuple(card):
    tuple_rep = {
        Card(Card.Suit.CLUB, Card.Face.ACE): (0, 0, 0, 1, 7),  # A♣
        Card(Card.Suit.CLUB, Card.Face.KING): (0, 0, 0, 1, 5),  # K♣
        Card(Card.Suit.CLUB, Card.Face.QUEEN): (0, 0, 0, 1, 4),  # Q♣
        Card(Card.Suit.CLUB, Card.Face.JACK): (0, 0, 0, 1, 8),  # J♣
        Card(Card.Suit.CLUB, Card.Face.TEN): (0, 0, 0, 1, 6),  # 10♣
        Card(Card.Suit.CLUB, Card.Face.NINE): (0, 0, 0, 1, 3),  # 9♣
        Card(Card.Suit.CLUB, Card.Face.EIGHT): (0, 0, 0, 1, 2),  # 8♣
        Card(Card.Suit.CLUB, Card.Face.SEVEN): (0, 0, 0, 1, 1),  # 7♣
        Card(Card.Suit.SPADE, Card.Face.ACE): (0, 0, 1, 0, 7),  # A♠
        Card(Card.Suit.SPADE, Card.Face.KING): (0, 0, 1, 0, 5),  # K♠
        Card(Card.Suit.SPADE, Card.Face.QUEEN): (0, 0, 1, 0, 4),  # Q♠
        Card(Card.Suit.SPADE, Card.Face.JACK): (0, 0, 1, 0, 8),  # J♠
        Card(Card.Suit.SPADE, Card.Face.TEN): (0, 0, 1, 0, 6),  # 10♠
        Card(Card.Suit.SPADE, Card.Face.NINE): (0, 0, 1, 0, 3),  # 9♠
        Card(Card.Suit.SPADE, Card.Face.EIGHT): (0, 0, 1, 0, 2),  # 8♠
        Card(Card.Suit.SPADE, Card.Face.SEVEN): (0, 0, 1, 0, 1),  # 7♠
        Card(Card.Suit.HEARTS, Card.Face.ACE): (0, 1, 0, 0, 7),  # A♥
        Card(Card.Suit.HEARTS, Card.Face.KING): (0, 1, 0, 0, 5),  # K♥
        Card(Card.Suit.HEARTS, Card.Face.QUEEN): (0, 1, 0, 0, 4),  # Q♥
        Card(Card.Suit.HEARTS, Card.Face.JACK): (0, 1, 0, 0, 8),  # J♥
        Card(Card.Suit.HEARTS, Card.Face.TEN): (0, 1, 0, 0, 6),  # 10♥
        Card(Card.Suit.HEARTS, Card.Face.NINE): (0, 1, 0, 0, 3),  # 9♥
        Card(Card.Suit.HEARTS, Card.Face.EIGHT): (0, 1, 0, 0, 2),  # 8♥
        Card(Card.Suit.HEARTS, Card.Face.SEVEN): (0, 1, 0, 0, 1),  # 7♥
        Card(Card.Suit.DIAMOND, Card.Face.ACE): (1, 0, 0, 0, 7),  # A♦
        Card(Card.Suit.DIAMOND, Card.Face.KING): (1, 0, 0, 0, 5),  # K♦
        Card(Card.Suit.DIAMOND, Card.Face.QUEEN): (1, 0, 0, 0, 4),  # Q♦
        Card(Card.Suit.DIAMOND, Card.Face.JACK): (1, 0, 0, 0, 8),  # J♦
        Card(Card.Suit.DIAMOND, Card.Face.TEN): (1, 0, 0, 0, 6),  # 10♦
        Card(Card.Suit.DIAMOND, Card.Face.NINE): (1, 0, 0, 0, 3),  # 9♦
        Card(Card.Suit.DIAMOND, Card.Face.EIGHT): (1, 0, 0, 0, 2),  # 8♦
        Card(Card.Suit.DIAMOND, Card.Face.SEVEN): (1, 0, 0, 0, 1)  # 7♦
    }
    converted_card = tuple_rep[card]

    return converted_card


# def convert_one_hot_to_card(card):
#     vector_rep = {
#         1: Card(Card.Suit.CLUB, Card.Face.ACE),  # A♣
#         2: Card(Card.Suit.CLUB, Card.Face.KING),  # K♣
#         3: Card(Card.Suit.CLUB, Card.Face.QUEEN),  # Q♣
#         4: Card(Card.Suit.CLUB, Card.Face.JACK),  # J♣
#         5: Card(Card.Suit.CLUB, Card.Face.TEN),  # 10♣
#         6: Card(Card.Suit.CLUB, Card.Face.NINE),  # 9♣
#         7: Card(Card.Suit.CLUB, Card.Face.EIGHT),  # 8♣
#         8: Card(Card.Suit.CLUB, Card.Face.SEVEN),  # 7♣
#         9: Card(Card.Suit.SPADE, Card.Face.ACE),  # A♠
#         10: Card(Card.Suit.SPADE, Card.Face.KING),  # K♠
#         11: Card(Card.Suit.SPADE, Card.Face.QUEEN),  # Q♠
#         12: Card(Card.Suit.SPADE, Card.Face.JACK),  # J♠
#         13: Card(Card.Suit.SPADE, Card.Face.TEN),  # 10♠
#         14: Card(Card.Suit.SPADE, Card.Face.NINE),  # 9♠
#         15: Card(Card.Suit.SPADE, Card.Face.EIGHT),  # 8♠
#         16: Card(Card.Suit.SPADE, Card.Face.SEVEN),  # 7♠
#         17: Card(Card.Suit.HEARTS, Card.Face.ACE),  # A♥
#         18: Card(Card.Suit.HEARTS, Card.Face.KING),  # K♥
#         19: Card(Card.Suit.HEARTS, Card.Face.QUEEN),  # Q♥
#         20: Card(Card.Suit.HEARTS, Card.Face.JACK),  # J♥
#         21: Card(Card.Suit.HEARTS, Card.Face.TEN),  # 10♥
#         22: Card(Card.Suit.HEARTS, Card.Face.NINE),  # 9♥
#         23: Card(Card.Suit.HEARTS, Card.Face.EIGHT),  # 8♥
#         24: Card(Card.Suit.HEARTS, Card.Face.SEVEN),  # 7♥
#         25: Card(Card.Suit.DIAMOND, Card.Face.ACE),  # A♦
#         26: Card(Card.Suit.DIAMOND, Card.Face.KING),  # K♦
#         27: Card(Card.Suit.DIAMOND, Card.Face.QUEEN),  # Q♦
#         28: Card(Card.Suit.DIAMOND, Card.Face.JACK),  # J♦
#         29: Card(Card.Suit.DIAMOND, Card.Face.TEN),  # 10♦
#         30: Card(Card.Suit.DIAMOND, Card.Face.NINE),  # 9♦
#         31: Card(Card.Suit.DIAMOND, Card.Face.EIGHT),  # 8♦
#         32: Card(Card.Suit.DIAMOND, Card.Face.SEVEN)  # 7♦
#     }
#     converted_card = vector_rep[card]
#
#     return converted_card


def convert_numerical_to_card(card):
    # in the beginning, the card values start at 0, but 0s are used to pad the states -> need for other representation
    vector_rep = {
        0: Card(Card.Suit.CLUB, Card.Face.ACE),  # A♣
        1: Card(Card.Suit.CLUB, Card.Face.KING),  # K♣
        2: Card(Card.Suit.CLUB, Card.Face.QUEEN),  # Q♣
        3: Card(Card.Suit.CLUB, Card.Face.JACK),  # J♣
        4: Card(Card.Suit.CLUB, Card.Face.TEN),  # 10♣
        5: Card(Card.Suit.CLUB, Card.Face.NINE),  # 9♣
        6: Card(Card.Suit.CLUB, Card.Face.EIGHT),  # 8♣
        7: Card(Card.Suit.CLUB, Card.Face.SEVEN),  # 7♣
        8: Card(Card.Suit.SPADE, Card.Face.ACE),  # A♠
        9: Card(Card.Suit.SPADE, Card.Face.KING),  # K♠
        10: Card(Card.Suit.SPADE, Card.Face.QUEEN),  # Q♠
        11: Card(Card.Suit.SPADE, Card.Face.JACK),  # J♠
        12: Card(Card.Suit.SPADE, Card.Face.TEN),  # 10♠
        13: Card(Card.Suit.SPADE, Card.Face.NINE),  # 9♠
        14: Card(Card.Suit.SPADE, Card.Face.EIGHT),  # 8♠
        15: Card(Card.Suit.SPADE, Card.Face.SEVEN),  # 7♠
        16: Card(Card.Suit.HEARTS, Card.Face.ACE),  # A♥
        17: Card(Card.Suit.HEARTS, Card.Face.KING),  # K♥
        18: Card(Card.Suit.HEARTS, Card.Face.QUEEN),  # Q♥
        19: Card(Card.Suit.HEARTS, Card.Face.JACK),  # J♥
        20: Card(Card.Suit.HEARTS, Card.Face.TEN),  # 10♥
        21: Card(Card.Suit.HEARTS, Card.Face.NINE),  # 9♥
        22: Card(Card.Suit.HEARTS, Card.Face.EIGHT),  # 8♥
        23: Card(Card.Suit.HEARTS, Card.Face.SEVEN),  # 7♥
        24: Card(Card.Suit.DIAMOND, Card.Face.ACE),  # A♦
        25: Card(Card.Suit.DIAMOND, Card.Face.KING),  # K♦
        26: Card(Card.Suit.DIAMOND, Card.Face.QUEEN),  # Q♦
        27: Card(Card.Suit.DIAMOND, Card.Face.JACK),  # J♦
        28: Card(Card.Suit.DIAMOND, Card.Face.TEN),  # 10♦
        29: Card(Card.Suit.DIAMOND, Card.Face.NINE),  # 9♦
        30: Card(Card.Suit.DIAMOND, Card.Face.EIGHT),  # 8♦
        31: Card(Card.Suit.DIAMOND, Card.Face.SEVEN)  # 7♦
    }
    converted_card = vector_rep[card]

    return converted_card


# the card representation: one token represents one card which is encoded as a vector
# convert data to following encoding:
# ♦, ♥, ♠, ♣, {7, 8, 9, Q, K, 10, A, J}
all_cards = [
    [0, 0, 0, 1, 7],  # A♣
    [0, 0, 0, 1, 5],  # K♣
    [0, 0, 0, 1, 4],  # Q♣
    [0, 0, 0, 1, 8],  # J♣
    [0, 0, 0, 1, 6],  # 10♣
    [0, 0, 0, 1, 3],  # 9♣
    [0, 0, 0, 1, 2],  # 8♣
    [0, 0, 0, 1, 1],  # 7♣
    [0, 0, 0, 1, 7],  # A♠
    [0, 0, 0, 1, 5],  # K♠
    [0, 0, 1, 0, 4],  # Q♠
    [0, 0, 1, 0, 8],  # J♠
    [0, 0, 1, 0, 6],  # 10♠
    [0, 0, 1, 0, 3],  # 9♠
    [0, 0, 1, 0, 2],  # 8♠
    [0, 0, 1, 0, 1],  # 7♠
    [0, 1, 0, 0, 7],  # A♥
    [0, 1, 0, 0, 5],  # K♥
    [0, 1, 0, 0, 4],  # Q♥
    [0, 1, 0, 0, 8],  # J♥
    [0, 1, 0, 0, 6],  # 10♥
    [0, 1, 0, 0, 3],  # 9♥
    [0, 1, 0, 0, 2],  # 8♥
    [0, 1, 0, 0, 1],  # 7♥
    [1, 0, 0, 0, 7],  # A♦
    [1, 0, 0, 0, 5],  # K♦
    [1, 0, 0, 0, 4],  # Q♦
    [1, 0, 0, 0, 8],  # J♦
    [1, 0, 0, 0, 6],  # 10♦
    [1, 0, 0, 0, 3],  # 9♦
    [1, 0, 0, 0, 2],  # 8♦
    [1, 0, 0, 0, 1]  # 7♦
]
