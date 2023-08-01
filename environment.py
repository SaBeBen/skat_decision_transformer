import numpy as np

from card_representation_conversion import convert_card_to_vec, convert_tuple_to_card

from game.game import Game
from game.game_state_machine import GameStateMachine
from game.game_variant import GameVariantSuit
from game.state.game_state_bid import BidCallAction, BidPassAction, PickUpSkatAction, DeclareGameVariantAction
from game.state.game_state_start import GameStateStart, StartGameAction
from game.state.game_state_play import PlayCardAction, SurrenderAction
from model.card import Card

from model.player import Player

# position co-player (3) + trump (4) + last trick (3 * act_dim) + open cards (2 * act_dim) + hand cards (12 * act_dim)
state_dim = 92

# card representation is a vector
act_dim = 5


# convert the trump from as commonly represented in one number to a (categorical) vector
def get_trump(trump, enc="cat"):
    if enc == "cat":
        # if categorical encoding of suits is activated
        return [1 if trump == 12 or trump == 24 else 0, 1 if trump == 11 or trump == 24 else 0,
                1 if trump == 10 or trump == 24 else 0, 1 if trump == 9 or trump == 24 else 0]
    elif 8 < trump < 13:
        return trump - 9
    else:
        return trump


# calculates the win probability solely based on the starting hand cards of a player
# a game is considered winnable if the kinback score of the own cards is at least 8.5 points
def calculate_kinback_scheme(hand_cards, pos):
    score = 0

    # hand cards are sorted by colours and value
    # we want to find out the most promising suit
    trump_suits = [0, 0, 0, 0]
    i = 0
    for card in hand_cards:
        if card.face is Card.Face.JACK:
            trump_suits = [count + 1 for count in trump_suits]
        # each possible trump counts as 1 point
        trump_suits[card.suit.value] += 1
        if card.face is Card.Face.ACE or card.face is Card.Face.TEN:
            # each trump ace and ten count as 2 points
            trump_suits[card.suit.value] += 1

        i += 1

    # each trump guaranteed to win a trick is worth 1 point
    # in a suit game, there are 11 trump cards -> if you have 6 or more trump cards,
    # you are guaranteed to win trumps - (11 - trumps) tricks
    score += score - (11 - score) if score >= 6 else 0

    non_trump_suits = [0, 0, 0, 0]
    # i = 0

    # in the non-trump cards:
    # ace + 10 + K      -> 2.5 points
    # ace + 10          -> 2 points
    # Ace + K           -> 1.5 points
    # Ace + _           -> 1 point
    # 10 + K            -> 1 point
    # 10 | 10 + x       -> 0.5 points
    # K + D + x         -> 0.5 points

    for i in range(len(hand_cards)):
        if hand_cards[i].face is Card.Face.ACE:
            if i < 9 and hand_cards[i + 1].face is Card.Face.TEN and hand_cards[i].suit == hand_cards[i + 1].suit:
                if i < 8 and hand_cards[i + 2].face is Card.Face.KING and hand_cards[i].suit == hand_cards[i + 1].suit:
                    non_trump_suits[hand_cards[i].suit.value] += 2.5
                    # skip the next two cards
                    i += 2
                else:
                    non_trump_suits[hand_cards[i].suit.value] += 2
                    # skip the next card
                    i += 1
            elif i < 9 and hand_cards[i + 1].face is Card.Face.KING and hand_cards[i].suit == hand_cards[i + 1].suit:
                non_trump_suits[hand_cards[i].suit.value] += 1.5
                # skip the next card
                i += 1
            else:
                non_trump_suits[hand_cards[i].suit.value] += 1
        elif hand_cards[i].face is Card.Face.TEN:
            non_trump_suits[hand_cards[i].suit.value] += 0.5
            if i < 9 and hand_cards[i + 1].face is Card.Face.KING and hand_cards[i].suit == hand_cards[i + 1].suit:
                non_trump_suits[hand_cards[i].suit.value] += 0.5
                # skip the next card
                i += 1
        elif i < 8 and hand_cards[i].face is Card.Face.KING and hand_cards[i].suit == hand_cards[i + 1].suit \
                and hand_cards[i + 1].face is Card.Face.QUEEN and hand_cards[i].suit == hand_cards[i + 2].suit \
                and hand_cards[i + 2].face is not Card.Face.JACK:
            non_trump_suits[hand_cards[i].suit.value] += 0.5
            # skip the next two cards
            i += 2

        i += 1

    # when sitting in forehand -> 0.5 points
    if pos == 1:
        score += 0.5

    # trump = trump_suits.index(max(trump_suits))
    highest_score = 0
    trump = 0

    # determine the best trump suit by calculating the Kinback score for every suit (indexed by eval_trump)
    for eval_trump in range(4):
        score = non_trump_suits[(eval_trump + 1) % 4] + non_trump_suits[(eval_trump + 2) % 4] + \
                non_trump_suits[(eval_trump + 3) % 4] + trump_suits[eval_trump]
        if score >= highest_score:
            highest_score = score
            trump = eval_trump

    return score, trump + 9


class Env:
    def __init__(self):
        # self.device = torch.device("cuda")
        # Name the players with placeholders to recognize them during evaluation and debugging
        self.player1 = Player(1, "Alice")
        self.player2 = Player(2, "Bob")
        self.player3 = Player(3, "Carol")
        self.game = Game([self.player1, self.player2, self.player3])
        self.state_machine = GameStateMachine(GameStateStart(self.game))

        # StartGameAction not necessary when initialising the cards manually
        # self.state_machine.handle_action(StartGameAction())

        self.action_space = act_dim
        self.observation_space = state_dim

        self.state = None

    def _get_state(self):
        return self.state

    def reset(self, current_player):
        # self.game.dealer += 1

        self.game.reset()
        self.state_machine = GameStateMachine(GameStateStart(self.game))
        # DONE: Problem: How do we initialise the seating and bidding?

        # shuffles and gives out cards
        self.state_machine.handle_action(StartGameAction())

        highest_score, soloist_pos, i = 0, 0, 0

        soloist = None
        trump = -1
        for player in self.game.players:
            i += 1
            # lets the player with the best scoring cards play
            score, trump_player = calculate_kinback_scheme(player.cards, i)
            if score > highest_score:
                highest_score = score
                soloist = player
                trump = trump_player

        # soloist.type = Player.Type.DECLARER
        self.state_machine.handle_action(BidCallAction(soloist, 18))
        self.state_machine.handle_action(BidPassAction(self.game.players[(soloist.get_id()) % 3], 18))  # id starts at 1
        self.state_machine.handle_action(BidPassAction(self.game.players[(soloist.get_id() + 1) % 3], 18))

        self.state_machine.handle_action(PickUpSkatAction(soloist))

        self.state_machine.handle_action(DeclareGameVariantAction(soloist, GameVariantSuit(trump)))

        # determine the position of players
        pos_p = [0, 0, 0]

        if soloist is current_player:
            # initialise positions of defender as -1
            pos_p[(current_player.get_id() + 1) % 3], pos_p[(current_player.get_id() + 2) % 3] = -1, -1
        elif soloist.get_id() == (current_player.get_id() + 1 % 3):
            pos_p[(current_player.get_id() + 1) % 3] = -1
            pos_p[(current_player.get_id() + 2) % 3] = 1
        else:
            pos_p[(current_player.get_id() + 1) % 3] = 1
            pos_p[(current_player.get_id() + 2) % 3] = -1

        # get the current trump
        trump_enc = get_trump(trump)

        # get the cards of the last trick and convert them to the card representation
        last_trick = [[0] * act_dim, [0] * act_dim]

        # get the open cards and convert them to the card representation
        open_cards = [[0] * act_dim, [0] * act_dim, [0] * act_dim]

        # update hand cards
        hand_cards = [convert_card_to_vec(card) for card in current_player.cards]

        # pad the Skat on the defenders hands if the agent is one
        if current_player is not soloist:
            hand_cards.extend([[0] * act_dim, [0] * act_dim])

        game_state = np.concatenate([pos_p, trump_enc, last_trick, open_cards, hand_cards], axis=None)

        self.state = game_state

        return game_state

    def step(self, card, current_player):
        # if the action is surrendering
        if card[0] == -2:
            self.state_machine.handle_action(SurrenderAction(player=current_player))
            reward = current_player.current_trick_points
            # pad the game state with 0s as a game-terminating signal
            game_state = [[0] * state_dim]  # * (12 - self.game.round)]
            # self.reset(current_player)
            return game_state, reward, True

        done = (self.game.round == 10)

        # pass action to the game state machine
        self.state_machine.handle_action(PlayCardAction(player=current_player, card=convert_tuple_to_card(card)))

        # update the reward, only the last points of the trick are relevant
        reward = self.player1.current_trick_points

        # determine the position of players
        pos_p = [0, 0, 0]

        if self.game.get_declarer() is current_player:
            pos_p[(current_player.get_id() + 1) % 3] = -1
            pos_p[(current_player.get_id() + 2) % 3] = -1
        elif self.game.get_declarer().get_id() == (current_player.get_id() + 1 % 3):
            pos_p[(current_player.get_id() + 1) % 3] = -1
            pos_p[(current_player.get_id() + 2) % 3] = 1
        else:
            pos_p[(current_player.get_id() + 1) % 3] = 1
            pos_p[(current_player.get_id() + 2) % 3] = -1

        # get the current trump
        trump_enc = get_trump(self.game.game_variant.get_trump())

        # get the cards of the last trick and convert them to the card representation
        last_trick = [convert_card_to_vec(card) for card in self.game.get_last_trick_cards()] if self.game.round != 0 \
            else [[0] * act_dim, [0] * act_dim]

        # get the open cards and convert them to the card representation
        open_cards = [convert_card_to_vec(card) for card in self.game.trick.get_open_cards()]

        # update hand cards
        hand_cards = [convert_card_to_vec(card) for card in current_player.cards]

        game_state = np.concatenate([pos_p, trump_enc, last_trick, open_cards, hand_cards], axis=None)

        self.state = game_state

        # Return state, reward, done
        return game_state, reward, done

    # def train(self):
    #     self.training = True
    #
    # def eval(self):
    #     self.training = False
