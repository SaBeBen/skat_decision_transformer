import numpy as np

import exceptions
from card_representation_conversion import convert_one_hot_to_card, convert_card_to_vector, \
    convert_one_hot_to_vector, convert_tuple_to_card

from game.game import Game
from game.game_state_machine import GameStateMachine
from game.game_variant import GameVariantSuit
from game.state.game_state_bid import BidCallAction, BidPassAction, PickUpSkatAction, DeclareGameVariantAction, \
    PutDownSkatAction
from game.state.game_state_start import GameStateStart, StartGameAction
from game.state.game_state_play import PlayCardAction, SurrenderAction
from model.card import Card

from model.player import Player

# card representation is a vector
act_dim = 12

card_dim = 5

# for padding of hand_cards, it is the maximum size with a compressed card representation
max_hand_len = 16 + act_dim  # 12 * card_dim  # 28

# position co-player (3) + trump (4) + last trick (3 * card_dim)
# + open cards (2 * card_dim) + hand cards (12 * card_dim)
state_dim = 3 + 2 + 4 + 3 * card_dim + 2 * card_dim + max_hand_len  # 12 * card_dim


# convert the trump from as commonly represented in one number to a (categorical) vector
def get_trump_enc(trump, enc="cat"):
    if enc == "cat":
        # if categorical encoding of suits is activated
        return [1 if trump == 9 or trump == 24 else 0, 1 if trump == 10 or trump == 24 else 0,
                1 if trump == 11 or trump == 24 else 0, 1 if trump == 12 or trump == 24 else 0]
    elif 8 < trump < 13:
        return trump - 9
    else:
        return trump


# calculates the win probability solely based on the starting hand cards of a player
# a game is considered winnable if the kinback score of the own cards is at least 8.5 points
def calculate_kinback_scheme(hand_cards, pos):
    kb_score = 0

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
    kb_score += kb_score - (11 - kb_score) if kb_score >= 6 else 0

    non_trump_suits = [0, 0, 0, 0]
    # i = 0

    # in the non-trump cards:
    # ace + 10 + K      -> 2.5 points
    # ace + 10          -> 2 points
    # ace + K           -> 1.5 points
    # ace + _           -> 1 point
    # 10 + K            -> 1 point
    # 10 | 10 + x       -> 0.5 points
    # K + D + x         -> 0.5 points

    for i in range(len(hand_cards)):
        if hand_cards[i].face is Card.Face.KING:
            if i < 9 and hand_cards[i + 1].face is Card.Face.TEN and hand_cards[i].suit == hand_cards[i + 1].suit:
                if i < 8 and hand_cards[i + 2].face is Card.Face.ACE and hand_cards[i].suit == hand_cards[i + 2].suit:
                    # ACE + 10 + K
                    non_trump_suits[hand_cards[i].suit.value] += 2.5
                    # skip the next two cards
                    i += 2
                else:
                    # 10 + K
                    non_trump_suits[hand_cards[i].suit.value] += 1
                    # skip the next two cards
                    i += 1
            elif i > 1 and hand_cards[i - 1].face is Card.Face.QUEEN and hand_cards[i].suit == hand_cards[i - 1].suit \
                    and hand_cards[i].suit == hand_cards[i - 2].suit and hand_cards[i - 2].face is not Card.Face.JACK:
                # K + D + x
                non_trump_suits[hand_cards[i].suit.value] += 0.5
            elif i < 9 and hand_cards[i + 1].face is Card.Face.ACE and hand_cards[i].suit == hand_cards[i + 1].suit:
                # A + K
                non_trump_suits[hand_cards[i].suit.value] += 1.5
                # skip the next two cards
                i += 1
        elif hand_cards[i].face is Card.Face.TEN:
            non_trump_suits[hand_cards[i].suit.value] += 0.5
            if i < 9 and hand_cards[i + 1].face is Card.Face.ACE and hand_cards[i].suit == hand_cards[i + 1].suit:
                # ACE + 10
                non_trump_suits[hand_cards[i].suit.value] += 2
                # skip the next card
                i += 1
            else:
                # 10 | 10 + x
                non_trump_suits[hand_cards[i].suit.value] += 0.5
        elif hand_cards[i].face is Card.Face.ACE:
            # A + _
            non_trump_suits[hand_cards[i].suit.value] += 1

    # when sitting in forehand -> 0.5 points
    if pos == 1:
        kb_score += 0.5

    # trump = trump_suits.index(max(trump_suits))
    highest_score = 0
    trump = 0

    # determine the best trump suit by calculating the Kinback score for every suit (indexed by eval_trump)
    for eval_trump in range(4):
        kb_score = non_trump_suits[(eval_trump + 1) % 4] + non_trump_suits[(eval_trump + 2) % 4] + \
                   non_trump_suits[(eval_trump + 3) % 4] + trump_suits[eval_trump]
        if kb_score >= highest_score:
            highest_score = kb_score
            trump = eval_trump

    return kb_score, trump + 9


def get_hand_cards(current_player):
    hand_cards = []
    last_card = None
    for card in current_player.cards:
        if last_card is not None and card.suit == last_card.suit:
            hand_cards += [card.face.value]
        else:
            hand_cards += convert_card_to_vector(card)
        last_card = card
        # hand_cards += convert_card_to_vector(card)
    hand_cards += [0] * (max_hand_len - len(hand_cards))

    return hand_cards


def initialise_hand_cards(game, current_player, current_player2, current_player3):
    # ...instead, initialise the hand cards
    current_player.set_cards(
        [convert_one_hot_to_card(card) for card in
         game[4 + 10 * current_player.get_id():14 + 10 * current_player.get_id()].tolist()])
    current_player2.set_cards(
        [convert_one_hot_to_card(card) for card in
         game[4 + 10 * current_player2.get_id():14 + 10 * current_player2.get_id()].tolist()])
    current_player3.set_cards(
        [convert_one_hot_to_card(card) for card in
         game[4 + 10 * current_player3.get_id():14 + 10 * current_player3.get_id()].tolist()])

    # sort the cards to make hands reproducible, improve readability for attention mechanism (and humans)
    current_player.cards.sort()
    current_player2.cards.sort()
    current_player3.cards.sort()


class Env:
    def __init__(self):
        # self.device = torch.device("cuda")
        # Name the players with placeholders to recognize them during evaluation and debugging
        self.player1 = Player(0, "Alice")
        self.player2 = Player(1, "Bob")
        self.player3 = Player(2, "Carol")
        self.game = Game([self.player1, self.player2, self.player3])
        self.state_machine = GameStateMachine(GameStateStart(self.game))

        self.action_space = act_dim
        self.observation_space = state_dim
        self.card_size = card_dim
        self.point_rewards = False

        self.state = None

        self.hand = False
        self.skat_down = []
        self.pos_p = None
        self.trump_enc = None
        self.score = [0, 0]

        self.current_player = None
        self.last_trick = None
        self.skat_put = [False, False]
        self.trick = 1
        self.skat_and_cs = []

    def _get_state(self):
        return self.state

    def get_game_points(self):
        game_points = 94
        # TODO: check the game variant
        # self.trump_enc
        # TODO: chekc if hand was played
        # self.hand
        # TODO: calculate the card points
        # card_points

        # game_points =

        return game_points

    def reset(self, current_player_id, game_first_state=None, meta_and_cards_game=None):
        # self.game.dealer += 1

        self.game.reset()

        # DONE: Problem: How do we initialise the seating and bidding?
        if game_first_state is None:
            self.state_machine = GameStateMachine(GameStateStart(self.game))
            # shuffles and gives out cards
            self.state_machine.handle_action(StartGameAction())

            highest_score, soloist_pos, i = 0, 0, 0

            soloist = None
            trump = -1
            for player in self.game.players:
                i += 1
                # lets the player with the best scoring cards play
                kb_score, trump_player = calculate_kinback_scheme(player.cards, i)
                if kb_score > highest_score:
                    highest_score = kb_score
                    soloist = player
                    trump = trump_player

            self.current_player = soloist
            # soloist.type = Player.Type.DECLARER
            self.state_machine.handle_action(BidCallAction(soloist, 18))
            self.state_machine.handle_action(
                BidPassAction(self.game.players[(soloist.get_id()) % 3], 18))  # id starts at 1
            self.state_machine.handle_action(BidPassAction(self.game.players[(soloist.get_id() + 1) % 3], 18))

            self.state_machine.handle_action(PickUpSkatAction(soloist))

            self.current_player.cards.sort()

            self.state_machine.handle_action(DeclareGameVariantAction(soloist, GameVariantSuit(trump)))

        # else:
        # StartGameAction not necessary with an already initialised game
        # state of game after Skat was picked up is used

        # making independent of env

        # Option 1: (preferred)
        # pass 1st game state from data in dt_train to initialise all vars and pass it back

        # Option 2: pass every necessary var

        # self.game = game_env.game
        # self.state_machine = game_env.state_machine
        # self.skat_down = game_env.skat_down
        # self.skat_and_cs = game_env.skat_and_cs
        # self.player1 = game_env.player1
        # self.player2 = game_env.player2
        # self.player3 = game_env.player3
        # self.hand = game_env.hand

        # game_state = pos_p + score + trump_enc + last_trick + open_cards + get_hand_cards(current_player)

        self.pos_p = game_first_state[:3]

        self.trump_enc = game_first_state[3:6]

        self.current_player = self.game.players[current_player_id]

        initialise_hand_cards(meta_and_cards_game,
                              self.current_player,
                              self.game.players[current_player_id + 1 % 3],
                              self.game.players[current_player_id + 2 % 3])

        # # Problem: Karten der anderen Spieler
        # if self.pos_p[0] == 0:
        #     # read in the hand_cards
        #     convert_tuple_to_card()
        #     self.game.players[current_player_id].set_cards(game_state[])
        # elif self.pos_p[1] == 0:
        #     convert_tuple_to_card()
        #     self.game.players[current_player_id].set_cards(game_state[])
        # else:
        #     convert_tuple_to_card()
        #     self.game.players[current_player_id].set_cards(game_state[])

        # set the current player on the copied environment

        # determine the position of players
        # pos_p = [0, 0, 0]
        #
        # if self.game.get_declarer().get_id() == current_player_id:
        #     # initialise positions of defender as -1
        #     pos_p[(current_player_id + 1) % 3], pos_p[(current_player_id + 2) % 3] = -1, -1
        # elif self.game.get_declarer().get_id() == (current_player_id + 1 % 3):
        #     pos_p[(current_player_id + 1) % 3] = -1
        #     pos_p[(current_player_id + 2) % 3] = 1
        # else:
        #     pos_p[(current_player_id + 1) % 3] = 1
        #     pos_p[(current_player_id + 2) % 3] = -1
        #
        # # set the player positions and roles
        # self.pos_p = pos_p
        #
        # # get the current trump
        # self.trump_enc = get_trump_enc(self.game.game_variant.get_trump())
        #
        # # set the current player on the copied environment
        # self.current_player = self.game.players[current_player_id]

        self.skat_put = [True, True]

        # # update status
        # self.skat_put[0] = False
        #
        # # put down the first Skat card
        # if not self.hand:
        #     if self.current_player is self.game.get_declarer():
        #         self.score[0] = Card.get_value(self.skat_down[0])
        #         # TODO: self.game.skat instead of init in data pipeline, in general: optimize variables
        #
        #     try:
        #         self.state_machine.handle_action(PutDownSkatAction(self.game.get_declarer(), self.skat_down[0]))
        #     except ValueError:
        #         raise exceptions.InvalidPlayerMove(
        #             f"Player {self.current_player.get_id()} is not holding card {self.skat_down[0]}! "
        #             f"The hand cards are {self.current_player.cards}")
        #
        #     self.current_player.cards.sort()
        #
        # # update status
        # self.skat_put[1] = False
        #
        # # put down the second Skat card
        # if not self.hand:
        #
        #     if self.current_player is self.game.get_declarer():
        #         self.score[0] += Card.get_value(self.skat_down[1])
        #
        #     try:
        #         self.state_machine.handle_action(PutDownSkatAction(self.game.get_declarer(), self.skat_down[1]))
        #     except ValueError:
        #         raise exceptions.InvalidPlayerMove(
        #             f"Player {self.current_player.get_id()} is not holding card {self.skat_down[1]}! "
        #             f"The hand cards are {self.current_player.cards}")

        # self.state_machine.handle_action(DeclareGameVariantAction(self.game.get_declarer(), self.game.game_variant))

        # if self.current_player is self.game.get_declarer():
        #     # get the cards of the last trick and convert them to the card representation
        #     self.last_trick = convert_card_to_vector(self.skat_down[0]) + convert_card_to_vector(self.skat_down[1]) + [
        #         0] * card_dim  # [0] * card_dim * 3
        # else:
        #     self.last_trick = [0] * card_dim * 3
        #
        # if self.game.get_first_seat() is self.current_player:
        #     # get the open cards and convert them to the card representation
        #     open_cards = [0] * card_dim * 2
        # elif self.game.get_second_seat() is self.current_player:
        #     open_cards = convert_one_hot_to_vector(self.skat_and_cs[3 * 1 - 1]) + [0] * card_dim
        # else:
        #     open_cards = convert_one_hot_to_vector(self.skat_and_cs[3 * 1 - 1]) + convert_one_hot_to_vector(
        #         self.skat_and_cs[3 * 1])

        # open_cards = [0] * card_dim * 2

        self.last_trick = [0] * card_dim * 3

        # pad the Skat on the defenders hands if the agent is one
        # if current_player_id != soloist.get_id() or (current_player_id == soloist.get_id() and self.hand):
        #     hand_cards += card_dim * [0] * 2

        # game_state = self.pos_p + self.score + self.trump_enc + self.last_trick + open_cards + get_hand_cards(
        #     self.current_player)
        #
        # self.last_trick = self.last_trick = convert_one_hot_to_vector(
        #     self.skat_and_cs[3 * 1 - 1]) + convert_one_hot_to_vector(
        #     self.skat_and_cs[3 * 1]) + convert_one_hot_to_vector(self.skat_and_cs[3 * 1 + 1])

        self.state = game_first_state

        return np.array(game_first_state)

    def step(self, card_index):
        # TODO: make step independent of existing env, use states, reset can be made independent (only change trump_enc)

        # if the action is surrendering
        if card_index[0] == -2:
            self.state_machine.handle_action(SurrenderAction(player=self.current_player))
            reward = self.current_player.current_trick_points
            # pad the game state with 0s as a game-terminating signal
            game_state = [[0] * state_dim]  # * (12 - self.game.round)]
            # self.reset(current_player)
            return game_state, reward, True

        # default reward of 0
        reward = 0

        open_cards = []

        # the game is finished after 10 rounds
        done = (self.game.round == 10)

        #  padding from right to left: cards in env are in the same order

        # split Skat putting into two actions:
        # After the first action the first card of the Skat will not be visible in the state,
        # after the second action the second card of the Skat will not be visible in the state
        if self.skat_put[0]:
            # update status
            self.skat_put[0] = False

            # put down the second Skat card
            if not self.hand:

                if self.current_player is self.game.get_declarer():
                    # # first Skat card as action
                    self.skat_down.append(self.current_player.cards[card_index.index(1)])
                    #
                    # # add the card value of second Skat card as reward
                    reward = Card.get_value(self.skat_down[0])

                    self.score[0] = Card.get_value(self.skat_down[0])
                    # TODO: self.game.skat instead of init in data pipeline, in general: optimize variables

                try:
                    self.state_machine.handle_action(PutDownSkatAction(self.game.get_declarer(), self.skat_down[0]))
                except ValueError:
                    raise exceptions.InvalidPlayerMove(
                        f"Player {self.current_player.get_id()} is not holding card {self.skat_down[0]}! "
                        f"The hand cards are {self.current_player.cards}")

                self.current_player.cards.sort()

            # last_trick = [0] * card_dim + [0] * card_dim + [0] * card_dim

            open_cards = [0] * card_dim + [0] * card_dim

        elif self.skat_put[1]:
            # update status
            self.skat_put[1] = False

            # put down the second Skat card
            if not self.hand:

                if self.current_player is self.game.get_declarer():
                    # second Skat card as action
                    self.skat_down.append(self.current_player.cards[card_index.index(1)])
                    #
                    # # add the card value of second Skat card as reward
                    reward = Card.get_value(self.skat_down[1])

                    self.score[0] += Card.get_value(self.skat_down[1])

                try:
                    self.state_machine.handle_action(PutDownSkatAction(self.game.get_declarer(), self.skat_down[1]))
                except ValueError:
                    raise exceptions.InvalidPlayerMove(
                        f"Player {self.current_player.get_id()} is not holding card {self.skat_down[1]}! "
                        f"The hand cards are {self.current_player.cards}")

            # after Skat putting, every player has 10 cards which need to be padded
            # this state will be the third state, namely the one after Skat putting

            # last_trick = [0] * card_dim + [0] * card_dim + [0] * card_dim

            open_cards = [0] * card_dim + [0] * card_dim

            self.state_machine.handle_action(DeclareGameVariantAction(self.game.get_declarer(), self.game.game_variant))

            # separate ifs for next open cards
            if self.game.trick.get_current_player() == self.current_player:
                # in position of first player, there are no open cards
                open_cards = [0] * card_dim + [0] * card_dim
            if self.game.trick.get_next_player(self.current_player) == self.current_player:
                # in position of the second player, there is one open card
                open_cards = convert_one_hot_to_vector(self.skat_and_cs[3 * (self.trick) - 1]) + [0] * card_dim
            else:
                # in position of the third player, there are two open cards
                open_cards = convert_one_hot_to_vector(
                    self.skat_and_cs[3 * (self.trick) - 1]) + convert_one_hot_to_vector(
                    self.skat_and_cs[3 * (self.trick)])
        else:
            # # if Skat is put in reset
            # if self.trick == 1 and self.current_player is self.game.get_declarer():
            #     reward += Card.get_value(self.skat_down[0]) + Card.get_value(self.skat_down[1])

            # select the card on the players hand
            card = self.current_player.cards[card_index.index(1)]

            # if the player sits in the front this trick
            if self.game.trick.get_current_player() == self.current_player:
                # iterates over players, each time PlayCardAction is called the role of the current player rotates
                self.state_machine.handle_action(
                    PlayCardAction(player=self.current_player, card=card))
            else:
                # iterates over players, each time PlayCardAction is called the role of the current player rotates
                self.state_machine.handle_action(
                    PlayCardAction(player=self.game.trick.get_current_player(),
                                   card=convert_one_hot_to_card(self.skat_and_cs[3 * self.trick - 1])))

            # if the player sits in the middle this trick
            if self.game.trick.get_current_player() == self.current_player:
                # iterates over players, each time PlayCardAction is called the role of the current player rotates
                self.state_machine.handle_action(PlayCardAction(player=self.current_player, card=card))
            else:
                # iterates over players, each time PlayCardAction is called the role of the current player rotates
                self.state_machine.handle_action(
                    PlayCardAction(player=self.game.trick.get_current_player(),
                                   card=convert_one_hot_to_card(self.skat_and_cs[3 * self.trick])))

            # if the player sits in the rear this trick
            if self.game.trick.get_current_player() == self.current_player:
                # iterates over players, each time PlayCardAction is called the role of the current player rotates
                self.state_machine.handle_action(PlayCardAction(player=self.current_player, card=card))
            else:
                # iterates over players, each time PlayCardAction is called the role of the current player rotates
                self.state_machine.handle_action(
                    PlayCardAction(player=self.game.trick.get_current_player(),
                                   card=convert_one_hot_to_card(self.skat_and_cs[3 * self.trick + 1])))

            # TODO: surrender
            reward += self.current_player.current_trick_points

            self.trick += 1

            # separate ifs for next open cards
            if self.game.trick.get_current_player() == self.current_player:
                # in position of first player, there are no open cards
                open_cards = [0] * card_dim + [0] * card_dim
            if self.game.trick.get_next_player(self.current_player) == self.current_player:
                # in position of the second player, there is one open card
                open_cards = convert_one_hot_to_vector(self.skat_and_cs[3 * (self.trick) - 1]) + [0] * card_dim
            else:
                # in position of the third player, there are two open cards
                open_cards = convert_one_hot_to_vector(
                    self.skat_and_cs[3 * (self.trick) - 1]) + convert_one_hot_to_vector(
                    self.skat_and_cs[3 * (self.trick)])

        self.score[1] += self.game.get_last_trick_points() if self.current_player.current_trick_points == 0 else 0
        self.score[0] += self.current_player.current_trick_points

        # TODO: get game_points
        if done:
            if self.current_player.type != Player.Type.DECLARER:
                self.score[1] += Card.get_value(self.skat_down[0]) + Card.get_value(self.skat_down[1])
            game_points = self.get_game_points()

            if self.hand:
                # add reward in hand game of two unseen Skat cards
                reward += Card.get_value(self.skat_down[0]) + Card.get_value(self.skat_down[1])

            if self.point_rewards:
                # if point_rewards add card points on top of achieved points...
                if self.current_player.type == Player.Type.DECLARER:
                    # add the points to the soloist
                    reward = 0.9 * game_points + reward * 0.1  # rewards[-1] + soloist_points
                else:
                    # subtract the game points
                    reward = -0.9 * game_points + reward * 0.1  # rewards[-1] + soloist_points
            else:
                # ...otherwise, give a 0 reward for lost and a positive reward for won games
                reward *= 1 if game_points > 0 else 0

        game_state = self.pos_p + self.score + self.trump_enc + self.last_trick + open_cards + get_hand_cards(
            self.current_player)

        if not self.skat_put[1] and not done:
            self.last_trick = convert_one_hot_to_vector(
                self.skat_and_cs[3 * self.trick - 1]) + convert_one_hot_to_vector(
                self.skat_and_cs[3 * self.trick]) + convert_one_hot_to_vector(self.skat_and_cs[3 * self.trick + 1])

        self.state = game_state

        # Return state, reward, done
        return np.array(game_state), reward, done

    # def train(self):
    #     self.training = True
    #
    # def eval(self):
    #     self.training = False

    def set_current_player(self, player):
        self.current_player = player
