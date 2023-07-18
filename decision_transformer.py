import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DecisionTransformerModel, DecisionTransformerConfig

from game.game import Game
from game.game_state_machine import GameStateMachine
from game.game_variant import GameVariantSuit
from game.state.game_state_bid import DeclareGameVariantAction, PutDownSkatAction, BidCallAction, BidPassAction, \
    PickUpSkatAction
from game.state.game_state_start import GameStateStart, StartGameAction
from game.state.game_state_play import PlayCardAction

from model.player import Player
from model.card import Card

from transformers import TrainingArguments
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import Trainer


#
# def convert_vec_to_card(card):
#     vector_rep = {
#         [0, 0, 0, 1, 7, _]: Card(Card.Suit.CLUB, Card.Face.ACE),  # A♣
#         [0, 0, 0, 1, 5, _]: Card(Card.Suit.CLUB, Card.Face.KING),  # K♣
#         [0, 0, 0, 1, 4, _]: Card(Card.Suit.CLUB, Card.Face.QUEEN),  # Q♣
#         [0, 0, 0, 1, 8, 1]: Card(Card.Suit.CLUB, Card.Face.JACK),  # J♣
#         [0, 0, 0, 1, 6, _]: Card(Card.Suit.CLUB, Card.Face.TEN),  # 10♣
#         [0, 0, 0, 1, 3, _]: Card(Card.Suit.CLUB, Card.Face.NINE),  # 9♣
#         [0, 0, 0, 1, 2, _]: Card(Card.Suit.CLUB, Card.Face.EIGHT),  # 8♣
#         [0, 0, 0, 1, 1, _]: Card(Card.Suit.CLUB, Card.Face.SEVEN),  # 7♣
#         [0, 0, 0, 1, 7, _]: Card(Card.Suit.SPADE, Card.Face.ACE),  # A♠
#         [0, 0, 0, 1, 5, _]: Card(Card.Suit.SPADE, Card.Face.KING),  # K♠
#         [0, 0, 1, 0, 4, _]: Card(Card.Suit.SPADE, Card.Face.QUEEN),  # Q♠
#         [0, 0, 1, 0, 8, 1]: Card(Card.Suit.SPADE, Card.Face.JACK),  # J♠
#         [0, 0, 1, 0, 6, _]: Card(Card.Suit.SPADE, Card.Face.TEN),  # 10♠
#         [0, 0, 1, 0, 3, _]: Card(Card.Suit.SPADE, Card.Face.NINE),  # 9♠
#         [0, 0, 1, 0, 2, _]: Card(Card.Suit.SPADE, Card.Face.EIGHT),  # 8♠
#         [0, 0, 1, 0, 1, _]: Card(Card.Suit.SPADE, Card.Face.SEVEN),  # 7♠
#         [0, 1, 0, 0, 7, _]: Card(Card.Suit.HEARTS, Card.Face.ACE),  # A♥
#         [0, 1, 0, 0, 5, _]: Card(Card.Suit.HEARTS, Card.Face.KING),  # K♥
#         [0, 1, 0, 0, 4, _]: Card(Card.Suit.HEARTS, Card.Face.QUEEN),  # Q♥
#         [0, 1, 0, 0, 8, 1]: Card(Card.Suit.HEARTS, Card.Face.JACK),  # J♥
#         [0, 1, 0, 0, 6, _]: Card(Card.Suit.HEARTS, Card.Face.TEN),  # 10♥
#         [0, 1, 0, 0, 3, _]: Card(Card.Suit.HEARTS, Card.Face.NINE),  # 9♥
#         [0, 1, 0, 0, 2, _]: Card(Card.Suit.HEARTS, Card.Face.EIGHT),  # 8♥
#         [0, 1, 0, 0, 1, _]: Card(Card.Suit.HEARTS, Card.Face.SEVEN),  # 7♥
#         [1, 0, 0, 0, 7, _]: Card(Card.Suit.DIAMOND, Card.Face.ACE),  # A♦
#         [1, 0, 0, 0, 5, _]: Card(Card.Suit.DIAMOND, Card.Face.KING),  # K♦
#         [1, 0, 0, 0, 4, _]: Card(Card.Suit.DIAMOND, Card.Face.QUEEN),  # Q♦
#         [1, 0, 0, 0, 8, 1]: Card(Card.Suit.DIAMOND, Card.Face.JACK),  # J♦
#         [1, 0, 0, 0, 6, _]: Card(Card.Suit.DIAMOND, Card.Face.TEN),  # 10♦
#         [1, 0, 0, 0, 3, _]: Card(Card.Suit.DIAMOND, Card.Face.NINE),  # 9♦
#         [1, 0, 0, 0, 2, _]: Card(Card.Suit.DIAMOND, Card.Face.EIGHT),  # 8♦
#         [1, 0, 0, 0, 1, _]: Card(Card.Suit.DIAMOND, Card.Face.SEVEN)  # 7♦
#     }
#     converted_card = vector_rep[card]
#
#     return converted_card


def convert_card_to_one_hot(card):
    vector_rep = {
        Card(Card.Suit.CLUB, Card.Face.ACE): 0,  # A♣
        Card(Card.Suit.CLUB, Card.Face.KING): 1,  # K♣
        Card(Card.Suit.CLUB, Card.Face.QUEEN): 2,  # Q♣
        Card(Card.Suit.CLUB, Card.Face.JACK): 3,  # J♣
        Card(Card.Suit.CLUB, Card.Face.TEN): 4,  # 10♣
        Card(Card.Suit.CLUB, Card.Face.NINE): 5,  # 9♣
        Card(Card.Suit.CLUB, Card.Face.EIGHT): 6,  # 8♣
        Card(Card.Suit.CLUB, Card.Face.SEVEN): 7,  # 7♣
        Card(Card.Suit.SPADE, Card.Face.ACE): 8,  # A♠
        Card(Card.Suit.SPADE, Card.Face.KING): 9,  # K♠
        Card(Card.Suit.SPADE, Card.Face.QUEEN): 10,  # Q♠
        Card(Card.Suit.SPADE, Card.Face.JACK): 11,  # J♠
        Card(Card.Suit.SPADE, Card.Face.TEN): 12,  # 10♠
        Card(Card.Suit.SPADE, Card.Face.NINE): 12,  # 9♠
        Card(Card.Suit.SPADE, Card.Face.EIGHT): 14,  # 8♠
        Card(Card.Suit.SPADE, Card.Face.SEVEN): 15,  # 7♠
        Card(Card.Suit.HEARTS, Card.Face.ACE): 16,  # A♥
        Card(Card.Suit.HEARTS, Card.Face.KING): 17,  # K♥
        Card(Card.Suit.HEARTS, Card.Face.QUEEN): 18,  # Q♥
        Card(Card.Suit.HEARTS, Card.Face.JACK): 19,  # J♥
        Card(Card.Suit.HEARTS, Card.Face.TEN): 20,  # 10♥
        Card(Card.Suit.HEARTS, Card.Face.NINE): 21,  # 9♥
        Card(Card.Suit.HEARTS, Card.Face.EIGHT): 22,  # 8♥
        Card(Card.Suit.HEARTS, Card.Face.SEVEN): 23,  # 7♥
        Card(Card.Suit.DIAMOND, Card.Face.ACE): 24,  # A♦
        Card(Card.Suit.DIAMOND, Card.Face.KING): 25,  # K♦
        Card(Card.Suit.DIAMOND, Card.Face.QUEEN): 26,  # Q♦
        Card(Card.Suit.DIAMOND, Card.Face.JACK): 27,  # J♦
        Card(Card.Suit.DIAMOND, Card.Face.TEN): 28,  # 10♦
        Card(Card.Suit.DIAMOND, Card.Face.NINE): 29,  # 9♦
        Card(Card.Suit.DIAMOND, Card.Face.EIGHT): 30,  # 8♦
        Card(Card.Suit.DIAMOND, Card.Face.SEVEN): 31  # 7♦
    }
    converted_card = vector_rep[card]

    return converted_card


def convert_one_hot_to_card(card):
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
# ♦, ♥, ♠, ♣, {7, 8, 9, Q, K, 10, A, J}, T
all_cards = [
    [0, 0, 0, 1, 7, 0],  # A♣
    [0, 0, 0, 1, 5, 0],  # K♣
    [0, 0, 0, 1, 4, 0],  # Q♣
    [0, 0, 0, 1, 8, 1],  # J♣
    [0, 0, 0, 1, 6, 0],  # 10♣
    [0, 0, 0, 1, 3, 0],  # 9♣
    [0, 0, 0, 1, 2, 0],  # 8♣
    [0, 0, 0, 1, 1, 0],  # 7♣
    [0, 0, 0, 1, 7, 0],  # A♠
    [0, 0, 0, 1, 5, 0],  # K♠
    [0, 0, 1, 0, 4, 0],  # Q♠
    [0, 0, 1, 0, 8, 1],  # J♠
    [0, 0, 1, 0, 6, 0],  # 10♠
    [0, 0, 1, 0, 3, 0],  # 9♠
    [0, 0, 1, 0, 2, 0],  # 8♠
    [0, 0, 1, 0, 1, 0],  # 7♠
    [0, 1, 0, 0, 7, 0],  # A♥
    [0, 1, 0, 0, 5, 0],  # K♥
    [0, 1, 0, 0, 4, 0],  # Q♥
    [0, 1, 0, 0, 8, 1],  # J♥
    [0, 1, 0, 0, 6, 0],  # 10♥
    [0, 1, 0, 0, 3, 0],  # 9♥
    [0, 1, 0, 0, 2, 0],  # 8♥
    [0, 1, 0, 0, 1, 0],  # 7♥
    [1, 0, 0, 0, 7, 0],  # A♦
    [1, 0, 0, 0, 5, 0],  # K♦
    [1, 0, 0, 0, 4, 0],  # Q♦
    [1, 0, 0, 0, 8, 1],  # J♦
    [1, 0, 0, 0, 6, 0],  # 10♦
    [1, 0, 0, 0, 3, 0],  # 9♦
    [1, 0, 0, 0, 2, 0],  # 8♦
    [1, 0, 0, 0, 1, 0]  # 7♦
]


class Env:
    def __init__(self):
        # self.device = torch.device("cuda")
        # Name the players with placeholders to recognize them during evaluation and debugging
        self.player1 = Player(1, "Alice")
        self.player2 = Player(2, "Bob")
        self.player3 = Player(3, "Carol")
        self.game = Game([self.player1, self.player2, self.player3])
        self.state_machine = GameStateMachine(GameStateStart(self.game))
        # self.state_machine.handle_action(StartGameAction())
        # TODO: init cards of each
        self.action_space = 6
        self.observation_space = 16

    # def _get_state(self):
    #     state =
    #     return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

    def reset(self):
        # return torch.stack(list(self.state_buffer), 0)
        self.game.reset()

    def step(self, card):
        reward = 0

        # pass action to the game state machine
        self.state_machine.handle_action(PlayCardAction(player=self.player1, card=convert_one_hot_to_card(card)))

        # update the reward, only the last points of the trick are relevant
        reward += self.player1.current_trick_points

        self.game.get_declarer()

        state = self.player1.cards

        # Return state, reward
        return torch.stack(), reward

    # def train(self):
    #     self.training = True
    #
    # def eval(self):
    #     self.training = False


# TODO: exchange for sql query
# game_data = app.get_skat_data_wm()

skat_wm_cs_data_path = "C:/Users/sasch/Desktop/Uni/Bachelorarbeit/SaschaBenz/software/skat_decision_transformer/db_app/data/wm_skattisch_kf.CSV"

skat_wm_game_data_path = "C:/Users/sasch/Desktop/Uni/Bachelorarbeit/SaschaBenz/software/skat_decision_transformer/db_app/data/wm_skattisch_spiel.CSV"

# skat_data_cs_frame = np.asarray(np.loadtxt(skat_wm_cs_data_path, delimiter=",", dtype=int))

# skat_data_game_frame = np.asarray(np.loadtxt(skat_wm_game_data_path, delimiter=",", dtype=int))

# Only use the relevant information: Skat and 30 played cards
# skat_and_cs = skat_data_cs_frame[:, 1:33]

# GameID (0), PlayerFH (6), PlayerMH (7), PlayerRH (8), Card1 (9):Card30 (40), PlayerID (44), Game (45),
# special treatment: Won (55), Miscall (56), AllPassed (58), Surrendered (59)
# only needs cards of current player
# meta_and_cards = skat_data_game_frame[:, np.r_[0, 6, 7, 8, 9:40, 44, 45, 55, 56, 58, 59]]

# position of the team player with respect to own pos; if 0 -> soloist
# alternating players perspective = {FH/MH/RH}
# -> pos_tp
# -> hand_cards

# select available cudas for faster matrix computation
# device = torch.device("cuda")

# env = Env(device)

skat_wm_cs_data_frame = pd.read_csv(skat_wm_cs_data_path, header=None)


def get_data():
    skat_wm_cs_data = skat_wm_cs_data_frame

    skat_wm_cs_data.columns = ["GameID", "Sd1", "Sd2", "CNr0", "CNr1", "CNr2", "CNr3", "CNr4", "CNr5", "CNr6", "CNr7",
                               "CNr8", "CNr9", "CNr10", "CNr11", "CNr12", "CNr13", "CNr14", "CNr15", "CNr16", "CNr17",
                               "CNr18", "CNr19", "CNr20", "CNr21", "CNr22", "CNr23", "CNr24", "CNr25", "CNr26", "CNr27",
                               "CNr28", "CNr29", "CNr30", "CNr31", "SurrenderedAt"]

    # TODO: Surrendered and SurrenderedAT do not match
    # skat_wm_cs_data = skat_wm_cs_data_frame[skat_wm_cs_data["SurrenderedAt"] == -1]

    skat_and_cs = skat_wm_cs_data.to_numpy()

    skat_and_cs[:, 3:35] = np.argsort(skat_and_cs[:, 3:35], axis=1)

    # skat_and_cs = skat_and_cs[:, 1:33]

    # GameID (0), PlayerFH (6), PlayerMH (7), PlayerRH (8), Card1 (9):Card30 (38), Card31 (39): Card32 (40) = Skat,
    # PlayerID (44), Game (45),
    # special treatment: Won (55), Miscall (56), AllPassed (58), Surrendered (59)
    # only needs cards of current player
    colums = [0, 6, 7, 8] + list(range(9, 41)) + [44, 45, 55, 56, 58, 59]

    # meta_and_cards = np.asarray(np.loadtxt(skat_wm_game_data_path, delimiter=",", dtype=int, usecols=colums))

    skat_wm_game_data_frame = pd.read_csv(skat_wm_game_data_path, header=None)

    skat_wm_game_data_frame.columns = ["GameID", "IDGame", "IDTable", "IDVServer", "StartTime", "EndTime", "PlayerFH",
                                       "PlayerMH", "PlayerRH", "Card1", "Card2", "Card3", "Card4", "Card5", "Card6",
                                       "Card7", "Card8", "Card9", "Card10", "Card11", "Card12", "Card13", "Card14",
                                       "Card15", "Card16", "Card17", "Card18", "Card19", "Card20", "Card21", "Card22",
                                       "Card23", "Card24", "Card25", "Card26", "Card27", "Card28", "Card29", "Card30",
                                       "Card31", "Card32", "CallValueFH", "CallValueMH", "CallValueRH", "PlayerID",
                                       "Game",
                                       "With", "Without", "Hand", "Schneider", "SchneiderCalled", "Schwarz",
                                       "SchwarzCalled", "Ouvert", "PointsPlayer", "Won", "Miscall",
                                       "CardPointsPlayer", "AllPassed", "Surrendered", "PlayerPosAtTableFH",
                                       "PlayerPosAtTableMH", "PlayerPosAtTableRH"]

    skat_wm_game_data_frame = skat_wm_game_data_frame.iloc[:, colums]

    # TODO: only look at won games
    # only consider games where nobody surrendered, if someone surrendered, the rest of the cards are "taken"
    # in their order
    # exclude games where all players passed
    skat_wm_game_data_frame = skat_wm_game_data_frame[(skat_wm_game_data_frame["AllPassed"] == 0)]
    # & (skat_wm_game_data_frame["Won"] == 1)
    # & (skat_wm_game_data_frame["Surrendered"] == 0)

    meta_and_cards = skat_wm_game_data_frame.to_numpy()  # skat_wm_game_data_frame

    # position of the team player with respect to own pos; if 0 -> soloist
    # alternating players perspective = {FH/MH/RH}
    # -> pos_tp
    # -> hand_cards
    amount_games = 10

    game_state_table = [[]]

    actions_table = [[]]

    rewards_table = [[]]

    # select available cudas for faster matrix computation
    # device = torch.device("cuda")

    for game in meta_and_cards[:amount_games]:
        env = Env()

        # skip the bid process
        # env.state_machine.handle_action()

        print(game)
        game_state = []

        actions = []

        rewards = []

        # conversion1 = [convert_one_hot_to_card(card) for card in meta_and_cards[game, 4:14].to_list()]

        # skip start of the game (shuffling and dealing)
        env.state_machine.state_finished_handler()
        # instead initialise the hand cards
        env.player1.set_cards([convert_one_hot_to_card(card) for card in game[4:14].tolist()])
        env.player2.set_cards([convert_one_hot_to_card(card) for card in game[14:24].tolist()])
        env.player3.set_cards([convert_one_hot_to_card(card) for card in game[24:34].tolist()])

        # initialize the Skat
        skat_up = [game[34], game[35]]
        # env.state_machine.handle_action(PutDownSkatAction(env.player1, [convert_one_hot_to_card(skat[0]),
        #                                                                 convert_one_hot_to_card(skat[1])]))
        env.game.skat.extend([convert_one_hot_to_card(skat_up[0]), convert_one_hot_to_card(skat_up[1])])

        # ID of current solo player
        player_ID = game[-6]

        # trump is defined by the game the soloist plays
        trump = game[-5]

        # we fixate the player on an index in the data set and
        # TODO: rotate
        agent_player = game[1]

        # convert DB encoding to env encoding
        suit = Card.Suit(trump - 9).name

        # env.state_machine.handle_action(DeclareGameVariantAction(env.player1, GameVariantSuit(suit)))
        # env.game.game_variant = GameVariantSuit(suit)
        # skip the bid process
        # env.state_machine.state_finished_handler()

        # TODO: 12 cards
        if agent_player == player_ID:
            soloist = env.player1
            env.player1.type = Player.Type.DECLARER
            env.player2.type = Player.Type.DEFENDER
            env.player3.type = Player.Type.DEFENDER
            env.state_machine.handle_action(BidCallAction(env.player1, 18))
            env.state_machine.handle_action(BidPassAction(env.player2, 18))
            env.state_machine.handle_action(BidPassAction(env.player3, 18))
            pos_tp = 0
        else:
            env.player1.type = Player.Type.DEFENDER
            if player_ID == game[2]:
                soloist = env.player2
                pos_tp = 2
                env.player2.type = Player.Type.DECLARER
                env.player3.type = Player.Type.DEFENDER
                env.state_machine.handle_action(BidCallAction(env.player2, 18))
                env.state_machine.handle_action(BidPassAction(env.player1, 18))
                env.state_machine.handle_action(BidPassAction(env.player3, 18))
            else:
                soloist = env.player3
                pos_tp = 1
                env.player3.type = Player.Type.DECLARER
                env.player2.type = Player.Type.DEFENDER
                env.state_machine.handle_action(BidCallAction(env.player3, 18))
                env.state_machine.handle_action(BidPassAction(env.player1, 18))
                env.state_machine.handle_action(BidPassAction(env.player2, 18))

        # pick up the Skat
        env.state_machine.handle_action(PickUpSkatAction(soloist))

        skat_down = [convert_one_hot_to_card(skat_and_cs[game[0] - 1, 1]),
                     convert_one_hot_to_card(skat_and_cs[game[0] - 1, 2])]

        # put Skat down
        env.state_machine.handle_action(PutDownSkatAction(soloist, skat_down))

        # TODO: 12 cards
        if env.player1.type == Player.Type.DECLARER:
            # if agent is player, select Skat
            actions.extend([skat_and_cs[game[0] - 1, 1], skat_and_cs[game[0] - 1, 2]])
            rewards.extend([skat_down[0].get_value(), skat_down[1].get_value()])
            skat_down.append(-1)
            last_trick = [skat_and_cs[game[0] - 1, 1], skat_and_cs[game[0] - 1, 2]]  # Skat and padding in the beginning
        else:
            actions.extend([-1, -1])
            rewards.extend([0, 0])
            last_trick = [-1, -1, -1]

        # announce game variant
        env.state_machine.handle_action(DeclareGameVariantAction(soloist, GameVariantSuit(suit)))

        open_cards = [-1, -1]  # there is no card revealed during Skat putting

        hand_cards = [convert_card_to_one_hot(card) for card in env.player1.cards]

        game_state = np.concatenate([pos_tp, trump, last_trick, open_cards, hand_cards], dtype=np.int8, axis=None)

        for trick in range(1, 11):
            print(f"current trick {trick}")

            # if the player sits in the front this trick
            if env.game.trick.leader == env.player1:
                # in position of first player, there are no open cards
                open_cards = [-1, -1]

                # convert each card to the desired encoding
                hand_cards = [convert_card_to_one_hot(card) for card in env.player1.cards]

                # pad the cards to a length of 12
                hand_cards = np.pad(hand_cards, (trick, 0))

                game_state = np.concatenate([game_state, pos_tp, trump, last_trick, open_cards, hand_cards],
                                            dtype=np.int8, axis=None)

                actions.append(skat_and_cs[game[0] - 1, 3 * trick])

            env.state_machine.handle_action(
                PlayCardAction(player=env.game.trick.leader,
                               card=convert_one_hot_to_card(skat_and_cs[game[0] - 1, 3 * trick])))

            # if the player sits in the middle this trick
            if env.game.trick.get_current_player() == env.player1:
                # in position of the second player, there is one open card
                open_cards = [skat_and_cs[game[0] - 1, 3 * trick], -1]

                # convert each card to the desired encoding
                hand_cards = [convert_card_to_one_hot(card) for card in env.player1.cards]

                # pad the cards to a length of 12
                hand_cards = np.pad(hand_cards, (trick, 0))

                game_state = np.concatenate([game_state, pos_tp, trump, last_trick, open_cards, hand_cards],
                                            dtype=np.int8, axis=None)

                actions.append(skat_and_cs[game[0] - 1, 3 * trick + 1])

            env.state_machine.handle_action(
                PlayCardAction(player=env.game.trick.get_current_player(),
                               card=convert_one_hot_to_card(skat_and_cs[game[0] - 1, 3 * trick + 1])))

            # if the player sits in the rear this trick
            if env.game.trick.get_current_player() == env.player1:
                # in position of the third player, there are two open cards
                open_cards = [skat_and_cs[game[0] - 1, 3 * trick],
                              skat_and_cs[game[0] - 1, 3 * trick + 1]]

                # convert each card to the desired encoding
                hand_cards = [convert_card_to_one_hot(card) for card in env.player1.cards]

                # pad the cards to a length of 12
                hand_cards = np.pad(hand_cards, (trick, 0))

                game_state = np.concatenate([game_state, pos_tp, trump, last_trick, open_cards, hand_cards],
                                            dtype=np.int8, axis=None)

                actions.append(skat_and_cs[game[0] - 1, 3 * trick + 2])

            env.state_machine.handle_action(
                PlayCardAction(player=env.game.trick.get_current_player(),
                               card=convert_one_hot_to_card(skat_and_cs[game[0] - 1, 3 * trick + 2])))

            rewards.append(env.player1.current_trick_points)

            last_trick = [skat_and_cs[game[0] - 1, 3 * trick]] + [skat_and_cs[game[0] - 1, 3 * trick + 1]] + [
                skat_and_cs[game[0] - 1, 3 * trick + 2]]

        # in the end of each game, insert the states, actions and rewards
        # insert states
        # game_state_table[game] = [game[0] - 1] + [agent_player] + game_state
        game_state_table[game] = np.concatenate(([game[0] - 1, agent_player, game_state]), axis=None)
        # insert actions
        # actions_table[game] = [game[0] - 1] + [agent_player] + actions
        game_state_table[game] = np.concatenate([game[0] - 1, agent_player, actions], axis=None)
        # insert rewards
        # rewards_table[game] = [game[0] - 1] + [agent_player] + rewards
        game_state_table[game] = np.concatenate([[game[0] - 1], agent_player, rewards], axis=None)

    return game_state_table, actions_table, rewards_table


# the dataset is already tokenized in the database

# position co-player + last trick + open cards + hand cards = 1 + 3 + 2 + 10
state_dim = 16

# card representation is a vector of 6 elements
act_dim = 6

TARGET_RETURN = 61

# --------------------------------------------------------------------------------------------------------------------
#
# # select available cudas for faster matrix computation
# device = torch.device("cuda")
#
# # one could use pretrained, but in our case we need our own model
# configuration = DecisionTransformerConfig(state_dim=state_dim, act_dim=act_dim)
# model = DecisionTransformerModel(configuration)
#
# training_args = TrainingArguments(
#     output_dir="training_output",
#     learning_rate=2e-5,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=2,
# )
#
# # split dataset into train and test
# skat_train, skat_test = train_test_split([], test_size=0.2, random_state=0)
#
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=skat_train,
#     eval_dataset=skat_test,
# )  # doctest: +SKIP
#
# trainer.train()
#
# # ----------------------------------------------------------------------------------------
#
# # evaluation
# model = model.to(device)
# model.eval()
#
# env = Env(device)
#
#
# # Function that gets an action from the model using autoregressive prediction
# # with a window of the previous 20 timesteps.
# def get_action(model, states, actions, rewards, returns_to_go, timesteps):
#     # This implementation does not condition on past rewards
#
#     states = states.reshape(1, -1, model.config.state_dim)
#     actions = actions.reshape(1, -1, model.config.act_dim)
#     returns_to_go = returns_to_go.reshape(1, -1, 1)
#     timesteps = timesteps.reshape(1, -1)
#
#     # The prediction is conditioned on up to 20 previous time-steps
#     states = states[:, -model.config.max_length:]
#     actions = actions[:, -model.config.max_length:]
#     returns_to_go = returns_to_go[:, -model.config.max_length:]
#     timesteps = timesteps[:, -model.config.max_length:]
#
#     # pad all tokens to sequence length, this is required if we process batches
#     padding = model.config.max_length - states.shape[1]
#     attention_mask = torch.cat([torch.zeros(padding), torch.ones(states.shape[1])])
#     attention_mask = attention_mask.to(dtype=torch.long).reshape(1, -1)
#     states = torch.cat([torch.zeros((1, padding, state_dim)), states], dim=1).float()
#     actions = torch.cat([torch.zeros((1, padding, act_dim)), actions], dim=1).float()
#     returns_to_go = torch.cat([torch.zeros((1, padding, 1)), returns_to_go], dim=1).float()
#     timesteps = torch.cat([torch.zeros((1, padding), dtype=torch.long), timesteps], dim=1)
#
#     # perform the prediction
#     state_preds, action_preds, return_preds = model(
#         states=states,
#         actions=actions,
#         rewards=rewards,
#         returns_to_go=returns_to_go,
#         timesteps=timesteps,
#         attention_mask=attention_mask,
#         return_dict=False, )
#     return action_preds[0, -1]
#
#
# # This was normalized during training
# MAX_EPISODE_LENGTH = 1000
# scale = 1
#
# # state_mean = np.array()
# # state_std = np.array()
# #
# # state_mean = torch.from_numpy(state_mean)
# # state_std = torch.from_numpy(state_std)
#
# # build the environment for the evaluation
# state = env.reset()  # TODO
# target_return = torch.tensor(TARGET_RETURN).float().reshape(1, 1)
# states = torch.from_numpy(state).reshape(1, state_dim).float()
# actions = torch.zeros((0, act_dim)).float()
# rewards = torch.zeros(0).float()
# timesteps = torch.tensor(0).reshape(1, 1).long()
#
# # take steps in the environment (evaluation, not training)
# for t in range(MAX_EPISODE_LENGTH):
#     # add zeros for actions as input for the current time-step
#     actions = torch.cat([actions, torch.zeros((1, act_dim))], dim=0)
#     rewards = torch.cat([rewards, torch.zeros(1)])
#
#     # predicting the action to take
#     action = get_action(model,
#                         states,  # - state_mean) / state_std,
#                         actions,
#                         rewards,
#                         target_return,
#                         timesteps)
#     actions[-1] = action
#     action = action.detach().numpy()
#
#     # interact with the environment based on this action
#     state, reward, done, _ = env.step(action)  # TODO
#
#     cur_state = torch.from_numpy(state).reshape(1, state_dim)
#     states = torch.cat([states, cur_state], dim=0)
#     rewards[-1] = reward
#
#     pred_return = target_return[0, -1] - (reward / scale)
#     target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
#     timesteps = torch.cat([timesteps, torch.ones((1, 1)).long() * (t + 1)], dim=1)
#
#     if done:
#         break
#
# # state = env.reset()
# # states = torch.from_numpy(state).reshape(1, 1, state_dim).to(device=device, dtype=torch.float32)
# # actions = torch.zeros((1, 1, act_dim), device=device, dtype=torch.float32)
# # rewards = torch.zeros(1, 1, device=device, dtype=torch.float32)
# # target_return = torch.tensor(TARGET_RETURN, dtype=torch.float32).reshape(1, 1)
# # timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
# # attention_mask = torch.zeros(1, 1, device=device, dtype=torch.float32)
# #
# # # forward pass
# # with torch.no_grad():
# #     state_preds, action_preds, return_preds = model(
# #         states=states,
# #         actions=actions,
# #         rewards=rewards,
# #         returns_to_go=target_return,
# #         timesteps=timesteps,
# #         attention_mask=attention_mask,
# #         return_dict=False,
# #     )
