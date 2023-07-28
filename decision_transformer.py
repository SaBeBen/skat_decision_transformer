from math import floor

import numpy as np
import pandas as pd

from card_representation_conversion import convert_one_hot_to_card, convert_one_hot_to_vector, convert_card_to_vec
from game.game import Game
from game.game_state_machine import GameStateMachine
from game.game_variant import GameVariantSuit, GameVariantGrand, GameVariantNull
from game.state.game_state_bid import DeclareGameVariantAction, PutDownSkatAction, BidCallAction, BidPassAction, \
    PickUpSkatAction
from game.state.game_state_start import GameStateStart
from game.state.game_state_play import PlayCardAction

from model.player import Player
from model.card import Card


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
        return reward

    # def train(self):
    #     self.training = True
    #
    # def eval(self):
    #     self.training = False


# position co-player (1) + trump (4) + last trick (3 * act_dim) + open cards (2 * act_dim) + hand cards (12 * act_dim)
state_dim = 92

# card representation is a vector
act_dim = 5


def get_trump(trump, enc="cat"):
    if enc == "cat":
        # if categorical encoding of suits is activated
        return [1 if trump == 12 or trump == 24 else 0, 1 if trump == 11 or trump == 24 else 0,
                1 if trump == 10 or trump == 24 else 0, 1 if trump == 9 or trump == 24 else 0]
    elif 8 < trump < 13:
        return trump - 9
    else:
        return trump


def get_game(game="wc"):
    possible_championships = ["wc", "gc", "gtc", "bl", "rc"]

    if game not in possible_championships:
        raise ValueError(f"The championship {game} does not exist in the database.")

    skat_cs_path = f"db_app/data/{game}_card_sequence.CSV"

    skat_game_path = f"db_app/data/{game}_game.CSV"

    skat_cs_data = pd.read_csv(skat_cs_path, header=None)

    skat_cs_data.columns = ["GameID", "Sd1", "Sd2", "CNr0", "CNr1", "CNr2", "CNr3", "CNr4", "CNr5", "CNr6", "CNr7",
                            "CNr8", "CNr9", "CNr10", "CNr11", "CNr12", "CNr13", "CNr14", "CNr15", "CNr16", "CNr17",
                            "CNr18", "CNr19", "CNr20", "CNr21", "CNr22", "CNr23", "CNr24", "CNr25", "CNr26", "CNr27",
                            "CNr28", "CNr29", "CNr30", "CNr31", "SurrenderedAt"]

    # TODO: Surrendered and SurrenderedAT do not match
    # skat_cs_data = skat_cs_data_frame[skat_cs_data["SurrenderedAt"] == -1]

    # GameID (0), PlayerFH (6), PlayerMH (7), PlayerRH (8), Card1 (9):Card30 (38), Card31 (39): Card32 (40) = Skat,
    # PlayerID (44), Game (45), Hand(48), PointsPlayer (54), Won (55), Miscall (56), AllPassed (58), Surrendered (59)
    # only needs cards of current player
    colums = [0, 6, 7, 8] + list(range(9, 41)) + [44, 45, 48, 54, 55, 56, 58, 59]

    skat_game_data = pd.read_csv(skat_game_path, header=None)

    skat_game_data.columns = ["GameID", "IDGame", "IDTable", "IDVServer", "StartTime", "EndTime", "PlayerFH",
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

    skat_game_data = skat_game_data.iloc[:, colums]

    # if someone surrendered, the rest of the cards are in the cs data in their order

    # exclude games where all players passed
    skat_game_data = skat_game_data[(skat_game_data["AllPassed"] == 0)]
    # & (skat_game_data["Won"] == 1)
    # & (skat_game_data["Surrendered"] == 0)

    # exclude grand and null games
    skat_game_data = skat_game_data[(skat_game_data["Game"] != 24)
                                    & (skat_game_data["Game"] != 23)
                                    & (skat_game_data["Game"] != 35)
                                    & (skat_game_data["Game"] != 46)
                                    & (skat_game_data["Game"] != 59)]

    # get rid of certain games by merging
    merged_game = pd.merge(skat_game_data, skat_cs_data, how="inner", on="GameID")

    skat_game_data = merged_game.loc[:, :"Surrendered"]

    skat_and_cs = merged_game.iloc[:, -35:]

    skat_and_cs = skat_and_cs.to_numpy()

    """
    Sort the data set according to the playing order:
    We have an array where each column is a card, each row is a game, and the numbers indicate when the card was played
    We want an array where each row is a game, each column is a position in the game, and the values indicate cards
    """
    skat_and_cs[:, 2:34] = np.argsort(skat_and_cs[:, 2:34], axis=1)

    return skat_game_data.to_numpy(), skat_and_cs


def surrender(won, current_player, soloist_points, trick, game_state, actions, rewards):
    # if game was surrendered by defenders out of the perspective of soloist or
    # if game was surrendered by soloist out of the perspective of a defender
    if won:
        # add reward and player points as last reward
        rewards.append(current_player.current_trick_points + soloist_points)
        actions.extend([0] * act_dim)
    else:
        # action of surrendering
        actions.append([-2] * act_dim)
        # default behaviour (gives a negative reward here)
        rewards.append(current_player.current_trick_points)

    # pad the states, actions and rewards with 0s
    game_state = np.concatenate([game_state, ([0] * state_dim) * (10 - trick)])
    actions = actions + [[0] * act_dim] * (9 - trick)
    rewards = rewards + [0] * (10 - trick)

    return game_state, actions, rewards


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(len(x) - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum


def initialise_hand_cards(game, current_player, current_player2, current_player3, i):
    # TODO: access the player card with fast search, for instance binary search
    # ...instead, initialise the hand cards
    current_player.set_cards(
        [convert_one_hot_to_card(card) for card in
         game[4 + 10 * i:14 + 10 * i].tolist()])
    current_player2.set_cards(
        [convert_one_hot_to_card(card) for card in
         game[4 + ((10 + 10 * i) % 30):4 + ((20 + 10 * i - 1) % 30 + 1)].tolist()])
    current_player3.set_cards(
        [convert_one_hot_to_card(card) for card in
         game[4 + ((20 + 10 * i) % 30):4 + ((30 + 10 * i - 1) % 30 + 1)].tolist()])

    # sort the cards to make hands reproducible, improve readability for attention mechanism (and humans)
    current_player.cards.sort()
    current_player2.cards.sort()
    current_player3.cards.sort()


def get_states_actions_rewards(championship="wc", amount_games=1000, point_rewards=False):
    meta_and_cards, skat_and_cs = get_game(game=championship)

    # position of the team player with respect to own pos; if 0 -> soloist
    # alternating players perspective = {FH/MH/RH}
    # -> pos_tp
    # -> hand_cards

    game_state_table = [[] * state_dim * 10] * amount_games * 3

    actions_table = [[] * act_dim * 10] * amount_games * 3

    rewards_table = [[] * 10] * amount_games * 3

    # rtg_table = [[] * 10] * amount_games * 3
    # timesteps_table = [[] * 10] * amount_games * 3
    # mask_table = [[] * 10] * amount_games * 3

    # # episode length for padding
    # max_len = 12
    #
    # # scale for rtgs
    # scale = 1

    # use an own index to access the card sequence data, as the GameID is left out
    cs_index = 0

    for game in meta_and_cards[:amount_games, :]:

        for i in range(3):
            # print(f"Player {i} is currently transferred")
            env = Env()

            # rotate players to get every perspective of play
            players = [env.player1, env.player2, env.player3]

            current_player = players[i]

            current_player2 = players[(i + 1) % 3]

            current_player3 = players[(i + 2) % 3]

            states, actions, rewards, rtg, timesteps, mask = [], [], [], [], [], []

            # player_id: ID of current solo player
            # trump: game the soloist plays
            # hand: binary encoding whether hand was played
            # soloist_points: points the soloist receives for playing a certain game
            player_id, trump, hand, soloist_points = game[-8], game[-7], game[-6], game[-5]

            won = ((current_player is Player.Type.DECLARER) and game[-4]) or \
                  ((current_player is Player.Type.DEFENDER) and not game[-4])

            # we fixate the player on an index in the data set and rotate over the index
            agent_player = game[i + 1]

            # initialize the Skat
            skat_up = [convert_one_hot_to_card(game[34]), convert_one_hot_to_card(game[35])]

            # put down Skat...
            skat_down = [convert_one_hot_to_card(skat_and_cs[cs_index, 0]),
                         convert_one_hot_to_card(skat_and_cs[cs_index, 1])]

            # categorical encoding of trump suit color
            # if a grand is played --> [1, 1, 1, 1]
            # if a null game is played --> [0, 0, 0, 0]     # TODO: implement null ouvert
            trump_enc = get_trump(trump)

            # if a game was surrendered, the amount of tricks played before surrender is stored in surrendered trick
            surrendered_trick = floor(skat_and_cs[cs_index, -1] / 3)

            # skip start of the game (shuffling and dealing)
            env.state_machine.state_finished_handler()

            initialise_hand_cards(game, current_player, current_player2, current_player3, i)

            # initialise Skat in the environment
            env.game.skat.extend([skat_up[0], skat_up[1]])

            # initialise roles, encode positions and simulate bidding
            if agent_player == player_id:

                # encode the position of the team partner: 0 for self, 1 for team, -1 for opponent
                pos_tp = [-1, 0, -1]  # 0
                # used to encode position of agent for game identification
                agent_player = i
                soloist = current_player

                current_player.type = Player.Type.DECLARER
                current_player2.type = Player.Type.DEFENDER
                current_player3.type = Player.Type.DEFENDER
                env.state_machine.handle_action(BidCallAction(current_player, 18))
                env.state_machine.handle_action(BidPassAction(current_player2, 18))
                env.state_machine.handle_action(BidPassAction(current_player3, 18))
            else:
                current_player.type = Player.Type.DEFENDER
                if player_id == game[1 + (i + 1) % 3]:
                    pos_tp = [1, 0, -1]  # 2
                    agent_player = (i + 1) % 3
                    soloist = current_player2

                    current_player2.type = Player.Type.DECLARER
                    current_player3.type = Player.Type.DEFENDER
                    env.state_machine.handle_action(BidCallAction(current_player2, 18))
                    env.state_machine.handle_action(BidPassAction(current_player, 18))
                    env.state_machine.handle_action(BidPassAction(current_player3, 18))
                else:
                    pos_tp = [-1, 0, 1]  # 1
                    agent_player = (i + 2) % 3
                    soloist = current_player3

                    current_player3.type = Player.Type.DECLARER
                    current_player2.type = Player.Type.DEFENDER
                    env.state_machine.handle_action(BidCallAction(current_player3, 18))
                    env.state_machine.handle_action(BidPassAction(current_player, 18))
                    env.state_machine.handle_action(BidPassAction(current_player2, 18))

            # there is no card revealed during Skat putting
            open_cards = [[0] * act_dim, [0] * act_dim]

            # during Skat selection, there is no last trick,
            # it will only be the last trick for the soloist when putting the Skat down
            last_trick = [[0] * act_dim, [0] * act_dim, [0] * act_dim]

            if not hand:
                # pick up the Skat
                env.state_machine.handle_action(PickUpSkatAction(soloist))

                # update hand cards: they will contain the Skat
                hand_cards = [convert_card_to_vec(card) for card in current_player.cards]

                if soloist != current_player:
                    # pad the current cards to a length of 12, if agent does not pick up Skat
                    hand_cards.extend([[0] * act_dim, [0] * act_dim])

                # ...in the game state
                game_state = np.concatenate([pos_tp, trump_enc, last_trick, open_cards, hand_cards],
                                            # dtype=np.int8,
                                            axis=None)

                # ...in the environment
                env.state_machine.handle_action(PutDownSkatAction(soloist, skat_down))
            else:
                # update hand cards: they will not contain the Skat
                hand_cards = [convert_card_to_vec(card) for card in current_player.cards]

                # pad the current cards to a length of 12
                hand_cards.extend([[0] * act_dim, [0] * act_dim])

                game_state = np.concatenate([pos_tp, trump_enc, last_trick, open_cards, hand_cards],
                                            # dtype=np.int8,
                                            axis=None)

            # ...put down Skat in the data (s, a, r)
            # each Skat card needs its own action (due to act_dim)
            if current_player.type == Player.Type.DECLARER and not hand:

                # the last trick is the put Skat and padding in the beginning TODO: uniform padding and
                last_trick = [convert_one_hot_to_vector(skat_and_cs[cs_index, 0]), [0] * act_dim, [0] * act_dim]

                game_state = np.concatenate([game_state, pos_tp, trump_enc, last_trick, open_cards, hand_cards],
                                            # dtype=np.int8,
                                            axis=None)

                # the last trick is the put Skat and padding in the beginning
                last_trick = [convert_one_hot_to_vector(skat_and_cs[cs_index, 0]),
                              convert_one_hot_to_vector(skat_and_cs[cs_index, 1]),
                              [0] * act_dim]

                # it is not necessary to simulate sequential putting with actions and rewards
                # instantly get rewards of the put Skat
                rewards.extend([skat_down[0].get_value(), skat_down[1].get_value()])

                # if agent is player, select Skat by putting down two cards
                actions.extend([convert_one_hot_to_vector(skat_and_cs[cs_index, 0]),
                                convert_one_hot_to_vector(skat_and_cs[cs_index, 1])])
            else:
                # the process of Skat selection is not visible for defenders,
                # thus it is padded and the defenders cards do not change
                actions.extend([[0] * act_dim, [0] * act_dim])
                rewards.extend([0, 0])
                last_trick = [[0] * act_dim, [0] * act_dim, [0] * act_dim]

                game_state = np.concatenate([game_state, pos_tp, trump_enc, last_trick, open_cards, hand_cards],
                                            # dtype=np.int8,
                                            axis=None)

            # TODO: uniform padding with 0 -> cards from 1-32

            # declare the game variant, TODO: implement null rewards
            if trump == 0 or trump == 35 or trump == 46 or trump == 59:
                # null
                env.state_machine.handle_action(DeclareGameVariantAction(soloist, GameVariantNull()))
            elif trump == 24:
                # grand
                env.state_machine.handle_action(DeclareGameVariantAction(soloist, GameVariantGrand()))
            else:
                # convert DB encoding to env encoding
                suit = Card.Suit(trump - 9).name
                # announce game variant
                env.state_machine.handle_action(DeclareGameVariantAction(soloist, GameVariantSuit(suit)))

            # if the game is surrendered instantly
            if surrendered_trick == 0:
                game_state, actions, rewards = \
                    surrender(won, current_player, soloist_points, 0, game_state, actions, rewards)
            else:
                # iterate over each trick
                for trick in range(1, 11):

                    # if the player sits in the front this trick
                    if env.game.trick.leader == current_player:
                        # in position of first player, there are no open cards
                        open_cards = [[0] * act_dim, [0] * act_dim]

                        # convert each card to the desired encoding
                        hand_cards = [convert_card_to_vec(card) for card in current_player.cards]

                        # pad the cards to a length of 12
                        hand_cards = [act_dim * [0]] * (trick + 1) + hand_cards

                        game_state = np.concatenate([game_state, pos_tp, trump_enc, last_trick, open_cards, hand_cards],
                                                    # dtype=np.int8,
                                                    axis=None)

                        actions.append(convert_one_hot_to_vector(skat_and_cs[cs_index, 3 * trick - 1]))

                    env.state_machine.handle_action(
                        PlayCardAction(player=env.game.trick.leader,
                                       card=convert_one_hot_to_card(skat_and_cs[cs_index, 3 * trick - 1])))

                    # if the player sits in the middle this trick
                    if env.game.trick.get_current_player() == current_player:
                        # in position of the second player, there is one open card
                        open_cards = [convert_one_hot_to_vector(skat_and_cs[cs_index, 3 * trick - 1]), [0] * act_dim]

                        # convert each card to the desired encoding
                        hand_cards = [convert_card_to_vec(card) for card in current_player.cards]

                        # pad the cards to a length of 12
                        hand_cards = [act_dim * [0]] * (trick + 1) + hand_cards

                        game_state = np.concatenate([game_state, pos_tp, trump_enc, last_trick, open_cards, hand_cards],
                                                    # dtype=np.int8,
                                                    axis=None)

                        actions.append(convert_one_hot_to_vector(skat_and_cs[cs_index, 3 * trick]))

                    env.state_machine.handle_action(
                        PlayCardAction(player=env.game.trick.get_current_player(),
                                       card=convert_one_hot_to_card(skat_and_cs[cs_index, 3 * trick])))

                    # if the player sits in the rear this trick
                    if env.game.trick.get_current_player() == current_player:
                        # in position of the third player, there are two open cards
                        open_cards = [convert_one_hot_to_vector(skat_and_cs[cs_index, 3 * trick - 1]),
                                      convert_one_hot_to_vector(skat_and_cs[cs_index, 3 * trick])]

                        # convert each card to the desired encoding
                        hand_cards = [convert_card_to_vec(card) for card in current_player.cards]

                        # pad the cards to a length of 12
                        hand_cards = [act_dim * [0]] * (trick + 1) + hand_cards

                        game_state = np.concatenate([game_state, pos_tp, trump_enc, last_trick, open_cards, hand_cards],
                                                    # dtype=np.int8,
                                                    axis=None)

                        actions.append(convert_one_hot_to_vector(skat_and_cs[cs_index, 3 * trick + 1]))

                    env.state_machine.handle_action(
                        PlayCardAction(player=env.game.trick.get_current_player(),
                                       card=convert_one_hot_to_card(skat_and_cs[cs_index, 3 * trick + 1])))

                    last_trick = [convert_one_hot_to_vector(skat_and_cs[cs_index, 3 * trick])] + \
                                 [convert_one_hot_to_vector(skat_and_cs[cs_index, 3 * trick])] + \
                                 [convert_one_hot_to_vector(skat_and_cs[cs_index, 3 * trick + 1])]

                    # check if game was surrendered at this trick
                    if surrendered_trick == trick:
                        game_state, actions, rewards = \
                            surrender(won, current_player, soloist_points, trick, game_state, actions, rewards)
                        break
                    else:
                        rewards.append(current_player.current_trick_points)

                    # tlen = trick
                    # # following lines are adapted from huggingface
                    # rtg.append(discount_cumsum(rewards, gamma=1.0).reshape((1, -1, 1)))
                    # timesteps.append(np.arange(0, tlen).reshape(1, -1))
                    # mask.append(np.concatenate([np.zeros((1, 12 - trick)), np.ones((1, i))], axis=1))
                    #
                    # # pad the rtg, timesteps and mask
                    # rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
                    # timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
                    # mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

                # if hand is played, adding the Skat points in the end of the game simulates not knowing them
                if hand:
                    if pos_tp == 0:
                        rewards[-1] += (skat_up[0].get_value()
                                        + skat_up[1].get_value())
                    else:
                        rewards[-1] -= (skat_up[0].get_value()
                                        + skat_up[1].get_value())

            # reward system:
            if point_rewards:
                # if point_rewards add card points on top of achieved points...
                if current_player.type == Player.Type.DECLARER:
                    # add the points to the soloist (soloist points can be negative)
                    rewards[-1] += soloist_points
                else:
                    # subtract the game points (soloist points can be negative)
                    rewards[-1] -= soloist_points
            else:
                # ...otherwise, give a negative reward for lost and a positive reward for won games
                rewards[-1] *= 1 if won else -1

            # in the end of each game, insert the states, actions and rewards
            # with composite primary keys game_id and player perspective (1: forehand, 2: middle-hand, 3: rear-hand)
            # insert states
            game_state_table[3 * cs_index + i] = np.array_split([float(i) for i in game_state], 12)
            # insert actions
            actions_table[3 * cs_index + i] = actions  # np.array_split([float(i) for i in actions], 12)
            # insert rewards
            rewards_table[3 * cs_index + i] = [float(i) for i in rewards]
            # np.array_split([float(i) for i in rewards], 12)

            # rtg_table[3 * cs_index + i] = torch.from_numpy(np.concatenate(rtg, axis=0)).float()
            # timesteps_table[3 * cs_index + i] = torch.from_numpy(np.concatenate(timesteps, axis=0)).long()
            # mask_table[3 * cs_index + i] = torch.from_numpy(np.concatenate(mask, axis=0)).float()

        cs_index = cs_index + 1

    return {
        "states": game_state_table,
        "actions": actions_table,
        "rewards": rewards_table,
        # "returns_to_go": rtg_table,
        # "timesteps": timesteps_table,
        # "attention_mask": mask_table,
    }


# the dataset is already tokenized in the database


TARGET_RETURN = 61

# --------------------------------------------------------------------------------------------------------------------
# from https://huggingface.co/blog/train-decision-transformers
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
