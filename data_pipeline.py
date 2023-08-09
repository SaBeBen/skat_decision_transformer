import copy
from math import floor

import numpy as np
import pandas as pd
from tqdm import tqdm

from card_representation_conversion import convert_one_hot_to_card, convert_card_to_vector, \
    convert_one_hot_to_vector
from environment import Env, get_trump_enc

from game.game_variant import GameVariantSuit, GameVariantGrand, GameVariantNull
from game.state.game_state_bid import DeclareGameVariantAction, PutDownSkatAction, BidCallAction, BidPassAction, \
    PickUpSkatAction
from game.state.game_state_play import PlayCardAction

from model.player import Player
from model.card import Card

# actions are a vector which encodes the cards to select categorically
act_dim = 12

# cards are a vector, first 4 entries are a one-hot encoding of the suit, the fifth is the face encoded by an integer
card_dim = 5

# position co-player (3) + trump (4) + last trick (3 * card_dim)
# + open cards (2 * card_dim) + hand cards (12 * card_dim)
state_dim = 3 + 4 + 3 * card_dim + 2 * card_dim + 12 * card_dim


def get_game(game="wc"):
    possible_championships = ["wc", "gc", "gtc", "bl", "rc"]

    if game not in possible_championships:
        raise ValueError(f"The championship {game} does not exist in the database.")

    skat_cs_path = f"data/{game}_card_sequence.CSV"

    skat_game_path = f"data/{game}_game.CSV"

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
    columns = [0, 6, 7, 8] + list(range(9, 41)) + [44, 45, 48, 54, 55, 56, 58, 59]

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

    skat_game_data = skat_game_data.iloc[:, columns]

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
        actions.append([0] * act_dim)
    else:
        # action of surrendering
        actions.append([-2] * act_dim)
        # default behaviour (gives a negative reward here)
        rewards.append(current_player.current_trick_points)

    # pad the states, actions and rewards with 0s
    game_state += ([0] * state_dim) * (10 - trick)
    actions = actions + [[0] * act_dim] * (9 - trick)
    rewards = rewards + [0] * (10 - trick)

    return game_state, actions, rewards


def initialise_hand_cards(game, current_player, current_player2, current_player3, i):
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


def declare_game_variant(env, trump):
    # declare the game variant, TODO: implement null rewards
    if trump == 0 or trump == 35 or trump == 46 or trump == 59:
        # null
        env.state_machine.handle_action(DeclareGameVariantAction(env.game.get_declarer(), GameVariantNull()))
    elif trump == 24:
        # grand
        env.state_machine.handle_action(DeclareGameVariantAction(env.game.get_declarer(), GameVariantGrand()))
    else:
        # convert DB encoding to env encoding
        suit = Card.Suit(trump - 9).name
        # announce game variant
        env.state_machine.handle_action(DeclareGameVariantAction(env.game.get_declarer(), GameVariantSuit(suit)))


def get_hand_cards(current_player, trick):
    # convert each card to the desired encoding
    hand_cards = []
    for card in current_player.cards:
        hand_cards += convert_card_to_vector(card)

    # pad the cards to a length of 12
    hand_cards += card_dim * [0] * (trick + 1)

    return hand_cards


def get_states_actions_rewards(championship="wc", amount_games=1000, point_rewards=False, game_index=-1):
    meta_and_cards, skat_and_cs = get_game(game=championship)

    # meta_and_cards = [meta_and_cards[game_index],]

    # skat_and_cs = [skat_and_cs[game_index],]

    # position of the team player with respect to own pos; if 0 -> soloist
    # alternating players perspective = {FH/MH/RH}
    # -> pos_p
    # -> hand_cards
    fs_one_game = None

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

    for game in tqdm(meta_and_cards[:amount_games, :]):  # meta_and_cards:

        for i in range(3):

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
            trump_enc = get_trump_enc(trump)

            # if a game was surrendered, the amount of tricks played before surrender is stored in surrendered trick
            surrendered_trick = floor(skat_and_cs[cs_index, -1] / 3)

            # skip start of the game (shuffling and dealing)
            env.state_machine.state_finished_handler()

            # map the cards to the hands of the players
            initialise_hand_cards(game, current_player, current_player2, current_player3, i)

            # initialise Skat in the environment
            env.game.skat.extend([skat_up[0], skat_up[1]])

            # encode the position of the players: 0 for self, 1 for team, -1 for opponent
            pos_p = [0, 0, 0]

            # pos_p[i] = 0

            # initialise roles, encode positions and simulate bidding
            if agent_player == player_id:
                # if the perspective of the agent is the soloist

                # used to encode position of agent for game identification
                # agent_player = i

                # encode the position of the players: 0 for self, 1 for team, -1 for opponent
                pos_p[(i + 1) % 3] = -1
                pos_p[(i + 2) % 3] = -1

                current_player.type = Player.Type.DECLARER
                current_player2.type = Player.Type.DEFENDER
                current_player3.type = Player.Type.DEFENDER
                env.state_machine.handle_action(BidCallAction(current_player, 18))
                env.state_machine.handle_action(BidPassAction(current_player2, 18))
                env.state_machine.handle_action(BidPassAction(current_player3, 18))
            else:
                current_player.type = Player.Type.DEFENDER
                if player_id == game[1 + (i + 1) % 3]:
                    # agent_player = (i + 1) % 3

                    pos_p[(i + 1) % 3] = -1
                    pos_p[(i + 2) % 3] = 1

                    current_player2.type = Player.Type.DECLARER
                    current_player3.type = Player.Type.DEFENDER
                    env.state_machine.handle_action(BidCallAction(current_player2, 18))
                    env.state_machine.handle_action(BidPassAction(current_player, 18))
                    env.state_machine.handle_action(BidPassAction(current_player3, 18))
                else:
                    # agent_player = (i + 2) % 3

                    pos_p[(i + 1) % 3] = 1
                    pos_p[(i + 2) % 3] = -1

                    current_player3.type = Player.Type.DECLARER
                    current_player2.type = Player.Type.DEFENDER
                    env.state_machine.handle_action(BidCallAction(current_player3, 18))
                    env.state_machine.handle_action(BidPassAction(current_player, 18))
                    env.state_machine.handle_action(BidPassAction(current_player2, 18))

            won = ((current_player.type == Player.Type.DECLARER) and game[-4]) or \
                  ((current_player.type == Player.Type.DEFENDER) and not game[-4])
            # won = False

            # there is no card revealed during Skat putting
            # open_cards = [[0] * card_dim, [0] * card_dim]
            open_cards = [0] * card_dim + [0] * card_dim

            # during Skat selection, there is no last trick,
            # it will only be the last trick for the soloist when putting the Skat down
            # last_trick = [[0] * card_dim, [0] * card_dim, [0] * card_dim]
            last_trick = [0] * card_dim + [0] * card_dim + [0] * card_dim

            if not hand:
                # pick up the Skat
                env.state_machine.handle_action(PickUpSkatAction(env.game.get_declarer()))

                current_player.cards.sort()

                # if a single game is selected for evaluation, create a deep copy of it directly after Skat pick up
                if game_index == 3 * cs_index + i:
                    fs_one_game = copy.deepcopy(env)
                    fs_one_game.skat_up = skat_up
                    fs_one_game.skat_down = skat_down
                    fs_one_game.skat_and_cs = skat_and_cs[cs_index]
                    # fs_one_game.suit = trump

                # update hand cards: they will contain the Skat
                # hand_cards = [convert_card_to_tuple(card) for card in current_player.cards]
                hand_cards = []
                for card in current_player.cards:
                    hand_cards += convert_card_to_vector(card)

                if env.game.get_declarer() != current_player:
                    # pad the current cards to a length of 12, if agent does not pick up Skat
                    hand_cards += [0] * card_dim + [0] * card_dim

                # first game state
                game_state = pos_p + trump_enc + last_trick + open_cards + hand_cards

                # ...put down Skat one by one
                # each Skat card needs its own action (due to fixed dimensions)
                if current_player.type == Player.Type.DECLARER:
                    skat1 = convert_one_hot_to_card(skat_and_cs[cs_index, 0])
                    skat2 = convert_one_hot_to_card(skat_and_cs[cs_index, 1])

                    # categorical encoding of played card as action: put first card
                    cat_action = [0] * 12
                    cat_action[current_player.cards.index(skat1)] = 1
                    actions.append(cat_action)

                    # put down first Skat card in the environment
                    env.state_machine.handle_action(PutDownSkatAction(env.game.get_declarer(), skat_down[0]))

                    # the last trick is the put Skat and padding in the beginning
                    last_trick = convert_card_to_vector(skat1) + [0] * card_dim + [0] * card_dim

                    hand_cards = []
                    for card in current_player.cards:
                        hand_cards += convert_card_to_vector(card)

                    hand_cards += card_dim * [0]

                    game_state += pos_p + trump_enc + last_trick + open_cards + hand_cards

                    # categorical encoding of played card as action: put second card
                    cat_action = [0] * 12
                    cat_action[current_player.cards.index(skat2)] = 1
                    actions.append(cat_action)

                    # put down second Skat card in the environment
                    env.state_machine.handle_action(PutDownSkatAction(env.game.get_declarer(), skat_down[1]))

                    # the last trick is the put Skat and padding in the beginning
                    last_trick = convert_card_to_vector(skat1) + convert_card_to_vector(skat2) + [0] * card_dim

                    hand_cards = []
                    for card in current_player.cards:
                        hand_cards += convert_card_to_vector(card)

                    hand_cards += card_dim * [0] * 2

                    # it is not necessary to simulate sequential putting with actions and rewards
                    # instantly get rewards of the put Skat
                    rewards.extend([skat_down[0].get_value(), skat_down[1].get_value()])

                else:
                    # the process of Skat selection is not visible for defenders,
                    # thus it is padded and the defenders cards do not change

                    # there is no action during the Skat putting for the defenders
                    actions += [[0] * act_dim, [0] * act_dim]
                    rewards += [0, 0]

                    game_state += pos_p + trump_enc + last_trick + open_cards + hand_cards

                    # put Skat down in the environment
                    env.state_machine.handle_action(PutDownSkatAction(env.game.get_declarer(), skat_down))
            else:
                # if hand is played

                # if a single game is selected for evaluation, create a deep copy of it
                if game_index == 3 * cs_index + i:
                    fs_one_game = copy.deepcopy(env)
                    fs_one_game.skat_down = skat_up
                    fs_one_game.hand = True
                    fs_one_game.skat_and_cs = skat_and_cs[cs_index]
                    # fs_one_game.suit = trump

                # update hand cards: they will not contain the Skat
                hand_cards = []
                for card in current_player.cards:
                    hand_cards += convert_card_to_vector(card)

                # pad the cards to a length of 12
                hand_cards += card_dim * [0] * 2

                # there is no action during the Skat putting when playing hand
                actions.extend([[0] * act_dim, [0] * act_dim])
                rewards.extend([0, 0])

                # if hand is played, there are two identical game states from the perspective of every player
                game_state = pos_p + trump_enc + last_trick + open_cards + hand_cards

                game_state += pos_p + trump_enc + last_trick + open_cards + hand_cards

            # declare the game variant
            declare_game_variant(env, trump)

            # safe the game variant for the evaluated game
            if game_index == 3 * cs_index + i:
                fs_one_game.game.game_variant = env.game.game_variant

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
                        open_cards = [0] * card_dim + [0] * card_dim

                        game_state += pos_p + trump_enc + last_trick + open_cards + get_hand_cards(current_player,
                                                                                                   trick)
                        cat_action = [0] * 12
                        cat_action[current_player.cards.index(
                            convert_one_hot_to_card(skat_and_cs[cs_index, 3 * trick - 1]))] = 1
                        actions.append(cat_action)

                    # iterates over players, each time PlayCardAction is called the role of the current player rotates
                    env.state_machine.handle_action(
                        PlayCardAction(player=env.game.trick.leader,
                                       card=convert_one_hot_to_card(skat_and_cs[cs_index, 3 * trick - 1])))

                    # if the player sits in the middle this trick
                    if env.game.trick.get_current_player() == current_player:
                        # in position of the second player, there is one open card
                        open_cards = convert_one_hot_to_vector(skat_and_cs[cs_index, 3 * trick - 1]) + [
                            0] * card_dim

                        game_state += pos_p + trump_enc + last_trick + open_cards + get_hand_cards(current_player,
                                                                                                   trick)
                        cat_action = [0] * 12
                        cat_action[
                            current_player.cards.index(convert_one_hot_to_card(skat_and_cs[cs_index, 3 * trick]))] = 1
                        actions.append(cat_action)

                    # iterates over players, each time PlayCardAction is called the role of the current player rotates
                    env.state_machine.handle_action(
                        PlayCardAction(player=env.game.trick.get_current_player(),
                                       card=convert_one_hot_to_card(skat_and_cs[cs_index, 3 * trick])))

                    # if the player sits in the rear this trick
                    if env.game.trick.get_current_player() == current_player:
                        # in position of the third player, there are two open cards
                        open_cards = convert_one_hot_to_vector(skat_and_cs[cs_index, 3 * trick - 1]) + \
                                     convert_one_hot_to_vector(skat_and_cs[cs_index, 3 * trick])

                        game_state += pos_p + trump_enc + last_trick + open_cards + get_hand_cards(current_player,
                                                                                                   trick)

                        cat_action = [0] * 12
                        cat_action[current_player.cards.index(
                            convert_one_hot_to_card(skat_and_cs[cs_index, 3 * trick + 1]))] = 1
                        actions.append(cat_action)

                    # iterates over players, each time PlayCardAction is called the role of the current player rotates
                    env.state_machine.handle_action(
                        PlayCardAction(player=env.game.trick.get_current_player(),
                                       card=convert_one_hot_to_card(skat_and_cs[cs_index, 3 * trick + 1])))

                    last_trick = convert_one_hot_to_vector(
                        skat_and_cs[cs_index, 3 * trick - 1]) + convert_one_hot_to_vector(
                        skat_and_cs[cs_index, 3 * trick]) + convert_one_hot_to_vector(
                        skat_and_cs[cs_index, 3 * trick + 1])

                    # check if game was surrendered at this trick
                    if surrendered_trick == trick:
                        game_state, actions, rewards = \
                            surrender(won, current_player, soloist_points, trick, game_state, actions, rewards)
                        break
                    else:
                        rewards.append(current_player.current_trick_points)

                # if hand is played, adding the Skat points in the end of the game simulates not knowing them
                if hand:
                    if pos_p == 0:
                        rewards[-1] += (skat_up[0].get_value()
                                        + skat_up[1].get_value())
                    else:
                        rewards[-1] -= (skat_up[0].get_value()
                                        + skat_up[1].get_value())

            # reward system:
            if point_rewards:
                # if point_rewards add card points on top of achieved points...
                # soloist points can be negative if she lost
                if current_player.type == Player.Type.DECLARER:
                    # add the points to the soloist
                    rewards[-1] = 0.9 * soloist_points + rewards[-1] * 0.1  # rewards[-1] + soloist_points
                else:
                    # subtract the game points
                    rewards[-1] = -0.9 * soloist_points + rewards[-1] * 0.1  # rewards[-1] + soloist_points
            else:
                # ...otherwise, give a 0 reward for lost and a positive reward for won games
                rewards[-1] *= (1 if won else 0)

            # in the end of each game, insert the states, actions and rewards
            # with composite primary keys game_id and player perspective (1: forehand, 2: middle-hand, 3: rear-hand)
            # insert states
            game_state_table[3 * cs_index + i] = np.array_split([float(i) for i in game_state], 12)
            # insert actions
            actions_table[3 * cs_index + i] = actions  # np.array_split([float(i) for i in actions], 12)
            # insert rewards
            rewards_table[3 * cs_index + i] = [[i] for i in rewards]
            # np.array_split([float(i) for i in rewards], 12)

        cs_index = cs_index + 1

    return {
        "states": game_state_table,
        "actions": actions_table,
        "rewards": rewards_table,
    }, fs_one_game
