import copy
from math import floor

import numpy as np
import pandas as pd

from tqdm import tqdm

from card_representation_conversion import convert_numerical_to_card, convert_card_to_enc, convert_numerical_to_enc
from environment import Env, get_trump_enc, initialise_hand_cards, get_hand_cards, act_dim, initialise_encoding
from exceptions import InvalidPlayerMove

from game.game_variant import GameVariantSuit, GameVariantGrand, GameVariantNull
from game.state.game_state_bid import DeclareGameVariantAction, PutDownSkatAction, BidCallAction, BidPassAction, \
    PickUpSkatAction
from game.state.game_state_play import PlayCardAction

from model.player import Player
from model.card import Card

possible_championships = ["wc", "gc", "gtc", "bl", "rc"]


def get_game(game="wc", games_indices=slice(0, -1)):
    if game not in possible_championships:
        raise ValueError(f"The championship {game} does not exist in the database.")

    skat_cs_path = f"data/{game}_card_sequence.CSV"

    skat_game_path = f"data/{game}_game.CSV"

    skat_cs_data = pd.read_csv(skat_cs_path, header=None)

    skat_cs_data.columns = ["GameID", "Sd1", "Sd2", "CNr0", "CNr1", "CNr2", "CNr3", "CNr4", "CNr5", "CNr6", "CNr7",
                            "CNr8", "CNr9", "CNr10", "CNr11", "CNr12", "CNr13", "CNr14", "CNr15", "CNr16", "CNr17",
                            "CNr18", "CNr19", "CNr20", "CNr21", "CNr22", "CNr23", "CNr24", "CNr25", "CNr26", "CNr27",
                            "CNr28", "CNr29", "CNr30", "CNr31", "SurrenderedAt"]

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

    # GameID (0), PlayerFH (6), PlayerMH (7), PlayerRH (8), Card1 (9):Card30 (38), Card31 (39): Card32 (40) = Skat,
    # PlayerID (44), Game (45), Hand(48), PointsPlayer (54), Won (55), Miscall (56), AllPassed (58), Surrendered (59)
    # only needs cards of current player
    columns = [0, 6, 7, 8] + list(range(9, 41)) + [44, 45, 48, 54, 55, 56, 58, 59]

    skat_game_data = skat_game_data.iloc[:, columns]

    # if someone surrendered, the rest of the cards are in the cs data in their order

    # exclude games where all players passed
    skat_game_data = skat_game_data[(skat_game_data["AllPassed"] == 0) & (skat_game_data["Surrendered"] == 0)]
    # & (skat_game_data["Won"] == 1)]

    # exclude grand and null games
    skat_game_data = skat_game_data[(skat_game_data["Game"] != 24)
                                    & (skat_game_data["Game"] != 23)
                                    & (skat_game_data["Game"] != 35)
                                    & (skat_game_data["Game"] != 46)
                                    & (skat_game_data["Game"] != 59)]

    skat_cs_data = skat_cs_data[skat_cs_data["SurrenderedAt"] == -1]

    # get rid of certain games by merging
    merged_game = pd.merge(skat_game_data, skat_cs_data, how="inner", on="GameID")

    merged_game = merged_game.iloc[games_indices]

    # if game == 'gc':
    #     merged_game = merged_game.drop(merged_game[(merged_game["GameID"] == 167545)
    #                                                | (merged_game["GameID"] == 167546)].index)

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


def surrender(won, current_player, soloist_points, trick, game_state, actions, rewards, state_dim):
    # if game was surrendered by defenders out of the perspective of soloist or
    # if game was surrendered by soloist out of the perspective of a defender
    if won:
        # add reward and player points as last reward
        rewards.append(current_player.current_trick_points + soloist_points)
        actions.append([0] * act_dim)
    else:
        # action of surrendering
        actions.append([1] * act_dim)
        # default behaviour (gives a negative reward here)
        rewards.append(current_player.current_trick_points)

    # pad the states, actions and rewards with 0s
    game_state += ([0] * state_dim) * (10 - trick)
    actions = actions + [[0] * act_dim] * (9 - trick)
    rewards = rewards + [0] * (10 - trick)

    return game_state, actions, rewards


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


def get_states_actions_rewards(
        championship="wc",
        games_indices=slice(0, 1000),
        point_rewards=False,
        game_index=-1,
        perspective=(0, 1, 2),
        card_enc='mixed_comp'
):
    meta_and_cards, skat_and_cs = get_game(game=championship, games_indices=games_indices)

    card_dim, max_hand_len, state_dim = initialise_encoding(card_enc)

    # position of the team player with respect to own pos; if 0 -> soloist
    # alternating players perspective = {FH/MH/RH}
    # -> pos_p
    # -> hand_cards
    fs_one_game = None

    amount_games = len(meta_and_cards)

    if not isinstance(perspective, tuple):
        perspective = [perspective]
        len_p = 1
    else:
        len_p = len(perspective)

    game_state_table = [[] * state_dim * 10] * amount_games * len_p

    actions_table = [[] * act_dim * 10] * amount_games * len_p

    rewards_table = [[] * 10] * amount_games * len_p

    # use an own index to access the card sequence data, as the GameID is left out
    cs_index = 0

    skip = False

    card_error_games = []

    for game in tqdm(meta_and_cards):

        for i in perspective:

            if skip:
                skip = False
                print(f"broke out of game {game[0]}")
                card_error_games.append(game[0])
                break

            env = Env(enc=card_enc)

            # player_id: ID of current solo player
            # trump: game the soloist plays
            # hand: binary encoding whether hand was played
            # soloist_points: points the soloist receives for playing a certain game
            player_id, trump, hand, soloist_points = game[-8], game[-7], game[-6], game[-5]

            # categorical encoding of trump suit color
            # if a grand is played --> [1, 1, 1, 1]
            # if a null game is played --> [0, 0, 0, 0]     # TODO: implement null ouvert
            trump_enc = get_trump_enc(trump)

            # if a game was surrendered, the amount of tricks played before surrender is stored in surrendered trick
            surrendered_trick = floor(skat_and_cs[cs_index, -1] / 3)

            # skip start of the game (shuffling and dealing)
            env.state_machine.state_finished_handler()

            # initialize the Skat
            skat_up = [convert_numerical_to_card(game[34]), convert_numerical_to_card(game[35])]

            # put down Skat...
            skat_down = [convert_numerical_to_card(skat_and_cs[cs_index, 0]),
                         convert_numerical_to_card(skat_and_cs[cs_index, 1])]

            # initialise Skat in the environment
            env.game.skat.extend([skat_up[0], skat_up[1]])

            # encode the position of the players: 0 for self, 1 for team, -1 for opponent
            pos_p = [0, 0, 0]

            # keeps track of the score, first entry describes the points of the agent's team
            score = [0, 0]

            # rotate players to get every perspective of play
            players = [env.player1, env.player2, env.player3]

            current_player = players[i]

            current_player2 = players[(i + 1) % 3]

            current_player3 = players[(i + 2) % 3]

            states, actions, rewards, rtg, timesteps, mask, game_state = [], [], [], [], [], [], []

            # we fixate the player on an index in the data set and rotate over the index
            agent_player = game[i + 1]

            # map the cards to the hands of the players
            initialise_hand_cards(game, current_player, current_player2, current_player3)

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

            # there is no card revealed during Skat putting
            # open_cards = [[0] * card_dim, [0] * card_dim]
            open_cards = [0] * card_dim + [0] * card_dim

            # during Skat selection, there is no last trick,
            # it will only be the last trick for the soloist when putting the Skat down
            # last_trick = [[0] * card_dim, [0] * card_dim, [0] * card_dim]
            last_trick = [0] * card_dim + [0] * card_dim + [0] * card_dim

            skat1 = convert_numerical_to_card(skat_and_cs[cs_index, 0])
            skat2 = convert_numerical_to_card(skat_and_cs[cs_index, 1])

            if not hand:
                # pick up the Skat
                env.state_machine.handle_action(PickUpSkatAction(env.game.get_declarer()))

                current_player.cards.sort()

                # if a single game is selected for evaluation, create a deep copy of it directly after Skat pick up
                if game_index == 3 * cs_index + i:
                    fs_one_game = copy.deepcopy(env)
                    # fs_one_game.skat_up = skat_up
                    fs_one_game.skat_and_cs = skat_and_cs[cs_index]
                    # fs_one_game.suit = trump

                # first game state + score
                game_state = pos_p + score + trump_enc + last_trick + open_cards + get_hand_cards(current_player,
                                                                                                  encoding=card_enc)

                # ...put down Skat one by one
                # each Skat card needs its own action (due to fixed dimensions)
                if current_player.type == Player.Type.DECLARER:
                    # categorical encoding of played card as action: put first card
                    cat_action = [0] * act_dim
                    try:
                        cat_action[current_player.cards.index(skat1)] = 1
                        actions.append(cat_action)
                    except ValueError:
                        skip = True
                        break

                    try:
                        # put down first Skat card in the environment
                        env.state_machine.handle_action(PutDownSkatAction(env.game.get_declarer(), skat_down[0]))
                    except ValueError or InvalidPlayerMove:
                        skip = True
                        break

                    # the soloist knows how many points he possesses through the Skat,
                    # defenders do not have this information
                    score[0] += skat1.get_value()

                    # the last trick is the put Skat and padding in the beginning
                    last_trick = convert_card_to_enc(skat1, encoding=card_enc) + [0] * card_dim + [0] * card_dim

                    game_state += pos_p + score + trump_enc + last_trick + open_cards + get_hand_cards(
                        current_player, encoding=card_enc)

                    # categorical encoding of played card as action: put second card
                    cat_action = [0] * act_dim
                    try:
                        cat_action[current_player.cards.index(skat2)] = 1
                        actions.append(cat_action)
                    except ValueError:
                        skip = True
                        break

                    try:
                        # put down second Skat card in the environment
                        env.state_machine.handle_action(PutDownSkatAction(env.game.get_declarer(), skat_down[1]))
                    except ValueError or InvalidPlayerMove:
                        skip = True
                        break

                    # the soloist knows how many points he possesses through the Skat,
                    # defenders do not have this information
                    score[0] += skat2.get_value()

                    # the last trick is the put Skat and padding in the beginning
                    last_trick = convert_card_to_enc(skat1, encoding=card_enc) + convert_card_to_enc(
                        skat2, encoding=card_enc) + [0] * card_dim

                    # it is not necessary to simulate sequential putting with actions and rewards
                    # instantly get rewards of the put Skat
                    rewards.extend([skat_down[0].get_value(), skat_down[1].get_value()])

                else:
                    # the process of Skat selection is not visible for defenders,
                    # thus it is padded and the defenders cards do not change

                    # there is no action during the Skat putting for the defenders
                    actions += [[0] * act_dim, [0] * act_dim]
                    rewards += [0, 0]

                    game_state += pos_p + score + trump_enc + last_trick + open_cards + get_hand_cards(
                        current_player, encoding=card_enc)

                    try:
                        # put Skat down in the environment
                        env.state_machine.handle_action(PutDownSkatAction(env.game.get_declarer(), skat_down))
                    except ValueError or InvalidPlayerMove:
                        skip = True
                        break
            else:
                # if hand is played

                # if a single game is selected for evaluation, create a deep copy of it
                if game_index == 3 * cs_index + i:
                    fs_one_game = copy.deepcopy(env)
                    fs_one_game.hand = True
                    fs_one_game.skat_and_cs = skat_and_cs[cs_index]
                    # fs_one_game.suit = trump

                # there is no action during the Skat putting when playing hand
                actions.extend([[0] * act_dim, [0] * act_dim])
                rewards.extend([0, 0])

                # if hand is played, there are two identical game states from the perspective of every player

                game_state = pos_p + score + trump_enc + last_trick + open_cards + get_hand_cards(current_player,
                                                                                                  encoding=card_enc)

                game_state += pos_p + score + trump_enc + last_trick + open_cards + get_hand_cards(current_player,
                                                                                                   encoding=card_enc)

            # declare the game variant
            declare_game_variant(env, trump)

            # safe the game variant for the evaluated game
            if game_index == 3 * cs_index + i:
                fs_one_game.game.game_variant = env.game.game_variant
                fs_one_game.skat_down = copy.deepcopy(skat_down)

            # if the game is surrendered instantly
            if surrendered_trick == 0:
                game_state, actions, rewards = \
                    surrender(won, current_player, soloist_points, 0, game_state, actions, rewards, state_dim)
            else:
                # iterate over each trick
                for trick in range(1, 11):

                    # the first score shows the current players score
                    score[1] += env.game.get_last_trick_points() if current_player.current_trick_points == 0 else 0
                    score[0] += current_player.current_trick_points

                    game_state += pos_p + score + trump_enc + last_trick

                    # if the player sits in the front of this trick
                    if env.game.trick.leader == current_player:
                        # in position of first player, there are no open cards
                        open_cards = [0] * card_dim + [0] * card_dim

                        game_state += open_cards + get_hand_cards(current_player, encoding=card_enc)

                        cat_action = [0] * act_dim
                        try:
                            cat_action[current_player.cards.index(
                                convert_numerical_to_card(skat_and_cs[cs_index, 3 * trick - 1]))] = 1
                            actions.append(cat_action)
                        except ValueError:
                            skip = True
                            break

                    try:
                        # iterates over players
                        # each time PlayCardAction is called the role of the current player rotates
                        env.state_machine.handle_action(
                            PlayCardAction(player=env.game.trick.leader,
                                           card=convert_numerical_to_card(skat_and_cs[cs_index, 3 * trick - 1])))
                    except ValueError or InvalidPlayerMove:
                        skip = True
                        break

                    # if the player sits in the middle of this trick
                    if env.game.trick.get_current_player() == current_player:
                        # in position of the second player, there is one open card
                        open_cards = convert_numerical_to_enc(skat_and_cs[cs_index, 3 * trick - 1],
                                                              encoding=card_enc) + [0] * card_dim

                        game_state += open_cards + get_hand_cards(current_player, encoding=card_enc)

                        cat_action = [0] * act_dim
                        try:
                            cat_action[
                                current_player.cards.index(
                                    convert_numerical_to_card(skat_and_cs[cs_index, 3 * trick]))] = 1
                            actions.append(cat_action)
                        except ValueError:
                            skip = True
                            break

                    try:
                        # iterates over players
                        # each time PlayCardAction is called the role of the current player rotates
                        env.state_machine.handle_action(
                            PlayCardAction(player=env.game.trick.get_current_player(),
                                           card=convert_numerical_to_card(skat_and_cs[cs_index, 3 * trick])))
                    except ValueError or InvalidPlayerMove:
                        skip = True
                        break

                    # if the player sits in the rear of this trick
                    if env.game.trick.get_current_player() == current_player:
                        # in position of the third player, there are two open cards
                        open_cards = convert_numerical_to_enc(skat_and_cs[cs_index, 3 * trick - 1],
                                                              encoding=card_enc) + \
                                     convert_numerical_to_enc(skat_and_cs[cs_index, 3 * trick], encoding=card_enc)

                        game_state += open_cards + get_hand_cards(current_player, encoding=card_enc)

                        cat_action = [0] * act_dim
                        try:
                            cat_action[current_player.cards.index(
                                convert_numerical_to_card(skat_and_cs[cs_index, 3 * trick + 1]))] = 1
                            actions.append(cat_action)
                        except ValueError:
                            skip = True
                            break

                    try:
                        # iterates over players
                        # each time PlayCardAction is called the role of the current player rotates
                        env.state_machine.handle_action(
                            PlayCardAction(player=env.game.trick.get_current_player(),
                                           card=convert_numerical_to_card(skat_and_cs[cs_index, 3 * trick + 1])))
                    except ValueError or InvalidPlayerMove:
                        skip = True
                        break

                    last_trick = convert_numerical_to_enc(
                        skat_and_cs[cs_index, 3 * trick - 1], encoding=card_enc) + convert_numerical_to_enc(
                        skat_and_cs[cs_index, 3 * trick], encoding=card_enc) + convert_numerical_to_enc(
                        skat_and_cs[cs_index, 3 * trick + 1], encoding=card_enc)

                    # check if game was surrendered at this trick
                    if surrendered_trick == trick:
                        game_state, actions, rewards = \
                            surrender(won, current_player, soloist_points, trick, game_state, actions, rewards,
                                      state_dim)
                        break
                    else:
                        rewards.append(current_player.current_trick_points)

                        if trick == 1 and not hand:
                            rewards[-1] += skat1.get_value() + skat2.get_value()

                # if hand is played, adding the Skat points in the end of the game simulates not knowing them
                if hand:
                    skat_points = (skat_up[0].get_value() + skat_up[1].get_value())
                    if current_player.type == Player.Type.DECLARER:
                        score[0] += skat_points
                        rewards[-1] += skat_points
                    else:
                        score[1] += skat_points
                        rewards[-1] -= skat_points
                else:
                    # make the card points of the Skat visible for the defenders in the end of the game
                    if current_player.type != Player.Type.DECLARER:
                        score[1] += skat1.get_value() + skat2.get_value()

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
                rewards[-1] *= 1  # (1 if won else 0)

            # in the end of each game, insert the states, actions and rewards
            # with composite primary keys game_id and player perspective (1: forehand, 2: middle-hand, 3: rear-hand)
            # insert states
            game_state_table[3 * cs_index + i] = np.array_split(game_state, act_dim)  # [float(i) for i in game_state]
            # insert actions
            actions_table[3 * cs_index + i] = actions  # np.array_split([float(i) for i in actions], 12)
            # insert rewards
            rewards_table[3 * cs_index + i] = [[i] for i in rewards]

        cs_index = cs_index + 1

    return {
        "states": game_state_table,
        "actions": actions_table,
        "rewards": rewards_table,
    }, card_error_games

# if __name__ == '__main__':
# for championship in possible_championships:
#     point_rewards = True
#     print(f"Reading in championship {championship}")
#     data, _ = get_states_actions_rewards(championship,
#                                          games_indices=slice(0, -1),
#                                          point_rewards=point_rewards)
#     data_frame = pd.DataFrame(data)
#     data_train, data_test = train_test_split(data_frame, train_size=0.8, random_state=42)
#     dataset = DatasetDict({"train": Dataset.from_dict(data_train),
#                            "test": Dataset.from_dict(data_test)})
#     dataset.save_to_disk(f"./datasets/{championship}_without_surr_and_passed-pr_{point_rewards}")
