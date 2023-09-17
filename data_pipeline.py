from typing import Tuple

import numpy as np
import pandas as pd
from datasets import DatasetDict, Dataset
from sklearn.model_selection import train_test_split

from tqdm import tqdm

from dt_skat_environment.card_representation_conversion import convert_numerical_to_card, convert_card_to_enc, convert_numerical_to_enc
from dt_skat_environment.environment import Env, get_trump_enc, initialise_hand_cards, get_hand_cards, ACT_DIM, get_dims_in_enc

from dt_skat_environment.game_engine.game.state.game_state_bid import PutDownSkatAction, BidCallAction, BidPassAction, \
    PickUpSkatAction
from dt_skat_environment.game_engine.game.state.game_state_play import PlayCardAction

from dt_skat_environment.game_engine.model.player import Player

POSSIBLE_CHAMPIONSHIPS = ["wc", "bl", "gc", "gtc", "rc"]


# To see more detailed description of raw data, read README.md in data.
def get_games(championship="wc",
              games_indices=slice(0, -1),
              include_grand=False,
              include_surr=False
              ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads in championships stored in CSV format in data folder.

    :param championship: Championship to read the data from.
    :param games_indices: The games to be selected after filtering games.
     Filtering includes all passed and depending on include_grand and include_surr those too.
    :param include_grand: Whether to include grand games.
    :param include_surr: Whether to include surrendered games.
    :return: First: NumPy array of meta and card information. Secondly: Numpy array of played card sequence.
    """

    if championship not in POSSIBLE_CHAMPIONSHIPS:
        raise ValueError(f"The championship {championship} does not exist in the database.")

    skat_cs_path = f"data/{championship}_card_sequence.CSV"

    skat_game_path = f"data/{championship}_game.CSV"

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
    # PlayerID (44), Game (45), Hand (48), Schneider (49), SchneiderCalled (50), Schwarz (51), SchwarzCalled (52),
    # Ouvert (53), PointsPlayer (54), Won (55), Miscall (56), AllPassed (58), Surrendered (59)
    # only needs cards of current player
    columns = [0, 6, 7, 8] + list(range(9, 41)) + [44, 45, 48, 49, 50, 51, 52, 53, 54, 55, 56, 58, 59]

    skat_game_data = skat_game_data.iloc[:, columns]

    # if someone surrendered, the rest of the cards are in the cs data in their order,
    # so we have to tell the exact position of surrender

    # exclude games where all players passed and those with a miscall
    # the latter leads to sloppy and unpredictable behaviour and is not noted correctly,
    # e.g. GameID544205 and IDGame 8xIN8X3LIMxICPx (for "Skat-Archiv")
    # only rules out 0.19 % of the games in wc (see analysis)
    skat_game_data = skat_game_data[(skat_game_data["AllPassed"] == 0)
                                    & (skat_game_data["Miscall"] == 0)]

    if not include_surr:
        skat_game_data = skat_game_data[(skat_game_data["Surrendered"] == 0)]
        skat_cs_data = skat_cs_data[(skat_cs_data["SurrenderedAt"] <= -1)]

    if not include_grand:
        skat_game_data = skat_game_data[(skat_game_data["Game"] != 24)]

    # exclude grand and null games
    skat_game_data = skat_game_data[
        (skat_game_data["Game"] != 23)
        & (skat_game_data["Game"] != 35)
        & (skat_game_data["Game"] != 46)
        & (skat_game_data["Game"] != 59)]

    # get rid of games that only exist in one of the tables by merging (can occur in inconsistent data)
    merged_game = pd.merge(skat_game_data, skat_cs_data, how="inner", on="GameID")

    merged_game = merged_game.iloc[games_indices]

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


def get_states_actions_rewards(
        championship="wc",
        games_indices=slice(0, 1000),
        point_rewards=False,
        perspective=(0, 1, 2),
        card_enc='mixed_comp',
        include_grand=False,
        include_surr=False
):
    """
    Load championship log data from csv files, replay the games, while at the same time transforming it into our
    state, action and reward representation.
    Only processes non-surrendered suit games, but is able to output all variants.


    :param championship: The championship to prepare. Input: abbreviation of the championship. Only "wc" works due to
    corrupt logging in the database.
    :param games_indices: The games to load.
    :param point_rewards: Use of the Seeger-Fabian score for the last reward. If False, uses a simple reward based
    on card points and game success.
    :param perspective: The player(s)' perspectives. Cannot
    :param card_enc: The card encoding to use. "one-hot", "mixed", "mixed_comp" and "one-hot_comp" are possible.
    :param include_grand: Whether to include grand games.
    :param include_surr: Whether to include surrendered games.

    :return: Dict of NumPy Array of states, actions and rewards. Also returns integer GameIDs from corrupt games.
    Additionally returns raw game tables for evaluation: Namely first states, meta and initial card information,
    correct actions in indexing format and the card sequence.
    """

    # load the meta information, initial card configuration (meta_and_cards) and the course of the game (skat_and_cs)
    meta_and_cards, skat_and_cs = get_games(championship=championship,
                                            games_indices=games_indices,
                                            include_grand=include_grand,
                                            include_surr=include_surr)

    card_dim, max_hand_len, state_dim = get_dims_in_enc(card_enc)

    amount_games = len(meta_and_cards)

    if not isinstance(perspective, tuple):
        # for perspective of first player
        perspective = [perspective]
        len_p = 1
    else:
        len_p = len(perspective)

    # tables to load the states, actions and rewards
    game_state_table = [[] * state_dim * 12] * amount_games * len_p
    actions_table = [[] * ACT_DIM * 12] * amount_games * len_p
    rewards_table = [[] * 12] * amount_games * len_p

    # use an own index to access the card sequence data, as the GameID is left out
    cs_index = 0

    # for skipping games where an illegal move occured, no game in skipped in the wc
    skip = False
    # return the ids of the corrupted games
    card_error_games = []

    for game in tqdm(meta_and_cards):

        for i in perspective:

            if skip:
                # handling of malicious logging in data
                skip = False
                print(f"broke out of game {game[0]} at cs_index {cs_index}")
                card_error_games.append(game[0])
                break

            env = Env(enc=card_enc)

            # player_id: ID of current solo player
            # trump: game the soloist plays
            # hand: binary encoding whether hand was played
            # soloist_points: points the soloist receives for playing a certain game
            player_id, trump, hand, soloist_points = game[-13], game[-12], game[-11], game[-5]

            # additional levels
            schneider, schneider_called, schwarz, schwarz_called, ouvert = game[-10], game[-9], game[-8], game[-7], \
                game[-6]

            # only represent the called levels in the state, the agent should not know if they are achieved
            game_level_bonus = [hand + schneider_called + schwarz_called + ouvert]

            # categorical encoding of trump suit color
            # if a grand is played --> [1, 1, 1, 1]
            # if a null game is played --> [0, 0, 0, 0]
            trump_enc = get_trump_enc(trump)

            # if a game was surrendered, the amount of cards played before surrender is stored in surrendered_card
            surrendered_card = skat_and_cs[cs_index, -1]

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
                    pos_p[(i + 1) % 3] = -1
                    pos_p[(i + 2) % 3] = 1

                    current_player2.type = Player.Type.DECLARER
                    current_player3.type = Player.Type.DEFENDER
                    env.state_machine.handle_action(BidCallAction(current_player2, 18))
                    env.state_machine.handle_action(BidPassAction(current_player, 18))
                    env.state_machine.handle_action(BidPassAction(current_player3, 18))
                else:
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

                # if the game is surrendered instantly
                if surrendered_card == 0:
                    game_state, actions, rewards = \
                        env.surrender(won, current_player, soloist_points, 0, game_state, actions, rewards, state_dim)
                    # in the end of each game, insert the states, actions and rewards
                    # with composite primary keys game_id and player perspective
                    # (1: forehand, 2: middle-hand, 3: rear-hand)
                    # insert states
                    game_state_table[3 * cs_index + i] = np.array_split(np.array(game_state, dtype=np.int16), ACT_DIM)
                    # insert actions
                    actions_table[3 * cs_index + i] = np.array(actions, dtype=np.uint8)
                    # insert rewards
                    rewards_table[3 * cs_index + i] = np.array([[i] for i in rewards], dtype=np.int16)
                    break

                env.game.get_declarer().cards.sort()

                if current_player.type == Player.Type.DECLARER:
                    put_card = [1]
                else:
                    put_card = [0]

                # first game state + game_level_bonus
                game_state = pos_p + put_card + score + trump_enc + last_trick + open_cards \
                             + get_hand_cards(current_player, encoding=card_enc)

                # ...put down Skat one by one
                # each Skat card needs its own action (due to fixed dimensions)
                if current_player.type == Player.Type.DECLARER:
                    # The declarer should put a card when not playing hand
                    put_card = [1]

                    # categorical encoding of played card as action: put first card
                    cat_action = [0] * ACT_DIM
                    try:
                        cat_action[current_player.cards.index(skat1)] = 1
                        actions.append(cat_action)
                    except:  # ValueError:
                        game_state, actions, rewards = \
                            env.surrender(won, current_player, soloist_points, 0, game_state, actions, rewards,
                                          state_dim)
                        # in the end of each game, insert the states, actions and rewards
                        # with composite primary keys game_id and player perspective
                        # (1: forehand, 2: middle-hand, 3: rear-hand)
                        # insert states
                        game_state_table[3 * cs_index + i] = np.array_split(np.array(game_state, dtype=np.int16),
                                                                            ACT_DIM)
                        # insert actions
                        actions_table[3 * cs_index + i] = np.array(actions, dtype=np.uint8)
                        # insert rewards
                        rewards_table[3 * cs_index + i] = np.array([[i] for i in rewards], dtype=np.int16)
                        break

                    try:
                        # put down first Skat card in the environment
                        env.state_machine.handle_action(PutDownSkatAction(env.game.get_declarer(), skat_down[0]))
                    except:  # ValueError or InvalidPlayerMove:
                        skip = True
                        game_state, actions, rewards = \
                            env.surrender(won, current_player, soloist_points, 0, game_state, actions, rewards,
                                          state_dim)
                        # in the end of each game, insert the states, actions and rewards
                        # with composite primary keys game_id and player perspective
                        # (1: forehand, 2: middle-hand, 3: rear-hand)
                        # insert states
                        game_state_table[3 * cs_index + i] = np.array_split(np.array(game_state, dtype=np.int16),
                                                                            ACT_DIM)
                        # insert actions
                        actions_table[3 * cs_index + i] = np.array(actions, dtype=np.uint8)
                        # insert rewards
                        rewards_table[3 * cs_index + i] = np.array([[i] for i in rewards], dtype=np.int16)
                        break

                    # the soloist knows how many points he possesses through the Skat,
                    # defenders do not have this information
                    score[0] += skat1.get_value()

                    # the last trick is the put Skat and padding in the beginning
                    last_trick = convert_card_to_enc(skat1, encoding=card_enc) + [0] * card_dim + [0] * card_dim

                    # + game_level_bonus
                    game_state += pos_p + put_card + score + trump_enc + last_trick + open_cards + get_hand_cards(
                        current_player, encoding=card_enc)

                    # categorical encoding of played card as action: put second card
                    cat_action = [0] * ACT_DIM
                    try:
                        cat_action[current_player.cards.index(skat2)] = 1
                        actions.append(cat_action)
                    except:  # ValueError:
                        skip = True
                        game_state, actions, rewards = \
                            env.surrender(won, current_player, soloist_points, 0, game_state, actions, rewards,
                                          state_dim)
                        # in the end of each game, insert the states, actions and rewards
                        # with composite primary keys game_id and player perspective
                        # (1: forehand, 2: middle-hand, 3: rear-hand)
                        # insert states
                        game_state_table[3 * cs_index + i] = np.array_split(np.array(game_state, dtype=np.int16),
                                                                            ACT_DIM)
                        # insert actions
                        actions_table[3 * cs_index + i] = np.array(actions, dtype=np.uint8)
                        # insert rewards
                        rewards_table[3 * cs_index + i] = np.array([[i] for i in rewards], dtype=np.int16)
                        break

                    try:
                        # put down second Skat card in the environment
                        env.state_machine.handle_action(PutDownSkatAction(env.game.get_declarer(), skat_down[1]))
                    except:  # ValueError or InvalidPlayerMove:
                        skip = True
                        game_state, actions, rewards = \
                            env.surrender(won, current_player, soloist_points, 0, game_state, actions, rewards,
                                          state_dim)
                        # in the end of each game, insert the states, actions and rewards
                        # with composite primary keys game_id and player perspective
                        # (1: forehand, 2: middle-hand, 3: rear-hand)
                        # insert states
                        game_state_table[3 * cs_index + i] = np.array_split(np.array(game_state, dtype=np.int16),
                                                                            ACT_DIM)
                        # insert actions
                        actions_table[3 * cs_index + i] = np.array(actions, dtype=np.uint8)
                        # insert rewards
                        rewards_table[3 * cs_index + i] = np.array([[i] for i in rewards], dtype=np.int16)
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
                    actions += [[0] * ACT_DIM, [0] * ACT_DIM]
                    rewards += [0, 0]

                    # + game_level_bonus
                    game_state += pos_p + put_card + score + trump_enc + last_trick + open_cards + get_hand_cards(
                        current_player, encoding=card_enc)

                    try:
                        # put Skat down in the environment
                        env.state_machine.handle_action(PutDownSkatAction(env.game.get_declarer(), skat_down))
                    except:  # ValueError or InvalidPlayerMove:
                        skip = True
                        game_state, actions, rewards = \
                            env.surrender(won, current_player, soloist_points, 0, game_state, actions, rewards,
                                          state_dim)
                        # in the end of each game, insert the states, actions and rewards
                        # with composite primary keys game_id and player perspective
                        # (1: forehand, 2: middle-hand, 3: rear-hand)
                        # insert states
                        game_state_table[3 * cs_index + i] = np.array_split(np.array(game_state, dtype=np.int16),
                                                                            ACT_DIM)
                        # insert actions
                        actions_table[3 * cs_index + i] = np.array(actions, dtype=np.uint8)
                        # insert rewards
                        rewards_table[3 * cs_index + i] = np.array([[i] for i in rewards], dtype=np.int16)
                        break
            else:
                # if the game is surrendered instantly
                if surrendered_card == 0:
                    game_state, actions, rewards = \
                        env.surrender(won, current_player, soloist_points, 0, game_state, actions, rewards, state_dim)
                    # in the end of each game, insert the states, actions and rewards
                    # with composite primary keys game_id and player perspective
                    # (1: forehand, 2: middle-hand, 3: rear-hand)
                    # insert states
                    game_state_table[3 * cs_index + i] = np.array_split(np.array(game_state, dtype=np.int16), ACT_DIM)
                    # insert actions
                    actions_table[3 * cs_index + i] = np.array(actions, dtype=np.uint8)
                    # insert rewards
                    rewards_table[3 * cs_index + i] = np.array([[i] for i in rewards], dtype=np.int16)
                    break

                # if hand is played, show to not put a card
                put_card = [0]

                # there is no action during the Skat putting when playing hand
                actions.extend([[0] * ACT_DIM, [0] * ACT_DIM])
                rewards.extend([0, 0])

                # if hand is played, there are two identical game states from the perspective of every player
                # + game_level_bonus
                game_state = pos_p + put_card + score + trump_enc + last_trick + open_cards \
                             + get_hand_cards(current_player, encoding=card_enc)

                # + game_level_bonus
                game_state += pos_p + put_card + score + trump_enc + last_trick + open_cards \
                              + get_hand_cards(current_player, encoding=card_enc)

            # declare the game variant
            env.declare_game_variant(trump, hand, schneider_called, schwarz_called, ouvert)

            # iterate over each trick
            for trick in range(1, 11):

                # the first score shows the current players score
                score[1] += env.game.get_last_trick_points() if current_player.current_trick_points == 0 else 0
                score[0] += current_player.current_trick_points

                # always put a card in non-surrendered games
                game_state += pos_p + [1] + score + trump_enc + last_trick

                # check if game was surrendered at this card
                if surrendered_card == ((trick - 1) * 3):
                    game_state, actions, rewards = \
                        env.surrender(won, current_player, soloist_points, trick, game_state, actions, rewards,
                                      state_dim)
                    break

                # if the player sits in the front of this trick
                if env.game.trick.leader == current_player:
                    # in position of first player, there are no open cards
                    open_cards = [0] * card_dim + [0] * card_dim

                    game_state += open_cards + get_hand_cards(current_player, encoding=card_enc)

                    cat_action = [0] * ACT_DIM
                    try:
                        cat_action[current_player.cards.index(
                            convert_numerical_to_card(skat_and_cs[cs_index, 3 * trick - 1]))] = 1
                        actions.append(cat_action)
                    except:  # ValueError:
                        skip = True
                        game_state, actions, rewards = \
                            env.surrender(won, current_player, soloist_points, trick, game_state, actions, rewards,
                                          state_dim)
                        break

                try:
                    # iterates over players
                    # each time PlayCardAction is called the role of the current player rotates
                    env.state_machine.handle_action(
                        PlayCardAction(player=env.game.trick.leader,
                                       card=convert_numerical_to_card(skat_and_cs[cs_index, 3 * trick - 1])))
                except:  # ValueError or InvalidPlayerMove:
                    skip = True
                    game_state, actions, rewards = \
                        env.surrender(won, current_player, soloist_points, trick, game_state, actions, rewards,
                                      state_dim)
                    break

                # check if game was surrendered at this card
                if surrendered_card == ((trick - 1) * 3 + 1):
                    game_state, actions, rewards = \
                        env.surrender(won, current_player, soloist_points, trick, game_state, actions, rewards,
                                      state_dim)
                    break
                # if the player sits in the middle of this trick
                if env.game.trick.get_current_player() == current_player:
                    # in position of the second player, there is one open card
                    open_cards = convert_numerical_to_enc(skat_and_cs[cs_index, 3 * trick - 1],
                                                          encoding=card_enc) + [0] * card_dim

                    game_state += open_cards + get_hand_cards(current_player, encoding=card_enc)

                    cat_action = [0] * ACT_DIM
                    try:
                        cat_action[
                            current_player.cards.index(
                                convert_numerical_to_card(skat_and_cs[cs_index, 3 * trick]))] = 1
                        actions.append(cat_action)
                    except:  # ValueError:
                        skip = True
                        game_state, actions, rewards = \
                            env.surrender(won, current_player, soloist_points, trick, game_state, actions, rewards,
                                          state_dim)
                        break

                try:
                    # iterates over players
                    # each time PlayCardAction is called the role of the current player rotates
                    env.state_machine.handle_action(
                        PlayCardAction(player=env.game.trick.get_current_player(),
                                       card=convert_numerical_to_card(skat_and_cs[cs_index, 3 * trick])))
                except:  # ValueError or InvalidPlayerMove:
                    skip = True
                    game_state, actions, rewards = \
                        env.surrender(won, current_player, soloist_points, trick, game_state, actions, rewards,
                                      state_dim)
                    break

                # check if game was surrendered at this card
                if surrendered_card == ((trick - 1) * 3 + 2):
                    game_state, actions, rewards = \
                        env.surrender(won, current_player, soloist_points, trick, game_state, actions, rewards,
                                      state_dim)
                    break
                # if the player sits in the rear of this trick
                if env.game.trick.get_current_player() == current_player:
                    # in position of the third player, there are two open cards
                    open_cards = convert_numerical_to_enc(skat_and_cs[cs_index, 3 * trick - 1],
                                                          encoding=card_enc) + \
                                 convert_numerical_to_enc(skat_and_cs[cs_index, 3 * trick], encoding=card_enc)

                    game_state += open_cards + get_hand_cards(current_player, encoding=card_enc)

                    cat_action = [0] * ACT_DIM
                    try:
                        cat_action[current_player.cards.index(
                            convert_numerical_to_card(skat_and_cs[cs_index, 3 * trick + 1]))] = 1
                        actions.append(cat_action)
                    except:  # ValueError:
                        skip = True
                        game_state, actions, rewards = \
                            env.surrender(won, current_player, soloist_points, trick, game_state, actions, rewards,
                                          state_dim)
                        break

                try:
                    # iterates over players
                    # each time PlayCardAction is called the role of the current player rotates
                    env.state_machine.handle_action(
                        PlayCardAction(player=env.game.trick.get_current_player(),
                                       card=convert_numerical_to_card(skat_and_cs[cs_index, 3 * trick + 1])))
                except:  # ValueError or InvalidPlayerMove:
                    skip = True
                    game_state, actions, rewards = \
                        env.surrender(won, current_player, soloist_points, trick, game_state, actions, rewards,
                                      state_dim)
                    break

                last_trick = convert_numerical_to_enc(
                    skat_and_cs[cs_index, 3 * trick - 1], encoding=card_enc) + convert_numerical_to_enc(
                    skat_and_cs[cs_index, 3 * trick], encoding=card_enc) + convert_numerical_to_enc(
                    skat_and_cs[cs_index, 3 * trick + 1], encoding=card_enc)

                # check if game was surrendered at this card
                if surrendered_card == (trick * 3 + 3):
                    game_state, actions, rewards = \
                        env.surrender(won, current_player, soloist_points, trick, game_state, actions, rewards,
                                      state_dim)
                    break
                else:
                    rewards.append(current_player.current_trick_points)

                    if trick == 1 and not hand and current_player.type == Player.Type.DECLARER:
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

                # soloist_points are calculated by the extended Seeger and Fabian system
                if current_player.type == Player.Type.DECLARER:
                    # add the points to the soloist
                    rewards[-1] = 0.9 * soloist_points + rewards[-1] * 0.1
                elif env.game.has_declarer_won(env.game.skat[0].get_value() + env.game.skat[1].get_value() ):
                    # give 40 discounted score points to the defenders after Fabian-Seeger score
                    rewards[-1] = 0.9 * 40 + rewards[-1] * 0.1
            else:
                # ...otherwise, give a 0 reward for lost and a positive reward for won games
                rewards[-1] *= 1 if won else 0

            # in the end of each game, insert the states, actions and rewards
            # with composite primary keys game_id and player perspective (1: forehand, 2: middle-hand, 3: rear-hand)
            # insert states
            game_state_table[3 * cs_index + i] = np.array_split(np.array(game_state, dtype=np.int16), ACT_DIM)
            # insert actions
            actions_table[3 * cs_index + i] = np.array(actions, dtype=np.uint8)
            # insert rewards
            rewards_table[3 * cs_index + i] = np.array([[i] for i in rewards], dtype=np.int16)

        cs_index = cs_index + 1

    first_states = [game_state_table[i][0] for i in range(len(game_state_table))]
    # first_states = None

    return {
        "states": game_state_table,
        "actions": actions_table,
        "rewards": rewards_table,
    }, card_error_games, first_states, \
        meta_and_cards, actions_table, skat_and_cs


if __name__ == '__main__':
    # this is a bit messy, as some championship logs are corrupt and produce errors while replaying
    # the wc is only flawed when loading surrendered games, as the logging is not consistent
    # in the position of surrender
    # Furthermore, there are experiments to faster load starting configurations for the self-play

    # possible_cs = ["gc", "gtc", "rc"]  # "bl", "gc",

    championship = "wc"

    card_encodings = ["one-hot"]  # , "mixed_comp", "one-hot_comp", "mixed"]
    # for championship in POSSIBLE_CHAMPIONSHIPS:
    # point_rewards = True
    print(f"Reading in championship {championship}...")

    for point_rewards in [True]:
        for enc in card_encodings:
            print(f"...with {enc} encoding...")
            # card_dim, max_hand_len, state_dim = get_dims_in_enc(enc)

            data, _, first_states, meta_and_cards, actions_table, skat_and_cs = get_states_actions_rewards(
                championship,
                include_surr=False,
                include_grand=False,
                games_indices=slice(
                    0, -1),
                point_rewards=point_rewards,
                card_enc=enc)

            data_df = pd.DataFrame(data)
            # first_states_df = np.repeat(first_states, 3, axis=0)
            first_states_df = pd.DataFrame(first_states)
            # to match the input of the games from every perspective
            meta_and_cards = np.repeat(meta_and_cards, 3, axis=0)
            meta_and_cards_df = pd.DataFrame(meta_and_cards)

            # We only need the test portion of first states and meta_and_cards for the two online evaluations.
            # We do not load the dataset from the disk in the manual evaluation, because the manual evaluation is
            # only for a shallow analysis and debugging
            data_train, data_test, _, first_states_test, _, meta_and_cards_test = train_test_split(
                data_df, first_states_df, meta_and_cards_df, train_size=0.8, random_state=42)

            # data_train, data_test = train_test_split(data_frame, train_size=0.8, random_state=42)

            dataset = DatasetDict({"train": Dataset.from_dict(data_train),
                                   "test": Dataset.from_dict(data_test),
                                   "first_states_test": Dataset.from_dict(first_states_test),
                                   "meta_and_cards_test": Dataset.from_dict(meta_and_cards_test)
                                   })
            dataset.save_to_disk(
                f"./datasets/{championship}-surr_grand-pr_{point_rewards}-{enc}-card_put")
