import mysql.connector
import logging

import numpy as np
import pandas as pd
from flask import Flask

from data_pipeline import convert_one_hot_to_card, Env, convert_card_to_one_hot
from game.state.game_state_play import PlayCardAction
from model.player import Player

app = Flask(__name__)

skat_wm_cs_data_path = "data/wc_card_sequence.CSV"
skat_wm_game_data_path = "data/wc_game.CSV"

logger = logging.getLogger("mysql.connector")
logger.setLevel(logging.WARNING)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s- %(message)s")

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

file_handler = logging.FileHandler("py_mysql_app.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


@app.route('/')
def hello_world():
    return 'Hello, Docker!'


@app.route('/get_skat_data_wc_cs')
def get_skat_data_wc_cs():
    mydb = mysql.connector.connect(
        host="mysqldb",
        user="root",
        password="Pgooelx15",
        database="skat_db"
    )
    cursor = mydb.cursor()
    logger.info(f"User \"root\" accesses the WC card sequence table in the skat_db.")

    try:
        # query the card sequence per game
        cursor.execute("SELECT * FROM skat_data_wc_cs LIMIT 3;")
        # row_headers = [x[0] for x in cursor.description]  # this will extract row headers
    except (mysql.connector.Error, mysql.connector.Warning) as e:
        print(e)
        return None
    finally:
        results = cursor.fetchall()
        cursor.close()
        return results


@app.route('/get_skat_data_wc')
def get_skat_data_wc():
    mydb = mysql.connector.connect(
        host="mysqldb",
        user="root",
        password="Pgooelx15",
        database="skat_db"
    )
    cursor = mydb.cursor()
    logger.info(f"User \"root\" accesses the WC data table in the skat_db.")

    states = actions = rewards = []

    championship = skat_wm_game_data_path.split(".")[0].split("/")[-1]

    try:
        # query the card sequence per game
        cursor.execute(f"SELECT * FROM states_{championship} LIMIT 10;")
        states = cursor.fetchall()

        cursor.execute(f"SELECT * FROM actions_{championship} LIMIT 10;")
        actions = cursor.fetchall()

        cursor.execute(f"SELECT * FROM rewards_{championship} LIMIT 10;")
        rewards = cursor.fetchall()
    except (mysql.connector.Error, mysql.connector.Warning) as e:
        print(e)
        return None
    finally:
        # results = cursor.fetchall()
        cursor.close()
        return {"states": states,
                "actions": actions,
                "rewards": rewards}


def upload_game_course(cursor, cs_path, card_format):
    # extract table name from path
    corresponding_table = cs_path.split(".")[0].split("/")[-1]
    try:
        # create a table for the card sequence (cs)
        cursor.execute(f"DROP TABLE IF EXISTS {corresponding_table}")  # skat_data_wc_cs
        cursor.execute(f"CREATE TABLE {corresponding_table} ("
                       f"GameID INT NOT NULL, "
                       f"Sd1 TINYINT, Sd2 TINYINT, CNr0 TINYINT, CNr1 TINYINT, CNr2 TINYINT, CNr3 TINYINT,"
                       f"CNr4 TINYINT, CNr5 TINYINT, CNr6 TINYINT, CNr7 TINYINT, CNr8 TINYINT, "
                       f"CNr9 TINYINT, CNr10 TINYINT, CNr11 TINYINT, CNr12 TINYINT, CNr13 TINYINT,"
                       f" CNr14 TINYINT, CNr15 TINYINT, CNr16 TINYINT, CNr17 TINYINT, CNr18 TINYINT, "
                       f"CNr19 TINYINT, CNr20 TINYINT, CNr21 TINYINT, CNr22 TINYINT, CNr23 TINYINT, "
                       f"CNr24 TINYINT, CNr25 TINYINT, CNr26 TINYINT, CNr27 TINYINT, CNr28 TINYINT, "
                       f"CNr29 TINYINT, CNr30 TINYINT, CNr31 TINYINT, SurrenderedAt TINYINT, PRIMARY KEY(GameID)"
                       f")")
    except (mysql.connector.Error, mysql.connector.Warning) as e:
        print(e)
        return None

    # open the file and store it as a np array
    skat_wm_cs_data = np.asarray(np.loadtxt(cs_path, delimiter=",", dtype=int))

    """
    Sort the data set according to the playing order:
    We have an array where each column is a card, each row is a game, and the numbers indicate when the card was played
    We want an array where each row is a game, each column is a position in the game, and the values indicate cards
    """
    skat_wm_cs_data[:, 3:35] = np.argsort(skat_wm_cs_data[:, 3:35], axis=1)

    cs_data_insert_query = f"INSERT INTO {corresponding_table} " \
                           f"(GameID, Sd1, Sd2 , CNr0, CNr1, CNr2, CNr3, CNr4, CNr5, CNr6, CNr7, CNr8," \
                           f"CNr9, CNr10, CNr11, CNr12, CNr13, CNr14, CNr15, CNr16, CNr17, CNr18, CNr19," \
                           f"CNr20, CNr21, CNr22, CNr23, CNr24, CNr25 ,CNr26 ,CNr27 ,CNr28, CNr29," \
                           f"CNr30, CNr31, SurrenderedAt)" \
                           f"VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s," \
                           f" %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"

    # reads in every row and inputs them into the table
    for row in skat_wm_cs_data:
        try:
            cursor.execute(cs_data_insert_query, row.tolist())
        except (mysql.connector.Error, mysql.connector.Warning) as e:
            print(e)
            return None


def upload_game(cursor, game_path, card_format):
    # extract table name from path
    corresponding_table = game_path.split(".")[0].split("/")[-1]

    # GameID (0), PlayerPos, Card1 (9):Card30 (38), Skat (39):Skat (40)  Game (45),
    # special treatment: Won (55), Miscall (56), AllPassed (58), Surrendered (59), PointsPlayer (54)
    #  PlayerFH (6), PlayerMH (7), PlayerRH (8), PlayerID (44)
    # only needs cards of current player
    try:
        # create a table for the card sequence (cs)
        cursor.execute(f"DROP TABLE IF EXISTS {corresponding_table}")  # skat_data_wc_cs
        cursor.execute(f"CREATE TABLE {corresponding_table} ("
                       f"GameID INT NOT NULL, PlayerPos TINYINT,"
                       f"CNr1 TINYINT, CNr2 TINYINT, CNr3 TINYINT,"
                       f"CNr4 TINYINT, CNr5 TINYINT, CNr6 TINYINT, CNr7 TINYINT, CNr8 TINYINT, "
                       f"CNr9 TINYINT, CNr10 TINYINT, CNr11 TINYINT, CNr12 TINYINT, CNr13 TINYINT,"
                       f" CNr14 TINYINT, CNr15 TINYINT, CNr16 TINYINT, CNr17 TINYINT, CNr18 TINYINT, "
                       f"CNr19 TINYINT, CNr20 TINYINT, CNr21 TINYINT, CNr22 TINYINT, CNr23 TINYINT, "
                       f"CNr24 TINYINT, CNr25 TINYINT, CNr26 TINYINT, CNr27 TINYINT, CNr28 TINYINT, "
                       f"CNr29 TINYINT, CNr30 TINYINT, Skat1 TINYINT, Skat2 TINYINT, Game SMALLINT,"
                       f"PointsPlayer SMALLINT, Won BIT, Miscall BIT, AllPassed BIT, Surrendered BIT, "
                       f"PRIMARY KEY(GameID)"
                       f")")
    except (mysql.connector.Error, mysql.connector.Warning) as e:
        print(e)
        return None

    # open the file and store it as a np array
    skat_wm_game_data = np.asarray(np.loadtxt(game_path, delimiter=",", dtype=int))

    cs_data_insert_query = f"INSERT INTO {corresponding_table} " \
                           f"(GameID, PlayerPos, Card1, Card2, Card3, Card4, Card5, Card6, Card7," \
                           f" Card8, Card9, Card10, Card11, Card12, Card13, Card14, Card15, Card16, Card17, Card18," \
                           f" Card19, Card20, Card21, Card22, Card23, Card24, Card25, Card26, Card27, Card28, Card29," \
                           f" Card30, Skat1, Skat2, Game, PointsPlayer, Won, Miscall, AllPassed, Surrendered)" \
                           f"VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s " \
                           f" %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"

    # reads in every row and inputs them into the table
    for row in skat_wm_game_data:
        pos = 0
        if row[44] == row[7]:
            pos = 1
        elif row[44] == row[8]:
            pos = 2

        # GameID (0), PlayerPos, Card1 (9):Card30 (38), Skat (39):Skat (40), Game (45), PointsPlayer (54), Won (55),
        # Miscall (56), AllPassed (58), Surrendered (59)
        row = row[0] + [pos] + row[9:41] + row[45] + row[54] + row[55] + row[56] + row[58] + row[59]
        try:
            cursor.execute(cs_data_insert_query, row.tolist())
        except (mysql.connector.Error, mysql.connector.Warning) as e:
            print(e)
            return None


def convert_one_hot_to_card_init(card):
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

def create_dataset(cursor, game_path):
    game_table = game_path.split(".")[0].split("/")[-1]

    cursor.execute(f"DROP TABLE IF EXISTS states_{game_table}")
    # [pos_tp] + [trump] + last_trick + open_cards + env.player1.cards
    cursor.execute(f"CREATE TABLE states_{game_table} ("
                   f"GameID INT NOT NULL, Player TINYINT, CoPlayerPos TINYINT, Trump TINYINT, "
                   f"LastT1 TINYINT, LastT2 TINYINT, LastT3 TINYINT,"
                   f"OC1 TINYINT, OC2 TINYINT, "
                   f"CNr1 TINYINT, CNr2 TINYINT, CNr3 TINYINT, CNr4 TINYINT, CNr5 TINYINT, CNr6 TINYINT, "
                   f"CNr7 TINYINT, CNr8 TINYINT, CNr9 TINYINT, CNr10 TINYINT, CNr11 TINYINT, CNr12 TINYINT, "
                   f"PRIMARY KEY(GameID, Player)"
                   f")")

    states_insert_query = f"INSERT INTO states_{game_table}" \
                          f"(GameID, Player, CoPlayerPos, Trump, LastT1, LastT2, LastT3, OC1, OC2, CNr1, CNr2, " \
                          f"CNr3, CNr4, CNr5, CNr6, CNr7, CNr8, CNr9, CNr10, CNr11, CNr12)" \
                          f"VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"

    cursor.execute(f"DROP TABLE IF EXISTS actions_{game_table}")
    cursor.execute(f"CREATE TABLE actions_{game_table} ("
                   f"GameID INT NOT NULL, Player TINYINT, Action1 TINYINT, Action2 TINYINT, Action3 TINYINT, "
                   f"Action4 TINYINT, Action5 TINYINT, Action6 TINYINT, Action7 TINYINT, Action8 TINYINT,"
                   f"Action9 TINYINT, Action10 TINYINT,"
                   f"PRIMARY KEY(GameID, Player)"
                   f")")

    actions_insert_query = f"INSERT INTO actions_{game_table}" \
                           f"(GameID, Player, Action1, Action2, Action3, Action4, Action5, Action6," \
                           f" Action7, Action8, Action9, Action10, Action11, Action12)" \
                           f"VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"

    cursor.execute(f"DROP TABLE IF EXISTS rewards_{game_table}")
    cursor.execute(f"CREATE TABLE rewards_{game_table} ("
                   f"GameID INT NOT NULL, , Player TINYINT, Reward1 TINYINT, Reward2 TINYINT, Reward3 TINYINT,"
                   f"Reward4 TINYINT, Reward5 TINYINT, Reward6 TINYINT, Reward7 TINYINT, Reward8 TINYINT,"
                   f"Reward9 TINYINT, Reward10 TINYINT, Reward11 TINYINT, Reward12 TINYINT,"
                   f"PRIMARY KEY(GameID)"
                   f")")

    rewards_insert_query = f"INSERT INTO rewards_{game_table}" \
                           f"(GameID,  Player, Reward1, Reward2, Reward3, Reward4, Reward5, Reward6," \
                           f"Reward7, Reward8, Reward9, Reward10, Reward11, Reward12)" \
                           f"VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"

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
                            "CNr18", "CNr19", "CNr20", "CNr21", "CNr22", "CNr23", "CNr24", "CNr25", "CNr26",
                            "CNr27",
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
        actions.append(0)
    else:
        # action of surrendering
        actions.append(-2)
        # default behaviour (gives a negative reward here)
        rewards.append(current_player.current_trick_points)

    # pad the states with 0s
    game_state = np.pad(game_state, (0, (11 - trick - 1) * state_dim))

    actions = np.pad(actions, (0, 11 - trick - 2))

    # pad the rewards with 0s
    rewards = np.pad(rewards, (0, 11 - trick - 1))

    return game_state, actions, rewards

def get_states_actions_rewards(championship="wc", amount_games=1000, point_rewards=False):
    meta_and_cards, skat_and_cs = get_game(game=championship)

    # position of the team player with respect to own pos; if 0 -> soloist
    # alternating players perspective = {FH/MH/RH}
    # -> pos_tp
    # -> hand_cards

    game_state_table = [[] * state_dim * 10] * amount_games * 3

    actions_table = [[] * act_dim * 10] * amount_games * 3

    rewards_table = [[] * 10] * amount_games * 3

    # select available cudas for faster matrix computation
    # device = torch.device("cuda")

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
            skat_up = [convert_one_hot_to_card_init(game[34]), convert_one_hot_to_card_init(game[35])]

            # put down Skat...
            skat_down = [convert_one_hot_to_card_init(skat_and_cs[cs_index, 0]),
                         convert_one_hot_to_card_init(skat_and_cs[cs_index, 1])]

            # categorical encoding of trump suit color
            # if a grand is played --> [1, 1, 1, 1]
            # if a null game is played --> [0, 0, 0, 0]     # TODO: implement null ouvert
            trump_enc = get_trump(trump)

            # if a game was surrendered, the amount of tricks played before surrender is stored in surrendered trick
            surrendered_trick = floor(skat_and_cs[cs_index, -1] / 3)

            # skip start of the game (shuffling and dealing)
            env.state_machine.state_finished_handler()

            # ...instead, initialise the hand cards
            current_player.set_cards(
                [convert_one_hot_to_card_init(card) for card in
                 game[4 + 10 * i:14 + 10 * i].tolist()])
            current_player2.set_cards(
                [convert_one_hot_to_card_init(card) for card in
                 game[4 + ((10 + 10 * i) % 30):4 + ((20 + 10 * i - 1) % 30 + 1)].tolist()])
            current_player3.set_cards(
                [convert_one_hot_to_card_init(card) for card in
                 game[4 + ((20 + 10 * i) % 30):4 + ((30 + 10 * i - 1) % 30 + 1)].tolist()])

            # sort the cards to make hands reproducible, improve readability for attention mechanism (and humans)
            current_player.cards.sort()
            current_player2.cards.sort()
            current_player3.cards.sort()
            # TODO: access the player card with fast search, for instance binary search

            # initialise Skat in the environment
            env.game.skat.extend([skat_up[0], skat_up[1]])

            # initialise roles, encode positions and simulate bidding
            if agent_player == player_id:

                # encode the position of the team partner
                pos_tp = 0
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
                    pos_tp = 2
                    agent_player = (i + 1) % 3
                    soloist = current_player2

                    current_player2.type = Player.Type.DECLARER
                    current_player3.type = Player.Type.DEFENDER
                    env.state_machine.handle_action(BidCallAction(current_player2, 18))
                    env.state_machine.handle_action(BidPassAction(current_player, 18))
                    env.state_machine.handle_action(BidPassAction(current_player3, 18))
                else:
                    pos_tp = 1
                    agent_player = (i + 2) % 3
                    soloist = current_player3

                    current_player3.type = Player.Type.DECLARER
                    current_player2.type = Player.Type.DEFENDER
                    env.state_machine.handle_action(BidCallAction(current_player3, 18))
                    env.state_machine.handle_action(BidPassAction(current_player, 18))
                    env.state_machine.handle_action(BidPassAction(current_player2, 18))

            # there is no card revealed during Skat putting
            open_cards = [-1, -1]

            # during Skat selection, there is no last trick,
            # it will only be the last trick for the soloist when putting the Skat down
            last_trick = [-1, -1, -1]

            if not hand:
                # pick up the Skat
                env.state_machine.handle_action(PickUpSkatAction(soloist))

                # update hand cards: they will contain the Skat
                hand_cards = [convert_card_to_one_hot(card) for card in current_player.cards]

                if soloist != current_player:
                    # pad the current cards to a length of 12, if agent does not pick up Skat
                    hand_cards.extend([0, 0])

                # ...in the game state
                game_state = np.concatenate([pos_tp, trump_enc, last_trick, open_cards, hand_cards],
                                            # dtype=np.int8,
                                            axis=None)

                # ...in the environment
                env.state_machine.handle_action(PutDownSkatAction(soloist, skat_down))
            else:
                # update hand cards: they will not contain the Skat
                hand_cards = [convert_card_to_one_hot(card) for card in current_player.cards]

                # pad the current cards to a length of 12
                hand_cards.extend([0, 0])

                game_state = np.concatenate([pos_tp, trump_enc, last_trick, open_cards, hand_cards],
                                            # dtype=np.int8,
                                            axis=None)

            # ...put down Skat in the data (s, a, r)
            # each Skat card needs its own action (due to act_dim)
            if current_player.type == Player.Type.DECLARER and not hand:

                # the last trick is the put Skat and padding in the beginning
                last_trick = [skat_and_cs[cs_index, 0], -1, -1]

                game_state = np.concatenate([game_state, pos_tp, trump_enc, last_trick, open_cards, hand_cards],
                                            # dtype=np.int8,
                                            axis=None)

                # the last trick is the put Skat and padding in the beginning
                last_trick = [skat_and_cs[cs_index, 0], skat_and_cs[cs_index, 1], -1]

                # it is not necessary to simulate sequential putting with actions and rewards
                # instantly get rewards of the put Skat
                rewards.extend([skat_down[0].get_value(), skat_down[1].get_value()])

                # if agent is player, select Skat by putting down two cards
                actions.extend([skat_and_cs[cs_index, 0], skat_and_cs[cs_index, 1]])
            else:
                # the process of Skat selection is not visible for defenders,
                # thus it is padded and the defenders cards do not change
                actions.extend([0, 0])
                rewards.extend([0, 0])
                last_trick = [0, 0, 0]

                game_state = np.concatenate([game_state, pos_tp, trump_enc, last_trick, open_cards, hand_cards],
                                            # dtype=np.int8,
                                            axis=None)

            # TODO: uniform padding with 0 -> cards from 1-32

            # hand_cards = [convert_card_to_one_hot(card) for card in current_player.cards]
            # hand_cards.extend([0, 0])
            #
            # # add starting state
            # game_state = np.concatenate([game_state, pos_tp, trump_enc, last_trick, open_cards, hand_cards],
            #                             dtype=np.int8,
            #                             axis=None)

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
                        open_cards = [-1, -1]

                        # convert each card to the desired encoding
                        hand_cards = [convert_card_to_one_hot(card) for card in current_player.cards]

                        # pad the cards to a length of 12
                        hand_cards = np.pad(hand_cards, (trick + 1, 0))

                        game_state = np.concatenate(
                            [game_state, pos_tp, trump_enc, last_trick, open_cards, hand_cards],
                            # dtype=np.int8,
                            axis=None)

                        actions.append(skat_and_cs[cs_index, 3 * trick - 1])

                    env.state_machine.handle_action(
                        PlayCardAction(player=env.game.trick.leader,
                                       card=convert_one_hot_to_card_init(skat_and_cs[cs_index, 3 * trick - 1])))

                    # if the player sits in the middle this trick
                    if env.game.trick.get_current_player() == current_player:
                        # in position of the second player, there is one open card
                        open_cards = [skat_and_cs[cs_index, 3 * trick - 1], -1]

                        # convert each card to the desired encoding
                        hand_cards = [convert_card_to_one_hot(card) for card in current_player.cards]

                        # pad the cards to a length of 12
                        hand_cards = np.pad(hand_cards, (trick + 1, 0))

                        game_state = np.concatenate(
                            [game_state, pos_tp, trump_enc, last_trick, open_cards, hand_cards],
                            # dtype=np.int8,
                            axis=None)

                        actions.append(skat_and_cs[cs_index, 3 * trick])

                    env.state_machine.handle_action(
                        PlayCardAction(player=env.game.trick.get_current_player(),
                                       card=convert_one_hot_to_card_init(skat_and_cs[cs_index, 3 * trick])))

                    # if the player sits in the rear this trick
                    if env.game.trick.get_current_player() == current_player:
                        # in position of the third player, there are two open cards
                        open_cards = [skat_and_cs[cs_index, 3 * trick - 1],
                                      skat_and_cs[cs_index, 3 * trick]]

                        # convert each card to the desired encoding
                        hand_cards = [convert_card_to_one_hot(card) for card in current_player.cards]

                        # pad the cards to a length of 12
                        hand_cards = np.pad(hand_cards, (trick + 1, 0))

                        game_state = np.concatenate(
                            [game_state, pos_tp, trump_enc, last_trick, open_cards, hand_cards],
                            # dtype=np.int8,
                            axis=None)

                        actions.append(skat_and_cs[cs_index, 3 * trick + 1])

                    env.state_machine.handle_action(
                        PlayCardAction(player=env.game.trick.get_current_player(),
                                       card=convert_one_hot_to_card_init(skat_and_cs[cs_index, 3 * trick + 1])))

                    last_trick = [skat_and_cs[cs_index, 3 * trick]] + [skat_and_cs[cs_index, 3 * trick]] + [
                        skat_and_cs[cs_index, 3 * trick + 1]]

                    # check if game was surrendered at this trick
                    if surrendered_trick == trick:
                        game_state, actions, rewards = \
                            surrender(won, current_player, soloist_points, trick, game_state, actions, rewards)
                        break
                    else:
                        rewards.append(current_player.current_trick_points)

                if hand:
                    if pos_tp == 0:
                        # if hand is played, add the Skat points in the end of the game to simulate not knowing it
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
            # insert states
            cursor.execute(states_insert_query, [game, agent_player] + [float(i) for i in game_state])
            # insert actions
            cursor.execute(actions_insert_query, [game, agent_player] + [float(i) for i in actions])
            # insert rewards
            cursor.execute(rewards_insert_query, [game, agent_player] + [float(i) for i in rewards])

        cs_index = cs_index + 1


"""The following function initializes a database called skat_db, 
reads in the data and inserts it into the corresponding table"""


@app.route('/db_init')
def db_init():
    # connect to the MySQL database
    mydb = mysql.connector.connect(
        host="mysqldb",
        user="root",
        password="Pgooelx15"
    )
    # create a cursor to operate on the db
    cursor = mydb.cursor()

    try:
        # create the database
        cursor.execute("DROP DATABASE IF EXISTS skat_db")
        cursor.execute("CREATE DATABASE skat_db")
        cursor.execute("USE skat_db")
    except (mysql.connector.Error, mysql.connector.Warning) as e:
        print(e)
        return None

    # upload the game course
    upload_game_course(cursor, skat_wm_cs_data_path, card_format="one_hot")
    # create_dataset(cursor, skat_wm_game_data_path)

    # commit the changes, irrelevant if autocommit is activated
    mydb.commit()
    # closes the cursor IO
    cursor.close()

    return 'init database'


if __name__ == "__main__":
    # print(db_init())
    # print(get_skat_data_wm())
    app.run(host='0.0.0.0')
