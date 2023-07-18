import mysql.connector
import logging

import numpy as np
import pandas as pd
from flask import Flask

from decision_transformer import convert_one_hot_to_card, Env, convert_card_to_one_hot
from game.state.game_state_play import PlayCardAction
from model.player import Player

app = Flask(__name__)

skat_wm_cs_data_path = "data/wm_skattisch_kf.CSV"
skat_wm_game_data_path = "data/wm_skattisch_spiel.CSV"

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

    # Only use the relevant information: Skat and 30 played cards
    # skat_and_cs = np.asarray(np.loadtxt(skat_wm_cs_data_path, delimiter=",", dtype=int,
    #                                     usecols=list(range(1, 33))))

    skat_wm_cs_data_frame = pd.read_csv(skat_wm_cs_data_path, header=None)

    skat_wm_cs_data = skat_wm_cs_data_frame

    skat_wm_cs_data.columns = ["GameID", "Sd1", "Sd2", "CNr0", "CNr1", "CNr2", "CNr3", "CNr4", "CNr5", "CNr6", "CNr7",
                               "CNr8", "CNr9", "CNr10", "CNr11", "CNr12", "CNr13", "CNr14", "CNr15", "CNr16", "CNr17",
                               "CNr18", "CNr19", "CNr20", "CNr21", "CNr22", "CNr23", "CNr24", "CNr25", "CNr26", "CNr27",
                               "CNr28", "CNr29", "CNr30", "CNr31", "SurrenderedAt"]

    # TODO: Surrendered and SurrenderedAT do not match
    skat_wm_cs_data = skat_wm_cs_data_frame[skat_wm_cs_data["SurrenderedAt"] == -1]

    skat_and_cs = skat_wm_cs_data.to_numpy()

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
    skat_wm_game_data_frame = skat_wm_game_data_frame[(skat_wm_game_data_frame["AllPassed"] == 0)
                                                      & (skat_wm_game_data_frame["Won"] == 1)
                                                      & (skat_wm_game_data_frame["Surrendered"] == 0)]

    meta_and_cards = skat_wm_game_data_frame.to_numpy()

    # position of the team player with respect to own pos; if 0 -> soloist
    # alternating players perspective = {FH/MH/RH}
    # -> pos_tp
    # -> hand_cards
    amount_games = 10

    # select available cudas for faster matrix computation
    # device = torch.device("cuda")

    env = Env()

    for game in meta_and_cards[:amount_games, :]:

        game_state = []

        actions = []

        rewards = []

        # initialise the hand cards
        env.player1.set_cards(meta_and_cards[game, 4:14].apply(lambda card: convert_one_hot_to_card(card)))
        env.player2.set_cards(skat_and_cs[game, 14:24].apply(lambda card: convert_one_hot_to_card(card)))
        env.player3.set_cards(skat_and_cs[game, 24:34].apply(lambda card: convert_one_hot_to_card(card)))

        # initialize the Skat
        skat = [meta_and_cards[game, 39], meta_and_cards[game, 40]]
        env.game.skat.extend(
            [convert_one_hot_to_card(skat[0]), convert_one_hot_to_card(skat[1])])

        # ID of current solo player
        player_ID = meta_and_cards[game, -6]

        # trump is defined by the game the soloist plays
        trump = meta_and_cards[game, -5]

        # we fixate the player on an index in the data set and
        # TODO: rotate
        agent_player = meta_and_cards[game, 1]

        if agent_player == player_ID:
            env.player1.type = Player.Type.DECLARER
            pos_tp = 0
        else:
            env.player1.type = Player.Type.DEFENDER
            if player_ID == meta_and_cards[game, 2]:
                pos_tp = 2
            else:
                pos_tp = 1

        if env.player1.type == Player.Type.DECLARER:
            # if agent is player, select Skat

            actions.extend(skat)
            rewards.extend([convert_one_hot_to_card(skat_and_cs[game, 0]).get_value(),
                            convert_one_hot_to_card(skat_and_cs[game, 1]).get_value()])
            last_trick = skat.append(-1)  # Skat and padding in the beginning
        else:
            actions.extend([-1, -1])
            rewards.extend([0, 0])
            last_trick = [-1, -1, -1]

        open_cards = [-1, -1]  # there is no card revealed during Skat putting

        hand_cards = [convert_card_to_one_hot(card) for card in env.player1.cards]

        game_state = [pos_tp] + [trump] + last_trick + open_cards + hand_cards

        for trick in range(1, 11):

            # if the player sits in the front this trick
            if env.game.trick.leader == env.player1:
                open_cards = [-1, -1]  # in position of first player, there are no open cards

                # convert each card to the desired encoding
                hand_cards = [convert_card_to_one_hot(card) for card in env.player1.cards]

                # pad the cards to a length of 12
                hand_cards = np.pad(hand_cards, (trick, 0))

                game_state += [pos_tp] + [trump] + last_trick + open_cards + hand_cards

                actions.append(skat_and_cs[game, 3 * trick - 1])


            env.state_machine.handle_action(
                PlayCardAction(player=env.game.trick.leader, card=skat_and_cs[game, 3 * trick - 1]))

            # if the player sits in the middle this trick
            if env.game.trick.get_current_player() == env.player1:
                open_cards = [skat_and_cs[game, 3 * trick - 1],
                              [-1]]  # in position of the second player, there is one open card

                # convert each card to the desired encoding
                hand_cards = [convert_card_to_one_hot(card) for card in env.player1.cards]

                # pad the cards to a length of 12
                hand_cards = np.pad(hand_cards, (trick, 0))

                game_state += [pos_tp] + [trump] + last_trick + open_cards + hand_cards

                actions.append(skat_and_cs[game, 3 * trick])

            env.state_machine.handle_action(
                PlayCardAction(player=env.game.trick.get_current_player(), card=skat_and_cs[game, 3 * trick]))

            # if the player sits in the rear this trick
            if env.game.trick.get_current_player() == env.player1:
                open_cards = [skat_and_cs[game, 3 * trick - 1],
                              skat_and_cs[game, 3 * trick]]  # in position of the third player, there are two open cards

                # convert each card to the desired encoding
                hand_cards = [convert_card_to_one_hot(card) for card in env.player1.cards]

                # pad the cards to a length of 12
                hand_cards = np.pad(hand_cards, (trick, 0))

                game_state += [pos_tp] + [trump] + last_trick + open_cards + hand_cards

                actions.append(skat_and_cs[game, 3 * trick + 1])

            env.state_machine.handle_action(
                PlayCardAction(player=env.game.trick.get_current_player(), card=skat_and_cs[game, 3 * trick + 1]))

            rewards.append(env.player1.current_trick_points)

            last_trick = [skat_and_cs[game, 3 * trick - 1]] + [skat_and_cs[game, 3 * trick]] + [
                skat_and_cs[game, 3 * trick + 1]]

        # in the end of each game, insert the states, actions and rewards
        # insert states
        cursor.execute(states_insert_query, [game, agent_player] + game_state)
        # insert actions
        cursor.execute(actions_insert_query, [game, agent_player] + actions)
        # insert rewards
        cursor.execute(rewards_insert_query, [game, agent_player])


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
