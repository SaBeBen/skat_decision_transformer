import mysql.connector
import logging

import numpy as np
from flask import Flask

app = Flask(__name__)

skat_wm_cs_data_path = "data/wm_skattisch_kf.CSV"

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
def get_skat_data_wm():
    mydb = mysql.connector.connect(
        host="mysqldb",
        user="root",
        password="Pgooelx15",
        database="skat_db"
    )
    cursor = mydb.cursor()

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


def upload_game_course(cursor, cs_path, card_format):
    # extract table name from path
    corresponding_table = cs_path.split(".")[0].split("/")[-1]
    try:
        # create a table for the card sequence (cs)
        cursor.execute(f"DROP TABLE IF EXISTS {corresponding_table}")  # skat_data_wc_cs
        cursor.execute(f"CREATE TABLE {corresponding_table} ("
                       f"GameID INT NOT NULL, "
                       f"Sd1 DECIMAL(2), Sd2 DECIMAL(2), CNr0 DECIMAL(2), CNr1 DECIMAL(2), CNr2 DECIMAL(2), CNr3 DECIMAL(2),"
                       f"CNr4 DECIMAL(2), CNr5 DECIMAL(2), CNr6 DECIMAL(2), CNr7 DECIMAL(2), CNr8 DECIMAL(2), "
                       f"CNr9 DECIMAL(2), CNr10 DECIMAL(2), CNr11 DECIMAL(2), CNr12 DECIMAL(2), CNr13 DECIMAL(2),"
                       f" CNr14 DECIMAL(2), CNr15 DECIMAL(2), CNr16 DECIMAL(2), CNr17 DECIMAL(2), CNr18 DECIMAL(2), "
                       f"CNr19 DECIMAL(2), CNr20 DECIMAL(2), CNr21 DECIMAL(2), CNr22 DECIMAL(2), CNr23 DECIMAL(2), "
                       f"CNr24 DECIMAL(2), CNr25 DECIMAL(2), CNr26 DECIMAL(2), CNr27 DECIMAL(2), CNr28 DECIMAL(2), "
                       f"CNr29 DECIMAL(2), CNr30 DECIMAL(2), CNr31 DECIMAL(2), SurrenderedAt DECIMAL(2), PRIMARY KEY(GameID)"
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
    try:
        # create a table for the card sequence (cs)
        cursor.execute(f"DROP TABLE IF EXISTS {corresponding_table}")  # skat_data_wc_cs
        cursor.execute(f"CREATE TABLE {corresponding_table} ("
                       f"GameID INT NOT NULL, "
                       f"Sd1 DECIMAL(2), Sd2 DECIMAL(2), CNr0 DECIMAL(2), CNr1 DECIMAL(2), CNr2 DECIMAL(2), CNr3 DECIMAL(2),"
                       f"CNr4 DECIMAL(2), CNr5 DECIMAL(2), CNr6 DECIMAL(2), CNr7 DECIMAL(2), CNr8 DECIMAL(2), "
                       f"CNr9 DECIMAL(2), CNr10 DECIMAL(2), CNr11 DECIMAL(2), CNr12 DECIMAL(2), CNr13 DECIMAL(2),"
                       f" CNr14 DECIMAL(2), CNr15 DECIMAL(2), CNr16 DECIMAL(2), CNr17 DECIMAL(2), CNr18 DECIMAL(2), "
                       f"CNr19 DECIMAL(2), CNr20 DECIMAL(2), CNr21 DECIMAL(2), CNr22 DECIMAL(2), CNr23 DECIMAL(2), "
                       f"CNr24 DECIMAL(2), CNr25 DECIMAL(2), CNr26 DECIMAL(2), CNr27 DECIMAL(2), CNr28 DECIMAL(2), "
                       f"CNr29 DECIMAL(2), CNr30 DECIMAL(2), CNr31 DECIMAL(2), SurrenderedAt DECIMAL(2), "
                       f"PRIMARY KEY(GameID)"
                       f")")
    except (mysql.connector.Error, mysql.connector.Warning) as e:
        print(e)
        return None

    # open the file and store it as a np array
    skat_wm_game_data = np.asarray(np.loadtxt(game_path, delimiter=",", dtype=int))

    cs_data_insert_query = f"INSERT INTO {corresponding_table} " \
                           f"(GameID, PlayerFH, PlayerMH, PlayerRH, Card1, Card2, Card3, Card4, Card5, Card6, Card7," \
                           f" Card8, Card9, Card10, Card11, Card12, Card13, Card14, Card15, Card16, Card17, Card18," \
                           f" Card19, Card20, Card21, Card22,Card23, Card24, Card25, Card26, Card27, Card28, Card29," \
                           f" Card30, Card31, Card32, PlayerID, Game, SurrenderedAt)" \
                           f"VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s," \
                           f" %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"

    "GameID", \
        # out  -- "IDGame", "IDTable", "IDVServer", "StartTime", "EndTime",\
    "PlayerFH", "PlayerMH", "PlayerRH", \
 \
    "Card1", "Card2", "Card3", "Card4", "Card5", "Card6",
    "Card7", "Card8", "Card9", "Card10", "Card11", "Card12", "Card13", "Card14",
    "Card15", "Card16", "Card17", "Card18", "Card19", "Card20", "Card21", "Card22",
    "Card23", "Card24", "Card25", "Card26", "Card27", "Card28", "Card29", "Card30",
    "Card31", "Card32", \
 \
    # out "CallValueFH", "CallValueMH", "CallValueRH", \
    "PlayerID", "Game",
    "With", "Without", "Hand", "Schneider", "SchneiderCalled", "Schwarz",
    "SchwarzCalled", "Ouvert", "PointsPlayer", "Won", "Miscall",
    "CardPointsPlayer", "AllPassed", "Surrendered", \
    "PlayerPosAtTableFH", "PlayerPosAtTableMH", "PlayerPosAtTableRH"

    # reads in every row and inputs them into the table
    for row in skat_wm_game_data:
        try:
            cursor.execute(cs_data_insert_query, row.tolist())
        except (mysql.connector.Error, mysql.connector.Warning) as e:
            print(e)
            return None


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

    # commit the changes, irrelevant if autocommit is activated
    mydb.commit()
    # closes the cursor IO
    cursor.close()

    return 'init database'


if __name__ == "__main__":
    # print(db_init())
    # print(get_skat_data_wm())
    app.run(host='0.0.0.0')
