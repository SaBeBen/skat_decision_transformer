
import pandas as pd
import random
from sklearn.model_selection import train_test_split

# %%
"""The dataset of the sequence the cards (cs) were played follows"""
skat_bl_cs_data_path = "C:/Users/sasch/Desktop/Uni/Bachelorarbeit/SaschaBenz/data/dl/bl_skattisch_kf.CSV"

skat_bl_cs_data_frame = pd.read_csv(skat_bl_cs_data_path, header=None)

# TODO: convert data to following encoding:
# ♦, ♥, ♠, ♣, {7, 8, 9, Q, K, 10, A}, J/T
skat_bl_cs_data = skat_bl_cs_data_frame

skat_bl_cs_data.columns = ["GameID", "Sd1", "Sd2", "KNr0", "KNr1", "KNr2", "KNr3", "KNr4", "KNr5", "KNr6", "KNr7",
                           "KNr8", "KNr9", "KNr10", "KNr11", "KNr12", "KNr13", "KNr14", "KNr15", "KNr16", "KNr17",
                           "KNr18", "KNr19", "KNr20", "KNr21", "KNr22", "KNr23", "KNr24", "KNr25", "KNr26", "KNr27",
                           "KNr28", "KNr29", "KNr30", "KNr31", "SurrenderedAt"]

head = skat_bl_cs_data.head(n=10)
print(head)

# %%
"""The table dataset follows"""
skat_bl_table_data_path = "C:/Users/sasch/Desktop/Uni/Bachelorarbeit/SaschaBenz/data/dl/bl_skattisch.CSV"

skat_bl_table_data_frame = pd.read_csv(skat_bl_table_data_path, header=None)

skat_bl_table_data_frame.columns =["IDTable", "Name", "Number", "PlayerID1", "PlayerID2", "PlayerID3", "PlayerID4",
                                    "Player1", "Player2", "Player3", "Player4", "Date", "IDVServer", "Series"]

skat_bl_table_data = skat_bl_table_data_frame

head = skat_bl_table_data.head(n=10)
print(head)

# %%

"""The game dataset follows"""

skat_bl_game_data_path = "C:/Users/sasch/Desktop/Uni/Bachelorarbeit/SaschaBenz/data/dl/bl_skattisch_spiel.CSV"

skat_bl_game_data_frame = pd.read_csv(skat_bl_game_data_path, header=None)

skat_bl_game_data_frame.columns = ["GameID", "IDGame", "IDTable", "IDVServer", "StartTime", "EndTime", "PlayerFH",
                                   "PlayerMH", "PlayerBH", "Card1", "Card2", "Card3", "Card4", "Card5", "Card6",
                                   "Card7", "Card8", "Card9", "Card10", "Card11", "Card12", "Card13", "Card14",
                                   "Card15", "Card16", "Card17", "Card18", "Card19", "Card20", "Card21", "Card22",
                                   "Card23", "Card24", "Card25", "Card26", "Card27", "Card28", "Card29", "Card30",
                                   "Card31", "Card32", "ReizwertVH", "ReizwertMH", "ReizwertHH", "SpielerID", "Spiel",
                                   "Mit", "Ohne", "Hand", "Schneider", "SchneiderAngesagt", "Schwarz",
                                   "SchwarzAngesagt", "Overt", "PunkteSpieler", "Gewonnen", "Ueberreizt",
                                   "SpielerAugen", "Eingepasst", "Aufgabe", "SpielerPosAmTischVH",
                                   "SpielerPosAmTischMH", "SpielerPosAmTischHH"]
# Sanity Checks

skat_bl_game_data_frame.isna.sum()

skat_bl_game_data_frame.__sizeof__()

skat_bl_game_data_frame["Gewonnen"].value_counts()

skat_bl_game_data_frame["Gewonnen"].value_counts()

# TODO: convert data to following encoding:
# ♦, ♥, ♠, ♣, {7, 8, 9, Q, K, 10, A}, J/T
skat_bl_game_data = skat_bl_game_data_frame

head = skat_bl_game_data.head(n=10)
print(head)

# create train and test sets
# skat_train, skat_test = train_test_split(skat_data, test_size=0.2, random_state=0)

# If data set is unsorted and randomly distributed
# train_data = skat_data[:n*0.8]
# test_data = skat_data[n*0.2:]



