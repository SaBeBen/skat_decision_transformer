#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split

#%%

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

#%%

# Sanity Checks

#%%

print(skat_bl_cs_data_frame.isna().sum().sum())

#%%

print(skat_bl_cs_data_frame["GameID"].size)

#%%

print(skat_bl_cs_data_frame["GameID"].duplicated().any())

#%%

skat_bl_table_data = skat_bl_cs_data_frame

head = skat_bl_cs_data.head(n=10)
print(head)


#%%

"""The table dataset follows"""
skat_bl_table_data_path = "C:/Users/sasch/Desktop/Uni/Bachelorarbeit/SaschaBenz/data/dl/bl_skattisch.CSV"

skat_bl_table_data_frame = pd.read_csv(skat_bl_table_data_path, header=None)

skat_bl_table_data_frame.columns = ["IDTable", "Name", "Number", "PlayerID1", "PlayerID2", "PlayerID3", "PlayerID4",
                                    "Player1", "Player2", "Player3", "Player4", "Date", "IDVServer", "Series"]

skat_bl_table_data_frame

#%%

# Sanity Checks

print(skat_bl_table_data_frame.isna().sum())

#%%

print(skat_bl_table_data_frame["IDTable"].size)

#%%

print(skat_bl_table_data_frame["IDTable"].duplicated().any())

#%%

head = skat_bl_table_data_frame.head(n=10)
print(head)

#%%

"""The game dataset follows"""

skat_bl_game_data_path = "C:/Users/sasch/Desktop/Uni/Bachelorarbeit/SaschaBenz/data/dl/bl_skattisch_spiel.CSV"

skat_bl_game_data_frame = pd.read_csv(skat_bl_game_data_path, header=None)

skat_bl_game_data_frame.columns = ["GameID", "IDGame", "IDTable", "IDVServer", "StartTime", "EndTime", "PlayerFH",
                                   "PlayerMH", "PlayerBH", "Card1", "Card2", "Card3", "Card4", "Card5", "Card6",
                                   "Card7", "Card8", "Card9", "Card10", "Card11", "Card12", "Card13", "Card14",
                                   "Card15", "Card16", "Card17", "Card18", "Card19", "Card20", "Card21", "Card22",
                                   "Card23", "Card24", "Card25", "Card26", "Card27", "Card28", "Card29", "Card30",
                                   "Card31", "Card32", "CallValueFH", "CallValueMH", "CallValueBH", "PlayerID", "Game",
                                   "With", "Without", "Hand", "Schneider", "SchneiderCalled", "Schwarz",
                                   "SchwarzCalled", "Overt", "PointsPlayer", "Won", "Miscall",
                                   "CardPointsPlayer", "AllPassed", "Surrendered", "PlayerPosAtTableFH",
                                   "PlayerPosAtTableMH", "PlayerPosAtTableBH"]

#%%

head = skat_bl_game_data_frame.head(n=10)
print(head)

#%% Sanity Checks

skat_bl_game_data_frame[["CallValueFH", "CallValueMH", "CallValueBH", "PlayerID", "Game",
                         "With", "Without", "Hand", "Schneider", "SchneiderCalled", "Schwarz",
                         "SchwarzCalled", "Overt", "PointsPlayer", "Won", "Miscall",
                         "CardPointsPlayer", "AllPassed", "Surrendered"]].describe()

#%%

skat_bl_game_data_frame[["CallValueFH", "CallValueMH", "CallValueBH", "PlayerPosAtTableFH", "PlayerPosAtTableMH",
                         "PlayerPosAtTableBH"]].describe()

#%%

print(skat_bl_game_data_frame.isna().sum().sum())

#%%

print(skat_bl_game_data_frame["GameID"].size)

#%%

print(skat_bl_game_data_frame["GameID"].duplicated().any())

#%%

print(skat_bl_game_data_frame["IDGame"].duplicated().any())

#%%

skat_bl_game_data_frame["Won"].value_counts(normalize=True)

#%%

skat_bl_game_data_frame["Schneider"].value_counts(normalize=True)

#%%

skat_bl_game_data_frame["Schwarz"].value_counts(normalize=True)

#%%

skat_bl_game_data_frame["Game"].value_counts()

#%%

skat_bl_game_data_frame["AllPassed"].value_counts(normalize=True)

#%%

skat_bl_game_data_frame["Miscall"].value_counts(normalize=True)

#%%

skat_bl_game_data_frame["Surrendered"].value_counts(normalize=True)



#%%

# Data Visualisation

#%%

# Analysis of games
game_variants = skat_bl_game_data_frame["Game"].value_counts(normalize=True)
game_variants = game_variants.rename(
    index={24: "Grand", 12: "Cross", 11: "Spades", 10: "Hearts", 9: "Diamonds", 0: "AllPassed", 23: "Null",
           46: "Null Overt", 59: "Null Overt Hand", 35: "Null Hand"})
game_variants["Null"] += game_variants["Null Overt"] + game_variants["Null Hand"] + game_variants["Null Overt Hand"]
game_variants = game_variants.drop(labels=["Null Overt", "Null Hand", "Null Overt Hand"])

#%%

# Create a pie plot showing the relative occurrence of game variants in all games
game_variants.plot(kind="pie", y="Game", autopct='%1.0f%%')
plt.savefig("graphics/all_games_pie_bl.png")
plt.show()


#%%

# Filter the games that were won
games_won = skat_bl_game_data_frame[skat_bl_game_data_frame["Won"] == 1]

game_variants_won = games_won["Game"].value_counts(normalize=True)
game_variants_won = game_variants_won.rename(
    index={24: "Grand", 12: "Cross", 11: "Spades", 10: "Hearts", 9: "Diamonds", 0: "AllPassed", 23: "Null",
           46: "Null Overt", 59: "Null Overt Hand", 35: "Null Hand"})
game_variants_won.loc["AllPassed"] = 0.0
game_variants_won

#%%

# Filter the games that were lost
games_lost = skat_bl_game_data_frame[skat_bl_game_data_frame["Won"] == 0]

game_variants_lost = games_lost["Game"].value_counts(normalize=True)
game_variants_lost = game_variants_lost.rename(
    index={24: "Grand", 12: "Cross", 11: "Spades", 10: "Hearts", 9: "Diamonds", 0: "AllPassed", 23: "Null",
           46: "Null Overt", 59: "Null Overt Hand", 35: "Null Hand"})
game_variants_lost

#%%

# Drop the games were all passed for the comparison with the won games
# game_variants_lost = game_variants_lost.drop(["AllPassed"])


#%%

# direct comparison of lost and won games
comp_won_lost = pd.DataFrame({"won": game_variants_won,
                              "lost": game_variants_lost})
comp_won_lost.plot.barh()
plt.savefig("graphics/comp_won_lost_bl.png")
plt.show()

#%%

game_variants_won["Null"] += game_variants_won["Null Overt"] + game_variants_won["Null Hand"] + game_variants_won["Null Overt Hand"]
game_variants_won = game_variants_won.drop(labels=["Null Overt", "Null Hand", "Null Overt Hand", "AllPassed"])

# Create a pie plot showing the relative occurrence of game variants in won games
game_variants_won.plot(kind="pie", y="Game", autopct='%1.0f%%')
plt.savefig("graphics/won_games_pie_bl.png")
plt.show()

#%%
# Data Cleansing

#%%

# Drop irrelevant information
games_clean = skat_bl_game_data_frame.drop(["IDVServer", "StartTime", "EndTime"], axis=1)


#%%

# convert data to following encoding:
# ♦, ♥, ♠, ♣, {7, 8, 9, Q, K, 10, A, J}, T
def convert(card, trump):
    vector_rep = {
        0: [0, 0, 0, 1, 7, 0],  # A♣
        1: [0, 0, 0, 1, 5, 0],  # K♣
        2: [0, 0, 0, 1, 4, 0],  # Q♣
        3: [0, 0, 0, 1, 8, 1],  # J♣
        4: [0, 0, 0, 1, 6, 0],  # 10♣
        5: [0, 0, 0, 1, 3, 0],  # 9♣
        6: [0, 0, 0, 1, 2, 0],  # 8♣
        7: [0, 0, 0, 1, 1, 0],  # 7♣
        8: [0, 0, 0, 1, 7, 0],  # A♠
        9: [0, 0, 0, 1, 5, 0],  # K♠
        10: [0, 0, 1, 0, 4, 0],  # Q♠
        11: [0, 0, 1, 0, 8, 1],  # J♠
        12: [0, 0, 1, 0, 6, 0],  # 10♠
        13: [0, 0, 1, 0, 3, 0],  # 9♠
        14: [0, 0, 1, 0, 2, 0],  # 8♠
        15: [0, 0, 1, 0, 1, 0],  # 7♠
        16: [0, 1, 0, 0, 7, 0],  # A♥
        17: [0, 1, 0, 0, 5, 0],  # K♥
        18: [0, 1, 0, 0, 4, 0],  # Q♥
        19: [0, 1, 0, 0, 8, 1],  # J♥
        20: [0, 1, 0, 0, 6, 0],  # 10♥
        21: [0, 1, 0, 0, 3, 0],  # 9♥
        22: [0, 1, 0, 0, 2, 0],  # 8♥
        23: [0, 1, 0, 0, 1, 0],  # 7♥
        24: [1, 0, 0, 0, 7, 0],  # A♦
        25: [1, 0, 0, 0, 5, 0],  # K♦
        26: [1, 0, 0, 0, 4, 0],  # Q♦
        27: [1, 0, 0, 0, 8, 1],  # J♦
        28: [1, 0, 0, 0, 6, 0],  # 10♦
        29: [1, 0, 0, 0, 3, 0],  # 9♦
        30: [1, 0, 0, 0, 2, 0],  # 8♦
        31: [1, 0, 0, 0, 1, 0]  # 7♦
    }
    converted_card = vector_rep[card]

    # check if the card is trump in a colour game
    if trump == 9:
        if converted_card[0] == 1:
            converted_card[-1] = 1
    elif trump == 10:
        if converted_card[1] == 1:
            converted_card[-1] = 1
    elif trump == 11:
        if converted_card[2] == 1:
            converted_card[-1] = 1
    elif trump == 12:
        if converted_card[3] == 1:
            converted_card[-1] = 1

    return converted_card


trump = 0

games_clean.loc[:, "Card1":"Card32"].head(n=10)
#
# games_clean.loc[:, "Card1":"Card32"] = games_clean.loc[:, "Card1":"Card32"].apply(
#     lambda card_list: card_list.apply(lambda card: convert(card, trump=trump)))


cards = ["Card1", "Card2", "Card3", "Card4", "Card5", "Card6", "Card7", "Card8", "Card9", "Card10", "Card11", "Card12",
         "Card13", "Card14", "Card15", "Card16", "Card17", "Card18", "Card19", "Card20", "Card21", "Card22",
         "Card23", "Card24", "Card25", "Card26", "Card27", "Card28", "Card29", "Card30", "Card31", "Card32"]
new_games_cards = pd.DataFrame(columns=cards)

r = range(len(games_clean["Card1"]))
r = range(100)

for j in cards:
    print("Check\n")
    for i in r:
        # games_clean[j][i] = convert(games_clean[j][i], games_clean["Game"][i])
        # games_clean.loc[i,j]
        new_games_cards.loc[i, j] = convert(games_clean.loc[i, j], games_clean["Game"][i])

##%%
# games_clean["Card1"][1]
# create train and test sets
# skat_train, skat_test = train_test_split(skat_data, test_size=0.2, random_state=0)

# If data set is unsorted and randomly distributed
# train_data = skat_data[:n*0.8]
# test_data = skat_data[n*0.2:]
