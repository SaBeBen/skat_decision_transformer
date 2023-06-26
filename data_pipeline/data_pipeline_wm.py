# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split

# %%

"""The dataset of the sequence the cards (cs) were played follows"""
skat_wm_cs_data_path = "C:/Users/sasch/Desktop/Uni/Bachelorarbeit/SaschaBenz/data/dl/wm_skattisch_kf.CSV"

skat_wm_cs_data_frame = pd.read_csv(skat_wm_cs_data_path, header=None)

# TODO: convert data to following encoding:
# ♦, ♥, ♠, ♣, {7, 8, 9, Q, K, 10, A}, J/T
skat_wm_cs_data = skat_wm_cs_data_frame

skat_wm_cs_data.columns = ["GameID", "Sd1", "Sd2", "CNr0", "CNr1", "CNr2", "CNr3", "CNr4", "CNr5", "CNr6", "CNr7",
                           "CNr8", "CNr9", "CNr10", "CNr11", "CNr12", "CNr13", "CNr14", "CNr15", "CNr16", "CNr17",
                           "CNr18", "CNr19", "CNr20", "CNr21", "CNr22", "CNr23", "CNr24", "CNr25", "CNr26", "CNr27",
                           "CNr28", "CNr29", "CNr30", "CNr31", "SurrenderedAt"]

# %%

skat_wm_cs_data.info(memory_usage="deep")

skat_wm_cs_data_frame.select_dtypes(include=['int'])
# converted_int = gl_int.apply(pd.to_numeric,downcast='unsigned')

# %%

# Sanity Checks

# %%

skat_wm_cs_data
print(skat_wm_cs_data.loc[1, :])

# %%

print(skat_wm_cs_data_frame.isna().sum().sum())

# %%

print(skat_wm_cs_data_frame["GameID"].size)

# %%

print(skat_wm_cs_data_frame["GameID"].duplicated().any())

# %%

print(skat_wm_cs_data_frame.loc[1, :])

# %%

skat_wm_table_data = skat_wm_cs_data_frame

head = skat_wm_cs_data.head(n=20)
print(head)

# %%

"""The table dataset follows"""
skat_wm_table_data_path = "C:/Users/sasch/Desktop/Uni/Bachelorarbeit/SaschaBenz/data/dl/wm_skattisch.CSV"

skat_wm_table_data_frame = pd.read_csv(skat_wm_table_data_path, header=None)

skat_wm_table_data_frame.columns = ["IDTable", "Name", "Number", "PlayerID1", "PlayerID2", "PlayerID3", "PlayerID4",
                                    "Player1", "Player2", "Player3", "Player4", "Date", "IDVServer", "Series"]

skat_wm_table_data_frame

# %%

# Sanity Checks

print(skat_wm_table_data_frame.isna().sum())

# %%

print(skat_wm_table_data_frame["IDTable"].size)

# %%

print(skat_wm_table_data_frame["IDTable"].duplicated().any())

# %%

head = skat_wm_table_data_frame.head(n=10)
print(head)

# %%

"""The game dataset follows"""

skat_wm_game_data_path = "C:/Users/sasch/Desktop/Uni/Bachelorarbeit/SaschaBenz/data/dl/wm_skattisch_spiel.CSV"

skat_wm_game_data_frame = pd.read_csv(skat_wm_game_data_path, header=None)

skat_wm_game_data_frame.columns = ["GameID", "IDGame", "IDTable", "IDVServer", "StartTime", "EndTime", "PlayerFH",
                                   "PlayerMH", "PlayerBH", "Card1", "Card2", "Card3", "Card4", "Card5", "Card6",
                                   "Card7", "Card8", "Card9", "Card10", "Card11", "Card12", "Card13", "Card14",
                                   "Card15", "Card16", "Card17", "Card18", "Card19", "Card20", "Card21", "Card22",
                                   "Card23", "Card24", "Card25", "Card26", "Card27", "Card28", "Card29", "Card30",
                                   "Card31", "Card32", "CallValueFH", "CallValueMH", "CallValueBH", "PlayerID", "Game",
                                   "With", "Without", "Hand", "Schneider", "SchneiderCalled", "Schwarz",
                                   "SchwarzCalled", "Overt", "PointsPlayer", "Won", "Miscall",
                                   "CardPointsPlayer", "AllPassed", "Surrendered", "PlayerPosAtTableFH",
                                   "PlayerPosAtTableMH", "PlayerPosAtTableBH"]

# %%

head = skat_wm_game_data_frame.head(n=20)
print(head)

# %%
print(skat_wm_game_data_frame.iloc[1, 0:2])

print(skat_wm_game_data_frame.iloc[1, 9:41])

print(skat_wm_game_data_frame.iloc[1, 59:60])

# %% Sanity Checks

skat_wm_game_data_frame[["CallValueFH", "CallValueMH", "CallValueBH", "PlayerID", "Game",
                         "With", "Without", "Hand", "Schneider", "SchneiderCalled", "Schwarz",
                         "SchwarzCalled", "Overt", "PointsPlayer", "Won", "Miscall",
                         "CardPointsPlayer", "AllPassed", "Surrendered"]].describe()

# %%

skat_wm_game_data_frame[["CallValueFH", "CallValueMH", "CallValueBH", "PlayerPosAtTableFH", "PlayerPosAtTableMH",
                         "PlayerPosAtTableBH"]].describe()

# %%

print(skat_wm_game_data_frame.isna().sum().sum())

# %%

print(skat_wm_game_data_frame["GameID"].size)

# %%

print(skat_wm_game_data_frame["GameID"].duplicated().any())

# %%

print(skat_wm_game_data_frame["IDGame"].duplicated().any())

# %%

skat_wm_game_data_frame["Won"].value_counts(normalize=True)

# %%

skat_wm_game_data_frame["Schneider"].value_counts(normalize=True)

# %%

skat_wm_game_data_frame["Schwarz"].value_counts(normalize=True)

# %%

skat_wm_game_data_frame["Game"].value_counts()

# %%

skat_wm_game_data_frame["AllPassed"].value_counts(normalize=True)

# %%

skat_wm_game_data_frame["Miscall"].value_counts(normalize=True)

# %%

skat_wm_game_data_frame["Surrendered"].value_counts(normalize=True)

# %%

# Data Visualisation

# %%

# Analysis of games
game_variants = skat_wm_game_data_frame["Game"].value_counts(normalize=True)
game_variants = game_variants.rename(
    index={24: "Grand", 12: "Cross", 11: "Spades", 10: "Hearts", 9: "Diamonds", 0: "AllPassed", 23: "Null",
           46: "Null Overt", 59: "Null Overt Hand", 35: "Null Hand"})
game_variants["Null"] += game_variants["Null Overt"] + game_variants["Null Hand"] + game_variants["Null Overt Hand"]
game_variants = game_variants.drop(labels=["Null Overt", "Null Hand", "Null Overt Hand"])

# %%

# Create a pie plot showing the relative occurrence of game variants in all games
game_variants.plot(kind="pie", y="Game", autopct='%1.0f%%')
plt.savefig("graphics/all_games_pie_wm.png")
plt.show()

# %%

# Filter the games that were won
games_won = skat_wm_game_data_frame[skat_wm_game_data_frame["Won"] == 1]

game_variants_won = games_won["Game"].value_counts(normalize=True)
game_variants_won = game_variants_won.rename(
    index={24: "Grand", 12: "Cross", 11: "Spades", 10: "Hearts", 9: "Diamonds", 0: "AllPassed", 23: "Null",
           46: "Null Overt", 59: "Null Overt Hand", 35: "Null Hand"})
game_variants_won.loc["AllPassed"] = 0.0
game_variants_won

# %%

# Filter the games that were lost
games_lost = skat_wm_game_data_frame[skat_wm_game_data_frame["Won"] == 0]

game_variants_lost = games_lost["Game"].value_counts(normalize=True)
game_variants_lost = game_variants_lost.rename(
    index={24: "Grand", 12: "Cross", 11: "Spades", 10: "Hearts", 9: "Diamonds", 0: "AllPassed", 23: "Null",
           46: "Null Overt", 59: "Null Overt Hand", 35: "Null Hand"})
game_variants_lost

# %%

# Drop the games were all passed for the comparison with the won games
# game_variants_lost = game_variants_lost.drop(["AllPassed"])


# %%

# direct comparison of lost and won games
comp_won_lost = pd.DataFrame({"won": game_variants_won,
                              "lost": game_variants_lost})
comp_won_lost.plot.barh()
plt.savefig("graphics/comp_won_lost_wm.png")
plt.show()
# %%

game_variants_won["Null"] += game_variants_won["Null Overt"] + game_variants_won["Null Hand"] + game_variants_won[
    "Null Overt Hand"]
game_variants_won = game_variants_won.drop(labels=["Null Overt", "Null Hand", "Null Overt Hand", "AllPassed"])

# Create a pie plot showing the relative occurrence of game variants in won games
game_variants_won.plot(kind="pie", y="Game", autopct='%1.0f%%')
plt.savefig("graphics/won_games_pie_wm.png")
plt.show()

# %%

# Data Cleansing

# %%

# Drop irrelevant information
games_clean = skat_wm_game_data_frame.drop(["IDVServer", "StartTime", "EndTime"], axis=1)


# %%
games_clean.info(memory_usage="deep")

games_clean.select_dtypes(include=['int'])

# %%
# Optimizing memory space
games_clean.iloc[:, 3:] = games_clean.iloc[:, 3:].apply(pd.to_numeric, downcast="unsigned")

# games_clean.iloc[:, 6:38] = games_clean.iloc[:, 6:38].apply(pd.to_numeric, downcast="")

games_clean.info(memory_usage="deep")
# %%
# convert data to following encoding:
# ♦, ♥, ♠, ♣, {7, 8, 9, Q, K, 10, A, J}, T
# with respect to their position
def convert_cs(cards, trump):
    converted_cs_cards = []

    # for every card in the cs data frame, convert the card by it's input and store it in
    # the order the cards were played
    for i in range(30):
        converted_cs_cards[3+cards.iloc[i]] = convert(i, trump=trump)

    return converted_cs_cards


# %%

# ONLY FOR THE GAME DATA FRAME
# convert cards to following encoding:
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
r = range(10000)

for j in cards:
    print("Check\n")
    for i in r:
        # games_clean[j][i] = convert(games_clean[j][i], games_clean["Game"][i])
        # games_clean.loc[i,j]
        new_games_cards.loc[i, j] = convert(games_clean.loc[i, j], games_clean["Game"][i])


# %%
new_games_cards["Card1"].info(memory_usage="deep")


# %%
# games_clean["Card1"][1]
new_games_cards

# create train and test sets
# skat_train, skat_test = train_test_split(skat_data, test_size=0.2, random_state=0)

# If data set is unsorted and randomly distributed
# train_data = skat_data[:n*0.8]
# test_data = skat_data[n*0.2:]
