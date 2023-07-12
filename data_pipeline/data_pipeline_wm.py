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
                                   "PlayerMH", "PlayerRH", "Card1", "Card2", "Card3", "Card4", "Card5", "Card6",
                                   "Card7", "Card8", "Card9", "Card10", "Card11", "Card12", "Card13", "Card14",
                                   "Card15", "Card16", "Card17", "Card18", "Card19", "Card20", "Card21", "Card22",
                                   "Card23", "Card24", "Card25", "Card26", "Card27", "Card28", "Card29", "Card30",
                                   "Card31", "Card32", "CallValueFH", "CallValueMH", "CallValueRH", "PlayerID", "Game",
                                   "With", "Without", "Hand", "Schneider", "SchneiderCalled", "Schwarz",
                                   "SchwarzCalled", "Ouvert", "PointsPlayer", "Won", "Miscall",
                                   "CardPointsPlayer", "AllPassed", "Surrendered", "PlayerPosAtTableFH",
                                   "PlayerPosAtTableMH", "PlayerPosAtTableRH"]

# %%

head = skat_wm_game_data_frame.head(n=20)
print(head)

# %%
print(skat_wm_game_data_frame.iloc[1, 0:2])

print(skat_wm_game_data_frame.iloc[1, 9:41])

print(skat_wm_game_data_frame.iloc[1, 59:60])

# %% Sanity Checks

skat_wm_game_data_frame[["CallValueFH", "CallValueMH", "CallValueRH", "PlayerID", "Game",
                         "With", "Without", "Hand", "Schneider", "SchneiderCalled", "Schwarz",
                         "SchwarzCalled", "Ouvert", "PointsPlayer", "Won", "Miscall",
                         "CardPointsPlayer", "AllPassed", "Surrendered"]].describe()

# %%

skat_wm_game_data_frame[["CallValueFH", "CallValueMH", "CallValueRH", "PlayerPosAtTableFH", "PlayerPosAtTableMH",
                         "PlayerPosAtTableRH"]].describe()

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
           46: "Null Ouvert", 59: "Null Ouvert Hand", 35: "Null Hand"})
game_variants["Null"] += game_variants["Null Ouvert"] + game_variants["Null Hand"] + game_variants["Null Ouvert Hand"]
game_variants = game_variants.drop(labels=["Null Ouvert", "Null Hand", "Null Ouvert Hand"])

# %%

# Create a pie plot showing the relative occurrence of game variants in all games
game_variants.plot(kind="pie", y="Game", autopct='%1.0f%%')
plt.savefig("graphics/all_games_pie_wm.png")
plt.show()


#%%

skat_wm_game_data_frame["Surrendered"].value_counts(normalize=True)

#%%
skat_wm_cs_data_frame["SurrenderedAt"].value_counts(normalize=True)

#%%
print(len(skat_wm_game_data_frame))

#%%
print(len(skat_wm_cs_data_frame))

#%%
skat_wm_cs_and_game_data = skat_wm_game_data_frame.join(skat_wm_cs_data, lsuffix='_caller', rsuffix='_other')
# %%

# Filter the games that were won
games_won = skat_wm_game_data_frame[skat_wm_game_data_frame["Won"] == 1]

game_variants_won = games_won["Game"].value_counts(normalize=True)
game_variants_won = game_variants_won.rename(
    index={24: "Grand", 12: "Cross", 11: "Spades", 10: "Hearts", 9: "Diamonds", 0: "AllPassed", 23: "Null",
           46: "Null Ouvert", 59: "Null Ouvert Hand", 35: "Null Hand"})
game_variants_won.loc["AllPassed"] = 0.0
game_variants_won

# %%

# Filter the games that were lost
games_lost = skat_wm_game_data_frame[skat_wm_game_data_frame["Won"] == 0]

game_variants_lost = games_lost["Game"].value_counts(normalize=True)
game_variants_lost = game_variants_lost.rename(
    index={24: "Grand", 12: "Cross", 11: "Spades", 10: "Hearts", 9: "Diamonds", 0: "AllPassed", 23: "Null",
           46: "Null Ouvert", 59: "Null Ouvert Hand", 35: "Null Hand"})
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

game_variants_won["Null"] += game_variants_won["Null Ouvert"] + game_variants_won["Null Hand"] + game_variants_won[
    "Null Ouvert Hand"]
game_variants_won = game_variants_won.drop(labels=["Null Ouvert", "Null Hand", "Null Ouvert Hand", "AllPassed"])

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