# %%

import pandas as pd
import matplotlib.pyplot as plt

# %%


# "IDGame" of the first 20 games, can be searched in https://skatarchiv.skat-spielen.de/ for a better view of the game

# 8x619M13Bx36

# 8x619M13BxQH
# 8x619M13Bx3U

# Game 4:
# 8x619M13Bx3C

# 8x619M13Bx30
# 8x619M13Bx32
# 8x619M13Bx9S

# 8x619M13Bx3B

# 8x619M13Bx3X
# 8x619M13Bx9V
# 8x619M13Bx99
# 8x619M13Bx9Y
# 8x619M13Bx38
# 8x619M13Bx31
# 8x619M13Bx3M
# 8x619M13Bx9Q
# 8x619M13Bx3E
# 8x619M13Bx3P
# 8x619M13Bx3J
# 8x619M13Bx9C
# 8x619M13Bx9M

"""The dataset of the card sequences (cs) follows"""

championships = ["wc", "bl", "gc", "gtc", "rc"]

championship = "wc"

cs_data_path = f"C:/Users/sasch/Desktop/Uni/Bachelorarbeit/SaschaBenz/software/skat_decision_transformer/" \
               f"data/{championship}_card_sequence.CSV"

cs_data = pd.read_csv(cs_data_path, header=None)

cs_data.columns = ["GameID", "Sd1", "Sd2", "CNr0", "CNr1", "CNr2", "CNr3", "CNr4", "CNr5", "CNr6", "CNr7",
                   "CNr8", "CNr9", "CNr10", "CNr11", "CNr12", "CNr13", "CNr14", "CNr15", "CNr16", "CNr17",
                   "CNr18", "CNr19", "CNr20", "CNr21", "CNr22", "CNr23", "CNr24", "CNr25", "CNr26", "CNr27",
                   "CNr28", "CNr29", "CNr30", "CNr31", "SurrenderedAt"]

# %%

cs_data.info(memory_usage="deep")

cs_data.select_dtypes(include=['int'])

# %%

# Sanity Checks

# %%

print(cs_data.loc[1, :])

# %%

print(cs_data.isna().sum().sum())

# %%

print(cs_data["GameID"].size)

# %%

print(cs_data["GameID"].duplicated().any())

# %%

print(cs_data.loc[1, :])

# %%

head = cs_data.head(n=20)
print(head)

# %%

"""The table dataset follows"""

table_path = f"C:/Users/sasch/Desktop/Uni/Bachelorarbeit/SaschaBenz/software/skat_decision_transformer/" \
               f"data/{championship}_table.CSV"

table_data = pd.read_csv(table_path, header=None)

table_data.columns = ["IDTable", "Name", "Number", "PlayerID1", "PlayerID2", "PlayerID3", "PlayerID4",
                      "Player1", "Player2", "Player3", "Player4", "Date", "IDVServer", "Series"]

table_data

# %%

# Sanity Checks

print(table_data.isna().sum())

# %%

print(table_data["IDTable"].size)

# %%

print(table_data["IDTable"].duplicated().any())

# %%

head = table_data.head(n=10)
print(head)

# %%

"""The game dataset follows"""

game_path = f"C:/Users/sasch/Desktop/Uni/Bachelorarbeit/SaschaBenz/software/skat_decision_transformer/" \
               f"data/{championship}_game.CSV"

game_data = pd.read_csv(game_path, header=None)

game_data.columns = ["GameID", "IDGame", "IDTable", "IDVServer", "StartTime", "EndTime", "PlayerFH",
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

head = game_data.head(n=20)
print(head)

# %% Sanity Checks

game_data[["CallValueFH", "CallValueMH", "CallValueRH", "PlayerID", "Game",
           "With", "Without", "Hand", "Schneider", "SchneiderCalled", "Schwarz",
           "SchwarzCalled", "Ouvert", "PointsPlayer", "Won", "Miscall",
           "CardPointsPlayer", "AllPassed", "Surrendered"]].describe()

# %%

game_data[["CallValueFH", "CallValueMH", "CallValueRH", "PlayerPosAtTableFH", "PlayerPosAtTableMH",
           "PlayerPosAtTableRH"]].describe()

# %%

print(game_data.isna().sum().sum())

# %%

print(game_data["GameID"].size)

# %%

print(game_data["GameID"].duplicated().any())

# %%

print(game_data["IDGame"].duplicated().any())

# %%

# Analysis

game_data["Won"].value_counts(normalize=True)

# %%

game_data["Schneider"].value_counts(normalize=True)

# %%

game_data["Schwarz"].value_counts(normalize=True)

# %%

game_data["Game"].value_counts()

# %%

game_data["AllPassed"].value_counts(normalize=True)

# %%

game_data["Miscall"].value_counts(normalize=True)

# %%

# Data Visualisation

# %%

# Analysis of games
game_variants = game_data["Game"].value_counts(normalize=True)
game_variants = game_variants.rename(
    index={24: "Grand", 12: "Cross", 11: "Spades", 10: "Hearts", 9: "Diamonds", 0: "AllPassed", 23: "Null",
           46: "Null Ouvert", 59: "Null Ouvert Hand", 35: "Null Hand"})
game_variants["Null"] += game_variants["Null Ouvert"] + game_variants["Null Hand"] + game_variants["Null Ouvert Hand"]
game_variants = game_variants.drop(labels=["Null Ouvert", "Null Hand", "Null Ouvert Hand"])

# %%

# Create a pie plot showing the relative occurrence of game variants in all games
game_variants.plot(kind="pie", y="Game", autopct='%1.0f%%')
plt.savefig(f"graphics/all_games_pie_{championship}.png")
plt.show()

# %%

game_data["Surrendered"].value_counts(normalize=True)

# %%
cs_data["SurrenderedAt"].value_counts(normalize=True)

# %%
print(game_data.shape[0])

# %%
print(cs_data.shape[0])

# %%
cs_and_game_data = pd.merge(game_data, cs_data, how="inner", on="GameID")

# %%

surrendered = cs_and_game_data[(cs_and_game_data["SurrenderedAt"] > -1)]

sur_and_null = cs_and_game_data[
    (cs_and_game_data["SurrenderedAt"] > -1) &
    ((cs_and_game_data["Game"] == 23) |
     (cs_and_game_data["Game"] == 46) |
     (cs_and_game_data["Game"] == 59) |
     (cs_and_game_data["Game"] == 35))]

sur_and_ouvert = cs_and_game_data[
    (cs_and_game_data["SurrenderedAt"] > -1) &
    (cs_and_game_data["Ouvert"] == 1)]

game_variants_sur = cs_and_game_data[(cs_and_game_data["SurrenderedAt"] > -1)]["Game"].value_counts(normalize=True)
game_variants_sur = game_variants_sur.rename(
    index={24: "Grand", 12: "Cross", 11: "Spades", 10: "Hearts", 9: "Diamonds", 0: "AllPassed", 23: "Null",
           46: "Null Ouvert", 59: "Null Ouvert Hand", 35: "Null Hand"})

# %%

game_level_sur_values = cs_and_game_data[(cs_and_game_data["SurrenderedAt"] > -1)].loc[:, "Hand":"Ouvert"]

# the configurations of Hand, Schneider, SchneiderCalled, Schwarz, SchwarzCalled, Ouvert of surrendered Games
# 0/0/0/0/0/0 means nothing was called
game_level_sur_conf = game_level_sur_values.value_counts(normalize=True)

# %%

game_variants_sur.plot(kind="pie", y="Game", autopct='%1.0f%%')
plt.ylabel("Surrendered Games")

plt.savefig(f"graphics/sur_games_pie_{championship}.png")
plt.show()

# %%
amount_sur = surrendered.shape[0]

amount_sur_and_null = sur_and_null.shape[0]

amount_sur_and_ouvert = sur_and_ouvert.shape[0]

ratio_surrendered_null = amount_sur - amount_sur_and_null

# %%

# Filter the games that were won
games_won = game_data[game_data["Won"] == 1]

game_variants_won = games_won["Game"].value_counts(normalize=True)
game_variants_won = game_variants_won.rename(
    index={24: "Grand", 12: "Cross", 11: "Spades", 10: "Hearts", 9: "Diamonds", 0: "AllPassed", 23: "Null",
           46: "Null Ouvert", 59: "Null Ouvert Hand", 35: "Null Hand"})
game_variants_won.loc["AllPassed"] = 0.0
game_variants_won

# %%

# Filter the games that were lost
games_lost = game_data[game_data["Won"] == 0]

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
plt.savefig(f"graphics/comp_won_lost_{championship}.png")
plt.show()
# %%

game_variants_won["Null"] += game_variants_won["Null Ouvert"] + game_variants_won["Null Hand"] + game_variants_won[
    "Null Ouvert Hand"]
game_variants_won = game_variants_won.drop(labels=["Null Ouvert", "Null Hand", "Null Ouvert Hand", "AllPassed"])

# Create a pie plot showing the relative occurrence of game variants in won games
game_variants_won.plot(kind="pie", y="Game", autopct='%1.0f%%')

plt.ylabel("Won Games")
plt.savefig(f"graphics/won_games_pie_{championship}.png")
plt.show()
