import pandas as pd
import dataframe_image as dfi
import os

# %%

columns_game = ["GameID", "IDGame", "IDTable", "IDVServer", "StartTime", "EndTime", "PlayerFH",
                "PlayerMH", "PlayerBH", "Card1", "Card2", "Card3", "Card4", "Card5", "Card6",
                "Card7", "Card8", "Card9", "Card10", "Card11", "Card12", "Card13", "Card14",
                "Card15", "Card16", "Card17", "Card18", "Card19", "Card20", "Card21", "Card22",
                "Card23", "Card24", "Card25", "Card26", "Card27", "Card28", "Card29", "Card30",
                "Card31", "Card32", "CallValueFH", "CallValueMH", "CallValueBH", "PlayerID", "Game",
                "With", "Without", "Hand", "Schneider", "SchneiderCalled", "Schwarz",
                "SchwarzCalled", "Overt", "PointsPlayer", "Won", "Miscall",
                "CardPointsPlayer", "AllPassed", "Surrendered", "PlayerPosAtTableFH",
                "PlayerPosAtTableMH", "PlayerPosAtTableBH"]

columns_cs = ["GameID", "Sd1", "Sd2", "CNr0", "CNr1", "CNr2", "CNr3", "CNr4", "CNr5", "CNr6", "CNr7",
              "CNr8", "CNr9", "CNr10", "CNr11", "CNr12", "CNr13", "CNr14", "CNr15", "CNr16", "CNr17",
              "CNr18", "CNr19", "CNr20", "CNr21", "CNr22", "CNr23", "CNr24", "CNr25", "CNr26", "CNr27",
              "CNr28", "CNr29", "CNr30", "CNr31", "SurrenderedAt"]

# to read in files in project, it is assumed the current working directory is the one of the data_analysis
cwd = os.getcwd()
data_dir = cwd.split("data_analysis")[0] + "data/"

# %%

POSSIBLE_CHAMPIONSHIPS = ["gtc", "rc", "bl", "gc", "wc"]

game_data_dict = {}
stats = {}

for cs in POSSIBLE_CHAMPIONSHIPS:
    game_data = pd.read_csv(f"{data_dir}/{cs}_game.CSV", header=None)
    cs_data = pd.read_csv(f"{data_dir}/{cs}_card_sequence.CSV", header=None)

    game_data.columns = columns_game
    cs_data.columns = columns_cs
    game_valid_surr = game_data.merge(cs_data[["GameID", "SurrenderedAt"]], on="GameID", how="left")

    # only in a few simple Null games which are played until the end, contradict the tables each other 
    # -> SurrenderedAt is correct
    # in the WC the SurrenderedAt is strictly -1, in other, like the rc, also -2
    temp_comp = game_valid_surr[(game_valid_surr["SurrenderedAt"] <= -1) & (game_valid_surr["Surrendered"] != 0)]

    game_data_dict.update({cs: game_valid_surr})

    surrendered_stats = game_data_dict[cs]["SurrenderedAt"].value_counts(normalize=True)
    surrendered_stats = surrendered_stats[surrendered_stats.index > -1]

    # update statistics with this championship cs
    stats.update({cs: [
        game_data_dict[cs]["GameID"].size, game_data_dict[cs]["Won"].value_counts(normalize=True)[1],
        game_data_dict[cs]["Schneider"].value_counts(normalize=True)[1],
        game_data_dict[cs]["Schwarz"].value_counts(normalize=True)[1],
        game_data_dict[cs]["AllPassed"].value_counts(normalize=True)[1],
        game_data_dict[cs]["Miscall"].value_counts(normalize=True)[1],
        sum(surrendered_stats)
    ]})

# %%
stat_frame = pd.DataFrame(stats,
                          ["# of Games", "Win rate", "Schneider", "Schwarz", "AllPassed", "Miscall", "Surrendered"])

# round every column
stat_frame = stat_frame.round(4)

# convert the rates to percentages
stat_frame[(stat_frame.index != "# of Games")] = stat_frame[(stat_frame.index != "# of Games")] * 100

# suppress scientific notation
stat_frame = stat_frame.astype(object)

dfi.export(stat_frame, "graphics/stat_table_all.png")
