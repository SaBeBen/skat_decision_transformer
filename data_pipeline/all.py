import pandas as pd
import plotly.figure_factory as ff
import dataframe_image as dfi

# %%
# Read dt data set
skat_dt_game_data_path = "C:/Users/sasch/Desktop/Uni/Bachelorarbeit/SaschaBenz/data/dl/dt_skattisch_spiel.CSV"

skat_dt_game_data_frame = pd.read_csv(skat_dt_game_data_path, header=None)

skat_dt_game_data_frame.columns = ["GameID", "IDGame", "IDTable", "IDVServer", "StartTime", "EndTime", "PlayerFH",
                                   "PlayerMH", "PlayerBH", "Card1", "Card2", "Card3", "Card4", "Card5", "Card6",
                                   "Card7", "Card8", "Card9", "Card10", "Card11", "Card12", "Card13", "Card14",
                                   "Card15", "Card16", "Card17", "Card18", "Card19", "Card20", "Card21", "Card22",
                                   "Card23", "Card24", "Card25", "Card26", "Card27", "Card28", "Card29", "Card30",
                                   "Card31", "Card32", "CallValueFH", "CallValueMH", "CallValueBH", "PlayerID", "Game",
                                   "With", "Without", "Hand", "Schneider", "SchneiderCalled", "Schwarz",
                                   "SchwarzCalled", "Overt", "PointsPlayer", "Won", "Miscall",
                                   "CardPointsPlayer", "AllPassed", "Surrendered", "PlayerPosAtTableFH",
                                   "PlayerPosAtTableMH", "PlayerPosAtTableBH"]

# Read rl data set
skat_rl_game_data_path = "C:/Users/sasch/Desktop/Uni/Bachelorarbeit/SaschaBenz/data/dl/rl_skattisch_spiel.CSV"

skat_rl_game_data_frame = pd.read_csv(skat_rl_game_data_path, header=None)

skat_rl_game_data_frame.columns = ["GameID", "IDGame", "IDTable", "IDVServer", "StartTime", "EndTime", "PlayerFH",
                                   "PlayerMH", "PlayerBH", "Card1", "Card2", "Card3", "Card4", "Card5", "Card6",
                                   "Card7", "Card8", "Card9", "Card10", "Card11", "Card12", "Card13", "Card14",
                                   "Card15", "Card16", "Card17", "Card18", "Card19", "Card20", "Card21", "Card22",
                                   "Card23", "Card24", "Card25", "Card26", "Card27", "Card28", "Card29", "Card30",
                                   "Card31", "Card32", "CallValueFH", "CallValueMH", "CallValueBH", "PlayerID", "Game",
                                   "With", "Without", "Hand", "Schneider", "SchneiderCalled", "Schwarz",
                                   "SchwarzCalled", "Overt", "PointsPlayer", "Won", "Miscall",
                                   "CardPointsPlayer", "AllPassed", "Surrendered", "PlayerPosAtTableFH",
                                   "PlayerPosAtTableMH", "PlayerPosAtTableBH"]

# Read wm data set
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

# Read dm data set
skat_dm_game_data_path = "C:/Users/sasch/Desktop/Uni/Bachelorarbeit/SaschaBenz/data/dl/dm_skattisch_spiel.CSV"

skat_dm_game_data_frame = pd.read_csv(skat_dm_game_data_path, header=None)

skat_dm_game_data_frame.columns = ["GameID", "IDGame", "IDTable", "IDVServer", "StartTime", "EndTime", "PlayerFH",
                                   "PlayerMH", "PlayerBH", "Card1", "Card2", "Card3", "Card4", "Card5", "Card6",
                                   "Card7", "Card8", "Card9", "Card10", "Card11", "Card12", "Card13", "Card14",
                                   "Card15", "Card16", "Card17", "Card18", "Card19", "Card20", "Card21", "Card22",
                                   "Card23", "Card24", "Card25", "Card26", "Card27", "Card28", "Card29", "Card30",
                                   "Card31", "Card32", "CallValueFH", "CallValueMH", "CallValueBH", "PlayerID", "Game",
                                   "With", "Without", "Hand", "Schneider", "SchneiderCalled", "Schwarz",
                                   "SchwarzCalled", "Overt", "PointsPlayer", "Won", "Miscall",
                                   "CardPointsPlayer", "AllPassed", "Surrendered", "PlayerPosAtTableFH",
                                   "PlayerPosAtTableMH", "PlayerPosAtTableBH"]

# Read bl data set
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

# %%

stats = {
    "dt": [skat_dt_game_data_frame["GameID"].size, skat_dt_game_data_frame["Won"].value_counts(normalize=True)[1],
           skat_dt_game_data_frame["Schneider"].value_counts(normalize=True)[1],
           skat_dt_game_data_frame["Schwarz"].value_counts(normalize=True)[1],
           skat_dt_game_data_frame["AllPassed"].value_counts(normalize=True)[1],
           skat_dt_game_data_frame["Miscall"].value_counts(normalize=True)[1],
           1 - skat_dt_game_data_frame["Surrendered"].value_counts(normalize=True)[0]
           ],
    "rl": [skat_rl_game_data_frame["GameID"].size, skat_rl_game_data_frame["Won"].value_counts(normalize=True)[1],
           skat_rl_game_data_frame["Schneider"].value_counts(normalize=True)[1],
           skat_rl_game_data_frame["Schwarz"].value_counts(normalize=True)[1],
           skat_rl_game_data_frame["AllPassed"].value_counts(normalize=True)[1],
           skat_rl_game_data_frame["Miscall"].value_counts(normalize=True)[1],
           1 - skat_rl_game_data_frame["Surrendered"].value_counts(normalize=True)[0]
           ],
    "bl": [skat_bl_game_data_frame["GameID"].size, skat_bl_game_data_frame["Won"].value_counts(normalize=True)[1],
           skat_bl_game_data_frame["Schneider"].value_counts(normalize=True)[1],
           skat_bl_game_data_frame["Schwarz"].value_counts(normalize=True)[1],
           skat_bl_game_data_frame["AllPassed"].value_counts(normalize=True)[1],
           skat_bl_game_data_frame["Miscall"].value_counts(normalize=True)[1],
           1 - skat_bl_game_data_frame["Surrendered"].value_counts(normalize=True)[0]
           ],
    "wm": [skat_wm_game_data_frame["GameID"].size, skat_wm_game_data_frame["Won"].value_counts(normalize=True)[1],
           skat_wm_game_data_frame["Schneider"].value_counts(normalize=True)[1],
           skat_wm_game_data_frame["Schwarz"].value_counts(normalize=True)[1],
           skat_wm_game_data_frame["AllPassed"].value_counts(normalize=True)[1],
           skat_wm_game_data_frame["Miscall"].value_counts(normalize=True)[1],
           1 - skat_wm_game_data_frame["Surrendered"].value_counts(normalize=True)[0]
           ],
    "dm": [skat_dm_game_data_frame["GameID"].size, skat_dm_game_data_frame["Won"].value_counts(normalize=True)[1],
           skat_dm_game_data_frame["Schneider"].value_counts(normalize=True)[1],
           skat_dm_game_data_frame["Schwarz"].value_counts(normalize=True)[1],
           skat_dm_game_data_frame["AllPassed"].value_counts(normalize=True)[1],
           skat_dm_game_data_frame["Miscall"].value_counts(normalize=True)[1],
           1 - skat_dm_game_data_frame["Surrendered"].value_counts(normalize=True)[0]
           ]
}

# %%
stat_frame = pd.DataFrame(stats,
                          ["# of Games", "Win rate", "Schneider", "Schwarz", "AllPassed", "Miscall", "Surrendered"])

# round every column
stat_frame = stat_frame.round(4)

# convert the rates to percentages
stat_frame.loc[:, stat_frame.columns != "# of Games"] = stat_frame.loc[:, stat_frame.columns != "# of Games"]*100

# suppress scientific notation
stat_frame = stat_frame.astype(object)

dfi.export(stat_frame, "graphics/stat_table_all.png")

