import pandas as pd
import dataframe_image as dfi

# %%

columns = ["GameID", "IDGame", "IDTable", "IDVServer", "StartTime", "EndTime", "PlayerFH",
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
# Read gtc data set
gtc_game_data_path = f"C:/Users/sasch/Desktop/Uni/Bachelorarbeit/SaschaBenz/software/skat_decision_transformer/" \
                     f"db_app/data/gtc_game.CSV"

gtc_game_data = pd.read_csv(gtc_game_data_path, header=None)

gtc_game_data.columns = columns

# Read rc data set
rc_game_data_path = f"C:/Users/sasch/Desktop/Uni/Bachelorarbeit/SaschaBenz/software/skat_decision_transformer/" \
                    f"db_app/data/rc_game.CSV"

rc_game_data = pd.read_csv(rc_game_data_path, header=None)

rc_game_data.columns = columns

# Read wc data set
wc_game_data_path = f"C:/Users/sasch/Desktop/Uni/Bachelorarbeit/SaschaBenz/software/skat_decision_transformer/" \
                    f"db_app/data/wc_game.CSV"

wc_game_data = pd.read_csv(wc_game_data_path, header=None)

wc_game_data.columns = columns

# Read gc data set
gc_game_data_path = f"C:/Users/sasch/Desktop/Uni/Bachelorarbeit/SaschaBenz/software/skat_decision_transformer/" \
                    f"db_app/data/gc_game.CSV"

gc_game_data = pd.read_csv(gc_game_data_path, header=None)

gc_game_data.columns = columns

# Read bl data set
bl_game_data_path = f"C:/Users/sasch/Desktop/Uni/Bachelorarbeit/SaschaBenz/software/skat_decision_transformer/" \
                    f"db_app/data/bl_game.CSV"

bl_game_data = pd.read_csv(bl_game_data_path, header=None)

bl_game_data.columns = columns

# %%

stats = {
    "gtc": [gtc_game_data["GameID"].size, gtc_game_data["Won"].value_counts(normalize=True)[1],
            gtc_game_data["Schneider"].value_counts(normalize=True)[1],
            gtc_game_data["Schwarz"].value_counts(normalize=True)[1],
            gtc_game_data["AllPassed"].value_counts(normalize=True)[1],
            gtc_game_data["Miscall"].value_counts(normalize=True)[1],
            1 - gtc_game_data["Surrendered"].value_counts(normalize=True)[0]
            ],
    "rc": [rc_game_data["GameID"].size, rc_game_data["Won"].value_counts(normalize=True)[1],
           rc_game_data["Schneider"].value_counts(normalize=True)[1],
           rc_game_data["Schwarz"].value_counts(normalize=True)[1],
           rc_game_data["AllPassed"].value_counts(normalize=True)[1],
           rc_game_data["Miscall"].value_counts(normalize=True)[1],
           1 - rc_game_data["Surrendered"].value_counts(normalize=True)[0]
           ],
    "bl": [bl_game_data["GameID"].size, bl_game_data["Won"].value_counts(normalize=True)[1],
           bl_game_data["Schneider"].value_counts(normalize=True)[1],
           bl_game_data["Schwarz"].value_counts(normalize=True)[1],
           bl_game_data["AllPassed"].value_counts(normalize=True)[1],
           bl_game_data["Miscall"].value_counts(normalize=True)[1],
           1 - bl_game_data["Surrendered"].value_counts(normalize=True)[0]
           ],
    "wc": [wc_game_data["GameID"].size, wc_game_data["Won"].value_counts(normalize=True)[1],
           wc_game_data["Schneider"].value_counts(normalize=True)[1],
           wc_game_data["Schwarz"].value_counts(normalize=True)[1],
           wc_game_data["AllPassed"].value_counts(normalize=True)[1],
           wc_game_data["Miscall"].value_counts(normalize=True)[1],
           1 - wc_game_data["Surrendered"].value_counts(normalize=True)[0]
           ],
    "gc": [gc_game_data["GameID"].size, gc_game_data["Won"].value_counts(normalize=True)[1],
           gc_game_data["Schneider"].value_counts(normalize=True)[1],
           gc_game_data["Schwarz"].value_counts(normalize=True)[1],
           gc_game_data["AllPassed"].value_counts(normalize=True)[1],
           gc_game_data["Miscall"].value_counts(normalize=True)[1],
           1 - gc_game_data["Surrendered"].value_counts(normalize=True)[0]
           ]
}

# %%
stat_frame = pd.DataFrame(stats,
                          ["# of Games", "Win rate", "Schneider", "Schwarz", "AllPassed", "Miscall", "Surrendered"])

# round every column
stat_frame = stat_frame.round(4)

# convert the rates to percentages
stat_frame.loc[:, stat_frame.columns != "# of Games"] = stat_frame.loc[:, stat_frame.columns != "# of Games"] * 100

# suppress scientific notation
stat_frame = stat_frame.astype(object)

dfi.export(stat_frame, "graphics/stat_table_all.png")
