{bl_skattisch, bl_skattisch_kf, bl_skattisch_spiel} for bl, dm, dt, rl, wm

The used data consists out of games played at the 
world championship (wc), 

German league “Bundesliga” (bl), 

German championship (gc) (in German “deutsche Meisterschaft”), 

a DOSKV championship where only members can participate, more regional (rc) (rl) 

and a German tandem championship (gtc). 

All of them use the same set of rules. 
As the bl is played in teams, single players play more defensively. 
Therefore, there are fewer wins and fewer grands played.

Consequently, the win rate in the bl is slightly worse in comparison with dm, wc and dt. 

[comment]: <> (insert win rate table)

Each dataset is partitioned into three tables, one table containing information about the table where the game is played, one with each game’s description including metadata of the players and the game, each player’s hand and the Skat.

In the following, the tables will be explained more in detail.

Table about the table

Each table entry consists of a table ID, a name, a number, the ID as a single integer of the first, second, third and if given of the fourth player. Then, the players non-integer ID including their initials and at least one number follows. The last entries are made of a date, an ID of the server and the series to which the table belongs to.
  

Pl, "PlayerID2", "PlayerID3", "PlayerID4",
"Player1", "Player2", "Player3", "Player4", "Date", "IDVServer", "Series"

Card sequence table

"GameID", "Sd1", "Sd2", "CNr0", "CNr1", "CNr2", "CNr3", "CNr4", "CNr5", "CNr6", "CNr7",
"CNr8", "CNr9", "CNr10", "CNr11", "CNr12", "CNr13", "CNr14", "CNr15", "CNr16", "CNr17",
"CNr18", "CNr19", "CNr20", "CNr21", "CNr22", "CNr23", "CNr24", "CNr25", "CNr26", "CNr27",
"CNr28", "CNr29", "CNr30", "CNr31", "SurrenderedAt"

Game table

"GameID", "IDGame", "IDTable", "IDVServer", "StartTime", "EndTime", "PlayerFH",
"PlayerMH", "PlayerBH", "Cards", "CallValueFH", "CallValueMH", "CallValueBH", "PlayerID", "Game",  "With", "Without", "Hand", "Schneider", "SchneiderCalled", "Schwarz",
"SchwarzCalled", "Overt", "PointsPlayer", "Won", "Miscall", "CardPointsPlayer", "AllPassed",
"Surrendered", "PlayerPosAtTableFH", "PlayerPosAtTableMH", "PlayerPosAtTableBH"