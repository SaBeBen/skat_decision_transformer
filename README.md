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


 FIXED: problem of context length max = 1024: see Solution 3

 atm of Problem (each Skat as action, card_dim 5, cards on hand): episode length = 105 * 12 = 1260

 problem at action 9 (10th action) (10 * 105 = 1050) (should select 3rd card, selects 1st):
 tensor([0.2189, 0.1059, 0.1479, 0.0586, 0.0595, 0.0583, 0.0585, 0.0587, 0.0584, 0.0586, 0.0583, 0.0586])

 Solution 1:
 Skat as first last trick
   hand_cards -= 2, 2 s, a, r less
   state_dim = -> 82
   episode length = (82 + 12 + 1) * 10 = 950 v

   But what if we want to encode the Skat as two separate actions?

 Solution 2:
 compress card representation with sorting by suit and further identifying by face
 example: [0, 1, 0, 0, 1, 5, 7, 8, 0, 0, 0, 0]
   -> spades 7, K, A, J, missing: 8, 9, Q, 10
   size per colour with padding -> 4 + 8 = 12
   size of whole hand 12 * 4 = 48
   state size = 3 + 4 + 3 * 5 + 2 * 5 + 48 = 80
   episode: (s + a + r) * 12 = (80 + 12 + 1) * 12 = 1116
   episode with Skat as first last trick: 93 * 10 = 930 v

 Solution 3: currently used
 Solution 2 + further compressing: do not pad the suits, only pad up to the max possible state length
   max_hand_length = 16 (encoding of colours) + 12 (all cards) = 28
 pad with zeros
 Example 1:
 [1, 0, 0, 0, 1, 5, 7, 8], [0, 1, 0, 0, 1, 5],
 [0, 0, 1, 0, 2, 3, 4], [0, 0, 0, 1, 1, 3, 8]
 Example 2:
 [1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8], [0] * (max_hand_length - 12)

   state size = 3 + 4 + 3 * 5 + 2 * 5 + 28 = 60
   episode length: (s + a + r) * 12 = (60 + 12 + 1) * 12 = 876 v

 Solution 4 (respects problem with loss function):
 Solution 3, but solely with one-hot encoded cards
   hand_length = 4 * 12 = 48
 Example 1:
 [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1], [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
 [0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1]

   state size = 3 + 4 + 3 * 12 + 2 * 12 + 48 = 115
   episode length: (s + a + r) * 12 = (115 + 12 + 1) * 12 = 1 536  x
