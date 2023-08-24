# Structure of Data

The used data consists out of games played at the 
world championship (wc), German league “Bundesliga” (bl), German championship (gc), a DOSKV championship where only members can participate (rc)
and a German tandem championship (gtc).

All of them use the same set of rules. 
As the bl is played in teams, single players play more defensively. 
Therefore, there are fewer wins and fewer grands played.

Consequently, the win rate in the bl is slightly worse in comparison with gc, wc and gtc. 

For each championship, there are three tables with following suffixes: 
- _skattisch: contains information about the respective championship Skat tables
- _skattisch_kf: contains information about the card sequence of the games
- _skattisch_spiel: contains information about metadata of the players and the game, each player’s hand and the Skat before pick up 

[comment]: <> (insert win rate table)


In the following, the entries of those tables will be explained more in detail.

# Data about the Skat table

Each table entry consists of a 
- "IDtable" (self-explanatory)
- "Name" (self-explanatory)
- "Number" (self-explanatory)
- "PlayerID1" - "PlayerID4": ID as a single integer of the first, second, third and if given of the fourth player. 
- "Player1" - "Player4": Then, the players non-integer ID including their initials and at least one number follows. 
- "Date": the date the table was played
- "IDVServer": ID of the server  
- "Series": the series to which the table belongs to.

The data about the Skat tables is irrelevant for our application.

# Card sequence table

The card sequence table is identified by a game ID. 
- "Sd1", "Sd2": The put down Skat cards
- "CNr0" - "CNr31": The following 32 columns each are fixed representation of a card:
  - They are sorted by colors in the order cross, spades, hearts, diamonds
  - Within the colours the order is from Ace, King, Queen, Jack over ten to seven
  - For instance, card number (CNr) 0 is the Ace of cross, CNr 11 represents the jack of spades
  - The entry of the columns in each game index the position in the game when the card was played
  - Example: CNr 3 entry: 18 -> The Jack of cross (CNr 3) was played as the 18th card 
  - CNrs with the entries 30 and 31 are the put Skat 
- "SurrenderedAt"*: The last entry is the position at which the game was surrendered, if this value is "-1", the game was not surrendered

*Note: the entries "SurrenderedAt" and "Surrendered" of the game table do not logically match, which can be explained by not consequently log redundantly. The entry "SurrenderedAt" is set correctly in surrendered games 

# Game table

The game table consists of the entries

- "GameID": the GameID
- "IDGame": IDGame, another ID which exists due to the config of the database 
- "IDTable": the table of the game 
- "IDVServer": the server of the game
- "StartTime", "EndTime" (self-explanatory)
- "PlayerFH","PlayerMH","PlayerBH": Player in forehand, mid-hand and rear-hand
- "Card1"-"Card32": 
  - Card1-Card10 belong to forehand, 
  - Card11-20 to mid-hand, 
  - Card21-30 to rear-hand
  - Card 31-32 Skat before pick up 
- "CallValueFH", "CallValueMH", "CallValueBH": the call values of the players**
- "PlayerID": the ID of the player declaring a game
- "Game": the game played, can take on following values
  - 0: Nobody played (all passed)
  - 9, 10, 11, 12: Diamonds, Hearts, Spades, Cross
  - 24: Grand
  - 23, 35, 46, 59: Null, Null hand, Null ouvert, Null ouvert hand, respectively
- "With", "Without": The highest trumps 
- "Hand", "Schneider", "SchneiderCalled", "Schwarz", "SchwarzCalled", "Ouvert": binary encoded playing mode
- "PointsPlayer": resulting points of player (tournament count)
- "Won", "Miscall": binary encodings of won and miscalled
- "CardPointsPlayer": the final points of the soloist's cards
- "AllPassed": whether all passed
- "Surrendered": whether the game was surrendered*
- "PlayerPosAtTableFH", "PlayerPosAtTableMH", "PlayerPosAtTableBH": the players positions at the table**

**Note: is inconsistent in database due to historic reasons, not relevant for our scope




 # FIXED: problem of context length max = 1024: see Solution 3

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
