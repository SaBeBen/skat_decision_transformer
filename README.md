# Skat Decision Transformer

This is the implementation developed during my Bachelor thesis.
The project aims to explore the capabilities of [Decision Transformers](https://github.com/kzl/decision-transformer/tree/master) (DTs)
in application to card game Skat.

It trains a DT on expert data and evaluates the trained model. For that, a data pipeline converts the 
information of the games into states, actions and returns. Then, a data collator applies batching, masking and produces 
the returns-to-go. The states, actions and returns-to-go are fed into a huggingface DT which is modified to log Skat 
relevant metrices and predictions.

The training can be reproduced with the following command:
````shell
python dt_train.py --championship 'wc' --games (0, 1000) 
````
For more information about the options:
````shell
python dt_train.py --help 
````


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
