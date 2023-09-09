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
python dt_train.py --championship 'wc' --games (0, 10000) 
````
For more information about the options:
````shell
python dt_train.py --help 
````
For playing against a pre-trained model, provide your starting position. It will rotate with each round. 
For example, to start in fore-hand:
````shell
python dt_train.py --play_as 0
````
