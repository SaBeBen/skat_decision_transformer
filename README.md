# Skat Decision Transformer

This is the implementation developed during my Bachelor thesis.
The project aims to explore the capabilities of [Decision Transformers](https://github.com/kzl/decision-transformer/tree/master) (DTs)
in application to card game Skat.

It trains a DT on expert data and evaluates the trained model. For that, a data pipeline converts the 
information of the games into states, actions and returns. Then, a data collator applies batching, masking and produces 
the returns-to-go. The states, actions and returns-to-go are fed into a huggingface DT which is modified to log Skat 
relevant metrices and predictions.
Further, you can evaluate pre-trained model on random plays or against yourself.
Note that playing is prioritized over evaluation over training. You cannot do all or two of them at once.

The training can be reproduced with the following command:
````shell
python experiments.py --championship "wc" --games 0 10000
````
For more information about the options:
````shell
python experiments.py --help 
````
For playing against a pre-trained model, provide your starting position. It will rotate with each round. 
For example, to start in fore-hand, play 36 games against the model trained on all non-surrendered suit WC games:
````shell
python experiments.py --play_as 0 --amount_games_to_play 36 --pretrained_model "games_all-encoding_one-hot-point_rewards_True-card_put-masked-Thu_Sep__7_22-41-35_2023"
````
To let the AI play against a random actor:
````shell
python experiments.py --play_as 0 --amount_games_to_play 2000 --random_player True --pretrained_model "games_all-encoding_one-hot-point_rewards_True-card_put-masked-Thu_Sep__7_22-41-35_2023"
````
To let it play against itself with German championship starting configurations:
````shell
python experiments.py --online_eval True --amount_games_to_play 2000 --pretrained_model "games_all-encoding_one-hot-point_rewards_True-card_put-masked-Thu_Sep__7_22-41-35_2023"
````
Note that due to the increased size of datasets, only the raw data tables and not the datasets are included in this version. 
The data_pipeline can however read in these datasets (takes up to several hours).