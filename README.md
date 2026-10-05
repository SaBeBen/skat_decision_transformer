# Skat Decision Transformer

This is the implementation developed during my Bachelor thesis.
The project aims to explore the capabilities
of [Decision Transformers](https://github.com/kzl/decision-transformer/tree/master) (DTs)
in application to card game Skat.

It trains a DT on expert data and evaluates the trained model. For that, a data pipeline converts the
information of the games into states, actions and returns. Then, a data collator applies batching, masking and produces
the returns-to-go. The states, actions and returns-to-go are fed into a Huggingface (HF) DT which is modified to log
Skat relevant metrics and predictions.
Further, you can evaluate pre-trained model on random plays or against yourself.
Note that playing is prioritized over evaluation over training. You cannot do all or two of them at once.

The project requires Python 3.10 or 3.11. To install the requirements, run pip install on the
[configuration file](requirements.txt): 
````shell
pip install -r requirements.txt
````

The tests can be run from the root of the repository with:

````shell
pytest
````

The training can be reproduced with the following command:

````shell
python experiments.py --championship "wc" --games 0 10000
````

By default, suit, grand and null games are used. The thesis only trained on suit games, which corresponds to
`--include_grand False --include_null False --reward_convention thesis`.

With `--hidden_card_head True`, the DT additionally learns where the cards it cannot see are (left opponent, right
opponent or Skat). The data pipeline records these locations for every time step and the cross-entropy of the
prediction is added to the loss with `--hidden_loss_weight`. The auxiliary task shapes the representation of the state
towards inferring the hidden cards, which is crucial in Skat.

`--augment_prob` permutes the suits of training games with the given probability
([trainer/augmentation.py](trainer/augmentation.py)): the non-trump suits in suit games and all suits in grand (without
the jacks) and null games. This does not change the course or the value of a game and multiplies the variety of the
expert data.

### Returns-to-go against the luck of the deal

In Skat, the return of a game depends heavily on the dealt cards, so a DT conditioned on raw returns partly learns to
be lucky instead of playing well. `--rtg_mode` replaces the return-to-go labels while the DT stays unchanged
([trainer/return_model.py](trainer/return_model.py)):

- `expected`: the expected return of the taken action given only what the player observed (the luck of unseen cards is
  averaged out),
- `deal_margin`: the achieved return minus the expected return of the deal, estimated by a model that sees all hands
  during training (never when playing).

The return models are trained by regression on the returns of the data and saved to `pretrained_models/return-*`:

````shell
python experiments.py --train_return_model deal_margin --games 0 -1 --num_epochs 20 --batch_size 512
python experiments.py --games 0 -1 --rtg_mode deal_margin --return_model "<return model>" --save_model True
````

The DT stores the return-to-go it is conditioned on when playing (a high quantile of the relabelled returns,
`--rtg_quantile`) in its config.

For more information about the options:

````shell
python experiments.py --help 
````

For playing against a pre-trained model, provide your starting position. It will rotate with each round.
For example, to start in forehand, play 36 games against the model trained on all non-surrendered suit WC games:

````shell
python experiments.py --play_as 0 --amount_games_to_play 36 --pretrained_model "games_all-encoding_one-hot-point_rewards_True-card_put-masked-Thu_Sep__7_22-41-35_2023"
````

To let the AI play against a random actor:

````shell
python experiments.py --play_as 0 --amount_games_to_play 2000 --random_player True --pretrained_model "games_all-encoding_one-hot-point_rewards_True-card_put-masked-Thu_Sep__7_22-41-35_2023"
````

To let it play against itself with German championship starting configurations:

````shell
python experiments.py --online_eval True --games 0 2 --pretrained_model "games_all-encoding_one-hot-point_rewards_True-card_put-masked-Thu_Sep__7_22-41-35_2023"
````

The included pre-trained models were trained with the reward convention of the thesis, in which the put Skat is
rewarded a second time in the first trick. Add `--reward_convention thesis` when playing with or evaluating them, so
that the returns-to-go during play match the ones seen in training. New models use the default convention.

Note that due to the increased size of datasets, only the raw data tables and not the datasets are included in this
version.
However, the data_pipeline can read in these data (takes up to several hours) and save them into a dataset directory:

````shell
python data_pipeline.py
````

## Bidding and game declaration

The auction and the choice of the game are learned by a second, return-conditioned Decision Transformer with a single
time step ([trainer/declaration.py](trainer/declaration.py)). Given the dealt hand, the seat and a target return, it
predicts the declaration (pass, a suit, grand or null, each from the hand or with the Skat). The highest bid of a player
is the value of its declaration, and the auction follows the order of the rules (middle-hand bids to fore-hand,
afterwards rear-hand bids to the winner). It is trained on the declarations of the data and the points they achieved:

````shell
python experiments.py --train_declaration True --games 0 -1 --num_epochs 20 --batch_size 256
````

The model is saved to `pretrained_models/declaration-*` and can be used with `--declaration_model` when playing
(`experiments.py`) or in the online fine-tuning, where `--finetune_declaration True` also fine-tunes it on the points
its declarations achieve in self-play. Without a declaration model, the player with the best cards plays a suit game.

## Online fine-tuning (Online Decision Transformer)

As proposed in the future work of the thesis, a pre-trained DT can be fine-tuned online via self-play following the
[Online Decision Transformer](https://arxiv.org/abs/2202.05607) (Zheng et al., 2022):

````shell
python online_training.py --pretrained_model "games_all-encoding_one-hot-point_rewards_True-card_put-masked-Thu_Sep__7_22-41-35_2023"
````

Each iteration of [online_training.py](online_training.py)
1. plays `--games_per_iter` games on random deals. Each seat is played by the learner with probability
   `--learner_seat_prob`, otherwise by a frozen opponent from a pool containing the pre-trained model and snapshots of
   the learner (`--snapshot_every`, `--pool_size`). The learner samples its actions from its policy, restricted to the
   legal cards of the game engine ([online/self_play.py](online/self_play.py)),
2. adds the learner's trajectories to a FIFO replay buffer ([online/replay_buffer.py](online/replay_buffer.py)), which
   is initially seeded with offline games (`--offline_games`). The returns-to-go are computed from the achieved rewards
   (hindsight return relabelling),
3. performs `--updates_per_iter` gradient steps with the ODT objective: the negative log-likelihood of the buffer's
   actions with a lower bound on the policy entropy (`--target_entropy`), enforced by a learned temperature
   ([online/odt.py](online/odt.py)),
4. every `--eval_every` iterations, compares the learner to the pre-trained model on `--eval_games` fixed deals
   (duplicate evaluation: every deal is played with the pre-trained model on all seats, with the learner as declarer,
   and with the learner as both defenders, see [online/evaluation.py](online/evaluation.py)).

The metrics are logged to TensorBoard in `training-logs/online-*` (`eval/delta/*` shows the improvement over the
pre-trained model), checkpoints are saved to `training_output/online-*`. The defaults are sized for a GPU server; for a
quick test on a CPU use e.g.
`--iterations 2 --games_per_iter 16 --updates_per_iter 20 --eval_games 20 --offline_games 0 100`.
See `python online_training.py --help` for all options.

## Security

The dependencies are pinned to versions without known vulnerabilities (checked with `pip-audit -r requirements.txt`).
The pre-trained models are stored as `model.safetensors` and loaded with `use_safetensors=True`, so loading a model
never unpickles data. Only load models and checkpoints from sources you trust.

# Project Structure

We modularize our project into a data pipeline, a Skat environment, a trainer and experiments. The Skat environment is
based on the existing skat_buddy implementation. Our trainer employs the HF DT implementation.
The card representation can be set with the option "--hand_encoding" when calling [experiments.py](experiments.py).
However, the pretrained model relies on the one-hot encoding and cannot be used with another.
If you want to change the state representation, you have to change the dimensions in the function "get_dims_in_enc" in
the [environment](dt_skat_environment/environment.py), [data_pipeline.py](data_pipeline.py) and the offset in the
creation of the masks in the [trainer](trainer/dt_trainer.py).

## Data Pipeline

We have to extract the representation of the game from the raw data and represent it in states, actions and rewards.
Multiple card representations are possible. The state and its dimension changes when providing different representations
as an argument in [experiments.py](experiments.py).
There are four different card encodings (with their argument to use them via "experiment.py --hand_encoding"):

- one-hot (one-hot)
- mixed (mixed)
- compressed one-hot (one-hot_comp)
- compressed mixed (mixed_comp)

The data pipeline also records the legal actions of the game engine for every time step. The data collator passes them
to the model as `action_mask`, which restricts the predictions to legal cards for every encoding and game variant.
Without these masks (e.g. datasets created before), the model falls back to a rule mask derived from the one-hot state
representation, which only supports suit games.

The data directory includes the raw data in csv format. The format of the data is explained in another README in
skat_decision_transformer/data.

The directory [data_analysis](data_analysis) contains scripts analyzing
a [certain championship](data_analysis/analysis_championships.py)
or [all championships](data_analysis/statistics_all_championships.py).
Both create plots saved in the directory data_analysis/graphics.

The directory datasets contains datasets with a test/train split generated by data_pipeline.py in arrow format for
training. To guarantee reproducibility, we use a seed for the random generation.
The existing datasets are loaded when at least 10000 games should be included in training.

[data_pipeline.py](data_pipeline.py) loads the raw data from the csv files, replays the games from the data on Skat
environment to validate the card playing actions and generates the state, action and reward representations.
It can also load the first states and meta and card data. This is only relevant to possibly load games more efficiently
for evaluation. However, we do not save the first states and meta and card information, because the several thousand
games for the online evaluation can be read in within a few minutes.

A data collator prepares the data during training and is called by a data loader. This
is more efficient than converting all games, as it only loads the ones used in training.

A script as [data_pipeline.py](data_pipeline.py) does not exist in the linked HF implementation, as the data already is
prepared for the data collator. We need the script for the game extraction from the raw data.

## Skat Environment

Our Skat environment dt_skat_environment the necessary conversion of card representation and game engine. The game
engine enables playing Skat and throws [exceptions](dt_skat_environment/game_engine/exceptions.py) when players attempt
illegal moves. It also checks for the correct order of [game states](dt_skat_environment/game_engine/game/state) being
start, bid, play and end.

## Trainer

Our trainer defines a data collator and adapts
the [Decision Transformer from HF](https://huggingface.co/docs/transformers/main/model_doc/decision_transformer) to our
Skat environment.
We adapt the collator partially, because
their [example use case](https://huggingface.co/blog/train-decision-transformers) of a Gym Cheetah differs from ours.
Theirs is a continuous environment, ours a discrete.
Furthermore, we limit the maximum episode length to 12 timesteps of states, actions and rewards per game, their
continuous and cyclic movement allows splitting the movement into more timesteps.

The trainer logs to [training-logs](training-logs) and saves training checkpoints to [training_output](training_output).
Furthermore, there is an option to save a model as a pre-trained model in [pre-trained_models](pretrained_models).
One pre-trained model exists with which experiments can be carried out. It was trained on all non-surrendered suit games
of the world championship data. 
294 752 games from every perspective with 240 epochs. 
We used it for our playing experiments.
Also, there are two smaller models which only include 20000 or 50000 of these non-surrendered suit games, also trained.
The configuration is the same as the one trained on all on these two.
All pre-trained models used masking during training. 

Plots can be generated by a [plotting_script](plots/plotting_script.py) from selected
[Tensorboard (TB) logs](plots/tensorboard_graphs). To create a plot, the TB log(s) should exist in
[TB graphs](plots/tensorboard_graphs). These logs are created in [training-logs](training-logs) during
training. To achieve this, copy the wanted logs into the [TB graphs](plots/tensorboard_graphs) and change the default
run_ids (log folder names) or provide them in the arguments when calling
the [plotting script](plots/plotting_script.py).

## Experiments

experiments.py defines the experiments. With the help of an argument parser, we can run the following procedures
(with their functions the main function calls):

- Training (run_training)
- (deactivated) manual evaluation (evaluate_in_env)
- Playing against
    - Itself (run_online_eval)
    - A random player (play_with_two)
    - It as a human (play_with_two)

# Workflow of Experiments

The workflow depends on the choice of the experiment listed below.
The functions to call are in brackets.

### Training (run_training)

1. Load dataset either from existing sources or new. Random seed enables reproducibility
2. Use HF train framework:
    1. Initialize data collator
    2. Initialize DT config
    3. Initialize model or load pre-trained model if given
    4. Initialize training arguments
    5. Set evaluation parameters
    6. Train model
    7. (optionally) save model (is not saved by default)

The HF trainer is part of our custom DT trainer. We alter the forward function to apply a mask for valid actions
specifically for the one-hot encoding. Further, we manipulate the logging in the forward, evaluate and training_step
methods to log own metrics, such as the probability of correct actions.

### Playing against itself (run_online_eval)

1. Read in arguments from parser calls
2. Load dimensions
3. Load pre-trained model
4. Play against itself over 12 timesteps from every perspective (eval_three_agents)

### Playing against a random player and against a human (play_with_two)

1. Read in arguments from parser calls
2. Load dimensions
3. Load pre-trained model
4. Play with two agents based on the pre-trained model over 12 timesteps from every perspective (play_with_two_agents)
    - Differs between a random player and human player through if-conditions while playing

### Manual implementation (evaluate_in_env) (deactivated, kept for debugging usage)

- Plays Skat in background. Thus, less efficient than integrated testing in training,
- Can be employed for debugging
