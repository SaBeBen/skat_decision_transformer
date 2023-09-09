"""
This Implementation is based on the Decision Transformer https://github.com/kzl/decision-transformer/tree/master.

To achieve a speed-up and more modularity, the Huggingface (HF) implementation of the Decision Transformer is adapted.
See https://huggingface.co/blog/train-decision-transformers for more information.
"""

import argparse
import math
# import resource
import time
from typing import List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data_collator import DecisionTransformerSkatDataCollator
from datasets import Dataset, DatasetDict, load_from_disk
from torch import nn
from torch import Tensor
from sklearn.model_selection import train_test_split

from transformers import TrainingArguments, DecisionTransformerConfig

import data_pipeline
from dt_trainer import TrainableDT, DTTrainer
from environment import ACT_DIM, get_dims_in_enc, Env, MAX_EPISODE_LENGTH


# %%
# def set_memory_limit(limit_gb):
#     limit_bytes = limit_gb * 1024 * 1024 * 1024  # Convert MB to bytes
#     resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
#
#
# set_memory_limit(64)


def run_training(args):
    # game specific arguments
    championship = args['championship']
    hand_encoding = args['hand_encoding']
    point_rewards = args['point_rewards']
    games_to_load = slice(args['games'][0], args['games'][1])
    perspective = args['perspective']

    # train specific arguments
    batch_size = args['batch_size']
    n_layer = args['n_layer']
    n_head = args['n_head']
    act_fct = args['activation_function']
    dropout = args['dropout']
    l_rate = args['learning_rate']
    weight_decay = args['weight_decay']
    warmup_ratio = args['warmup_ratio']
    num_train_epochs = args['num_epochs']
    logging_steps = args['logging_steps']
    use_cuda = args['use_cuda']
    save_model = args['save_model']
    pretrained_model_path = args['pretrained_model']
    eval_in_training = args['eval_in_training']

    if torch.cuda.is_available() and use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    card_dim, max_hand_len, state_dim = get_dims_in_enc(hand_encoding)

    episode_length = (state_dim + ACT_DIM + 1) * MAX_EPISODE_LENGTH

    # ensure that our sequence length fits in the context length
    if episode_length > 1024:
        context_length = 2048
    else:
        context_length = 1024

    load_dataset = games_to_load.stop == -1 or games_to_load.stop - games_to_load.start >= 10000

    if load_dataset:
        # "wc-without_surr_and_passed-pr_True-one-hot"
        dataset = load_from_disk(f"./datasets/wc-without_surr_and_passed-pr_{point_rewards}-{hand_encoding}-card_put")
        amount_games = (games_to_load.stop - games_to_load.start) * 3  # * 3 for every perspective
        dataset['train'] = Dataset.from_dict(dataset['train'][:math.floor(amount_games * 0.8)])
        dataset['test'] = Dataset.from_dict(dataset['test'][:math.floor(amount_games * 0.2)])
    else:
        print("\nLoading data...")
        data, _, first_states, meta_and_cards, actions_table, skat_and_cs = data_pipeline.get_states_actions_rewards(
            # first_states, meta_and_cards, actions_table, skat_and_cs
            championship=championship,
            games_indices=games_to_load,
            point_rewards=point_rewards,
            perspective=perspective,
            card_enc=hand_encoding)

        data_frame = pd.DataFrame(data)
        if data_frame.shape[0] == 1:
            # if only one game is given
            dataset = DatasetDict({"train": Dataset.from_dict(data),
                                   "test": Dataset.from_dict(data)})
        else:
            data_train, data_test = train_test_split(data_frame, train_size=0.8, random_state=42)  # 42
            dataset = DatasetDict({"train": Dataset.from_dict(data_train),
                                   "test": Dataset.from_dict(data_test)})

    dataset = dataset.with_format("torch")

    collator = DecisionTransformerSkatDataCollator(dataset["train"], batch_size)

    config = DecisionTransformerConfig(
        state_dim=state_dim,
        act_dim=ACT_DIM,
        action_tanh=False,  # do not apply the tanh fct on the output action
        activation_function=act_fct,
        n_head=n_head,
        n_layer=n_layer,
        max_ep_len=MAX_EPISODE_LENGTH,  # each episode is a game -> 12 tuples of s,a,r make up 1 game
        n_positions=context_length,
        scale_attn_weights=True,
        embd_pdrop=dropout,
        resid_pdrop=dropout,
        attn_pdrop=dropout,
        max_length=MAX_EPISODE_LENGTH,
    )

    if pretrained_model_path is not None:
        pretrained_model = TrainableDT.from_pretrained(pretrained_model_path)
        model = pretrained_model

        if model.config.state_dim != state_dim:
            raise ValueError("The current configuration and the model do not have the same state dimension.\n"
                             "This is caused by altering the state representation.\n"
                             "Please choose the newest model.")
    else:
        model = TrainableDT(config)

    model = model.to(device)

    current_time = time.asctime().replace(':', '-').replace(' ', '_')

    dir_name = rf"games_{games_to_load.start}-{games_to_load.stop}-encoding_{hand_encoding}-point_rewards_{point_rewards}-{current_time}"

    logging_dir = rf"./training-logs/{dir_name}"

    training_args = TrainingArguments(
        report_to=["tensorboard"],
        output_dir="training_output/",
        remove_unused_columns=False,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,
        learning_rate=l_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        optim="adamw_torch",
        max_grad_norm=0.1,
        logging_steps=logging_steps,
        logging_dir=logging_dir,
        # dataloader_drop_last=True,  # drop incomplete batches, prevents training of less than batch_size games
        # bf16=True,
        save_steps=5000
    )

    if eval_in_training:
        training_args.do_eval = True
        training_args.evaluation_strategy = "steps"
        training_args.eval_steps = 100

    trainer = DTTrainer(  # CustomDTTrainer
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=collator,
    )

    print(f"\nTraining on games {games_to_load.start, games_to_load.stop} with {hand_encoding} encoding...")
    trainer.train()

    # and loading it again
    # pretrained_model = TrainableDT.from_pretrained("./pretrained_models")
    # trainer.model = pretrained_model

    # if save_model:
    #     model.save_pretrained(rf"./pretrained_models/{dir_name}")

    # training_args.do_eval, training_args.evaluation_strategy, training_args.eval_steps = True, "steps", 10
    #
    # trainer.data_collator = DecisionTransformerSkatDataCollator(dataset["test"])
    #
    # evaluation_results = trainer.evaluate()
    #
    # print(evaluation_results)

    # first_states = dataset['test']['states'][]
    if not load_dataset:
        evaluate_in_env(model, point_rewards, state_dim,
                        card_enc=hand_encoding,
                        first_states=first_states,
                        meta_and_cards_game=meta_and_cards,
                        skat_and_cs=skat_and_cs,
                        correct_actions=actions_table)


# ------------------------------------------------------------------------------------

# Function that gets an action from the model using autoregressive prediction
# with a window of the previous 20 timesteps. We only need a maximum of 12 timesteps
def get_action(model, states, actions, rewards, returns_to_go, timesteps):
    # This implementation does not condition on past rewards

    states = states.reshape(1, -1, model.config.state_dim)
    actions = actions.reshape(1, -1, model.config.act_dim)
    returns_to_go = returns_to_go.reshape(1, -1, 1)
    timesteps = timesteps.reshape(1, -1)

    # The prediction is conditioned on up to 20 previous time-steps
    states = states[:, -model.config.max_length:]
    actions = actions[:, -model.config.max_length:]
    returns_to_go = returns_to_go[:, -model.config.max_length:]
    timesteps = timesteps[:, -model.config.max_length:]

    # pad all tokens to sequence length, this is required if we process batches
    padding = model.config.max_length - states.shape[1]
    attention_mask = torch.cat([torch.zeros(padding), torch.ones(states.shape[1])])
    attention_mask = attention_mask.to(dtype=torch.long).reshape(1, -1)
    states = torch.cat([torch.zeros((1, padding, model.config.state_dim)), states], dim=1).float()
    actions = torch.cat([torch.zeros((1, padding, ACT_DIM)), actions], dim=1).float()
    returns_to_go = torch.cat([torch.zeros((1, padding, 1)), returns_to_go], dim=1).float()
    timesteps = torch.cat([torch.zeros((1, padding), dtype=torch.long), timesteps], dim=1)

    # perform the prediction
    state_preds, action_preds, return_preds = model.original_forward(
        states=states,
        actions=actions,
        rewards=rewards,
        returns_to_go=returns_to_go,
        timesteps=timesteps,
        attention_mask=attention_mask,
        return_dict=False,
    )
    return action_preds[0, -1]


# def training_loop(num_epochs, num_batches, optimizer, collator: DecisionTransformerSkatDataCollator):
#     for epoch in range(num_epochs):
#         for _ in range(num_batches):
#             # state = env.reset(current_player_id=current_player)
#             batch = collator.__call__()
#             action_target = x
#
#             action_pred = get_action(model, )
#
#             # Backpropagation
#             optimizer.zero_grad()
#             loss.backward()
#             # Optimisation
#             optimizer.step()


def evaluate_in_env(model: TrainableDT,
                    point_reward: bool,
                    state_dim: int,
                    card_enc: str,
                    first_states: List[np.ndarray],
                    meta_and_cards_game: List[np.ndarray],
                    skat_and_cs: List[np.ndarray],
                    correct_actions: List) -> None:
    """
    Less efficient manual evaluation, actually plays Skat in background:

    This evaluation guides the agent through a game.
    If another card than in the data was played, the course of the game would shift,
    e.g. if the agent puts a different card in the Skat, he will miss this card during play and can
    eventually play other cards, as the distribution of suits on the hand changes.
    """

    scale = 1

    amount_games = math.floor(len(first_states) / 3)

    correct_actions_idx = np.argmax(correct_actions, axis=2)

    for game_idx in range(amount_games):

        for current_player in range(3):
            game_and_pl_idx = 3 * game_idx + current_player

            env = Env(card_enc)

            # build the environment for the evaluation
            state = env.reset(current_player,
                              game_first_state=first_states[game_and_pl_idx],
                              meta_and_cards_game=meta_and_cards_game[game_idx],
                              skat_and_cs=skat_and_cs[game_idx])

            # Whether to use the Seeger-Fabian score or not
            if point_reward:
                if env.current_player == env.game.get_declarer():
                    if sum(env.trump_enc) == 0:
                        game_points = env.game.game_variant.get_level()
                        target_return = game_points * 2
                    else:
                        if sum(env.trump_enc) == 4:
                            base_value = 24
                        else:
                            base_value = 9 + env.trump_enc.index(1)

                        level = 1 + env.game.game_variant.get_level()
                        # when playing as declarer the maximum points are doubled like some configurations
                        # of the DT paper (Chen et al., 2021)
                        # both are scaled to hinder reward hacking
                        target_return = 0.1 * 2 * 120 + 0.9 * (level * base_value * 2 + 50)
                else:
                    # it is enough to reach 60 points as defender to win and practically impossible to reach 120,
                    # thus, the rewards are doubled from 60 and the bonus of a loss of the declarer is added
                    # both are scaled to hinder reward hacking
                    target_return = 0.1 * 120 + 0.9 * 40
            else:
                if env.current_player == env.game.get_declarer():
                    # if the reward are only the card points
                    target_return = 2 * 120
                else:
                    # it is enough to reach 60 points as defender to win and practically impossible to reach 120,
                    # thus, the rewards are doubled from 60 and the bonus of a loss of the declarer is added
                    target_return = 120

            target_return = torch.tensor(target_return).float().reshape(1, 1)
            states = torch.from_numpy(state).reshape(1, state_dim).float()
            actions = torch.zeros((0, ACT_DIM)).float()
            actions_preds = torch.zeros((0, ACT_DIM)).float()
            rewards = torch.zeros(0).float()
            timesteps = torch.tensor(0).reshape(1, 1).long()

            # take steps in the environment (evaluation, not training)
            for t in range(MAX_EPISODE_LENGTH):

                # add zeros for actions as input for the current time-step
                actions = torch.cat([actions, torch.zeros((1, ACT_DIM))], dim=0)
                actions_preds = torch.cat([actions_preds, torch.zeros((1, ACT_DIM))], dim=0)
                rewards = torch.cat([rewards, torch.zeros(1)])

                # predicting the action to take
                action_pred = get_action(model,
                                         states,
                                         actions,
                                         rewards,
                                         target_return,
                                         timesteps)

                actions_preds[-1] = action_pred

                action = action_pred.detach().numpy()

                # get the index of the card with the highest probability
                card_index = np.argmax(action)

                # only select the best card
                action[:] = 0
                action[card_index] = 1

                action_target_idx = correct_actions_idx[game_and_pl_idx][t]
                card_pred = env.current_player.cards[card_index]
                card_target = env.current_player.cards[action_target_idx]

                if card_index != action_target_idx:
                    # put card from targets, if agent wants to select other
                    action = correct_actions[game_and_pl_idx][t]
                    print(f"Agent wanted to select {card_pred}, but target is {card_target}")

                actions[-1] = Tensor(action)

                # interact with the environment based on this action
                # reward_player is a tuple of the trick winner and her reward,
                # else there are difficulties with reward assignment
                state, reward, done = env.step(tuple(action))

                # print(f"Reward {t}: {reward}")

                cur_state = torch.from_numpy(state).reshape(1, state_dim)
                states = torch.cat([states, cur_state], dim=0)
                rewards[-1] = reward

                pred_return = target_return[0, -1] - (rewards[-1] / scale)
                target_return = torch.cat(
                    [target_return, pred_return.reshape(1, 1)], dim=1)
                timesteps = torch.cat(
                    [timesteps, torch.ones((1, 1)).long() * (t + 1)], dim=1)

                if done:
                    nll_fct = nn.NLLLoss()

                    action_targets = torch.from_numpy(correct_actions[game_and_pl_idx])

                    # mask zero actions for the loss, otherwise nll_loss = inf due to argmax selecting the first 0 as
                    # target index when no action is taken
                    mask_wo_zeros = action_targets.sum(dim=1)
                    mask_wo_zeros = mask_wo_zeros > 0

                    action_targets = action_targets[mask_wo_zeros]

                    actions_preds = actions_preds[mask_wo_zeros]

                    nll_loss = nll_fct(torch.log(actions_preds),
                                       torch.argmax(action_targets, dim=1))

                    # log
                    print(f"Loss of game {game_and_pl_idx}: {nll_loss}")

                    break


def run_online_eval(args):
    # game specific arguments
    championship = args['championship']
    hand_encoding = args['hand_encoding']
    point_rewards = args['point_rewards']

    card_dim, max_hand_len, state_dim = get_dims_in_enc(hand_encoding)
    pretrained_model = TrainableDT.from_pretrained(
        f"./pretrained_models/games_0-50000-encoding_one-hot-point_rewards_True-Mon_Sep__4_23-48-24_2023")

    pretrained_model.config.state_dim = state_dim

    eval_three_agents(pretrained_model,
                      point_reward=point_rewards,
                      state_dim=state_dim,
                      card_enc=hand_encoding,
                      amount_games=10
                      )


def eval_three_agents(model, point_reward, state_dim, card_enc, amount_games, first_game_states=None):
    scale = 1

    # env = environment.Env() if one_game is None else one_game

    # initialise s, a, r, t and raw action logits for every player
    states, actions, rewards, timesteps, actions_pred_eval = [[] * 3], [[] * 3], [[] * 3], [[] * 3], [[] * 3]

    # accelerator = Accelerator(gradient_accumulation_steps=1)
    # optimiser = torch.optim.Adam(model.config)
    # model = accelerator.prepare_model(model)

    # current_time = time.asctime().replace(':', '-').replace(' ', '_')
    #
    # dir_name = rf"online_training-episodes_{amount_games}-point_rewards_{point_reward}-{current_time}"
    #
    # logging_dir = rf"./training-logs/{dir_name}"
    #
    # training_args = TrainingArguments(
    #     report_to=["tensorboard"],
    #     output_dir="online_training_output/",
    #     remove_unused_columns=False,
    #     num_train_epochs=240,
    #     per_device_train_batch_size=32,
    #     per_device_eval_batch_size=32,
    #     gradient_accumulation_steps=4,
    #     optim="adamw_torch",
    #     max_grad_norm=0.1,
    #     logging_steps=50,
    #     logging_dir=logging_dir,
    #     save_steps=5000
    # )
    #
    # trainer = DTTrainer(  # CustomDTTrainer
    #     model=model,
    #     args=training_args
    # )
    #
    # trainer.training_step(model)

    for _ in tqdm(range(amount_games)):
        # TODO: rotate players after game
        env = Env(card_enc)

        # build the environment for the evaluation
        states = env.online_reset()

        target_return = [0] * 3

        # as the maximum reward differs for defenders and declarer,
        # we have to differ to set the reward to the double of the maximum possible
        for pl in range(3):
            if point_reward:
                if sum(env.trump_enc) == 0:
                    game_points = env.game.game_variant.get_level()
                    target_return[pl] = game_points * 2
                else:
                    if sum(env.trump_enc) == 4:
                        base_value = 24
                    else:
                        base_value = 9 + env.trump_enc.index(1)

                    level = 1 + env.game.game_variant.get_level()
                    target_return[pl] = level * base_value * 2

                if env.game.players[pl] == env.game.get_declarer():
                    target_return[pl] += 120 * 0.1 + target_return[pl] * 0.9
                else:
                    # Seeger Score gives defenders points for winning.
                    # We only play as 3 players -> 40 points if won as defender
                    target_return[pl] += 120 * 0.1 + 40 * 0.9
            else:
                # if the reward are only the card points
                target_return[pl] = 2 * 120
        # target_return = 2 * 120

        target_return = torch.tensor(target_return).float().reshape(3, 1)
        states = torch.from_numpy(states).reshape(3, 1, state_dim).float()
        actions = torch.zeros((3, 0, ACT_DIM)).float()
        actions_pred_eval = torch.zeros((3, 0, ACT_DIM)).float()
        rewards = torch.zeros(3, 0).float()
        timesteps = torch.tensor(0).reshape(1, 1).long()

        # take steps in the environment (evaluation, not training)
        for t in range(12):
            # add zeros for actions as input for the current time-step
            actions = torch.cat([actions, torch.zeros((3, 1, ACT_DIM))], dim=1)
            actions_pred_eval = torch.cat([actions_pred_eval, torch.zeros((3, 1, ACT_DIM))], dim=1)
            rewards = torch.cat([rewards, torch.zeros(3, 1)], dim=1)

            cur_state = torch.zeros(0, state_dim).float()

            cur_state = np.array([[0] * state_dim] * 3)
            cur_pred_return = torch.zeros(0, 1).float()

            for i in range(3):
                # use the first playing order as an order of playing when putting the Skat
                # note that there is no order of playing cards during Skat putting in the game, as only the declarer
                # takes actions
                # after the first trick, the order of playing can change depending on the last tricks winner

                # skat putting
                if t < 2 and i != env.game.get_declarer().get_id():
                    current_player = i

                    state = states[current_player, -1]
                    reward_player = [current_player, 0]

                    action_pred = get_action(model,
                                             states[current_player],
                                             actions[current_player],
                                             rewards[current_player],
                                             target_return[current_player],
                                             timesteps)
                    actions_pred_eval[current_player][-1] = action_pred

                    actions[current_player][-1] = Tensor([0] * 12)

                    # allow to play a card after Skat putting
                    if t == 1:
                        state[3] = 1

                    # cur_state = torch.cat([cur_state, state.reshape(1, state_dim)])
                    cur_state[current_player] = state

                else:
                    # start at the first seat, at the player besides the dealer
                    # if (t == 2) and (i == 0):
                    #     current_player = env.game.get_first_seat().id

                    current_player = env.game.trick.get_current_player()

                    # update the state of the current player to get the open cards
                    states[current_player, -1] = env.update_state(states[current_player, -1])

                    # predicting the action to take
                    action_pred = get_action(model,
                                             states[current_player],
                                             actions[current_player],
                                             rewards[current_player],
                                             target_return[current_player],
                                             timesteps)

                    actions_pred_eval[current_player][-1] = action_pred

                    action = action_pred.detach().numpy()

                    # action_pred are already soft-maxed and valid actions,
                    # we only need to prevent the argmax from choosing an all 0 action
                    if torch.sum(action_pred) != 0:
                        # get the index of the card with the highest probability
                        card_index = np.argmax(action)

                        # only select the best card
                        action[:] = 0
                        action[card_index] = 1
                    else:
                        card_index = -1

                    actions[current_player][-1] = Tensor(action)

                    acting_player = current_player

                    # interact with the environment based on this action
                    # reward_player is a tuple of the trick winner and reward to propagate it
                    # current_player is used to correctly iterate over the states in the order of play
                    state, reward_player, done, current_player = env.online_step(card_index, current_player)

                    # cur_state = torch.cat([cur_state, torch.from_numpy(state).reshape(1, state_dim)])
                    cur_state[acting_player] = state

                # states[current_player] = torch.cat([states[current_player], cur_state], dim=0)
                rewards[reward_player[0]][-1] = reward_player[1]

                pred_return = target_return[reward_player[0], -1] - (rewards[reward_player[0]][-1] / scale)
                cur_pred_return = torch.cat([cur_pred_return, pred_return.reshape(1, 1)])

            # update all states after each trick for the hand cards, score and last_trick
            target_return = torch.cat([target_return, cur_pred_return], dim=1)
            cur_state = torch.from_numpy(cur_state).reshape(3, 1, state_dim)
            states = torch.cat([states, cur_state], dim=1)
            timesteps = torch.cat([timesteps, torch.ones((1, 1)).long() * (t + 1)], dim=1)

            # TODO: loss and backward propagation

        for current_player in range(3):
            # for evaluation of unspecific games
            diff_target_reached = torch.sub(target_return[current_player, 0], target_return[current_player, -1])

            # MSE loss
            loss = diff_target_reached ** 2

            if env.game.get_declarer().id == current_player:
                print(f"{env.game.get_declarer()} achieved a score of {rewards[current_player]}")

            # model.zero_grad()

            mse_loss = torch.nn.MSELoss()
            loss = mse_loss(target_return[current_player, 0], target_return[current_player, -1])
            # accelerator.backward(loss)

            print(
                f"Difference of target reward and reached reward of player {current_player}: {diff_target_reached}")

    # return trained model
    return model


def play_with_two(args):
    # game specific arguments
    championship = args['championship']
    hand_encoding = args['hand_encoding']
    point_rewards = args['point_rewards']
    human_position = args['play_as']

    card_dim, max_hand_len, state_dim = get_dims_in_enc(hand_encoding)
    pretrained_model = TrainableDT.from_pretrained(
        f"./pretrained_models/games_0-50000-encoding_one-hot-point_rewards_True-Mon_Sep__4_23-48-24_2023")

    pretrained_model.config.state_dim = state_dim

    play_with_two_agents(pretrained_model,
                         point_reward=point_rewards,
                         state_dim=state_dim,
                         card_enc=hand_encoding,
                         amount_games=10,
                         human_player_start=human_position
                         )


def play_with_two_agents(model,
                         point_reward,
                         state_dim,
                         card_enc,
                         amount_games,
                         human_player_start,
                         first_game_states=None):
    scale = 1

    # initialise s, a, r, t and raw action logits for every player
    states, actions, rewards, timesteps, actions_pred_eval = [[] * 3], [[] * 3], [[] * 3], [[] * 3], [[] * 3]
    
    # The list points for the human player and the two agents
    list_points = [0, 0, 0]

    for j in tqdm(range(amount_games)):
        env = Env(card_enc)
        
        # rotate position of human player
        human_player_id = (human_player_start + j) % 3

        # build the environment for the evaluation
        states = env.online_reset()

        target_return = [0] * 3

        # as the maximum reward differs for defenders and declarer,
        # we have to differ to set the reward to the double of the maximum possible
        for pl in range(3):
            if point_reward:
                if sum(env.trump_enc) == 0:
                    game_points = env.game.game_variant.get_level()
                    target_return[pl] = game_points * 2
                else:
                    if sum(env.trump_enc) == 4:
                        base_value = 24
                    else:
                        base_value = 9 + env.trump_enc.index(1)

                    level = 1 + env.game.game_variant.get_level()
                    target_return[pl] = level * base_value * 2

                if env.game.players[pl] == env.game.get_declarer():
                    target_return[pl] += 120 * 0.1 + target_return[pl] * 0.9
                else:
                    # Seeger Score gives defenders points for winning.
                    # We only play as 3 players -> 40 points if won as defender
                    target_return[pl] += 120 * 0.1 + 40 * 0.9
            else:
                # if the reward are only the card points
                target_return[pl] = 2 * 120
        # target_return = 2 * 120

        target_return = torch.tensor(target_return).float().reshape(3, 1)
        states = torch.from_numpy(states).reshape(3, 1, state_dim).float()
        actions = torch.zeros((3, 0, ACT_DIM)).float()
        actions_pred_eval = torch.zeros((3, 0, ACT_DIM)).float()
        rewards = torch.zeros(3, 0).float()
        timesteps = torch.tensor(0).reshape(1, 1).long()

        # take steps in the environment (evaluation, not training)
        for t in range(12):
            # add zeros for actions as input for the current time-step
            actions = torch.cat([actions, torch.zeros((3, 1, ACT_DIM))], dim=1)
            actions_pred_eval = torch.cat([actions_pred_eval, torch.zeros((3, 1, ACT_DIM))], dim=1)
            rewards = torch.cat([rewards, torch.zeros(3, 1)], dim=1)

            cur_state = torch.zeros(0, state_dim).float()

            cur_state = np.array([[0] * state_dim] * 3)
            cur_pred_return = torch.zeros(0, 1).float()

            for i in range(3):
                # use the first playing order as an order of playing when putting the Skat
                # note that there is no order of playing cards during Skat putting in the game, as only the declarer
                # takes actions
                # after the first trick, the order of playing can change depending on the last tricks winner

                # skat putting
                if t < 2:
                    current_player = i
                    if current_player == env.game.get_declarer().get_id():

                        if current_player == human_player_id:
                            # define input action
                            print(f"\nYou play as the declarer (player {i}).\n"
                                  f"\n{env.game.game_variant.get_variant_name()} "
                                  f" Your cards are {env.game.players[i].cards}")

                            card_index = -1
                            while card_index > (12 - t) or card_index < 0:
                                card_index = input(
                                    f"Which card do you want to put in the Skat "
                                    f"(possible indices in range 0, {12 - t})?")
                                try:
                                    card_index = int(card_index)
                                except ValueError:
                                    print("Invalid input. Please enter a valid number.")

                            action = [0] * 12
                            action[card_index] = 1
                        else:
                            action_pred = get_action(model,
                                                     states[current_player],
                                                     actions[current_player],
                                                     rewards[current_player],
                                                     target_return[current_player],
                                                     timesteps)
                            # action_pred are already soft-maxed and valid actions,
                            # we only need to prevent the argmax from choosing an all 0 action

                            actions_pred_eval[current_player][-1] = action_pred

                            action = action_pred.detach().numpy()

                            if torch.sum(action_pred) != 0:
                                # get the index of the card with the highest probability
                                card_index = np.argmax(action)

                                # only select the best card
                                action[:] = 0
                                action[card_index] = 1
                            else:
                                card_index = -1

                        actions[current_player][-1] = Tensor(action)

                        # interact with the environment based on this action
                        # reward_player is a tuple of the trick winner and reward to propagate it
                        state, reward_player, done, _ = env.online_step(card_index, current_player)

                        cur_state[i] = state

                    else:
                        state = states[current_player, -1]
                        reward_player = [current_player, 0]

                        action_pred = get_action(model,
                                                 states[current_player],
                                                 actions[current_player],
                                                 rewards[current_player],
                                                 target_return[current_player],
                                                 timesteps)
                        actions_pred_eval[current_player][-1] = action_pred

                        actions[current_player][-1] = Tensor([0] * 12)

                        # allow to play a card after Skat putting
                        if t == 1:
                            state[3] = 1

                        # cur_state = torch.cat([cur_state, state.reshape(1, state_dim)])
                        cur_state[current_player] = state

                else:
                    # start at the first seat, at the player besides the dealer
                    # if (t == 2) and (i == 0):
                    #     current_player = env.game.get_first_seat().id

                    current_player = env.game.trick.get_current_player()

                    # update the state of the current player to get the open cards
                    states[current_player, -1] = env.update_state(states[current_player, -1])

                    if current_player == human_player_id:
                        # define input action
                        if t == 2:
                            print(f"\nYou play as player {human_player_id}. "
                                  f"\nYou are {env.game.players[human_player_id].get_role()}"
                                  f"\n{env.game.game_variant.get_variant_name()} "
                                  f"is played by player {env.game.get_declarer().get_id()}")

                        print(f"\nYour cards are {env.game.players[human_player_id].cards}")
                        print(f"The open cards are {env.game.trick.get_open_cards()}")

                        # to see what model would have chosen
                        action_pred = get_action(model,
                                                 states[human_player_id],
                                                 actions[human_player_id],
                                                 rewards[human_player_id],
                                                 target_return[human_player_id],
                                                 timesteps)

                        legal_indices = torch.nonzero(action_pred, as_tuple=True)
                        legal_indices = legal_indices[0].detach().numpy()

                        card_index = -1
                        card_index = input(
                            f"Which card do you want to select?"
                            f"\nPossible indices are {legal_indices}")
                        try:
                            card_index = int(card_index)
                        except ValueError:
                            print("Invalid input. Please enter a valid number.")

                        while not (card_index in legal_indices):
                            # predicting the action to take
                            card_index = input(
                                f"Invalid index (possible indices are {legal_indices}). Try again")

                        action = [0] * 12
                        action[card_index] = 1
                    else:
                        # predicting the action to take
                        action_pred = get_action(model,
                                                 states[current_player],
                                                 actions[current_player],
                                                 rewards[current_player],
                                                 target_return[current_player],
                                                 timesteps)

                        actions_pred_eval[current_player][-1] = action_pred

                        action = action_pred.detach().numpy()

                        # action_pred are already soft-maxed and valid actions,
                        # we only need to prevent the argmax from choosing an all 0 action
                        if torch.sum(action_pred) != 0:
                            # get the index of the card with the highest probability
                            card_index = np.argmax(action)

                            # only select the best card
                            action[:] = 0
                            action[card_index] = 1

                            print(f"Player {current_player} plays {env.game.players[current_player].cards[card_index]}")
                        else:
                            card_index = -1
                            print(f"Player {current_player} does not play a card.")

                    actions[current_player][-1] = Tensor(action)

                    acting_player = current_player

                    # interact with the environment based on this action
                    # reward_player is a tuple of the trick winner and reward to propagate it
                    # current_player is used to correctly iterate over the states in the order of play
                    state, reward_player, done, current_player = env.online_step(card_index, current_player)

                    # cur_state = torch.cat([cur_state, torch.from_numpy(state).reshape(1, state_dim)])
                    cur_state[acting_player] = state

                # states[current_player] = torch.cat([states[current_player], cur_state], dim=0)
                rewards[reward_player[0]][-1] = reward_player[1]

                pred_return = target_return[reward_player[0], -1] - (rewards[reward_player[0]][-1] / scale)
                cur_pred_return = torch.cat([cur_pred_return, pred_return.reshape(1, 1)])

            if t >= 2:
                print(f"Trick {t+1} was won by {env.game.trick.leader.id}. "
                      f"Cards were {env.game.trick.leader.trick_stack[t-1]}")

            # update all states after each trick for the hand cards, score and last_trick
            target_return = torch.cat([target_return, cur_pred_return], dim=1)
            cur_state = torch.from_numpy(cur_state).reshape(3, 1, state_dim)
            states = torch.cat([states, cur_state], dim=1)
            timesteps = torch.cat([timesteps, torch.ones((1, 1)).long() * (t + 1)], dim=1)

        print(f"\nFinished game {j}\n"
              f"\nYour final card points are: {env.game.players[human_player_id].sum_trick_values()}"
              f"\nYour game points {rewards[human_player_id][-1]}")
        
        # add to the scores of the players
        list_points[human_player_start] += rewards[(human_player_id - j) % 3][-1]
        list_points[(human_player_start + 1) % 3] += rewards[(human_player_id - j + 1) % 3][-1]
        list_points[(human_player_start + 2) % 3] += rewards[(human_player_id - j + 2) % 3][-1]
        
    return model


def file_check(rel_path: str):
    # check whether the pre-trained model file exists
    from os.path import exists

    rel_path = f"./pretrained_models/{rel_path}"
    if not exists(rel_path):
        raise argparse.ArgumentTypeError("Model file does not exist")
    else:
        return rel_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Skat_Decision_Transformer_trainer',
                                     description='Trains a Decision Transformer on a championship dataset using '
                                                 'Huggingface´s Decision Transformer.'
                                                 'A GPT2 model uses causal self-attention for training.'
                                                 'After training it autoregressively predicts the next action.',
                                     epilog='For more information, we refer to the underlying paper '
                                            '"Decision Transformer: Reinforcement Learning via Sequence Modeling" and'
                                            'their implementation '
                                            'https://github.com/kzl/decision-transformer/tree/master, as well as'
                                            ' Huggingface')
    parser.add_argument('--championship', '-cs', type=str, default='wc', choices=['wc'],
                        help='dataset of championship to select from. '
                             'Currently, only the world championship is available, as data of gc, gtc, bl and rc is'
                             ' contradictory.')
    parser.add_argument('--games', type=int, default=(0, 10), nargs="+",
                        help='The games to load. Note that if this value surpasses 10 000, the game ids are ignored '
                             'and the games are randomly loaded from a dataset.')
    parser.add_argument('--perspective', type=int, default=(0, 1, 2), nargs="+",
                        help='Get the game from the amount of perspectives.'
                             'Note that the dataset will be split into test and train,',
                        choices=[(0, 1, 2), (0, 1), 0])  # choices=[(0, 1, 2), (0, 1), (0, 2), (1, 2), 1, 2, 3])
    parser.add_argument('--point_rewards', type=bool, default=True,
                        help='whether to add points of the game to the card points as a reward')
    parser.add_argument('--hand_encoding', '-enc', type=str, default='one-hot',
                        choices=['mixed', 'mixed_comp', 'one-hot', 'one-hot_comp'],
                        help='The encoding of cards in the state.')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_layer', type=int, default=2, help='number of hidden layers')
    parser.add_argument('--n_head', type=int, default=1, help='number of attention heads')
    parser.add_argument('--activation_function', '-act_fct', type=str, default='tanh',
                        help='the activation function to use in the GPT2Model')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=240)
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--save_model', type=bool, default=False,
                        help='Whether to save the model. '
                             'If true, the trained model will be saved to "pretrained_models"')
    parser.add_argument('--logging_steps', type=int, default=20)
    parser.add_argument('--pretrained_model', type=file_check,
                        default="games_0-20000-encoding_one-hot-point_rewards_True-Tue_Sep__5_23-18-22_2023",
                        help="Takes relative path as argument for the pretrained model which should be used. "
                             "The model has to be stored in the folder 'pretrained_models'.")
    parser.add_argument('--eval_in_training', type=bool, default=True,
                        help="Whether to evaluate during training. Slows down training if activated."
                             "Evaluation takes place on the test portion of the dataset (#_games_to_load * 0.2).")
    parser.add_argument('--online_eval', type=bool, default=False,
                        help="Uses a pre-trained model to play online against itself."
                             " Rules out further training and evaluation of model.")
    parser.add_argument('--play_as', type=int, default=None, choices=[0, 1, 2],
                        help='If you want to play against the AI, provide your position. Player 0 is starting.'
                             'Play with two pre-trained agents. Rules out training and evaluation of model.')

    args = vars(parser.parse_args())

    if not args['play_as'] is None:
        play_with_two(args)
    elif args['online_eval']:
        run_online_eval(args)
    else:
        run_training(args)
