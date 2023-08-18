import random
from dataclasses import dataclass
from typing import Dict, Union, Tuple

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_metric
from torch import nn
from torch import Tensor

from transformers import DecisionTransformerModel, TrainingArguments, Trainer, DecisionTransformerConfig, \
    TrainerCallback
from transformers.integrations import TensorBoardCallback
from transformers.modeling_utils import unwrap_model
from transformers.models.decision_transformer.modeling_decision_transformer import DecisionTransformerOutput

import data_pipeline
import environment

# %%
# gameID 4: index 5
game_index = 4

games_interval = (game_index, game_index + 1)  # (0, 200)  # (0, 128)

print("Loading data...")
data, one_game = data_pipeline.get_states_actions_rewards(amount_games=games_interval[1],
                                                          point_rewards=False,
                                                          game_index=game_index)

# %%
dataset = DatasetDict({"train": Dataset.from_dict(data)})

# Done: 0 actions when not putting Skat should be changed due to malicious behaviour
# Done, but doesnt fix problem

# Done: evaluate one hot encoding with same outcome


# %%

# FIXED: problem of context length max = 1024: see Solution 3

# atm of Problem (each Skat as action, card_dim 5, cards on hand): episode length = 105 * 12 = 1260

# problem at action 9 (10th action) (10 * 105 = 1050) (should select 3rd card, selects 1st):
# tensor([0.2189, 0.1059, 0.1479, 0.0586, 0.0595, 0.0583, 0.0585, 0.0587, 0.0584, 0.0586, 0.0583, 0.0586])

# Solution 1:
# Skat as first last trick
#   hand_cards -= 2, 2 s, a, r less
#   state_dim = -> 82
#   episode length = (82 + 12 + 1) * 10 = 950 v

#   But what if we want to encode the Skat as two separate actions?

# Solution 2:
# compress card representation with sorting by suit and further identifying by face
# example: [0, 1, 0, 0, 1, 5, 7, 8, 0, 0, 0, 0]
#   -> spades 7, K, A, J, missing: 8, 9, Q, 10
#   size per colour with padding -> 4 + 8 = 12
#   size of whole hand 12 * 4 = 48
#   state size = 3 + 4 + 3 * 5 + 2 * 5 + 48 = 80
#   episode: (s + a + r) * 12 = (80 + 12 + 1) * 12 = 1116
#   episode with Skat as first last trick: 93 * 10 = 930 v

# Solution 3: currently used
# Solution 2 + further compressing: do not pad the suits, only pad up to the max possible state length
#   max_hand_length = 16 (encoding of colours) + 12 (all cards) = 28
# pad with zeros
# Example 1:
# [1, 0, 0, 0, 1, 5, 7, 8], [0, 1, 0, 0, 1, 5],
# [0, 0, 1, 0, 2, 3, 4], [0, 0, 0, 1, 1, 3, 8]
# Example 2:
# [1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8], [0] * (max_hand_length - 12)
#
#   state size = 3 + 4 + 3 * 5 + 2 * 5 + 28 = 60
#   episode length: (s + a + r) * 12 = (60 + 12 + 1) * 12 = 876 v

# Solution 4 (respects problem with loss function):
# Solution 3, but solely with one-hot encoded cards
#   hand_length = 4 * 12 = 48
# Example 1:
# [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1], [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
# [0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1]
#
#   state size = 3 + 4 + 3 * 12 + 2 * 12 + 48 = 115
#   episode length: (s + a + r) * 12 = (115 + 12 + 1) * 12 = 1 536  x

act_dim = 10
card_dim = 5

# for padding of hand_cards, it is the maximum size with a compressed card representation
max_hand_len = 16 + act_dim  # 12 * card_dim  # 28

# position co-player (3) + trump (4) + last trick (3 * card_dim)
# + open cards (2 * card_dim) + hand cards (12 * card_dim)
state_dim = 3 + 2 + 4 + 3 * card_dim + 2 * card_dim + max_hand_len  # 12 * card_dim


# device = torch.device("cuda") # "cpu"


# adapted from https://huggingface.co/blog/train-decision-transformers
@dataclass
class DecisionTransformerSkatDataCollator:
    return_tensors: str = "pt"  # pytorch
    max_len: int = 10  # subsets of the episode we use for training, our episode length is short
    state_dim: int = state_dim  # size of state space
    act_dim: int = act_dim  # size of action space
    max_ep_len: int = 10  # max episode length in the dataset
    scale: float = 1.0  # normalization of rewards/returns
    state_mean: np.array = None  # to store state means
    state_std: np.array = None  # to store state stds
    p_sample: np.array = None  # a distribution to take account trajectory lengths
    n_traj: int = 0  # to store the number of trajectories in the dataset TODO: do we need this?
    games_ind: tuple = (0, 0)

    def __init__(self, dataset, games_ind) -> None:
        self.act_dim = len(dataset[0]["actions"][0])
        self.state_dim = len(dataset[0]["states"][0])
        self.dataset = dataset
        self.games_ind = games_ind
        # calculate dataset stats for normalization of states
        states = []
        traj_lens = []
        for obs in dataset["states"]:
            states.extend(obs)
            traj_lens.append(len(obs))
        self.n_traj = len(traj_lens)
        states = np.vstack(states)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

        traj_lens = np.array(traj_lens)
        self.p_sample = traj_lens / sum(traj_lens)

    def get_batch_ind(self):
        # picks game indices
        # this picks the same game over *times* times
        # (WC GameID 4: Agent sits in rear hand as soloist)
        times = 32
        return np.tile(np.arange(self.games_ind[0], self.games_ind[1]), times)

    def _discount_cumsum(self, x, gamma):
        # weighted rewards are in the data set (get_states_actions_rewards)
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum

    def __call__(self, features):
        batch_size = 32  # len(features)

        batch_inds = self.get_batch_ind()  # self.state_dim * self.max_ep_len

        # this is a bit of a hack to be able to sample of a non-uniform distribution
        # we have a random pick of the data as a batch without controlling the shape,
        # batch_inds = np.random.choice(
        #     np.arange(self.n_traj),
        #     size=batch_size,
        #     replace=True,
        #     p=self.p_sample,  # reweights, so we sample according to timesteps
        # )

        # a batch of dataset features
        s, a, r, rtg, timesteps, mask, big_action_mask = [], [], [], [], [], [], []

        for ind in batch_inds:
            # for feature in features:

            feature = self.dataset[int(ind)]

            # why do we need a randint?
            # to be able to jump into one game -> predict from every position and improve training
            # TODO: jumping randomly into a surrendered game does not work well
            si = 0  # random.randint(0, len(feature["rewards"]) - 1)  # 0

            # get sequences from dataset
            s.append(np.array(feature["states"]
                              [si:self.max_len]).reshape((1, -1, self.state_dim)))
            a.append(np.array(feature["actions"][si:self.max_len]).reshape((1, -1, self.act_dim)))
            r.append(np.array(feature["rewards"][si:self.max_len]).reshape((1, -1, 1)))

            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= self.max_ep_len] = self.max_ep_len - 1  # padding cutoff
            rtg.append(
                self._discount_cumsum(np.array(feature["rewards"][si:]), gamma=0.99)[: s[-1].shape[1]
                ].reshape(1, -1, 1)  # TODO check the +1 removed here
            )
            if rtg[-1].shape[1] < s[-1].shape[1]:
                print("if true")
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]  # ind % 12

            # states, actions, rewards are already padded

            padding = np.zeros((1, self.max_len - tlen, self.state_dim))
            s[-1] = np.concatenate([padding, s[-1]], axis=1)

            # state normalization
            # s[-1] = (s[-1] - self.state_mean) / self.state_std

            a[-1] = np.concatenate(
                [np.zeros((1, self.max_len - tlen, self.act_dim)), a[-1]],
                axis=1,
            )
            r[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), r[-1]], axis=1)

            rtg[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), rtg[-1]], axis=1)  # / self.scale
            timesteps[-1] = np.concatenate([np.zeros((1, self.max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, self.max_len - tlen)), np.ones((1, tlen))], axis=1))
            # big_action_mask = [np.concatenate([np.ones((1, 10 - t)), np.zeros((1, t))], axis=1) for t in range(0, 10)]

        s = torch.from_numpy(np.concatenate(s, axis=0)).float()
        a = torch.from_numpy(np.concatenate(a, axis=0)).float()
        r = torch.from_numpy(np.concatenate(r, axis=0)).float()
        # d = torch.from_numpy(np.concatenate(d, axis=0))
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).float()
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).long()
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).float()
        # torch.from_numpy(np.concatenate(big_action_mask, axis=0)).float()  #

        return {
            "states": s,
            "actions": a,
            "rewards": r,
            "returns_to_go": rtg,
            "timesteps": timesteps,
            "attention_mask": mask,
        }


class TrainableDT(DecisionTransformerModel):
    def __init__(self, config):
        super().__init__(config)

    def originalForward(
            self,
            states=None,
            actions=None,
            rewards=None,
            returns_to_go=None,
            timesteps=None,
            attention_mask=None,
            output_hidden_states=None,
            output_attentions=None,
            return_dict=None,
            action_mask=None
    ) -> Union[Tuple, DecisionTransformerOutput]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # action_embeddings = action_embeddings * action_mask

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = (
            torch.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_length, self.hidden_size)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

        # TODO: change action attention mask action_mask
        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 3 * seq_length)
        )
        device = stacked_inputs.device
        # we feed in the input embeddings (not word indices as in NLP) to the model
        encoder_outputs = self.encoder(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            position_ids=torch.zeros(stacked_attention_mask.shape, device=device, dtype=torch.long),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        x = encoder_outputs[0]

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_return(x[:, 2])  # predict next return given state and action
        state_preds = self.predict_state(x[:, 2])  # predict next state given state and action
        action_preds = self.predict_action(x[:, 1])  # predict next action given state
        if not return_dict:
            return (state_preds, action_preds, return_preds)

        return DecisionTransformerOutput(
            last_hidden_state=encoder_outputs.last_hidden_state,
            state_preds=state_preds,
            action_preds=action_preds,
            return_preds=return_preds,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def forward(self, **kwargs):
        action_mask = [np.concatenate([np.ones((1, 10 - t)), np.zeros((1, t))], axis=1) for t in range(0, 10)]
        action_mask = torch.from_numpy(np.concatenate(action_mask, axis=0)).float()

        output = self.originalForward(**kwargs, action_mask=action_mask)  # super().forward(**kwargs)
        # add the DT loss
        action_preds = output[1]
        action_targets = kwargs["actions"]
        attention_mask = kwargs["attention_mask"]
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_targets = action_targets.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        soft_max = nn.Softmax(dim=1)

        action_preds_sm = action_preds  # soft_max(action_preds)

        action_preds_sm = action_preds_sm  # * action_targets

        debug_sm = soft_max(action_preds)[:10,:] * action_targets[:10,:]

        # cross entropy loss
        cross_ent_fct = nn.CrossEntropyLoss()  # reduction='sum')

        cross_ent_loss = cross_ent_fct(action_preds_sm, action_targets)

        loss = cross_ent_loss

        # forbidden_action_mask = [np.concatenate([np.zeros((1, 10-t)), np.ones((1, t))], axis=1)
        #                          for t in range(10 - action_preds_sm.shape[0], 10)]
        # forbidden_action_mask = torch.from_numpy(np.concatenate(forbidden_action_mask, axis=0)).float()
        #
        # mask2 = torch.from_numpy(np.zeros((1, action_preds_sm.shape[0] * action_preds_sm.shape[1])).reshape(action_preds_sm.shape[0], action_preds_sm.shape[1]))
        #
        # action_preds_forb = action_preds_sm * forbidden_action_mask

        # loss += sum(sum(abs(action_preds_forb))) * 0.1

        # state (experimental)
        state_preds = output[0]
        state_targets = kwargs["states"]
        attention_mask = kwargs["attention_mask"]
        state_dim = state_preds.shape[2]
        state_preds = state_preds.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0]
        state_targets = state_targets.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0]

        soft_max = nn.Softmax(dim=1)

        state_preds_sm = soft_max(state_preds)

        # cross entropy loss
        cross_ent_fct = nn.CrossEntropyLoss()

        state_preds_sm = state_preds_sm * state_targets

        state_loss = cross_ent_fct(state_preds_sm, state_targets)

        # reward (experimental)
        reward_preds = output[2]
        reward_targets = kwargs["rewards"]
        attention_mask = kwargs["attention_mask"]
        reward_dim = reward_preds.shape[2]
        reward_preds = reward_preds.reshape(-1, reward_dim)[attention_mask.reshape(-1) > 0]
        reward_targets = reward_targets.reshape(-1, reward_dim)[attention_mask.reshape(-1) > 0]

        mse_fct = nn.MSELoss()

        reward_loss = mse_fct(reward_preds, reward_targets)

        # soft_max_fct = nn.Softmax(1)  # TODO: softmax
        # soft_maxed = soft_max_fct(action_preds)

        # not possible due to parallel processing
        # # differs between 12 and 10 hand cards for defenders or hand games
        # hand_len = 12 if any(action_targets[0, 0, :]) else 10
        #
        # prob_each_act_correct = action_pred_sm * correct_actions
        #
        # prob_correct_action = sum(sum(prob_each_act_correct[12 - hand_len:])) / hand_len

        # only predicts the prob in context of 12 cards
        soft_max = nn.Softmax(dim=1)
        action_pred_sm = soft_max(action_preds)

        prob_each_act_correct = action_pred_sm * action_targets

        prob_correct_action = sum(sum(prob_each_act_correct)) / action_preds.shape[0]

        # print(f"probability of correct action: {prob_correct_action}")

        # problem of double logging. _maybe_log_save_evaluate logs already, but does not get passed additional metrics
        # logs: Dict[str, float] = {}
        # logs =
        # self.log()

        # manual logging
        # writer.add_scalar("Loss/train", loss,)  # problem: get episode

        return {"loss": loss, "probability of correct action": prob_correct_action}  # loss, prob_correct_action

    def original_forward(self, **kwargs):
        return super().forward(**kwargs)


# class CustomTBCallback(TensorBoardCallback):
#     def on_log(self, args, state, control, logs=None, **kwargs):

# class CustomDTTrainer(Trainer):
#
#     def compute_loss(self, model, inputs, return_outputs=False):
#         loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
#
#         if self._globalstep_last_logged == self.state.global_step:
#             logs: Dict[str, float] = {}
#
#             # # all_gather + mean() to get average loss over all processes
#             # tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
#             # # reset tr_loss to zero
#             # tr_loss -= tr_loss
#             for entry in outputs:
#                 logs.update({entry: round(float(outputs[entry].detach()), 4)})
#
#             # logs["loss"] = round(loss / (self.state.global_step - self._globalstep_last_logged), 4),
#             logs["learning_rate"] = self._get_learning_rate()
#
#             self.log(logs)
#
#         return loss  # if return_outputs else loss


# class LogCallback(TrainerCallback):
#     def on_evaluate(self, args, state, control, **kwargs):
#         # calculate loss here

# class OwnTensorboardCallback(TensorBoardCallback):
# class OwnCallback(TensorBoardCallback):
#     def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
#         if state.is_local_process_zero:
#             print(logs)

def compute_metrics(eval_pred):
    metric1 = load_metric("precision")
    metric2 = load_metric("recall")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision = metric1.compute(predictions=predictions, references=labels)["precision"]
    recall = metric2.compute(predictions=predictions, references=labels)["recall"]
    return {"precision": precision, "recall": recall}


collator = DecisionTransformerSkatDataCollator(dataset["train"], games_ind=games_interval)

config = DecisionTransformerConfig(
    state_dim=state_dim,
    act_dim=act_dim,
    action_tanh=False,  # do not apply the tanh fct on the output action
    activation_function="tanh",
    # n_head=2,
    # n_layer=2,
    max_ep_len=10,  # each episode is a game -> 12 tuples of s,a,r make up 1 game
    # vocab_size=1200,  # there are 32 cards + pos_tp + score +  + trump_enc
    n_positions=1024,
    scale_attn_weights=True,
    embd_pdrop=0.1,
    resid_pdrop=0.1
)
# TODO: how is the vocabulary defined?

model = TrainableDT(config)

# model.to(device)

# logging_files_name = "dt_training_{}_{}.log"
label_names = ["states", "actions", "rewards", "returns_to_go", "timesteps", "attention_mask"]

training_args = TrainingArguments(
    report_to=["tensorboard"],
    output_dir="training_output/",
    remove_unused_columns=False,
    num_train_epochs=250,
    per_device_train_batch_size=64,
    learning_rate=1e-3,
    weight_decay=1e-4,
    warmup_ratio=0.1,
    optim="adamw_torch",
    max_grad_norm=0.1,
    logging_steps=10,
    logging_dir="./training-logs",

    # do_eval=True,
    # evaluation_strategy="steps",
    # eval_steps=50,

    # no_cuda=True,
    # label_names=label_names
)

trainer = Trainer(  # CustomDTTrainer
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["train"],  # TODO: anderes Spiel
    data_collator=collator,
    compute_metrics=compute_metrics,
    # callbacks=[tensorboard_callback]
)

# writer = SummaryWriter()

print("Training...")
trainer.train()

# for tensorboard visualization:
# 1. rm -r ./training-logs/*
# 2. run training
# 3. tensorboard --logdir=./training-logs


# # Training loop with episode tracking
# current_episode = 0
# for epoch in range(round(training_args.num_train_epochs)):
#     for batch in trainer.get_train_dataloader():
#         # Increment episode number
#         current_episode += 1
#
#         # Perform forward and backward passes
#         loss = trainer.training_step(model, batch)
#
#         trainer.optimizer.step()
#         trainer.lr_scheduler.step()
#
#         writer.add_scalar("Loss/episode", loss, epoch)
#
#         # Log the episode number and loss
#         trainer.log_metrics({"episode": current_episode, "loss": loss.item()})
#
#     # Save the model at the end of each epoch
#     trainer.save_model()
#
# # Save the final model
# trainer.save_model()

# writer.flush()

# %%

# select available cudas for faster matrix computation
# device = torch.device("cuda")

model = model.to("cpu")

# trainer.evaluate(dataset["train"])

model.eval()


# env = environment.Env()


# Function that gets an action from the model using autoregressive prediction
# with a window of the previous 20 timesteps.
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
    states = torch.cat([torch.zeros((1, padding, state_dim)), states], dim=1).float()
    actions = torch.cat([torch.zeros((1, padding, act_dim)), actions], dim=1).float()
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
        return_dict=False, )
    return action_preds[0, -1]


MAX_EPISODE_LENGTH = 10
scale = 1

# state_mean = np.array()
# state_std = np.array()
#
# state_mean = torch.from_numpy(state_mean)
# state_std = torch.from_numpy(state_std)

state_mean = collator.state_mean.astype(np.float32)
state_std = collator.state_std.astype(np.float32)

state_mean = torch.from_numpy(state_mean).to(device="cpu")
state_std = torch.from_numpy(state_std).to(device="cpu")

# env = environment.Env() if one_game is None else one_game
env = environment.Env()

TARGET_RETURN = 2 * 120  # 102

for eval_game in range(1):  # dataset["test"]:

    # TODO: scale up
    # we need the others

    # build the environment for the evaluation
    state = env.reset(current_player_id=(game_index % 3), game_env=one_game)
    # game_states=dataset['train'][8]['states'])

    target_return = torch.tensor(TARGET_RETURN).float().reshape(1, 1)
    states = torch.from_numpy(state).reshape(1, state_dim).float()
    actions = torch.zeros((0, act_dim)).float()
    rewards = torch.zeros(0).float()
    timesteps = torch.tensor(0).reshape(1, 1).long()

    # take steps in the environment (evaluation, not training)
    for t in range(MAX_EPISODE_LENGTH):
        # add zeros for actions as input for the current time-step
        actions = torch.cat([actions, torch.zeros((1, act_dim))], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1)])

        # predicting the action to take
        action_pred = get_action(model,
                            states,  # - state_mean) / state_std,
                            actions,
                            rewards,
                            target_return,
                            timesteps)

        soft_max = nn.Softmax(dim=0)
        action_pred = soft_max(action_pred)

        print(f"Action {t}: {action_pred}")

        action = action_pred.detach().numpy()

        # hand cards within the state are padded from right to left after each action
        # mask the action
        # action[-t:] = 0

        valid_actions = action[:MAX_EPISODE_LENGTH - t]

        # get the index of the card with the highest probability
        card_index = np.argmax(valid_actions)

        # only select the best card
        action[:] = 0
        action[card_index] = 1

        actions[-1] = Tensor(action)

        # interact with the environment based on this action
        state, reward, done = env.step(tuple(action))

        print(f"Reward {t}: {reward}")

        cur_state = torch.from_numpy(state).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        pred_return = target_return[0, -1] - (reward / scale)
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat([timesteps, torch.ones((1, 1)).long() * (t + 1)], dim=1)

        if done:
            # for evaluation of unspecific games
            diff_target_reached = TARGET_RETURN - sum(rewards)

            print(f"Difference of target reward and reached reward: {diff_target_reached}")

            # for direct evaluation on specific known games
            actions_pred = actions.detach().numpy()

            actions_correct = dataset["train"]["actions"][game_index]

            hand_len = 10  # 12 if any(actions_correct[0]) else 10

            prob_each_act_correct = actions_pred * actions_correct

            prob_correct_action = sum(sum(prob_each_act_correct)) / hand_len  # [12 - hand_len:]

            print(f"Avg probability of correct action: {prob_correct_action}")
            break
