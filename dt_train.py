"""
This Implementation is based on the Decision Transformer https://github.com/kzl/decision-transformer/tree/master.

To achieve a speed-up and more modularity, the Huggingface (HF) implementation of the Decision Transformer is adapted.
See https://huggingface.co/blog/train-decision-transformers for more information.
"""

import argparse
import math
import random
# import resource
import time
from dataclasses import dataclass
from typing import Dict, Union, Optional, List, Any, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from datasets import Dataset, DatasetDict, load_from_disk
from torch import nn
from torch import Tensor
from sklearn.model_selection import train_test_split

from transformers import DecisionTransformerModel, TrainingArguments, Trainer, DecisionTransformerConfig

from transformers.trainer_utils import speed_metrics, IntervalStrategy
from transformers.utils import ModelOutput

import data_pipeline
from environment import ACT_DIM, get_dims_in_enc, Env

# %%
# def set_memory_limit(limit_gb):
#     limit_bytes = limit_gb * 1024 * 1024 * 1024  # Convert MB to bytes
#     resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
#
#
# set_memory_limit(64)

MAX_EPISODE_LENGTH = 12


@dataclass
class DecisionTransformerSkatDataCollator:
    return_tensors: str = "pt"  # pytorch
    max_len: int = MAX_EPISODE_LENGTH  # subsets of the episode we use for training, our episode length is short
    state_dim: int = 0  # size of state space
    act_dim: int = ACT_DIM  # size of action space
    max_ep_len: int = MAX_EPISODE_LENGTH  # max episode length in the dataset
    scale: float = 1.0  # normalization of rewards/returns
    state_mean: np.array = None  # to store state means
    state_std: np.array = None  # to store state stds
    p_sample: np.array = None  # a distribution to take account trajectory lengths
    n_traj: int = 0  # to store the number of trajectories in the dataset
    games_ind: tuple = (0, 0)

    def __init__(self, dataset, batch_size=32) -> None:
        self.act_dim = len(dataset[0]["actions"][0])
        self.state_dim = len(dataset[0]["states"][0])
        self.dataset = dataset
        self.batch_size = batch_size
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

    def _discount_cumsum(self, x, gamma):
        # weighted rewards are in the data set (get_states_actions_rewards)
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(discount_cumsum.shape[0] - 1)):
            # gamma as a discount factor to differ rewards temporarily
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum

    def __call__(self, features):
        # batch_inds = self.get_batch_ind()

        # features are already batched, but
        # we always want to put out batches of batch_size, len(features) can be < batch_size, then either we have issues
        # in forward or if we drop smaller batches, we can not train under 32 games (for overfitting)

        # this is a bit of a hack to be able to sample of a non-uniform distribution
        # we have a random pick of the data as a batch without controlling the shape
        batch_inds = np.random.choice(
            np.arange(len(features)),
            size=self.batch_size,
            replace=True,
            p=[1/len(features)]*len(features)
        )

        # a batch of dataset features
        s, a, r, rtg, timesteps, mask, big_action_mask = [], [], [], [], [], [], []

        for ind in batch_inds:

            feature = features[int(ind)]
            # feature = self.dataset[int(ind)]

            # why do we use a randint?
            # To be able to jump into one game -> predict from every position and improve training
            # jumping randomly into a surrendered game could not work well
            # ->  We encode whether card should be played in states (see data_pipeline)
            si = random.randint(1, len(feature["rewards"]))

            #  fixed frame of 12 timesteps
            #  we want to have a history starting from the beginning of the game
            #  and include the RTGs from the last reward

            #  --> to which extent is the attention mask implemented?
            # attention_mask is defined over time steps and the order in which the data is ordered in here

            # get sequences from dataset
            s.append(np.array(feature["states"]
                              [0:si]).reshape((1, -1, self.state_dim)))
            a.append(np.array(feature["actions"][0:si]).reshape((1, -1, self.act_dim)))
            r.append(np.array(feature["rewards"][0:si]).reshape((1, -1, 1)))

            timesteps.append(np.arange(0, s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= self.max_ep_len] = self.max_ep_len - 1  # padding cutoff
            rtg.append(
                self._discount_cumsum(np.array(feature["rewards"]), gamma=0.99)[: s[-1].shape[1]].reshape(1, -1, 1)
                # TODO check the +1 removed here
            )
            if rtg[-1].shape[1] < s[-1].shape[1]:
                print("if true")
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # trajectory length
            tlen = s[-1].shape[1]

            padding = np.zeros((1, self.max_len - tlen, self.state_dim))
            s[-1] = np.concatenate([padding, s[-1]], axis=1)

            a[-1] = np.concatenate(
                [np.zeros((1, self.max_len - tlen, self.act_dim)), a[-1]],
                axis=1,
            )
            r[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), r[-1]], axis=1)

            rtg[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), rtg[-1]], axis=1)  # / self.scale
            timesteps[-1] = np.concatenate([np.zeros((1, self.max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, self.max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).float()
        a = torch.from_numpy(np.concatenate(a, axis=0)).float()
        r = torch.from_numpy(np.concatenate(r, axis=0)).float()
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).float()
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).long()
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).float()

        return {
            "states": s,
            "actions": a,
            "rewards": r,
            "returns_to_go": rtg,
            "timesteps": timesteps,
            "attention_mask": mask,
        }


@dataclass
class CustomDecisionTransformerOutput(ModelOutput):
    """
    Adapted from HF's

    Custom class for model's outputs that also contains a pooling of the last hidden states and a transfer of the loss

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        state_preds (`torch.FloatTensor` of shape `(batch_size, sequence_length, state_dim)`):
            Environment state predictions
        action_preds (`torch.FloatTensor` of shape `(batch_size, sequence_length, action_dim)`):
            Model action predictions
        return_preds (`torch.FloatTensor` of shape `(batch_size, sequence_length, 1)`):
            Predicted returns for each state
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        loss
    """

    state_preds: torch.FloatTensor = None
    action_preds: torch.FloatTensor = None
    return_preds: torch.FloatTensor = None
    hidden_states: torch.FloatTensor = None
    attentions: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    loss: torch.FloatTensor = None


class TrainableDT(DecisionTransformerModel):
    def __init__(self, config):
        super().__init__(config)

    def original_forward(
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
    ) -> Union[Tuple, CustomDecisionTransformerOutput]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if it can be attended to, 0 if not
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

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = (
            torch.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_length, self.hidden_size)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

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

        # induces wrong behaviour for the prediction
        put_card_mask = states[:, :, 3].bool()
        action_preds = action_preds * put_card_mask.unsqueeze(2).repeat(1, 1, 12)

        sm = torch.nn.Softmax(dim=2)
        action_preds = sm(action_preds)

        action_targets_masked = actions.reshape(-1, ACT_DIM)[attention_mask.reshape(-1) > 0]
        action_preds_masked = action_preds.reshape(-1, ACT_DIM)[attention_mask.reshape(-1) > 0]
        # action_preds_l = action_preds.reshape(-1, ACT_DIM)[attention_mask.reshape(-1) > 0]

        # cross_ent_loss = nll_loss_fct(torch.log(action_preds_l), torch.argmax(action_targets, dim=1))

        # when argmax takes 0 actions it selects the first element as maximum -> there is no 1 -> inf loss
        # --> as the loss only looks at the ground truth and we mask 0 actions, we throw out the 0 actions for the loss
        mask_wo_zeros = action_targets_masked.sum(dim=1)
        mask_wo_zeros = mask_wo_zeros > 0

        targets_wo_zeros = action_targets_masked[mask_wo_zeros]
        preds_wo_zeros = action_preds_masked[mask_wo_zeros]

        nll_loss_fct = torch.nn.NLLLoss()
        loss_pure = nll_loss_fct(torch.log(preds_wo_zeros), torch.argmax(targets_wo_zeros, dim=1))
        loss = loss_pure

        # In the following, limiting behaviour is injected to set the rules of Skat.
        # This can be seen as a safety mechanism.
        # Masks are used to only allow legal actions.
        # The rules are included by looking at first open card, played cards and colour (including trump_enc),
        # and the possible hand length (prevent out-of-bounds actions)

        # Reminder: We use a one-hot encoding with the resulting game state
        # game_state = position co-player (3)  + score (2) + trump (4) + last trick (36)
        # + open cards (24) + hand cards (12 * 12)

        # A lot of indices and masks are used for parallel processing,
        # to translate it into simpler terms, it does the following:
        # If (open suit is on hand) or (trump is played and are jacks on hand)?
        # If yes -> valid actions are all cards with same suit or jacks when trump is lying
        # If not -> valid actions are all cards on the hand (called "in bounds")

        # if states.shape[0] == batch_size:
        trump_enc = states[:, :, 6:10]

        open_cards = states[:, :, 46:70]
        colour_trick = open_cards[:, :, :4]
        # jack_lying = open_cards[:, :, 11]

        # If jack is first card, the trump_enc matters, not the suit of the jack
        jack_played_as_first = open_cards[:, :, 11].reshape(batch_size, 12, 1).bool().to(self.device)

        # If jack is played at first, trump suit is the one of trick, else the suit of the first card
        # (batch, trick, colour)
        colour_trick = jack_played_as_first * trump_enc + ~jack_played_as_first * colour_trick

        # If colour does not exist on hand, it is padded with 0s
        # -> we can find out if suit is on hand by looking in state at fixed indices
        hand_cards = states[:, :, -144:].reshape(batch_size, 12, 12, 12).to(self.device)
        # (batch, trick, hand in trick, cards) -> each card is reduced to its suit/colour
        colours_on_hands = hand_cards[:, :, :, :4]

        # How can we find out if suit of the trick is on the hand?
        # -> create a mask for each card suit at trick t
        # the hand (sequence of colours can have a variable length)
        # the 12 repeated entries are a mask over the cards in each trick
        # (batch, trick, suit) -> (batch, trick, suit_in_trick * 12, suit)
        trick_mask = colour_trick.unsqueeze(2).repeat(1, 1, 12, 1).to(self.device)

        # Finally, find out if suit is on hand...
        is_colour_on_hand = trick_mask * colours_on_hands

        # ...and if trump is played, a jack:
        jack_on_hand = hand_cards[:, :, :, 11].bool()
        trump_played = torch.any(colour_trick * trump_enc, dim=2).to(self.device)  # + jack_lying
        jack_playable = jack_on_hand * trump_played.reshape(batch_size, 12, 1).to(self.device)

        # Creates a mask: Does the lying suit exist on the hand?
        # If no colour is laying, mask is merged with actions in bounds
        is_colour_on_hand = torch.any(is_colour_on_hand, dim=3).to(self.device)

        # Jacks also have a suit which can be falsely recognised by is_colour_on_hand
        # -> get Jacks out of is_colour_on_hand:
        is_colour_on_hand = is_colour_on_hand * (~jack_on_hand)

        # Playable cards to add to the trick, not to start it
        playable_cards_to_add = is_colour_on_hand + jack_playable.bool()

        # If colour is not on hand: mask is all cards excluding actions not on hand (OOB)
        hand_card_length = torch.sub(12, timesteps).to(self.device)

        # Mask is all cards excluding OOB actions
        actions_in_bounds = torch.arange(12).repeat(batch_size, 12).reshape(batch_size, 12, 12) < hand_card_length.view(-1, 12, 1)

        merging_mask = (~playable_cards_to_add).all(dim=2).to(self.device)

        # If no card is lying, action_in_bounds is valid
        # If at least one card is lying, playable_cards_to_add acknowledges the suit
        playable_cards = playable_cards_to_add.clone().to(self.device)
        playable_cards[merging_mask] = actions_in_bounds[merging_mask]

        # From modeling_decision_transformer.py (HF):
        # Since playable_cards is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        # playable_cards = playable_cards.to(dtype=self.dtype)  # fp16 compatibility
        # playable_cards = (1.0 - playable_cards) * torch.finfo(self.dtype).min

        possible_action_preds = playable_cards * action_preds

        # mask all actions were no card can be played, 1 if it can be played, 0 if not
        # e.g. in Hand games, as defenders in first two tricks (or side effect: due to the attention_mask)
        put_card_mask = states[:, :, 3].bool().to(self.device)

        # # starting from here until the loss fct, the loss is calculated
        # comb_mask = put_card_mask * attention_mask
        #
        # # for loss calculation: cut out action when player cannot play card
        # # softmax guarantees to take one action -> only cutting the actions works
        # possible_action_preds_clean = possible_action_preds.clone()
        # possible_action_preds_clean = possible_action_preds_clean[comb_mask > 0]
        #
        # final_valid_action_idx = torch.nonzero(possible_action_preds_clean, as_tuple=True)
        #
        # valid_pred_actions = possible_action_preds_clean[final_valid_action_idx]
        #
        # possible_action_targets = actions[comb_mask > 0]
        # valid_target_actions = possible_action_targets[final_valid_action_idx]

        # softmax has to be before the cross entropy loss

        # apply mask whether a card should be played based on game timestep
        action_preds = possible_action_preds * put_card_mask.unsqueeze(2).repeat(1, 1, 12)

        # action_preds_m = action_preds[attention_mask > 0]
        # # get indices of non-zero elements, as they have been ruled out
        # action_preds_idx = torch.nonzero(action_preds_m, as_tuple=True)
        #
        # # we have the actions and preds with the non-zero elements
        # # we want an attn mask which only covers the non-zero elements
        #
        # actions_m = actions[attention_mask > 0]
        #
        # action_preds_m = action_preds_m[action_preds_idx]
        # actions_m = actions_m[action_preds_idx]
        #
        # nll_loss_fct = torch.nn.NLLLoss()
        # cross_ent_loss = nll_loss_fct(torch.log(action_preds_m), torch.argmax(actions_m, dim=0))

        # action_loss_mask = attention_mask[action_preds_idx]
        #
        # actions_attended_loss = actions_l[action_loss_mask > 0]
        #
        # action_preds_l = action_preds[action_preds_idx]
        # action_preds_attended_loss = action_preds_l[action_loss_mask > 0]

        # for evaluation and online training
        if not return_dict:
            return state_preds, action_preds, return_preds

        return CustomDecisionTransformerOutput(
            last_hidden_state=encoder_outputs.last_hidden_state,
            state_preds=state_preds,
            action_preds=action_preds,
            return_preds=return_preds,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            loss=loss
        )

    def forward(self, **kwargs):
        output = self.original_forward(**kwargs)

        action_preds = output[1]
        action_targets = kwargs['actions']
        attention_mask = kwargs['attention_mask']
        action_targets = action_targets.reshape(-1, ACT_DIM)[attention_mask.reshape(-1) > 0]
        action_preds = action_preds.reshape(-1, ACT_DIM)[attention_mask.reshape(-1) > 0]

        # Cross-entropy loss is already calculated in original forward
        loss = output[-1]

        # cross entropy loss
        # cross_ent_fct = nn.CrossEntropyLoss()
        # cross_ent_loss = cross_ent_fct(action_preds, action_targets)

        # only predicts the prob in context of 12 cards
        # soft_max = nn.Softmax(dim=1)
        # action_pred_sm = soft_max(action_preds)

        # softmax is already applied on preds
        action_pred_sm = action_preds

        prob_each_act_correct = torch.mul(action_pred_sm, action_targets)

        # exclude Skat putting for defenders and hand games
        # amount of actions - amount of actions that select a card
        # unnecessary when showing rules
        amount_no_skat_action = action_targets.shape[0] - torch.sum(action_targets)

        # the accumulated probability of the correct action / total number of actions taken in targets
        prob_correct_action = torch.sum(prob_each_act_correct) / (action_targets.shape[0] - amount_no_skat_action)

        # we want to know to what probability the model actually chooses an action != target_action,
        # not the accumulated prob of (actions != target_action)
        action_taken = torch.argmax(action_pred_sm, dim=1)

        action_mask = torch.zeros_like(action_targets)
        action_mask[torch.arange(action_targets.shape[0]), action_taken] = 1

        wrong_action_taken = torch.mul(action_mask, ~action_targets.bool())

        # absolute amount of wrong cards being chosen, statistically exclude defending games
        # (first two actions are always wrong)
        amount_wrong_actions_taken = torch.sum(wrong_action_taken) - amount_no_skat_action

        # rate of wrong cards being chosen
        rate_wrong_action_taken = amount_wrong_actions_taken / action_targets.shape[0]

        # calculate subset of illegal actions:
        # actions selecting out of hand cards.
        # Example:
        # In the third trick (without Skat putting), only 7 cards can be selected,
        # actions could try to falsely select a 9th card
        # amount_past_tricks = kwargs["timesteps"][:, -1]
        timesteps = kwargs["timesteps"]
        hand_card_length = torch.sub(12, timesteps.reshape(-1)[attention_mask.reshape(-1) > 0])

        actions_oob = torch.nonzero(torch.div(action_taken, hand_card_length, rounding_mode='trunc'))
        # predictions have to be passed as tensors
        rate_oob_actions = Tensor([actions_oob.shape[0] / action_targets.shape[0]])

        # prob_illegal_action =

        return {"loss": loss,
                "probability_of_correct_action": prob_correct_action,
                "rate_wrong_action_taken": rate_wrong_action_taken,
                "rate_oob_actions": rate_oob_actions
                }


class DTTrainer(Trainer):
    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        output = self.evaluation_loop(
            eval_dataloader,
            description="Evaluation",
            # We want the predictions, as the metrics are passed through them
            # not only the loss
            prediction_loss_only=None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        # if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
        #     start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]

        # injected behaviour
        if len(output.predictions[0].shape) != 0:
            output.metrics.update(
                {"eval_loss: ": round(float(output.predictions[0][-1]), 4),
                 "eval_prob_correct_action": round(float(output.predictions[1][-1]), 4),
                 "eval_rate_wrong_action_taken": round(float(output.predictions[2][-1]), 4),
                 "eval_rate_oob_actions": round(float(output.predictions[3][-1]), 4)
                 }
            )
        else:
            # edge case of one game
            output.metrics.update(
                {"eval_loss: ": round(float(output.predictions[0]), 4),
                 "eval_prob_correct_action": round(float(output.predictions[1]), 4),
                 "eval_rate_wrong_action_taken": round(float(output.predictions[2]), 4),
                 "eval_rate_oob_actions": round(float(output.predictions[3]), 4)
                 }
            )

        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        # if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
        #     # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
        #     xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        # elif self.use_apex:
        # with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        #     scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        outputs["probability_of_correct_action"] = round(float(outputs["probability_of_correct_action"]), 4)
        outputs["loss"] = round(float(loss), 4)
        outputs["rate_wrong_action_taken"] = round(float(outputs["rate_wrong_action_taken"]), 4)
        outputs["rate_oob_actions"] = round(float(outputs["rate_oob_actions"]), 4)

        # problem of double logging. _maybe_log_save_evaluate logs already, but does not get passed additional metrics
        # this is a workaround to log own metrics during training
        if self.state.global_step == 1 and self.args.logging_first_step:
            self.control.should_log = True
        if self.args.logging_strategy == IntervalStrategy.STEPS and \
                self.state.global_step % self.args.logging_steps == 0:
            self.control.should_log = True

        if self.control.should_log:
            metrics: Dict[str, float] = {}

            tr_loss_step = round(float(loss.detach() / self.args.gradient_accumulation_steps), 4)

            # define own metrics
            metrics["tr_loss"] = tr_loss_step
            metrics["probability_of_correct_action"] = outputs["probability_of_correct_action"]
            metrics["rate_wrong_action_taken"] = outputs["rate_wrong_action_taken"]
            metrics["rate_oob_actions"] = outputs["rate_oob_actions"]

            self.log(metrics)

        return loss.detach() / self.args.gradient_accumulation_steps


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

    if games_to_load.stop == -1 or games_to_load.stop - games_to_load.start >= 10000:
        # "wc-without_surr_and_passed-pr_True-one-hot"
        dataset = load_from_disk(f"./datasets/wc-without_surr_and_passed-pr_{point_rewards}-{hand_encoding}")
        amount_games = (games_to_load.stop - games_to_load.start) * 3  # * 3 for every perspective
        dataset['train'] = Dataset.from_dict(dataset['train'][:math.floor(amount_games * 0.8)])
        dataset['test'] = Dataset.from_dict(dataset['test'][:math.floor(amount_games * 0.2)])
    else:
        print("\nLoading data...")
        game_index = 5
        data, _ = data_pipeline.get_states_actions_rewards(  # first_states, meta_and_cards, actions_table, skat_and_cs
            championship=championship,
            games_indices=games_to_load,
            point_rewards=point_rewards,
            game_index=game_index,
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
        # vocab_size=1200,  # there are 32 cards + pos_tp + score +  + trump_enc
        n_positions=context_length,
        scale_attn_weights=True,
        # embd_pdrop=dropout,
        resid_pdrop=dropout,
        attn_pdrop=dropout,
        max_length=MAX_EPISODE_LENGTH,
    )

    if pretrained_model_path is not None:
        pretrained_model = TrainableDT.from_pretrained(
            f"./pretrained_models/{pretrained_model_path}")
        model = pretrained_model
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
        training_args.do_eval = True,
        training_args.evaluation_strategy = "steps"
        training_args.eval_steps = 100

    trainer = DTTrainer(  # CustomDTTrainer
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=collator,
        # compute_metrics=compute_metrics,  # is not reachable without labels
        # callbacks=[tensorboard_callback]  # here or in "report_to" of args
    )

    print(f"\nTraining on games {games_to_load.start, games_to_load.stop} with {hand_encoding} encoding...")
    trainer.train()

    # for tensorboard visualization:
    # 1. rm -r ./training-logs/*
    # 2. run training
    # 3. tensorboard --logdir=./training-logs

    # for saving the model:
    # model.save_pretrained(
    #     rf"./pretrained_models/"
    #     rf"{time.asctime().replace(':', '-').replace(' ', '_')}-"
    #     rf"games_{games_to_load.start}-{games_to_load.stop}-sampled")
    # and loading it again
    # pretrained_model = TrainableDT.from_pretrained("./pretrained_models")
    # trainer.model = pretrained_model

    if save_model:
        model.save_pretrained(rf"./pretrained_models/{dir_name}")

    training_args.do_eval, training_args.evaluation_strategy, training_args.eval_steps = True, "steps", 10

    trainer.data_collator = DecisionTransformerSkatDataCollator(dataset["test"])

    evaluation_results = trainer.evaluate()

    print(evaluation_results)

    # pretrained_model = TrainableDT.from_pretrained(
    #     f"./pretrained_models/games_0-50000-encoding_one-hot-point_rewards_True-Mon_Sep__4_23-48-24_2023")
    #
    # pretrained_model.config.state_dim = state_dim
    # # pretrained_model.config.act_dim =
    #
    # train_online_dt(pretrained_model,
    #                 point_reward=point_rewards,
    #                 state_dim=state_dim,
    #                 card_enc=hand_encoding,
    #                 amount_games=10
    #                 )

    # first_states = dataset['test']['states'][]
    # if games_to_load.stop != -1:
    #     evaluate_in_env(model, point_rewards, state_dim,
    #                     card_enc=hand_encoding,
    #                     first_states=first_states,
    #                     meta_and_cards_game=meta_and_cards,
    #                     skat_and_cs=skat_and_cs,
    #                     correct_actions=actions_table)


# ------------------------------------------------------------------------------------
#                           Interactive Implementation

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
#             loss = nn.CrossEntropyLoss(action_pred, action_target)
#
#             # Backpropagation
#             optimizer.zero_grad()
#             loss.backward()
#             # Optimisation
#             optimizer.step()

# less efficient manual evaluation, actually plays Skat in background
# motivation:   ability to see the game and throw errors when AI plays game
#               expense in normal evaluation (forward) to find out whether card is legal
def evaluate_in_env(model,
                    point_reward,
                    state_dim,
                    card_enc,
                    first_states,
                    meta_and_cards_game,
                    skat_and_cs,
                    correct_actions):
    # TODO: test
    scale = 1

    # env = environment.Env() if one_game is None else one_game
    env = Env(card_enc)

    amount_games = math.floor(len(first_states) / 3)

    for game_idx in range(amount_games):
        if point_reward:
            if sum(env.trump_enc) == 0:
                game_points = env.game.game_variant.get_level()
                target_return = game_points * 2
            else:
                if sum(env.trump_enc) == 4:
                    base_value = 24
                else:
                    base_value = 9 + env.trump_enc.index(1)

                level = 1 + env.game.game_variant.get_level()
                target_return = level * base_value * 2
        else:
            # if the reward are only the card points
            target_return = 2 * 120

        for current_player in range(3):
            game_and_pl_idx = game_idx + current_player

            # build the environment for the evaluation
            state = env.reset(current_player, game_first_state=first_states[game_and_pl_idx],
                              meta_and_cards_game=meta_and_cards_game[game_and_pl_idx],
                              skat_and_cs=skat_and_cs[game_and_pl_idx])

            target_return = torch.tensor(target_return).float().reshape(1, 1)
            states = torch.from_numpy(state).reshape(1, state_dim).float()
            actions = torch.zeros((0, ACT_DIM)).float()
            actions_preds = torch.zeros((0, ACT_DIM)).float()
            rewards = torch.zeros(0).float()
            timesteps = torch.tensor(0).reshape(1, 1).long()

            print(f"Actions should be {torch.argmax(correct_actions[game_and_pl_idx])}")

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

                soft_max = nn.Softmax(dim=0)
                action_pred = soft_max(action_pred)

                actions_preds[-1] = action_pred

                # print(f"Action {t}: {action_pred}")

                action = action_pred.detach().numpy()

                # hand cards within the state are padded from right to left after each action
                # mask the action
                valid_actions = action[:MAX_EPISODE_LENGTH - t]

                # get the index of the card with the highest probability
                card_index = np.argmax(valid_actions)

                # only select the best card
                action[:] = 0
                action[card_index] = 1

                action_target = correct_actions[game_and_pl_idx][t]

                # if card_index != action_target.index(1):
                #     print(f"Action {t} incorrect, should be {action_target}")

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
                    game_loss = nn.NLLLoss(correct_actions[game_and_pl_idx], actions_preds)

                    # log
                    print(f"Loss of game {game_and_pl_idx}: {game_loss}")

                    break


# TODO: Implementation of Online DT
#  stated in ausschreibung: Idea of 3 models learning the game through self-play after learning it from expert data
#  alternative: see Online DT Algorithm 1 and 2 (see above for idea)

def train_online_dt(model, point_reward, state_dim, card_enc, amount_games):
    scale = 1

    # env = environment.Env() if one_game is None else one_game
    env = Env(card_enc)

    # initialise s, a, r, t and raw action logits for every player
    states, actions, rewards, timesteps, actions_pred_eval = [[] * 3], [[] * 3], [[] * 3], [[] * 3], [[] * 3]

    for games in tqdm(range(amount_games)):
        # TODO: rotate players after game

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
            cur_pred_return = torch.zeros(0, 1).float()

            for i in range(3):
                # use the first playing order as an order of playing when putting the Skat
                # note that there is no order of playing cards during Skat putting in the game, as only the declarer
                # takes actions
                # after the first trick, the order of playing can change depending on the last tricks winner
                if t <= 2:
                    current_player = i

                states[current_player, -1] = env.update_state(states[current_player, -1])
                # rewards[current_player] = torch.cat([rewards[current_player], torch.zeros(1)], dim=0)
                # predicting the action to take
                action_pred = get_action(model,
                                         states[current_player],
                                         actions[current_player],
                                         rewards[current_player],
                                         target_return[current_player],
                                         timesteps)

                # action_pred are already soft-maxed and valid actions,
                # we only need to prevent the argmax from choosing an all 0 action

                actions_pred_eval[current_player][-1] = action_pred

                # print(f"Action {t}: {action_pred}")

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
                # reward_player is a tuple of the trick winner and her reward,
                # else there are difficulties with reward assignment
                state, reward_player, done, current_player = env.online_step(card_index, current_player)

                # print(f"Reward {t}: {reward}")

                cur_state = torch.cat([cur_state, torch.from_numpy(state).reshape(1, state_dim)])

                # states[current_player] = torch.cat([states[current_player], cur_state], dim=0)
                rewards[reward_player[0]][-1] = reward_player[1]

                pred_return = target_return[current_player, -1] - (rewards[reward_player[0]][-1] / scale)
                cur_pred_return = torch.cat([cur_pred_return, pred_return.reshape(1, 1)])

            target_return = torch.cat([target_return, cur_pred_return], dim=1)
            states = torch.cat([states, cur_state.unsqueeze(1)], dim=1)
            timesteps = torch.cat([timesteps, torch.ones((1, 1)).long() * (t + 1)], dim=1)

            # TODO: loss and backward propagation

        for current_player in range(3):
            # for evaluation of unspecific games
            diff_target_reached = target_return[current_player] - sum(rewards[current_player])

            # MSE loss
            loss = diff_target_reached ** 2

            print(
                f"Difference of target reward and reached reward of player {current_player}: {diff_target_reached}")

    # return trained model
    return model


def model_file(rel_path: str):
    # check whether the pre-trained model file exists
    from os.path import exists
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
    parser.add_argument('--games', type=int, default=(1, 2), nargs="+",
                        help='the games to load')
    parser.add_argument('--perspective', type=int, default=(0), nargs="+",
                        help='get the game from the amount of perspectives.'
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
    parser.add_argument('--pretrained_model', type=model_file, default=None,
                        help="Takes relative path as argument for the pretrained model which should be used. "
                             "The model has to be stored in the folder 'pretrained_models'.")
    parser.add_argument('--eval_in_training', type=bool, default=True,
                        help="Whether to evaluate during training. Slows down training if activated."
                             "Evaluation takes place on the test portion of the dataset (#_games_to_load * 0.2).")

    args = parser.parse_args()

    run_training(vars(args))
