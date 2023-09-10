
import math
import time
from dataclasses import dataclass
from typing import Dict, Union, Optional, List, Any, Tuple

import torch

from datasets import Dataset
from torch import nn
from torch import Tensor

from transformers import DecisionTransformerModel, Trainer

from transformers.trainer_utils import speed_metrics, IntervalStrategy
from transformers.utils import ModelOutput

from environment import ACT_DIM


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

        # put_card_mask = states[:, :, 3].bool()
        # action_preds = action_preds * put_card_mask.unsqueeze(2).repeat(1, 1, 12)

        # sm = torch.nn.Softmax(dim=2)
        # action_preds = sm(action_preds)

        # action_targets_masked = actions.reshape(-1, ACT_DIM)[attention_mask.reshape(-1) > 0]
        # action_preds_masked = action_preds.reshape(-1, ACT_DIM)[attention_mask.reshape(-1) > 0]
        #
        # nll_loss_fct = torch.nn.NLLLoss()
        # loss_pure = nll_loss_fct(torch.log(action_preds_masked), torch.argmax(action_targets_masked, dim=1))
        # loss = loss_pure

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
        actions_in_bounds = torch.arange(12).repeat(batch_size, 12).reshape(batch_size, 12, 12).to(
            self.device) < hand_card_length.view(-1, 12, 1).to(self.device)

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
        playable_cards = playable_cards.to(dtype=self.dtype)  # fp16 compatibility
        playable_cards = (1.0 - playable_cards) * torch.finfo(self.dtype).min

        possible_action_preds = playable_cards + action_preds

        # mask all actions were no card can be played, 1 if it can be played, 0 if not
        # e.g. in Hand games, as defenders in first two tricks (or side effect: due to the attention_mask)
        put_card_mask = states[:, :, 3].bool().to(self.device)

        # # starting from here until the loss fct, the loss is calculated

        # softmax has to be before the cross entropy loss

        # action_targets_masked = actions * put_card_mask.unsqueeze(2).repeat(1, 1, 12) * playable_cards

        # put_card_mask = put_card_mask.unsqueeze(2).repeat(1, 1, 12).to(dtype=self.dtype)  # fp16 compatibility
        # put_card_mask = (1.0 - put_card_mask) * torch.finfo(self.dtype).min

        # TODO: comment when wanting to train without a mask
        # action_preds = possible_action_preds  # + put_card_mask

        # apply mask whether a card should be played based on game timestep
        # action_preds = possible_action_preds * put_card_mask.unsqueeze(2).repeat(1, 1, 12)

        # equal to target mask
        # mask_wo_zeros = action_preds.sum(dim=1)
        # mask_wo_zeros = mask_wo_zeros > 0

        # when argmax takes 0 actions it selects the first element as maximum -> there is no 1 -> inf loss
        # --> as the loss only looks at the ground truth, and we mask 0 actions, we throw out the 0 actions for the loss
        # effectively the same as put_card_mask
        # mask_wo_zeros = action_targets_masked.sum(dim=1)
        # mask_wo_zeros = mask_wo_zeros > 0
        #
        # # mask the targets and preds
        # targets_wo_zeros = action_targets_masked[mask_wo_zeros]
        # preds_wo_zeros = action_preds_masked[mask_wo_zeros]

        sm = torch.nn.Softmax(dim=2)
        action_preds_output = sm(action_preds)

        # more stable than sm + log
        action_preds_ls = torch.nn.functional.log_softmax(action_preds, dim=2)

        # produces NaNs
        # action_preds = action_preds * put_card_mask.unsqueeze(2).repeat(1, 1, 12)
        # produces NaNs
        # action_mask = action_preds != 0
        # comb_mask = action_mask * attention_mask.unsqueeze(2).repeat(1, 1, 12).bool()
        # temp = action_preds[comb_mask]
        # temp2 = actions[comb_mask]

        action_targets_masked = actions.reshape(-1, ACT_DIM)[attention_mask.reshape(-1) > 0]
        action_preds_masked = action_preds_ls.reshape(-1, ACT_DIM)[attention_mask.reshape(-1) > 0]

        nll_loss_fct = torch.nn.NLLLoss()
        loss_pure = nll_loss_fct(action_preds_masked, torch.argmax(action_targets_masked, dim=1))
        loss = loss_pure

        action_preds = action_preds_output  # * put_card_mask.unsqueeze(2).repeat(1, 1, 12)

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
                {"eval_loss": round(float(output.predictions[0][-1]), 4),
                 "eval_prob_correct_action": round(float(output.predictions[1][-1]), 4),
                 "eval_rate_wrong_action_taken": round(float(output.predictions[2][-1]), 4),
                 "eval_rate_oob_actions": round(float(output.predictions[3][-1]), 4)
                 }
            )
        else:
            # edge case of one game
            output.metrics.update(
                {"eval_loss": round(float(output.predictions[0]), 4),
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

