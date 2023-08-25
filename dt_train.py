import argparse
import math
import random
# import resource
import time
from dataclasses import dataclass
from typing import Dict, Union, Tuple, Optional, List, Any

import numpy as np
import pandas as pd
import torch

from datasets import Dataset, DatasetDict, load_dataset
from torch import nn
from torch import Tensor
from sklearn.model_selection import train_test_split

from transformers import DecisionTransformerModel, TrainingArguments, Trainer, DecisionTransformerConfig

from transformers.models.decision_transformer.modeling_decision_transformer import DecisionTransformerOutput
from transformers.trainer_utils import speed_metrics, IntervalStrategy

import data_pipeline
import environment

# %%
# def set_memory_limit(limit_gb):
#     limit_bytes = limit_gb * 1024 * 1024 * 1024  # Convert MB to bytes
#     resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
#
#
# set_memory_limit(16)

MAX_EPISODE_LENGTH = 12

act_dim = 12
card_dim = 5

# for padding of hand_cards, it is the maximum size with a compressed card representation
max_hand_len = 16 + act_dim  # 12 * card_dim

# position co-player (3) + score (2) + trump (4) + last trick (3 * card_dim)
# + open cards (2 * card_dim) + hand cards (12 * card_dim)
state_dim = 3 + 2 + 4 + 3 * card_dim + 2 * card_dim + max_hand_len  # 12 * card_dim

"""
This Implementation is based on the Decision Transformer https://github.com/kzl/decision-transformer/tree/master.

To achieve a speed up and more modularity, the Huggingface implementation of the Decision Transformer is adapted.
See https://huggingface.co/blog/train-decision-transformers for more information.
"""


@dataclass
class DecisionTransformerSkatDataCollator:
    return_tensors: str = "pt"  # pytorch
    max_len: int = MAX_EPISODE_LENGTH  # subsets of the episode we use for training, our episode length is short
    state_dim: int = state_dim  # size of state space
    act_dim: int = act_dim  # size of action space
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

    def get_batch_ind(self):
        # batch_size = len(features)
        # the batch size is per_device_train_batch_size*num_of_gpu, and the collator
        # makes batches of batch_size every dataset_size//batch_size times,
        # with the batch size being dataset_size%batch_size at the last call, just for the remaining data

        # this is a bit of a hack to be able to sample of a non-uniform distribution
        # we have a random pick of the data as a batch without controlling the shape,
        return np.random.choice(
            np.arange(self.n_traj),
            size=self.batch_size,
            replace=True,
            p=self.p_sample,  # reweights, so we sample according to timesteps
        )

    def _discount_cumsum(self, x, gamma):
        # weighted rewards are in the data set (get_states_actions_rewards)
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(discount_cumsum.shape[0] - 1)):
            # gamma as a discount factor to differ rewards temporarily
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum

    def __call__(self, features):
        batch_inds = self.get_batch_ind()

        # a batch of dataset features
        s, a, r, rtg, timesteps, mask, big_action_mask = [], [], [], [], [], [], []

        for ind in batch_inds:
            # for feature in features:

            feature = self.dataset[int(ind)]

            # why do we need a randint?
            # to be able to jump into one game -> predict from every position and improve training
            #  jumping randomly into a surrendered game could not work well
            si = random.randint(1, len(feature["rewards"]))

            #  fixed frame of 12 timesteps
            #  we want to have a history starting from the beginning of the game
            #  and include the rtgs from the last reward
            #  we had an implementation starting from the last timestep looking back

            #  does the self attention have to model the knowledge over time (mask with 1 from left to right)
            #  or chase the reward (mask with 1s from right to left)
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
                self._discount_cumsum(np.array(feature["rewards"]), gamma=1)[: s[-1].shape[1]].reshape(1, -1, 1)
                # TODO check the +1 removed here
            )
            if rtg[-1].shape[1] < s[-1].shape[1]:
                print("if true")
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            tlen = s[-1].shape[1]

            # states, actions, rewards are already padded

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
    ) -> Union[Tuple, DecisionTransformerOutput]:

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

        # injected behaviour
        action_preds = action_preds[attention_mask > 0]
        sm = nn.Softmax(dim=1)
        action_preds = sm(action_preds)

        if not return_dict:
            return state_preds, action_preds, return_preds

        return DecisionTransformerOutput(
            last_hidden_state=encoder_outputs.last_hidden_state,
            state_preds=state_preds,
            action_preds=action_preds,
            return_preds=return_preds,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def forward(self, **kwargs):
        output = self.originalForward(**kwargs)  # super().forward(**kwargs)
        # add the DT loss
        action_preds = output[1]
        action_targets = kwargs["actions"]
        attention_mask = kwargs["attention_mask"]
        # act_dim = action_preds.shape[1]  # 2

        # exclude actions that are not attended to
        action_preds = action_preds.reshape(-1, act_dim)  # [attention_mask.reshape(-1) > 0]
        action_targets = action_targets.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        # TODO: rather output softmax predictions and maybe have a sm layer (careful with attn mask)
        #  or calculate everything after prediction (closer to original implementation)
        # cross entropy loss
        # cross_ent_fct = nn.CrossEntropyLoss()

        # if the softmax is already applied in original forward
        nll_loss_fct = nn.NLLLoss()
        cross_ent_loss = nll_loss_fct(torch.log(action_preds), torch.argmax(action_targets, dim=1))

        # cross_ent_loss = cross_ent_fct(action_preds, action_targets)

        loss = cross_ent_loss

        # only predicts the prob in context of 12 cards
        # soft_max = nn.Softmax(dim=1)
        # action_pred_sm = soft_max(action_preds)

        # if softmax is already applied on preds
        action_pred_sm = action_preds

        prob_each_act_correct = action_pred_sm * action_targets

        # exclude Skat putting for defenders and hand games
        # amount of actions - amount of actions that select a card
        amount_no_skat_action = action_targets.shape[0] - torch.sum(action_targets)

        # the accumulated probability of the correct action / total number of actions taken in targets
        prob_correct_action = torch.sum(prob_each_act_correct) / (
                action_targets.shape[0] - amount_no_skat_action)

        # we want to know to what probability the model actually chooses an action != target_action,
        # not the accumulated prob of (actions != target_action)
        action_taken = torch.argmax(action_pred_sm, dim=1)

        action_mask = torch.zeros_like(action_targets)
        action_mask[torch.arange(action_targets.shape[0]), action_taken] = 1

        wrong_action_taken = action_mask * ~action_targets.bool()

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
        hand_card_length = 12 - timesteps.reshape(-1)[attention_mask.reshape(-1) > 0]

        actions_oob = torch.nonzero(torch.div(action_taken, hand_card_length, rounding_mode='trunc'))
        # predictions have to be passed as tensors
        rate_oob_actions = Tensor([actions_oob.shape[0] / action_targets.shape[0]])

        # TODO: include rules by looking at first open card and colour (including trump_enc), length
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
                 "prob_correct_action": round(float(output.predictions[1][-1]), 4),
                 "rate_wrong_action_taken": round(float(output.predictions[2][-1]), 4),
                 "rate_oob_actions": round(float(output.predictions[3][-1]), 4)
                 }
            )
        else:
            # edge case of one game
            output.metrics.update(
                {"eval_loss: ": round(float(output.predictions[0]), 4),
                 "prob_correct_action": round(float(output.predictions[1]), 4),
                 "rate_wrong_action_taken": round(float(output.predictions[2]), 4),
                 "rate_oob_actions": round(float(output.predictions[3]), 4)
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

        # if is_sagemaker_mp_enabled():
        #     loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
        #     return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        # elif self.use_apex:
        #     with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        #         scaled_loss.backward()
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


# class CustomTBCallback(TensorBoardCallback):
#     # the following method is introduced to prevent double logging
#     def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
#         # Log
#         control.should_log = False
#
#         # Evaluate
#         if (
#                 args.evaluation_strategy == IntervalStrategy.STEPS
#                 and state.global_step % args.eval_steps == 0
#                 and args.eval_delay <= state.global_step
#         ):
#             control.should_log = True
#             control.should_evaluate = True
#
#         # Save
#         if (
#                 args.save_strategy == IntervalStrategy.STEPS
#                 and args.save_steps > 0
#                 and state.global_step % args.save_steps == 0
#         ):
#             control.should_save = True
#
#         # End training
#         if state.global_step >= state.max_steps:
#             control.should_training_stop = True
#
#         return control


def run_training(args):
    # game specific arguments
    championship = args['championship']
    # data_format = args['data_format']
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
    device = args['device']
    device = torch.device(device)
    save_model = args['save_model']

    if games_to_load.stop == -1:
        dataset = load_dataset("./datasets/wc_without_surr_and_passed")
    else:
        print("Loading data...")
        game_index = 5
        data, _ = data_pipeline.get_states_actions_rewards(championship=championship,
                                                           games_indices=games_to_load,
                                                           point_rewards=point_rewards,
                                                           game_index=game_index,
                                                           perspective=perspective)

        data_frame = pd.DataFrame(data)
        data_train, data_test = train_test_split(data_frame, train_size=0.8, random_state=42)  # 42
        dataset = DatasetDict({"train": Dataset.from_dict(data_train),
                               "test": Dataset.from_dict(data_test)})

    # device = torch.device("cuda")  # "cpu"

    collator = DecisionTransformerSkatDataCollator(dataset["train"], batch_size)

    config = DecisionTransformerConfig(
        state_dim=state_dim,
        act_dim=act_dim,
        action_tanh=False,  # do not apply the tanh fct on the output action
        activation_function=act_fct,
        n_head=n_head,
        n_layer=n_layer,
        max_ep_len=MAX_EPISODE_LENGTH,  # each episode is a game -> 12 tuples of s,a,r make up 1 game
        # vocab_size=1200,  # there are 32 cards + pos_tp + score +  + trump_enc
        n_positions=1024,
        scale_attn_weights=True,
        # embd_pdrop=dropout,
        resid_pdrop=dropout,
        attn_pdrop=dropout,
        max_length=MAX_EPISODE_LENGTH,
    )

    # pretrained_model = TrainableDT.from_pretrained(
    #     "./pretrained_models/Tue_Aug_22_22-42-38_2023-games_1000-0-sampled")
    # model = pretrained_model

    model = TrainableDT(config)

    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)

    # model.to(device)

    training_args = TrainingArguments(
        report_to=["tensorboard"],
        output_dir="training_output/",
        remove_unused_columns=False,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=32,
        learning_rate=l_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        optim="adamw_torch",
        max_grad_norm=0.1,
        logging_steps=20,
        logging_dir=rf"./training-logs/"
                    rf"{time.asctime().replace(':', '-').replace(' ', '_')}-games_"
                    rf"{games_to_load.stop}-{games_to_load.start}-sampled",

        # do_eval=True,
        # evaluation_strategy="steps",
        # eval_steps=100,

        # no_cuda=True,
    )

    # tensorboard_callback = CustomTBCallback()

    trainer = DTTrainer(  # CustomDTTrainer
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=collator,
        # compute_metrics=compute_metrics,  # is not reachable without labels
        # callbacks=[tensorboard_callback]  # here or in "report_to" of args
    )

    print("Training...")
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
        model.save_pretrained(
            rf"./pretrained_models/"
            rf"{time.asctime().replace(':', '-').replace(' ', '_')}-"
            rf"games_{games_to_load.start}-{games_to_load.stop}-"
            rf"point_rewards_{point_rewards}")

    training_args.do_eval, training_args.evaluation_strategy, training_args.eval_steps = True, "steps", 10

    trainer.data_collator = DecisionTransformerSkatDataCollator(dataset["test"])

    evaluation_results = trainer.evaluate()

    print(evaluation_results)


# ------------------------------------------------------------------------------------
#                           Interactive Implementation

def interactive_env(model):
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

    scale = 1

    # env = environment.Env() if one_game is None else one_game
    env = environment.Env()

    # TODO: change to max point reward, if point reward is activated
    target_return = 2 * 120

    for games in range(1):

        for current_player in range(3):

            # build the environment for the evaluation
            state = env.reset(current_player_id=current_player)
            # game_states=dataset['train'][8]['states'])

            target_return = torch.tensor(target_return).float().reshape(1, 1)
            states = torch.from_numpy(state).reshape(1, state_dim).float()
            actions = torch.zeros((0, act_dim)).float()
            rewards = torch.zeros(0).float()
            timesteps = torch.tensor(0).reshape(1, 1).long()
            actions_pred_eval = torch.zeros((0, act_dim)).float()

            # take steps in the environment (evaluation, not training)
            for t in range(MAX_EPISODE_LENGTH):
                # add zeros for actions as input for the current time-step
                actions = torch.cat([actions, torch.zeros((1, act_dim))], dim=0)
                actions_pred_eval = torch.cat([actions_pred_eval, torch.zeros((1, act_dim))], dim=0)
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

                actions_pred_eval[-1] = action_pred

                print(f"Action {t}: {action_pred}")

                action = action_pred.detach().numpy()

                # hand cards within the state are padded from right to left after each action
                # mask the action
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
                    diff_target_reached = target_return - sum(rewards)

                    print(f"Difference of target reward and reached reward: {diff_target_reached}")
                    break


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
    parser.add_argument('--championship', '-cs', type=str, default='wc', choices=['wc', 'gc', 'gtc', 'bl', 'rc'],
                        help='dataset of championship to select from')
    # parser.add_argument('--data_format', type=str, default='mixed', choices=['one-hot', 'mixed'])
    parser.add_argument('--games', type=int, default=(0, 10), nargs="+",
                        help='the games to load')
    parser.add_argument('--perspective', type=int, default=(0, 1, 2), nargs="+",
                        help='get the game from the perspective of these players. ',
                        choices=[(0, 1, 2), (0, 1), (0, 2), (1, 2), (1), (2), (3)])
    parser.add_argument('--point_rewards', type=bool, default=False,
                        help='whether to add points of the game to the card points as a reward')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_layer', type=int, default=2, help='number of hidden layers')
    parser.add_argument('--n_head', type=int, default=1, help='number of attention heads')
    parser.add_argument('--activation_function', type=str, default='tanh',
                        help='the activation function to use in the GPT2Model')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=240)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_model', type=bool, default=False,
                        help='Whether to save the model. '
                             'If true, the trained model will be saved to "pretrained_models"')

    args = parser.parse_args()

    run_training(vars(args))
