import random
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm
from transformers import DecisionTransformerModel, TrainingArguments, Trainer, DecisionTransformerConfig, \
    DefaultDataCollator
# %%
import decision_transformer

from datasets import Dataset, DatasetDict

# %%
dataset = decision_transformer.get_states_actions_rewards(amount_games=10)

# %%
dataset = DatasetDict({"train": Dataset.from_dict(dataset)})

# %%
from transformers import DataCollatorWithPadding


# from datasets import load_dataset

# dataset = load_dataset("edbeeching/decision_transformer_gym_replay", "halfcheetah-expert-v2")


# TODO: tqdm for progress bar

def get_batch_ind():  # game_length
    # picks game indices
    # this picks the same game over *times* times
    # (WC GameID 4)
    times = 100
    return np.tile(np.arange(6, 7), times)

    # return np.random.choice(
    #     np.arange(n_traj),
    #     size=batch_size,
    #     replace=True,
    #     p=p_sample,  # reweights so we sample according to timesteps
    # )


# from https://huggingface.co/blog/train-decision-transformers
@dataclass
class DecisionTransformerSkatDataCollator:
    return_tensors: str = "pt"
    max_len: int = 12  # 13 subsets of the episode we use for training, our episode length is short
    # -> computationally cheap
    state_dim: int = 22  # 22 size of state space
    act_dim: int = 1  # 1 size of action space
    max_ep_len: int = 12  # 13 max episode length in the dataset
    scale: float = 12.0  # 13.0 normalization of rewards/returns
    state_mean: np.array = None  # to store state means
    state_std: np.array = None  # to store state stds
    p_sample: np.array = None  # a distribution to take account trajectory lengths
    n_traj: int = 0  # to store the number of trajectories in the dataset TODO: do we need this?

    def __init__(self, dataset) -> None:
        self.act_dim = len(dataset[0]["actions"][0])
        self.state_dim = len(dataset[0]["states"][0])
        self.dataset = dataset
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
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum

    def __call__(self, features):
        batch_size = len(features)

        # TODO: We want one game as a whole in a batch v
        # Done: normalization v
        # TODO: rtg, timesteps and mask v
        # TODO: scale rewards
        # Done: find a suit game which is easily won v

        # We don't need "train" and test within the dataset

        batch_inds = get_batch_ind()  # self.state_dim * self.max_ep_len

        # this is a bit of a hack to be able to sample of a non-uniform distribution
        # we have a random pick of the data as a batch without controlling the shape,
        # batch_inds = np.random.choice(
        #     np.arange(self.n_traj),
        #     size=batch_size,
        #     replace=True,
        #     p=self.p_sample,  # reweights, so we sample according to timesteps
        # )

        # a batch of dataset features
        s, a, r, rtg, timesteps, mask = [], [], [], [], [], []

        for ind in batch_inds:
            # for feature in features:

            feature = self.dataset[int(ind)]

            # why do we need a randint?
            # to be able to jump into one game -> predict from every position and improve training
            # TODO: jumping randomly into a surrendered game does not work well
            si = 0  # random.randint(0, len(feature["rewards"]) - 1)  # feature[]

            # get sequences from dataset
            s.append(np.array(feature["states"]  # feature[]
                              [si: self.max_len]).reshape((1, -1, self.state_dim)))
            a.append(np.array(feature["actions"][si:self.max_len]).reshape((1, -1, self.act_dim)))  # feature[]
            r.append(np.array(feature["rewards"][si:self.max_len]).reshape((1, -1, 1)))  # feature[]

            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))  # s[-1].shape[1]
            timesteps[-1][timesteps[-1] >= self.max_ep_len] = self.max_ep_len - 1  # padding cutoff
            rtg.append(
                self._discount_cumsum(np.array(feature["rewards"][si:]), gamma=1.0)[
                : s[-1].shape[1]  # TODO check the +1 removed here
                ].reshape(1, -1, 1)
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
                [np.ones((1, self.max_len - tlen, self.act_dim)) * -10.0, a[-1]],
                axis=1,
            )
            r[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), r[-1]], axis=1)

            rtg[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), rtg[-1]], axis=1) / self.scale
            timesteps[-1] = np.concatenate([np.zeros((1, self.max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, self.max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).float()
        a = torch.from_numpy(np.concatenate(a, axis=0)).float()
        r = torch.from_numpy(np.concatenate(r, axis=0)).float()
        # d = torch.from_numpy(np.concatenate(d, axis=0))
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

    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        # add the DT loss
        action_preds = output[1]
        action_targets = kwargs["actions"]
        attention_mask = kwargs["attention_mask"]
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_targets = action_targets.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = torch.mean((action_preds - action_targets) ** 2)

        return {"loss": loss}

    def original_forward(self, **kwargs):
        return super().forward(**kwargs)


# device = torch.device("cuda")
#
# target_return = torch.tensor(61, dtype=torch.float32).reshape(1, 1)
# timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)


training_args = TrainingArguments(
    output_dir="output/",
    remove_unused_columns=False,
    num_train_epochs=360,
    per_device_train_batch_size=64,
    learning_rate=1e-4,
    weight_decay=1e-4,
    warmup_ratio=0.1,
    optim="adamw_torch",
    max_grad_norm=0.25,
)

# collator = DecisionTransformerSkatDataCollator(dataset)
# collator = DefaultDataCollator()

configuration = DecisionTransformerConfig(state_dim=92,  # each state consist out of 22 numbers
                                          act_dim=5,  # each action consists out of one played card ()
                                          max_ep_len=12,  # each episode is a game -> 12 tuples of s,a,r make up 1 game
                                          vocab_size=35,  # there are 32 cards + pos_tp + trump + surrender
                                          # TODO: other encodings (like pos_tp)?
                                          )

# model = DecisionTransformerModel(configuration)

collator = DecisionTransformerSkatDataCollator(dataset["train"])

config = DecisionTransformerConfig(state_dim=collator.state_dim, act_dim=collator.act_dim)
model = TrainableDT(config)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    # eval_dataset=dataset["train"],
    data_collator=collator,
)

trainer.train()
