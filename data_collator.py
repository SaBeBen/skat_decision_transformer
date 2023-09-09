import random
from dataclasses import dataclass

import numpy as np
import torch

from environment import ACT_DIM, MAX_EPISODE_LENGTH


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
        # we do not need state normalisation with our one-hot encoding

    def _discount_cumsum(self, x, gamma):
        # weighted rewards are in the data set (get_states_actions_rewards)
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(discount_cumsum.shape[0] - 1)):
            # gamma as a discount factor to differ rewards temporarily
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum

    def __call__(self, features):
        # features are already batched, but
        # we always want to put out batches of batch_size, len(features) can be < batch_size, then either we have issues
        # in forward or if we drop smaller batches, we can not train under 32 games (for overfitting)

        # this is a bit of a hack to be able to sample of a non-uniform distribution
        # we have a random pick of the data as a batch without controlling the shape
        batch_inds = np.random.choice(
            np.arange(len(features)),
            size=self.batch_size,
            replace=True,
            p=[1 / len(features)] * len(features)
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
