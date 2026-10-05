"""
Return relabelling against the luck of the deal.

In Skat, the return of a game depends heavily on the dealt cards. A DT conditioned on raw returns-to-go partly learns
to "be lucky" instead of playing well (Paster et al., 2022). The DT itself stays unchanged; only the return-to-go labels
of the training data are replaced. The labels are estimated by small networks trained with Monte-Carlo regression on
the discounted returns-to-go of the data:

- "expected": R_t = Q(o_t, a_t), the expected return of the taken action given only what the player observed. The
  luck of unseen cards is averaged out, while the label still depends on the quality of the action.
- "deal_margin": R_t = G_t - V(deal), the achieved return relative to the expected return of the deal. V is a
  privileged model that sees all hands (only at training time, never when playing). The DT is conditioned on how much
  better than expected the cards were played.
"""

import json
import os
from typing import Dict

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from dt_skat_environment.environment import ACT_DIM, MAX_EPISODE_LENGTH, get_dims_in_enc

RTG_MODES = ("raw", "expected", "deal_margin")
GAMMA = 0.99  # as in the data collator
STATE_DIM = get_dims_in_enc("one-hot")[2]


def discounted_returns(rewards: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    returns = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def is_declarer(state: np.ndarray) -> bool:
    # both other players are opponents of the declarer
    return state[0] + state[1] + state[2] == -2


def _hidden_one_hot(hidden: np.ndarray) -> np.ndarray:
    """(…, 32) labels with the classes 0, 1, 2 and -100 -> (…, 96) one-hot, known cards are zeros."""
    one_hot = np.zeros(hidden.shape[:-1] + (32, 3), dtype=np.float32)
    for cls in range(3):
        one_hot[..., cls] = hidden == cls
    return one_hot.reshape(hidden.shape[:-1] + (96,))


def model_inputs(trajectory: Dict[str, np.ndarray], mode: str) -> np.ndarray:
    """Inputs of the return model for every time step of a trajectory."""
    states = np.asarray(trajectory["states"], dtype=np.float32).reshape(MAX_EPISODE_LENGTH, -1)
    if mode == "expected":
        actions = np.asarray(trajectory["actions"], dtype=np.float32).reshape(MAX_EPISODE_LENGTH, ACT_DIM)
        return np.concatenate([states, actions, np.eye(MAX_EPISODE_LENGTH, dtype=np.float32)], axis=1)
    if mode == "deal_margin":
        # the deal: the first state (own hand, position, game) and the locations of all other cards
        hidden = np.asarray(trajectory["hidden"]).reshape(MAX_EPISODE_LENGTH, 32)
        return np.concatenate([states[0], _hidden_one_hot(hidden[0])])[None]
    raise ValueError(f"Unknown mode {mode}")


def input_dim(mode: str) -> int:
    return STATE_DIM + ACT_DIM + MAX_EPISODE_LENGTH if mode == "expected" else STATE_DIM + 96


class ReturnModel(nn.Module):
    def __init__(self, mode: str, hidden_size: int = 256, n_layers: int = 3):
        super().__init__()
        if mode not in RTG_MODES[1:]:
            raise ValueError(f"Unknown mode {mode}")
        self.mode = mode
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        layers, dim = [], input_dim(mode)
        for _ in range(n_layers - 1):
            layers += [nn.Linear(dim, hidden_size), nn.ReLU()]
            dim = hidden_size
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)
        # normalisation of the regression targets
        self.register_buffer("return_mean", torch.zeros(1))
        self.register_buffer("return_std", torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1) * self.return_std + self.return_mean

    @torch.no_grad()
    def relabel(self, trajectory: Dict[str, np.ndarray]) -> np.ndarray:
        """:return: the returns-to-go of the trajectory (12,) with the relabelled values."""
        prediction = self(torch.from_numpy(model_inputs(trajectory, self.mode))).numpy()
        if self.mode == "expected":
            return prediction.astype(np.float32)
        returns = discounted_returns(np.asarray(trajectory["rewards"], dtype=np.float32).reshape(-1))
        return (returns - prediction[0]).astype(np.float32)

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        save_file(self.state_dict(), os.path.join(path, "model.safetensors"))
        with open(os.path.join(path, "config.json"), "w") as file:
            json.dump({"mode": self.mode, "hidden_size": self.hidden_size, "n_layers": self.n_layers}, file)

    @classmethod
    def load(cls, path: str) -> "ReturnModel":
        with open(os.path.join(path, "config.json")) as file:
            config = json.load(file)
        model = cls(**config)
        model.load_state_dict(load_file(os.path.join(path, "model.safetensors")))
        return model.eval()


def training_examples(trajectories, mode: str):
    inputs, targets = [], []
    for trajectory in trajectories:
        returns = discounted_returns(np.asarray(trajectory["rewards"], dtype=np.float32).reshape(-1))
        x = model_inputs(trajectory, mode)
        inputs.append(x)
        targets.append(returns if mode == "expected" else returns[:1])
    return np.concatenate(inputs), np.concatenate(targets)


def train_return_model(trajectories, mode: str, epochs: int = 20, batch_size: int = 512, learning_rate: float = 1e-3,
                       weight_decay: float = 1e-4, validation_split: float = 0.1, seed: int = 0, log=print):
    """Monte-Carlo regression of the (discounted) returns-to-go with the mean squared error."""
    torch.manual_seed(seed)
    x, y = training_examples(trajectories, mode)
    x, y = torch.from_numpy(x), torch.from_numpy(y)

    permutation = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))
    n_val = int(len(x) * validation_split)
    val_idx, train_idx = permutation[:n_val], permutation[n_val:]

    model = ReturnModel(mode)
    model.return_mean.fill_(float(y[train_idx].mean()))
    model.return_std.fill_(float(y[train_idx].std().clamp(min=1e-6)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        order = train_idx[torch.randperm(len(train_idx))]
        losses = []
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            loss = nn.functional.mse_loss((model(x[idx]) - model.return_mean) / model.return_std,
                                          (y[idx] - model.return_mean) / model.return_std)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            val = nn.functional.mse_loss(model(x[val_idx]), y[val_idx]).item() if n_val else float("nan")
        log(f"Return model epoch {epoch + 1}: normalised train loss {np.mean(losses):.4f}, validation MSE {val:.2f}")

    return model


def rtg_targets(trajectories, relabel, quantile: float = 0.9) -> Dict[str, float]:
    """
    Initial return-to-go for playing: a high quantile of the relabelled initial returns-to-go of each role.

    :param relabel: function mapping a trajectory to its returns-to-go.
    """
    initial = {"declarer": [], "defender": []}
    for trajectory in trajectories:
        role = "declarer" if is_declarer(np.asarray(trajectory["states"][0])) else "defender"
        initial[role].append(float(relabel(trajectory)[0]))
    return {role: float(np.quantile(values, quantile)) for role, values in initial.items() if values}
