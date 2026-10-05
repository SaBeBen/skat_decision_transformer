import glob
import os
import random

import numpy as np
import pytest
import torch

from dt_skat_environment.environment import Env
from online.self_play import play_game, compute_target_returns
from trainer.dt_trainer import TrainableDT

MODEL_DIRS = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "pretrained_models", "*")))


def dummy_batch(state_dim, batch_size=4, seq_len=12):
    torch.manual_seed(0)
    states = torch.zeros(batch_size, seq_len, state_dim)
    states[:, :, 3] = 1  # a card can be played
    states[:, :, 70:70 + 12 * 4] = 1  # some hand cards
    actions = torch.zeros(batch_size, seq_len, 12)
    actions[:, :, 0] = 1
    return dict(states=states, actions=actions, rewards=torch.zeros(batch_size, seq_len, 1),
                returns_to_go=torch.randn(batch_size, seq_len, 1) * 50,
                timesteps=torch.arange(seq_len).repeat(batch_size, 1),
                attention_mask=torch.ones(batch_size, seq_len, dtype=torch.long))


def test_pretrained_models_are_safetensors():
    assert MODEL_DIRS
    for model_dir in MODEL_DIRS:
        assert os.path.exists(os.path.join(model_dir, "model.safetensors"))
        assert not os.path.exists(os.path.join(model_dir, "pytorch_model.bin"))


@pytest.mark.parametrize("model_dir", MODEL_DIRS)
def test_forward_backward_is_finite(model_dir):
    model = TrainableDT.from_pretrained(model_dir, use_mask=True, use_safetensors=True)
    output = model.original_forward(**dummy_batch(model.config.state_dim), return_dict=True)

    assert torch.isfinite(output.loss)
    assert torch.isfinite(output.entropy)
    assert output.entropy >= 0

    # the ODT objective must not produce NaN gradients through masked actions
    (output.loss - 0.1 * output.entropy).backward()
    for param in model.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()


@pytest.mark.parametrize("model_dir", MODEL_DIRS)
def test_model_mask_agrees_with_game_engine(model_dir):
    model = TrainableDT.from_pretrained(model_dir, use_mask=True, use_safetensors=True).eval()

    for seed in range(5):
        random.seed(seed)
        env = Env("one-hot")
        first_states = env.online_reset()
        result = play_game(env, [model] * 3, first_states, compute_target_returns(env, True, 0.7), greedy=True)

        assert result["mask_mismatches"] == 0
        for trajectory in result["trajectories"]:
            assert trajectory["states"].shape == (12, model.config.state_dim)
            assert trajectory["actions"].shape == (12, 12)
            assert trajectory["rewards"].shape == (12,)
            # at most one card per time step
            assert set(np.unique(trajectory["actions"].sum(axis=1))) <= {0.0, 1.0}
