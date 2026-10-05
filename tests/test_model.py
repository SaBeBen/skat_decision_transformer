import glob
import os
import random

import numpy as np
import pytest
import torch

from dt_skat_environment.environment import Env
from online.self_play import play_game, compute_target_returns
from trainer.dt_trainer import TrainableDT
from trainer.inference import get_action

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


def test_forward_with_engine_mask():
    model = TrainableDT.from_pretrained(MODEL_DIRS[0], use_mask=False, use_safetensors=True)
    batch = dummy_batch(model.config.state_dim)
    action_mask = torch.zeros(4, 12, 12)
    action_mask[:, 2:, 3:5] = 1  # no legal action in the first two steps
    batch["actions"] = torch.zeros(4, 12, 12)
    batch["actions"][:, 2:, 4] = 1

    output = model.original_forward(**batch, return_dict=True, action_mask=action_mask)
    probs = output.action_preds
    assert torch.allclose(probs[:, 2:].sum(dim=2), torch.ones(4, 10))
    assert torch.all(probs[:, :2] == 0)
    assert torch.all(probs[:, 2:, :3] == 0) and torch.all(probs[:, 2:, 5:] == 0)

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

        for trajectory in result["trajectories"]:
            # the rule mask derived from the state allows only cards that are legal in the game engine
            for t in range(12):
                if not trajectory["legal"][t].any():
                    continue
                probs = get_action(model, torch.from_numpy(trajectory["states"][:t + 1]),
                                   torch.from_numpy(trajectory["actions"][:t + 1]), None,
                                   torch.full((t + 1,), 50.0), torch.arange(t + 1))
                assert np.all(trajectory["legal"][t][probs.numpy() > 0] == 1)

            assert trajectory["states"].shape == (12, model.config.state_dim)
            assert trajectory["actions"].shape == (12, 12)
            assert trajectory["rewards"].shape == (12,)
            # at most one card per time step
            assert set(np.unique(trajectory["actions"].sum(axis=1))) <= {0.0, 1.0}


def test_training_loss_is_the_action_nll():
    model = TrainableDT.from_pretrained(MODEL_DIRS[0], use_mask=True, use_safetensors=True)
    batch = dummy_batch(model.config.state_dim)
    output = model.original_forward(**batch, return_dict=True)
    assert torch.equal(model(**batch)["loss"], output.loss)
