import glob
import os

import numpy as np
import torch

from online.replay_buffer import TrajectoryReplayBuffer
from online_training import get_parser, run_online_training
from trainer.data_collator import DecisionTransformerSkatDataCollator
from trainer.dt_trainer import TrainableDT


def make_trajectory(value):
    return {"states": np.full((12, 214), value, dtype=np.float32),
            "actions": np.eye(12, dtype=np.float32),
            "rewards": np.full(12, value, dtype=np.float32)}


def test_replay_buffer_is_fifo():
    buffer = TrajectoryReplayBuffer(capacity=3)
    buffer.extend(make_trajectory(v) for v in range(5))

    assert len(buffer) == 3
    assert sorted(traj["rewards"][0] for traj in buffer.trajectories) == [2, 3, 4]


def test_replay_buffer_works_with_collator():
    buffer = TrajectoryReplayBuffer(capacity=10)
    buffer.extend(make_trajectory(v) for v in range(4))

    batch = DecisionTransformerSkatDataCollator(buffer, batch_size=8)(buffer.trajectories)

    assert batch["states"].shape == (8, 12, 214)
    assert batch["actions"].shape == (8, 12, 12)
    assert batch["returns_to_go"].shape == (8, 12, 1)


def test_online_training_smoke(tmp_path):
    model_name = os.path.basename(sorted(glob.glob("pretrained_models/*"))[-1])
    args = vars(get_parser().parse_args([
        "--pretrained_model", model_name,
        "--iterations", "2", "--games_per_iter", "3", "--updates_per_iter", "3", "--batch_size", "8",
        "--warmup_steps", "1", "--learning_rate", "1e-3",
        "--offline_games", "0", "0", "--eval_every", "2", "--eval_games", "2",
        "--snapshot_every", "1", "--pool_size", "2", "--save_every", "0", "--use_cuda", "false",
        "--log_dir", str(tmp_path / "logs"), "--output_dir", str(tmp_path / "out"),
    ]))

    learner = run_online_training(args)

    final = glob.glob(str(tmp_path / "out" / "*" / "final"))
    assert len(final) == 1
    reloaded = TrainableDT.from_pretrained(final[0], use_mask=True, use_safetensors=True)
    pretrained = TrainableDT.from_pretrained(args["pretrained_model"], use_mask=True, use_safetensors=True)

    # the learner was updated and saved correctly
    for p_learner, p_reloaded in zip(learner.state_dict().values(), reloaded.state_dict().values()):
        assert torch.equal(p_learner.cpu(), p_reloaded)
    assert any(not torch.equal(a, b) for a, b in zip(reloaded.state_dict().values(),
                                                     pretrained.state_dict().values()))
    assert glob.glob(str(tmp_path / "logs" / "*" / "events.out.tfevents.*"))
