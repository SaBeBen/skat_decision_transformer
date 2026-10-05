"""
Online fine-tuning of a pre-trained Skat Decision Transformer via self-play,
following the Online Decision Transformer (Zheng et al., 2022, https://arxiv.org/abs/2202.05607).

Each iteration
1. plays games on random deals; every seat is either the learner (sampling from its stochastic policy) or a frozen
   opponent from a pool (the pre-trained model and snapshots of the learner),
2. adds the trajectories of the learner's seats to a FIFO replay buffer (returns-to-go are relabelled with the
   achieved rewards by the data collator),
3. updates the learner on the buffer with the entropy-regularised ODT objective (online/odt.py),
4. periodically evaluates the learner against the pre-trained model on fixed deals (online/evaluation.py).
"""

import argparse
import copy
import os
import random
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import data_pipeline
from dt_skat_environment.environment import Env, get_dims_in_enc
from experiments import file_check
from online.evaluation import duplicate_evaluation
from online.odt import ODTUpdater
from online.replay_buffer import TrajectoryReplayBuffer
from online.self_play import play_game, compute_target_returns
from trainer.data_collator import DecisionTransformerSkatDataCollator
from trainer.dt_trainer import TrainableDT
from utils import str2bool

CARD_ENC = "one-hot"  # the rule mask of the model (and the pre-trained models) require the one-hot encoding


def collect_games(learner, pool, n_games, args, rng, game_counter):
    """Plays n_games self-play games and returns the learner's trajectories and statistics."""
    learner.eval()
    trajectories = []
    stats = {"declarer_points": [], "declarer_won": [], "learner_declarer_points": [], "learner_defender_won": [],
             "mask_mismatches": 0, "decisions": 0}

    for _ in range(n_games):
        # seeded deals for reproducibility, disjoint from the evaluation seeds
        random.seed(args["seed"] * 1_000_003 + game_counter)
        game_counter += 1

        env = Env(CARD_ENC)
        first_states = env.online_reset(point_rewards=args["point_rewards"])

        is_learner = rng.random(3) < args["learner_seat_prob"]
        if not is_learner.any():
            is_learner[rng.integers(3)] = True
        seat_models = [learner if is_learner[pl] else pool[rng.integers(len(pool))] for pl in range(3)]

        result = play_game(env, seat_models, first_states,
                           compute_target_returns(env, args["point_rewards"], args["rtg_scale"]),
                           greedy=False, temperature=args["temperature"], rng=rng,
                           reward_convention=args["reward_convention"])

        trajectories += [result["trajectories"][pl] for pl in range(3) if is_learner[pl]]

        declarer_id = result["declarer_id"]
        stats["declarer_points"].append(result["game_points"][declarer_id])
        stats["declarer_won"].append(result["declarer_won"])
        if is_learner[declarer_id]:
            stats["learner_declarer_points"].append(result["game_points"][declarer_id])
        if any(is_learner[pl] for pl in range(3) if pl != declarer_id):
            stats["learner_defender_won"].append(not result["declarer_won"])
        stats["mask_mismatches"] += result["mask_mismatches"]
        stats["decisions"] += result["decisions"]

    return trajectories, stats, game_counter


def run_online_training(args):
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    rng = np.random.default_rng(args["seed"])

    device = torch.device("cuda") if torch.cuda.is_available() and args["use_cuda"] else torch.device("cpu")

    _, _, state_dim = get_dims_in_enc(CARD_ENC)

    learner = TrainableDT.from_pretrained(args["pretrained_model"], use_mask=True, use_safetensors=True).to(device)
    if learner.config.state_dim != state_dim:
        raise ValueError("The pre-trained model does not use the one-hot state representation.")

    # the frozen pre-trained model is the first opponent and the baseline of the evaluation
    pretrained = copy.deepcopy(learner).eval()
    pool = [pretrained]

    buffer = TrajectoryReplayBuffer(args["buffer_size"])
    offline_games = slice(args["offline_games"][0], args["offline_games"][1])
    if offline_games.stop > offline_games.start:
        print(f"\nSeeding the replay buffer with games {offline_games.start, offline_games.stop} "
              f"of the {args['championship']}...")
        data, *_ = data_pipeline.get_states_actions_rewards(
            championship=args["championship"],
            games_indices=offline_games,
            point_rewards=args["point_rewards"],
            card_enc=CARD_ENC,
            reward_convention=args["reward_convention"],
            include_grand=args["include_grand"],
            include_null=args["include_null"])
        added = buffer.seed_from_offline(data, top_k=args["offline_top_k"])
        print(f"Added {added} offline trajectories.")

    updater = ODTUpdater(learner,
                         learning_rate=args["learning_rate"],
                         weight_decay=args["weight_decay"],
                         warmup_steps=args["warmup_steps"],
                         grad_clip=args["grad_clip"],
                         init_temperature=args["init_temperature"],
                         target_entropy=args["target_entropy"])

    current_time = time.asctime().replace(':', '-').replace(' ', '_')
    # the model name is logged as text instead of being part of the path (Windows path length limit)
    run_name = f"online-{current_time}"
    writer = SummaryWriter(os.path.join(args["log_dir"], run_name))
    writer.add_text("config", "\n\n".join(f"{key}: {value}" for key, value in args.items()))
    output_dir = os.path.join(args["output_dir"], run_name)

    collator = None
    game_counter = 0

    for iteration in range(1, args["iterations"] + 1):
        start = time.time()

        # 1. self-play with the stochastic policy of the learner
        trajectories, stats, game_counter = collect_games(learner, pool, args["games_per_iter"], args, rng,
                                                          game_counter)
        buffer.extend(trajectories)
        collect_time = time.time() - start

        # 2. ODT updates on the replay buffer
        if collator is None:
            collator = DecisionTransformerSkatDataCollator(buffer, args["batch_size"])

        update_logs = [updater.step(collator(buffer.trajectories)) for _ in range(args["updates_per_iter"])]

        step = iteration
        for key in update_logs[0]:
            writer.add_scalar(f"train/{key}", float(np.mean([log[key] for log in update_logs])), step)
        writer.add_scalar("selfplay/declarer_points", float(np.mean(stats["declarer_points"])), step)
        writer.add_scalar("selfplay/declarer_win_rate", float(np.mean(stats["declarer_won"])), step)
        if stats["learner_declarer_points"]:
            writer.add_scalar("selfplay/learner_declarer_points",
                              float(np.mean(stats["learner_declarer_points"])), step)
        if stats["learner_defender_won"]:
            writer.add_scalar("selfplay/learner_defender_win_rate",
                              float(np.mean(stats["learner_defender_won"])), step)
        writer.add_scalar("selfplay/mask_mismatch_rate", stats["mask_mismatches"] / max(stats["decisions"], 1), step)
        writer.add_scalar("buffer/size", len(buffer), step)
        writer.add_scalar("time/collect_s", collect_time, step)
        writer.add_scalar("time/iteration_s", time.time() - start, step)

        print(f"Iteration {iteration}: nll {np.mean([log['nll'] for log in update_logs]):.4f}, "
              f"entropy {np.mean([log['entropy'] for log in update_logs]):.4f}, "
              f"temperature {updater.temperature:.4f}, "
              f"self-play declarer win rate {np.mean(stats['declarer_won']):.3f}, "
              f"buffer {len(buffer)}, {time.time() - start:.1f}s")

        # 3. add a frozen snapshot of the learner to the pool of opponents
        if args["snapshot_every"] > 0 and iteration % args["snapshot_every"] == 0:
            pool.append(copy.deepcopy(learner).eval())
            if len(pool) > args["pool_size"]:
                # always keep the pre-trained model, drop the oldest snapshot
                pool.pop(1)

        # 4. duplicate evaluation against the pre-trained model
        if args["eval_games"] > 0 and (iteration % args["eval_every"] == 0 or iteration == args["iterations"]):
            learner.eval()
            metrics = duplicate_evaluation(learner, pretrained, args["eval_games"], seed=args["eval_seed"],
                                           point_rewards=args["point_rewards"], rtg_scale=args["eval_rtg_scale"],
                                           card_enc=CARD_ENC, reward_convention=args["reward_convention"])
            for key, value in metrics.items():
                writer.add_scalar(f"eval/{key}", value, step)
            print("Evaluation against the pre-trained model: " +
                  ", ".join(f"{key} {value:.3f}" for key, value in metrics.items()))

        if args["save_every"] > 0 and iteration % args["save_every"] == 0:
            learner.save_pretrained(os.path.join(output_dir, f"iteration_{iteration}"))

    learner.save_pretrained(os.path.join(output_dir, "final"))
    writer.close()
    print(f"\nSaved the fine-tuned model to {os.path.join(output_dir, 'final')}")

    return learner


def get_parser():
    parser = argparse.ArgumentParser(prog='online_training',
                                     description='Fine-tune a pre-trained Skat DT online via self-play '
                                                 '(Online Decision Transformer).')
    parser.add_argument('--pretrained_model', type=file_check, required=True,
                        help="Relative path of the pre-trained model in the folder 'pretrained_models' "
                             "(one-hot encoding).")
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--games_per_iter', type=int, default=256, help='Self-play games per iteration.')
    parser.add_argument('--updates_per_iter', type=int, default=300, help='Gradient steps per iteration.')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--buffer_size', type=int, default=20000, help='Capacity of the replay buffer in trajectories.')
    parser.add_argument('--championship', '-cs', type=str, default='wc', choices=['wc'],
                        help='Championship of the offline games used to seed the replay buffer.')
    parser.add_argument('--offline_games', type=int, nargs=2, default=(0, 2000),
                        help='Range of offline games to seed the replay buffer with. "0 0" disables seeding.')
    parser.add_argument('--include_grand', type=str2bool, default=True,
                        help='Whether to seed the replay buffer with grand games.')
    parser.add_argument('--include_null', type=str2bool, default=True,
                        help='Whether to seed the replay buffer with null games.')
    parser.add_argument('--offline_top_k', type=int, default=None,
                        help='Only seed the buffer with the top k offline trajectories by return (as in the ODT). '
                             'Note that this favours declarers, as their returns are higher.')
    parser.add_argument('--point_rewards', type=str2bool, default=True,
                        help='Whether to use discounted Seeger-Fabian score as rewards')
    parser.add_argument('--reward_convention', type=str, default='default', choices=['default', 'thesis'],
                        help='"thesis" reproduces the rewards the models of the thesis were trained with.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=5e-4)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--grad_clip', type=float, default=0.25)
    parser.add_argument('--target_entropy', type=float, default=0.4,
                        help='Lower bound of the policy entropy (nats) at time steps where a card can be played.')
    parser.add_argument('--init_temperature', type=float, default=0.1,
                        help='Initial value of the entropy temperature (Lagrange multiplier).')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature of the policy during self-play.')
    parser.add_argument('--rtg_scale', type=float, default=1.0,
                        help='Scale of the initial return-to-go for exploration (see compute_target_returns).')
    parser.add_argument('--eval_rtg_scale', type=float, default=0.7,
                        help='Scale of the initial return-to-go in the evaluation (0.7 as in the thesis).')
    parser.add_argument('--learner_seat_prob', type=float, default=0.8,
                        help='Probability that a seat is played by the learner instead of a frozen opponent.')
    parser.add_argument('--snapshot_every', type=int, default=10,
                        help='Add a frozen snapshot of the learner to the opponent pool every n iterations.')
    parser.add_argument('--pool_size', type=int, default=5,
                        help='Maximum number of frozen opponents including the pre-trained model.')
    parser.add_argument('--eval_every', type=int, default=5)
    parser.add_argument('--eval_games', type=int, default=500, help='Number of deals of the duplicate evaluation.')
    parser.add_argument('--eval_seed', type=int, default=1_000_000_007)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--use_cuda', type=str2bool, default=True)
    parser.add_argument('--log_dir', type=str, default='./training-logs')
    parser.add_argument('--output_dir', type=str, default='./training_output')
    return parser


if __name__ == '__main__':
    run_online_training(vars(get_parser().parse_args()))
