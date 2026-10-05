"""
Online fine-tuning of a pre-trained Skat Decision Transformer via self-play,
following the Online Decision Transformer (Zheng et al., 2022, https://arxiv.org/abs/2202.05607).

Each iteration
1. plays games on random deals; every seat is either the learner (sampling from its stochastic policy) or a frozen
   opponent from a pool (the pre-trained model, a rule-based player and snapshots of the learner, optionally
   sampled by prioritised fictitious self-play),
2. adds the trajectories of the learner's seats to a FIFO replay buffer (returns-to-go are relabelled with the
   achieved rewards by the data collator),
3. updates the learner on the buffer with the entropy-regularised ODT objective (online/odt.py),
4. periodically evaluates the learner against the pre-trained model on fixed deals (online/evaluation.py).
"""

import argparse
import copy
from collections import deque
import os
import random
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import data_pipeline
from dt_skat_environment.environment import Env, get_dims_in_enc
from experiments import file_check
from online.agents import RuleAgent
from online.evaluation import duplicate_evaluation
from online.odt import ODTUpdater
from online.replay_buffer import TrajectoryReplayBuffer
from online.self_play import play_games, compute_target_returns
from trainer.data_collator import DecisionTransformerSkatDataCollator
from trainer.dt_trainer import load_trainable_dt
from trainer.declaration import DeclarationDT, DeclarationPolicy
from trainer.return_model import ReturnModel
from utils import str2bool

CARD_ENC = "one-hot"  # the rule mask of the model (and the pre-trained models) require the one-hot encoding


class OpponentPool:
    """
    Frozen opponents of the learner. With prioritised fictitious self-play (PFSP), opponents against which the learner
    wins less often are sampled more often: p(opponent) ~ (1 - win rate) ^ exponent.
    """

    def __init__(self, fixed, sampling="uniform", exponent=2.0, max_snapshots=4, momentum=0.95):
        # fixed opponents (pre-trained model, rule-based player) are never removed
        self.opponents = list(fixed)
        self.n_fixed = len(fixed)
        self.win_rates = [0.5] * len(fixed)
        self.sampling = sampling
        self.exponent = exponent
        self.max_snapshots = max_snapshots
        self.momentum = momentum

    def __len__(self):
        return len(self.opponents)

    def sample(self, rng) -> int:
        if self.sampling == "pfsp":
            weights = (1.0 - np.asarray(self.win_rates)) ** self.exponent + 1e-3
            return int(rng.choice(len(self.opponents), p=weights / weights.sum()))
        return int(rng.integers(len(self.opponents)))

    def update(self, index: int, learner_won: bool) -> None:
        self.win_rates[index] = self.momentum * self.win_rates[index] + (1 - self.momentum) * float(learner_won)

    def add_snapshot(self, model) -> None:
        self.opponents.append(model)
        self.win_rates.append(0.5)
        if len(self.opponents) - self.n_fixed > self.max_snapshots:
            # drop the oldest snapshot
            self.opponents.pop(self.n_fixed)
            self.win_rates.pop(self.n_fixed)


class ReturnFilter:
    """
    Selective expert iteration: only keeps trajectories of the learner whose return reaches the running quantile of
    the recent returns of the same role (declarer or defender).
    """

    def __init__(self, quantile: float, window: int = 2000):
        self.quantile = quantile
        self.returns = {"declarer": deque(maxlen=window), "defender": deque(maxlen=window)}

    def keep(self, trajectory, role) -> bool:
        ret = float(np.sum(trajectory["rewards"]))
        history = self.returns[role]
        keep = len(history) < 50 or ret >= np.quantile(history, self.quantile)
        history.append(ret)
        return keep


def collect_games(learner, pool, n_games, args, rng, game_counter, declaration=None, declaration_examples=None,
                  return_filter=None):
    """
    Plays n_games self-play games and returns the learner's trajectories and statistics.

    :param pool: the OpponentPool.
    :param declaration: optional bidding policy to start the games.
    :param declaration_examples: list collecting (hand, declaration, achieved points) of every seat.
    :param return_filter: optional ReturnFilter for the learner's trajectories.
    """
    learner.eval()
    trajectories = []
    stats = {"declarer_points": [], "declarer_won": [], "learner_declarer_points": [], "learner_defender_won": [],
             "mask_mismatches": 0, "decisions": 0, "filtered": 0}

    envs, seats, first_states, targets, learner_seats, opponent_seats, choices = [], [], [], [], [], [], []
    for _ in range(n_games):
        # seeded deals for reproducibility, disjoint from the evaluation seeds
        random.seed(args["seed"] * 1_000_003 + game_counter)
        game_counter += 1

        env = Env(CARD_ENC)
        first_states.append(env.online_reset(point_rewards=args["point_rewards"], declaration=declaration))
        choices.append(list(declaration.last_choices) if declaration is not None else None)

        is_learner = rng.random(3) < args["learner_seat_prob"]
        if not is_learner.any():
            is_learner[rng.integers(3)] = True
        opponents = {pl: pool.sample(rng) for pl in range(3) if not is_learner[pl]}
        seat_models = [learner if is_learner[pl] else pool.opponents[opponents[pl]] for pl in range(3)]

        envs.append(env)
        seats.append(seat_models)
        targets.append(compute_target_returns(env, args["point_rewards"], args["rtg_scale"], seat_models))
        learner_seats.append(is_learner)
        opponent_seats.append(opponents)

    # all games are played in lockstep with batched predictions
    results = play_games(envs, seats, first_states, targets, greedy=False, temperature=args["temperature"], rng=rng,
                         reward_convention=args["reward_convention"])

    for result, is_learner, opponents, game_choices in zip(results, learner_seats, opponent_seats, choices):
        declarer_id = result["declarer_id"]
        for pl in range(3):
            if not is_learner[pl]:
                continue
            role = "declarer" if pl == declarer_id else "defender"
            if return_filter is None or return_filter.keep(result["trajectories"][pl], role):
                trajectories.append(result["trajectories"][pl])
            else:
                stats["filtered"] += 1

        # the outcome against the opponents if the learner only plays on one side
        learner_sides = {pl == declarer_id for pl in range(3) if is_learner[pl]}
        if len(learner_sides) == 1:
            learner_is_declarer = learner_sides.pop()
            learner_won = result["declarer_won"] == learner_is_declarer
            for pl, index in opponents.items():
                # only opponents of the other party
                if (pl == declarer_id) != learner_is_declarer:
                    pool.update(index, learner_won)

        if game_choices is not None and declaration_examples is not None:
            # hindsight: the declaration of each seat is labelled with the points it achieved
            declaration_examples += [(state, index, result["game_points"][pl])
                                     for pl, (state, index) in enumerate(game_choices)]

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

    head_kwargs = {}
    if args["hidden_card_head"]:
        head_kwargs = {"hidden_card_head": True, "hidden_loss_weight": args["hidden_loss_weight"]}
    learner = load_trainable_dt(args["pretrained_model"], use_mask=True, **head_kwargs).to(device)
    if learner.config.state_dim != state_dim:
        raise ValueError("The pre-trained model does not use the one-hot state representation.")

    # returns-to-go relabelled by a return model if the DT was trained with them
    rtg_relabeler = None
    return_model = args["return_model"] or getattr(learner.config, "return_model", None)
    if getattr(learner.config, "rtg_mode", "raw") != "raw" and return_model is not None:
        rtg_relabeler = ReturnModel.load(file_check(os.path.basename(os.path.normpath(return_model)))).relabel

    # the frozen pre-trained model is the first opponent and the baseline of the evaluation
    pretrained = copy.deepcopy(learner).eval()
    fixed_opponents = [pretrained] + ([RuleAgent()] if args["rule_opponent"] else [])
    pool = OpponentPool(fixed_opponents, sampling=args["opponent_sampling"], exponent=args["pfsp_exponent"],
                        max_snapshots=max(args["pool_size"] - len(fixed_opponents), 0))
    return_filter = ReturnFilter(args["buffer_filter_quantile"]) if args["buffer_filter_quantile"] > 0 else None

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

    # optional bidding with a declaration model, fine-tuned on its own outcomes
    declaration, declaration_baseline, declaration_optimizer = None, None, None
    declaration_examples = []
    if args["declaration_model"] is not None:
        declaration_model = DeclarationDT.from_pretrained(args["declaration_model"], use_safetensors=True).to(device)
        declaration = DeclarationPolicy(declaration_model, greedy=not args["finetune_declaration"],
                                        temperature=args["temperature"], rng=rng)
        # the evaluation bids with the initial model, so that the deals stay comparable
        declaration_baseline = DeclarationPolicy(copy.deepcopy(declaration_model).eval(), greedy=True)
        if args["finetune_declaration"]:
            declaration_optimizer = torch.optim.AdamW(declaration_model.parameters(), lr=args["learning_rate"])

    collator = None
    game_counter = 0

    for iteration in range(1, args["iterations"] + 1):
        start = time.time()

        # 1. self-play with the stochastic policy of the learner
        trajectories, stats, game_counter = collect_games(learner, pool, args["games_per_iter"], args, rng,
                                                          game_counter, declaration, declaration_examples,
                                                          return_filter)
        buffer.extend(trajectories)
        collect_time = time.time() - start

        # 2. ODT updates on the replay buffer
        if collator is None:
            collator = DecisionTransformerSkatDataCollator(buffer, args["batch_size"], augment_prob=args["augment_prob"],
                                                           seed=args["seed"], rtg_relabeler=rtg_relabeler)

        update_logs = [updater.step(collator(buffer.trajectories)) for _ in range(args["updates_per_iter"])]

        if declaration_optimizer is not None and declaration_examples:
            declaration_examples = declaration_examples[-args["buffer_size"]:]
            declaration_losses = []
            declaration.model.train()
            for _ in range(args["updates_per_iter"]):
                batch = [declaration_examples[k] for k in rng.integers(len(declaration_examples),
                                                                       size=args["batch_size"])]
                loss = declaration.model.loss(torch.from_numpy(np.stack([b[0] for b in batch])),
                                              torch.tensor([b[1] for b in batch]),
                                              torch.tensor([b[2] for b in batch], dtype=torch.float32))
                declaration_optimizer.zero_grad()
                loss.backward()
                declaration_optimizer.step()
                declaration_losses.append(loss.item())
            declaration.model.eval()
            writer.add_scalar("train/declaration_loss", float(np.mean(declaration_losses)), iteration)

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
        writer.add_scalar("buffer/filtered", stats["filtered"], step)
        for index, win_rate in enumerate(pool.win_rates):
            writer.add_scalar(f"pool/learner_win_rate_{index}", win_rate, step)
        writer.add_scalar("time/collect_s", collect_time, step)
        writer.add_scalar("time/iteration_s", time.time() - start, step)

        print(f"Iteration {iteration}: nll {np.mean([log['nll'] for log in update_logs]):.4f}, "
              f"entropy {np.mean([log['entropy'] for log in update_logs]):.4f}, "
              f"temperature {updater.temperature:.4f}, "
              f"self-play declarer win rate {np.mean(stats['declarer_won']):.3f}, "
              f"buffer {len(buffer)}, {time.time() - start:.1f}s")

        # 3. add a frozen snapshot of the learner to the pool of opponents
        if args["snapshot_every"] > 0 and iteration % args["snapshot_every"] == 0:
            pool.add_snapshot(copy.deepcopy(learner).eval())

        # 4. duplicate evaluation against the pre-trained model
        if args["eval_games"] > 0 and (iteration % args["eval_every"] == 0 or iteration == args["iterations"]):
            learner.eval()
            metrics = duplicate_evaluation(learner, pretrained, args["eval_games"], seed=args["eval_seed"],
                                           point_rewards=args["point_rewards"], rtg_scale=args["eval_rtg_scale"],
                                           card_enc=CARD_ENC, reward_convention=args["reward_convention"],
                                           declaration=declaration_baseline)
            for key, value in metrics.items():
                writer.add_scalar(f"eval/{key}", value, step)
            print("Evaluation against the pre-trained model: " +
                  ", ".join(f"{key} {value:.3f}" for key, value in metrics.items()))

        if args["save_every"] > 0 and iteration % args["save_every"] == 0:
            learner.save_pretrained(os.path.join(output_dir, f"iteration_{iteration}"))
            if declaration_optimizer is not None:
                declaration.model.save_pretrained(os.path.join(output_dir, f"declaration_iteration_{iteration}"))

    learner.save_pretrained(os.path.join(output_dir, "final"))
    if declaration_optimizer is not None:
        declaration.model.save_pretrained(os.path.join(output_dir, "declaration_final"))
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
    parser.add_argument('--hidden_card_head', type=str2bool, default=False,
                        help='Train an auxiliary head predicting the unseen cards (added if the model has none).')
    parser.add_argument('--hidden_loss_weight', type=float, default=0.5)
    parser.add_argument('--augment_prob', type=float, default=0.0,
                        help='Probability to permute the suits of a sampled game.')
    parser.add_argument('--return_model', type=file_check, default=None,
                        help="Return model in 'pretrained_models' to relabel the returns-to-go. Defaults to the one "
                             "the DT was trained with (config.return_model).")
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
    parser.add_argument('--declaration_model', type=file_check, default=None,
                        help="Declaration model in 'pretrained_models' for the auction. Without it, the player with "
                             "the best cards plays a suit game.")
    parser.add_argument('--finetune_declaration', type=str2bool, default=False,
                        help='Sample declarations and fine-tune the declaration model on the achieved points.')
    parser.add_argument('--learner_seat_prob', type=float, default=0.8,
                        help='Probability that a seat is played by the learner instead of a frozen opponent.')
    parser.add_argument('--snapshot_every', type=int, default=10,
                        help='Add a frozen snapshot of the learner to the opponent pool every n iterations.')
    parser.add_argument('--pool_size', type=int, default=6,
                        help='Maximum number of frozen opponents including the pre-trained model and the rule-based '
                             'player.')
    parser.add_argument('--rule_opponent', type=str2bool, default=True,
                        help='Add a rule-based player to the pool of opponents.')
    parser.add_argument('--opponent_sampling', type=str, default='pfsp', choices=['uniform', 'pfsp'],
                        help='Sample opponents uniformly or prioritised by the losses of the learner against them.')
    parser.add_argument('--pfsp_exponent', type=float, default=2.0)
    parser.add_argument('--buffer_filter_quantile', type=float, default=0.0,
                        help='Selective expert iteration: only add learner trajectories whose return reaches this '
                             'quantile of the recent returns of their role (0 keeps all).')
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
