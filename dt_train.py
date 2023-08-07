import random
from dataclasses import dataclass
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from torch import nn

from transformers import DecisionTransformerModel, TrainingArguments, Trainer, DecisionTransformerConfig
import data_pipeline
import environment

# %%
print("Loading data...")
data, one_game = data_pipeline.get_states_actions_rewards(amount_games=10,
                                                          # point_rewards=True
                                                          game_index=8)

# %%
dataset = DatasetDict({"train": Dataset.from_dict(data)})

# %%

state_dim = 92
act_dim = 12
card_dim = 5


# device = torch.device("cuda") # "cpu"


def get_batch_ind():  # game_length
    # picks game indices
    # this picks the same game over *times* times
    # (WC GameID 4: Agent sits in rear hand as soloist)
    times = 100
    return np.tile(np.arange(8, 9), times)

    # return np.random.choice(
    #     np.arange(n_traj),
    #     size=batch_size,
    #     replace=True,
    #     p=p_sample,  # reweights, so we sample according to timesteps
    # )


# from https://huggingface.co/blog/train-decision-transformers
@dataclass
class DecisionTransformerSkatDataCollator:
    return_tensors: str = "pt"
    max_len: int = 12  # subsets of the episode we use for training, our episode length is short
    state_dim: int = state_dim  # size of state space
    act_dim: int = act_dim  # size of action space
    max_ep_len: int = 12  # max episode length in the dataset
    scale: float = 12.0  # normalization of rewards/returns
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
        # weighted rewards are in the data set (get_states_actions_rewards)
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum

    def __call__(self, features):
        batch_size = len(features)

        # Done: We want one game as a whole in a batch v
        # Done: normalization v
        # Done: rtg, timesteps and mask v
        # Done: scale rewards
        # Done: find a suit game which is easily won v

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
            si = random.randint(0, len(feature["rewards"]) - 1)  # 0

            # get sequences from dataset
            s.append(np.array(feature["states"]
                              [si: self.max_len]).reshape((1, -1, self.state_dim)))
            a.append(np.array(feature["actions"][si:self.max_len]).reshape((1, -1, self.act_dim)))
            r.append(np.array(feature["rewards"][si:self.max_len]).reshape((1, -1, 1)))

            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= self.max_ep_len] = self.max_ep_len - 1  # padding cutoff
            rtg.append(
                self._discount_cumsum(np.array(feature["rewards"][si:]), gamma=1.0)[: s[-1].shape[1]
                ].reshape(1, -1, 1)  # TODO check the +1 removed here
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

            rtg[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), rtg[-1]], axis=1)  # / self.scale
            timesteps[-1] = np.concatenate([np.zeros((1, self.max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, self.max_len - tlen)), np.ones((1, tlen))], axis=1))
            # TODO: mask could stay the same, what is s[-1]?

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

        # mse
        # loss = torch.mean((action_preds - action_targets) ** 2)

        # L-2 norm
        # loss = torch.linalg.norm(action_preds - action_targets, ord=2)

        # cross entropy loss
        loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(action_preds, action_targets)

        # manual logging
        # writer.add_scalar("Loss/train", loss,)  # problem: get episode

        return {"loss": loss}

    def original_forward(self, **kwargs):
        return super().forward(**kwargs)


configuration = DecisionTransformerConfig(state_dim=state_dim,  # each state consist out of
                                          act_dim=act_dim,  # each action consists out of one played card ()
                                          max_ep_len=12,  # each episode is a game -> 12 tuples of s,a,r make up 1 game
                                          vocab_size=32,  # there are 32 cards + pos_tp + trump_enc + surrender
                                          # TODO: other encodings (like pos_tp)?
                                          )
# TODO: how is the vocabulary defined?

# model = DecisionTransformerModel(configuration)

collator = DecisionTransformerSkatDataCollator(dataset["train"])

config = DecisionTransformerConfig(state_dim=collator.state_dim, act_dim=collator.act_dim)
model = TrainableDT(config)

# model.to(device)

# logging_files_name = "dt_training_{}_{}.log"

training_args = TrainingArguments(
    report_to=["tensorboard"],
    output_dir="training_output/",
    remove_unused_columns=False,
    num_train_epochs=200,
    per_device_train_batch_size=64,
    learning_rate=1e-4,
    weight_decay=1e-4,
    warmup_ratio=0.1,
    optim="adamw_torch",
    max_grad_norm=0.25,
    logging_steps=10,
    logging_dir="./training-logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    # eval_dataset=dataset["train"][8],
    data_collator=collator,
    # callbacks=[tensorboard_callback]
)

# writer = SummaryWriter()

print("Training...")
trainer.train()

# for tensorboard visualization:
# 1. rm -r ./training-logs/*
# 2. run training
# 3. tensorboard --logdir=./training-logs


# # Training loop with episode tracking
# current_episode = 0
# for epoch in range(round(training_args.num_train_epochs)):
#     for batch in trainer.get_train_dataloader():
#         # Increment episode number
#         current_episode += 1
#
#         # Perform forward and backward passes
#         loss = trainer.training_step(model, batch)
#
#         trainer.optimizer.step()
#         trainer.lr_scheduler.step()
#
#         writer.add_scalar("Loss/episode", loss, epoch)
#
#         # Log the episode number and loss
#         trainer.log_metrics({"episode": current_episode, "loss": loss.item()})
#
#     # Save the model at the end of each epoch
#     trainer.save_model()
#
# # Save the final model
# trainer.save_model()

# writer.flush()

# %%

# select available cudas for faster matrix computation
# device = torch.device("cuda")

TARGET_RETURN = 102

# evaluation
model = model.to("cpu")
model.eval()

env = environment.Env()


# Function that gets an action from the model using autoregressive prediction
# with a window of the previous 20 timesteps.
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


MAX_EPISODE_LENGTH = 12
scale = 1

# state_mean = np.array()
# state_std = np.array()
#
# state_mean = torch.from_numpy(state_mean)
# state_std = torch.from_numpy(state_std)

state_mean = collator.state_mean.astype(np.float32)
state_std = collator.state_std.astype(np.float32)

state_mean = torch.from_numpy(state_mean).to(device="cpu")
state_std = torch.from_numpy(state_std).to(device="cpu")

# build the environment for the evaluation
state = env.reset(current_player_id=3, game_env=one_game)  # game_states=dataset['train'][8]['states'])  # TODO
target_return = torch.tensor(TARGET_RETURN).float().reshape(1, 1)
states = torch.from_numpy(state).reshape(1, state_dim).float()
actions = torch.zeros((0, act_dim)).float()
rewards = torch.zeros(0).float()
timesteps = torch.tensor(0).reshape(1, 1).long()

# take steps in the environment (evaluation, not training)
for t in range(MAX_EPISODE_LENGTH):
    # add zeros for actions as input for the current time-step
    actions = torch.cat([actions, torch.zeros((1, act_dim))], dim=0)
    rewards = torch.cat([rewards, torch.zeros(1)])

    # predicting the action to take
    action = get_action(model,
                        states,  # - state_mean) / state_std,
                        actions,
                        rewards,
                        target_return,
                        timesteps)

    print(f"action {t}: {action}")
    actions[-1] = action
    action = action.detach().numpy()

    # hand cards within the state are padded from right to left after each action
    # mask the action
    action[-t:] = 0

    valid_actions = action[:MAX_EPISODE_LENGTH - t]

    # TODO: mask to be only able to play legal cards

    # subset size of 6, wants to select a card already played/out of bounds: IndexError: list index out of range

    # when increasing the subset size from 6 to 12, 2nd Skat is put wrongly and Agent wants to make an invalid move

    # get the index of the card with the highest probability
    card_index = np.argmax(valid_actions)

    # only select the best card
    action[:] = 0
    action[card_index] = 1

    # interact with the environment based on this action
    state, reward, done = env.step(tuple(action))  # TODO

    cur_state = torch.from_numpy(state).reshape(1, state_dim)
    states = torch.cat([states, cur_state], dim=0)
    rewards[-1] = reward

    pred_return = target_return[0, -1] - (reward / scale)
    target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
    timesteps = torch.cat([timesteps, torch.ones((1, 1)).long() * (t + 1)], dim=1)

    if done:
        break
