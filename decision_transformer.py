import torch
import pandas as pd
from transformers import DecisionTransformerModel, DecisionTransformerConfig

from game.game import Game
from game.game_state_machine import GameStateMachine
from game.state.game_state_start import GameStateStart, StartGameAction

from model.player import Player
from model.card import Card

from transformers import TrainingArguments
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import Trainer


class Env:
    def __init__(self, current_device):
        self.device = current_device
        self.player1 = Player(1, "Alice")
        self.player2 = Player(2, "Bob")
        self.player3 = Player(3, "Carol")
        self.game = Game([self.player1, self.player2, self.player3])
        self.state_machine = GameStateMachine(GameStateStart(self.game))
        self.state_machine.handle_action(StartGameAction())
        self.bidding = True
        self.training = True  # Consistent with model training mode
        all_cards = [
            [0, 0, 0, 1, 7, 0],  # A♣
            [0, 0, 0, 1, 5, 0],  # K♣
            [0, 0, 0, 1, 4, 0],  # Q♣
            [0, 0, 0, 1, 8, 1],  # J♣
            [0, 0, 0, 1, 6, 0],  # 10♣
            [0, 0, 0, 1, 3, 0],  # 9♣
            [0, 0, 0, 1, 2, 0],  # 8♣
            [0, 0, 0, 1, 1, 0],  # 7♣
            [0, 0, 0, 1, 7, 0],  # A♠
            [0, 0, 0, 1, 5, 0],  # K♠
            [0, 0, 1, 0, 4, 0],  # Q♠
            [0, 0, 1, 0, 8, 1],  # J♠
            [0, 0, 1, 0, 6, 0],  # 10♠
            [0, 0, 1, 0, 3, 0],  # 9♠
            [0, 0, 1, 0, 2, 0],  # 8♠
            [0, 0, 1, 0, 1, 0],  # 7♠
            [0, 1, 0, 0, 7, 0],  # A♥
            [0, 1, 0, 0, 5, 0],  # K♥
            [0, 1, 0, 0, 4, 0],  # Q♥
            [0, 1, 0, 0, 8, 1],  # J♥
            [0, 1, 0, 0, 6, 0],  # 10♥
            [0, 1, 0, 0, 3, 0],  # 9♥
            [0, 1, 0, 0, 2, 0],  # 8♥
            [0, 1, 0, 0, 1, 0],  # 7♥
            [1, 0, 0, 0, 7, 0],  # A♦
            [1, 0, 0, 0, 5, 0],  # K♦
            [1, 0, 0, 0, 4, 0],  # Q♦
            [1, 0, 0, 0, 8, 1],  # J♦
            [1, 0, 0, 0, 6, 0],  # 10♦
            [1, 0, 0, 0, 3, 0],  # 9♦
            [1, 0, 0, 0, 2, 0],  # 8♦
            [1, 0, 0, 0, 1, 0]   # 7♦
        ]
        self.action_space = all_cards
        self.observation_space = [[], all_cards, [all_cards, all_cards]]

    # def _get_state(self):
    #     state =
    #     return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

    def reset(self):
        # return torch.stack(list(self.state_buffer), 0)
        self.game.reset()

    def step(self, action):
        reward = 0

        # if isinstance(action, ):
        #     self.state_machine.handle_action(action)
        # else:

        self.state_machine.handle_action(action)
        reward += self.player1.current_trick_points  # only the last points of the trick are relevant

        # Return state, reward
        return torch.stack(self.state_machine.current_state), reward

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


# from datasets import load_dataset


# select available cudas for faster matrix computation
device = torch.device("cuda")

# one could use pretrained, but in our case we need our own model
# model = DecisionTransformerModel.from_pretrained("edbeeching/decision-transformer-gym-hopper-medium")
# model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

configuration = DecisionTransformerConfig()
model = DecisionTransformerModel()

training_args = TrainingArguments(
    output_dir="training_output",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
)

skat_data_path = ""  # TODO: set path independently

skat_data_frame = pd.read_csv(skat_data_path)

dataset = skat_data_frame

# the dataset is already tokenized in the database

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)  # doctest: +SKIP

# evaluation
model = model.to(device)
model.eval()

env = Env(device)
# env = gym.make("Hopper-v3")
state_dim = env.observation_space
act_dim = env.action_space

TARGET_RETURN = 61


# state = env.reset()
# states = torch.from_numpy(state).reshape(1, 1, state_dim).to(device=device, dtype=torch.float32)
# actions = torch.zeros((1, 1, act_dim), device=device, dtype=torch.float32)
# rewards = torch.zeros(1, 1, device=device, dtype=torch.float32)
# target_return = torch.tensor(TARGET_RETURN, dtype=torch.float32).reshape(1, 1)
# timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
# attention_mask = torch.zeros(1, 1, device=device, dtype=torch.float32)
#
# # forward pass
# with torch.no_grad():
#     state_preds, action_preds, return_preds = model(
#         states=states,
#         actions=actions,
#         rewards=rewards,
#         returns_to_go=target_return,
#         timesteps=timesteps,
#         attention_mask=attention_mask,
#         return_dict=False,
#     )


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
    state_preds, action_preds, return_preds = model(
        states=states,
        actions=actions,
        rewards=rewards,
        returns_to_go=returns_to_go,
        timesteps=timesteps,
        attention_mask=attention_mask,
        return_dict=False, )
    return action_preds[0, -1]
