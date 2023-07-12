import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DecisionTransformerModel, DecisionTransformerConfig

from game.game import Game
from game.game_state_machine import GameStateMachine
from game.state.game_state_start import GameStateStart, StartGameAction
from game.state.game_state_play import PlayCardAction

from model.player import Player
from model.card import Card

from transformers import TrainingArguments
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import Trainer

from db_app import app


def convert_vec_to_card(card):
    vector_rep = {
        [0, 0, 0, 1, 7, _]: Card(Card.Suit.CLUB, Card.Face.ACE),  # A♣
        [0, 0, 0, 1, 5, _]: Card(Card.Suit.CLUB, Card.Face.KING),  # K♣
        [0, 0, 0, 1, 4, _]: Card(Card.Suit.CLUB, Card.Face.QUEEN),  # Q♣
        [0, 0, 0, 1, 8, 1]: Card(Card.Suit.CLUB, Card.Face.JACK),  # J♣
        [0, 0, 0, 1, 6, _]: Card(Card.Suit.CLUB, Card.Face.TEN),  # 10♣
        [0, 0, 0, 1, 3, _]: Card(Card.Suit.CLUB, Card.Face.NINE),  # 9♣
        [0, 0, 0, 1, 2, _]: Card(Card.Suit.CLUB, Card.Face.EIGHT),  # 8♣
        [0, 0, 0, 1, 1, _]: Card(Card.Suit.CLUB, Card.Face.SEVEN),  # 7♣
        [0, 0, 0, 1, 7, _]: Card(Card.Suit.SPADE, Card.Face.ACE),  # A♠
        [0, 0, 0, 1, 5, _]: Card(Card.Suit.SPADE, Card.Face.KING),  # K♠
        [0, 0, 1, 0, 4, _]: Card(Card.Suit.SPADE, Card.Face.QUEEN),  # Q♠
        [0, 0, 1, 0, 8, 1]: Card(Card.Suit.SPADE, Card.Face.JACK),  # J♠
        [0, 0, 1, 0, 6, _]: Card(Card.Suit.SPADE, Card.Face.TEN),  # 10♠
        [0, 0, 1, 0, 3, _]: Card(Card.Suit.SPADE, Card.Face.NINE),  # 9♠
        [0, 0, 1, 0, 2, _]: Card(Card.Suit.SPADE, Card.Face.EIGHT),  # 8♠
        [0, 0, 1, 0, 1, _]: Card(Card.Suit.SPADE, Card.Face.SEVEN),  # 7♠
        [0, 1, 0, 0, 7, _]: Card(Card.Suit.HEARTS, Card.Face.ACE),  # A♥
        [0, 1, 0, 0, 5, _]: Card(Card.Suit.HEARTS, Card.Face.KING),  # K♥
        [0, 1, 0, 0, 4, _]: Card(Card.Suit.HEARTS, Card.Face.QUEEN),  # Q♥
        [0, 1, 0, 0, 8, 1]: Card(Card.Suit.HEARTS, Card.Face.JACK),  # J♥
        [0, 1, 0, 0, 6, _]: Card(Card.Suit.HEARTS, Card.Face.TEN),  # 10♥
        [0, 1, 0, 0, 3, _]: Card(Card.Suit.HEARTS, Card.Face.NINE),  # 9♥
        [0, 1, 0, 0, 2, _]: Card(Card.Suit.HEARTS, Card.Face.EIGHT),  # 8♥
        [0, 1, 0, 0, 1, _]: Card(Card.Suit.HEARTS, Card.Face.SEVEN),  # 7♥
        [1, 0, 0, 0, 7, _]: Card(Card.Suit.DIAMOND, Card.Face.ACE),  # A♦
        [1, 0, 0, 0, 5, _]: Card(Card.Suit.DIAMOND, Card.Face.KING),  # K♦
        [1, 0, 0, 0, 4, _]: Card(Card.Suit.DIAMOND, Card.Face.QUEEN),  # Q♦
        [1, 0, 0, 0, 8, 1]: Card(Card.Suit.DIAMOND, Card.Face.JACK),  # J♦
        [1, 0, 0, 0, 6, _]: Card(Card.Suit.DIAMOND, Card.Face.TEN),  # 10♦
        [1, 0, 0, 0, 3, _]: Card(Card.Suit.DIAMOND, Card.Face.NINE),  # 9♦
        [1, 0, 0, 0, 2, _]: Card(Card.Suit.DIAMOND, Card.Face.EIGHT),  # 8♦
        [1, 0, 0, 0, 1, _]: Card(Card.Suit.DIAMOND, Card.Face.SEVEN)  # 7♦
    }
    converted_card = vector_rep[card]

    return converted_card


def convert_one_hot_to_card(card):
    vector_rep = {
        0: Card(Card.Suit.CLUB, Card.Face.ACE),  # A♣
        1: Card(Card.Suit.CLUB, Card.Face.KING),  # K♣
        2: Card(Card.Suit.CLUB, Card.Face.QUEEN),  # Q♣
        3: Card(Card.Suit.CLUB, Card.Face.JACK),  # J♣
        4: Card(Card.Suit.CLUB, Card.Face.TEN),  # 10♣
        5: Card(Card.Suit.CLUB, Card.Face.NINE),  # 9♣
        6: Card(Card.Suit.CLUB, Card.Face.EIGHT),  # 8♣
        7: Card(Card.Suit.CLUB, Card.Face.SEVEN),  # 7♣
        8: Card(Card.Suit.SPADE, Card.Face.ACE),  # A♠
        9: Card(Card.Suit.SPADE, Card.Face.KING),  # K♠
        10: Card(Card.Suit.SPADE, Card.Face.QUEEN),  # Q♠
        11: Card(Card.Suit.SPADE, Card.Face.JACK),  # J♠
        12: Card(Card.Suit.SPADE, Card.Face.TEN),  # 10♠
        13: Card(Card.Suit.SPADE, Card.Face.NINE),  # 9♠
        14: Card(Card.Suit.SPADE, Card.Face.EIGHT),  # 8♠
        15: Card(Card.Suit.SPADE, Card.Face.SEVEN),  # 7♠
        16: Card(Card.Suit.HEARTS, Card.Face.ACE),  # A♥
        17: Card(Card.Suit.HEARTS, Card.Face.KING),  # K♥
        18: Card(Card.Suit.HEARTS, Card.Face.QUEEN),  # Q♥
        19: Card(Card.Suit.HEARTS, Card.Face.JACK),  # J♥
        20: Card(Card.Suit.HEARTS, Card.Face.TEN),  # 10♥
        21: Card(Card.Suit.HEARTS, Card.Face.NINE),  # 9♥
        22: Card(Card.Suit.HEARTS, Card.Face.EIGHT),  # 8♥
        23: Card(Card.Suit.HEARTS, Card.Face.SEVEN),  # 7♥
        24: Card(Card.Suit.DIAMOND, Card.Face.ACE),  # A♦
        25: Card(Card.Suit.DIAMOND, Card.Face.KING),  # K♦
        26: Card(Card.Suit.DIAMOND, Card.Face.QUEEN),  # Q♦
        27: Card(Card.Suit.DIAMOND, Card.Face.JACK),  # J♦
        28: Card(Card.Suit.DIAMOND, Card.Face.TEN),  # 10♦
        29: Card(Card.Suit.DIAMOND, Card.Face.NINE),  # 9♦
        30: Card(Card.Suit.DIAMOND, Card.Face.EIGHT),  # 8♦
        31: Card(Card.Suit.DIAMOND, Card.Face.SEVEN)  # 7♦
    }
    converted_card = vector_rep[card]

    return converted_card


class Env:
    def __init__(self, current_device):
        self.device = current_device
        # Name the Players to recognize them during evaluation and debugging
        self.player1 = Player(1, "Alice")
        self.player2 = Player(2, "Bob")
        self.player3 = Player(3, "Carol")
        self.game = Game([self.player1, self.player2, self.player3])
        self.state_machine = GameStateMachine(GameStateStart(self.game))
        self.state_machine.handle_action(StartGameAction())
        self.bidding = True
        self.training = True  # Consistent with model training mode

        # the card representation: one token consists of one card which is encoded as a vector
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
            [1, 0, 0, 0, 1, 0]  # 7♦
        ]
        self.action_space = all_cards
        self.observation_space = [[], all_cards, [all_cards, all_cards]]

    # def _get_state(self):
    #     state =
    #     return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

    def reset(self):
        # return torch.stack(list(self.state_buffer), 0)
        self.game.reset()

    def step(self, card):
        reward = 0

        # if isinstance(action, ):
        #     self.state_machine.handle_action(action)
        # else:

        # pass action to the game state machine
        self.state_machine.handle_action(PlayCardAction(player=self.player1, card=convert_one_hot_to_card(card)))

        # update the reward
        reward += self.player1.current_trick_points  # only the last points of the trick are relevant

        # Return state, reward
        return torch.stack(self.state_machine.current_state), reward

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    # convert data to following encoding:
    # ♦, ♥, ♠, ♣, {7, 8, 9, Q, K, 10, A, J}, T


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

# TODO: exchange for sql query
# game_data = app.get_skat_data_wm()

skat_data_path = skat_wm_cs_data_path = "db_app/data/wm_skattisch_kf.CSV"

skat_data_frame = np.asarray(np.loadtxt(skat_wm_cs_data_path, delimiter=",", dtype=int))

# Only use the relevant information: Skat and 30 played cards
game_data = skat_data_frame[:, 1:33].apply_along_axis(lambda card: convert_one_hot_to_card(card)).flatten()

# the dataset is already tokenized in the database

# split dataset into train and test
skat_train, skat_test = train_test_split(game_data, test_size=0.2, random_state=0)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=skat_train,
    eval_dataset=skat_test,
)  # doctest: +SKIP

# evaluation
model = model.to(device)
model.eval()

env = Env(device)
state_dim = 16
act_dim = 6

TARGET_RETURN = 61


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


# This was normalized during training
MAX_EPISODE_LENGTH = 1000
scale = 1

# state_mean = np.array(
#     [1.3490015, -0.11208222, -0.5506444, -0.13188992, -0.00378754, 2.6071432,
#      0.02322114, -0.01626922, -0.06840388, -0.05183131, 0.04272673, ])
#
# state_std = np.array(
#     [0.15980862, 0.0446214, 0.14307782, 0.17629202, 0.5912333, 0.5899924,
#      1.5405099, 0.8152689, 2.0173461, 2.4107876, 5.8440027, ])
#
# state_mean = torch.from_numpy(state_mean)
# state_std = torch.from_numpy(state_std)

state = env.reset()  # TODO
target_return = torch.tensor(TARGET_RETURN).float().reshape(1, 1)
states = torch.from_numpy(state).reshape(1, state_dim).float()
actions = torch.zeros((0, act_dim)).float()
rewards = torch.zeros(0).float()
timesteps = torch.tensor(0).reshape(1, 1).long()

# take steps in the environment
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
    actions[-1] = action
    action = action.detach().numpy()

    # interact with the environment based on this action
    state, reward, done, _ = env.step(action)  # TODO

    cur_state = torch.from_numpy(state).reshape(1, state_dim)
    states = torch.cat([states, cur_state], dim=0)
    rewards[-1] = reward

    pred_return = target_return[0, -1] - (reward / scale)
    target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
    timesteps = torch.cat([timesteps, torch.ones((1, 1)).long() * (t + 1)], dim=1)

    if done:
        break

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
