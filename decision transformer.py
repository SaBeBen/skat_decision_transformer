import torch
import pandas as pd
from transformers import DecisionTransformerModel, DecisionTransformerConfig

from game.game import Game
from game.game_state_machine import GameStateMachine
from game.state.game_state_start import GameStateStart, StartGameAction
from model.player import Player

from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import Trainer

# from datasets import load_dataset


# select available cudas for faster matrix computation
device = torch.device("cuda")

# one could use pretrained, but in our case we need our own model
# model = DecisionTransformerModel.from_pretrained("edbeeching/decision-transformer-gym-hopper-medium")
# model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

configuration = DecisionTransformerConfig()
model = DecisionTransformerModel()


training_args = TrainingArguments(
    output_dir="trainingOutput",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
)


skat_data_path = ""

skat_data_frame = pd.read_csv(skat_data_path)

dataset = skat_data_frame

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_dataset(dataset):
    return tokenizer(dataset["text"])


dataset = dataset.map(tokenize_dataset, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)  # doctest: +SKIP

# evaluation
model = model.to(device)
model.eval()

env = gym.make("Hopper-v3")
state_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

state = env.reset()
states = torch.from_numpy(state).reshape(1, 1, state_dim).to(device=device, dtype=torch.float32)
actions = torch.zeros((1, 1, act_dim), device=device, dtype=torch.float32)
rewards = torch.zeros(1, 1, device=device, dtype=torch.float32)
target_return = torch.tensor(TARGET_RETURN, dtype=torch.float32).reshape(1, 1)
timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
attention_mask = torch.zeros(1, 1, device=device, dtype=torch.float32)

# forward pass
with torch.no_grad():
    state_preds, action_preds, return_preds = model(
        states=states,
        actions=actions,
        rewards=rewards,
        returns_to_go=target_return,
        timesteps=timesteps,
        attention_mask=attention_mask,
        return_dict=False,
    )


# ------------------------------------------------

class Env:
    def __init__(self, args):
        self.device = args.device
        player1 = Player(1, "Alice")
        player2 = Player(2, "Bob")
        player3 = Player(3, "Carol")
        self.game = Game([player1, player2, player3])
        self.state_machine = GameStateMachine(GameStateStart(self.game))
        self.state_machine.handle_action(StartGameAction())
        self.bidding = True
        self.training = True  # Consistent with model training mode

        # self.ale = atari_py.ALEInterface()
        # self.ale.setInt('random_seed', args.seed)
        # self.ale.setInt('max_num_frames_per_episode', args.max_episode_length)
        # self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
        # self.ale.setInt('frame_skip', 0)
        # self.ale.setBool('color_averaging', False)
        # self.ale.loadROM(atari_py.get_game_path(args.game))  # ROM loading must be done after setting options
        # actions = self.ale.getMinimalActionSet()

        # self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))

        # self.lives = 0  # Life counter (used in DeepMind training)
        # self.life_termination = False  # Used to check if resetting only from loss of life
        # self.window = args.history_length  # Number of frames to concatenate

        # self.state_buffer = deque([], maxlen=args.history_length)

    # def _get_state(self):
    #     state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
    #     return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

    # def _reset_buffer(self):
    #     for _ in range(self.window):
    #         self.state_buffer.append(torch.zeros(84, 84, device=self.device))

    def reset(self):
        # return torch.stack(list(self.state_buffer), 0)
        self.game.reset()

    def step(self, action):
        # TODO: handleAction in GameStateMachine?
        reward = 0

        # if self.bidding:
        #     self.state_machine.handle_action(action)
        # else:
        self.state_machine.handle_action(action)
        reward += self.game.  # self.ale.act(self.actions.get(action))

        # Return state, reward
        return torch.stack(self.state_machine.current_state), reward

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def action_space(self):
        return len(self.actions)
