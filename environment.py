import numpy as np

from card_representation_conversion import convert_one_hot_to_card, convert_one_hot_to_vector, convert_card_to_vec, \
    convert_vec_to_card
from data_pipeline import get_trump
from game.game import Game
from game.game_state_machine import GameStateMachine
from game.state.game_state_start import GameStateStart
from game.state.game_state_play import PlayCardAction, SurrenderAction

from model.player import Player


# position co-player (3) + trump (4) + last trick (3 * act_dim) + open cards (2 * act_dim) + hand cards (12 * act_dim)
state_dim = 92

# card representation is a vector
act_dim = 5


class Env:
    def __init__(self):
        # self.device = torch.device("cuda")
        # Name the players with placeholders to recognize them during evaluation and debugging
        self.player1 = Player(1, "Alice")
        self.player2 = Player(2, "Bob")
        self.player3 = Player(3, "Carol")
        self.game = Game([self.player1, self.player2, self.player3])
        self.state_machine = GameStateMachine(GameStateStart(self.game))
        # self.state_machine.handle_action(StartGameAction())
        self.action_space = act_dim
        self.observation_space = state_dim

    # def _get_state(self):
    #     state =
    #     return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

    def reset(self):
        # return torch.stack(list(self.state_buffer), 0)
        self.game.reset()

    def step(self, card, current_player):
        # if the action is surrendering
        if card[0] == -2:
            self.state_machine.handle_action(SurrenderAction(player=current_player))
            reward = current_player.current_trick_points
            # pad the game state with 0s as a game-terminating signal
            game_state = [[0] * state_dim]  # * (12 - self.game.round)]
            self.reset()
            return game_state, reward

        # pass action to the game state machine
        self.state_machine.handle_action(PlayCardAction(player=self.player1, card=convert_vec_to_card(card)))

        # update the reward, only the last points of the trick are relevant
        reward = self.player1.current_trick_points

        pos_p = [0, 0, 0]

        # determine the position of players
        if self.game.get_declarer() is current_player:
            pos_p[current_player.get_id()] = 0
            pos_p[(current_player.get_id() + 1) % 3] = -1
            pos_p[(current_player.get_id() + 2) % 3] = -1
        elif self.game.get_declarer().get_id() == (current_player.get_id() + 1 % 3):
            pos_p[current_player.get_id()] = 0
            pos_p[(current_player.get_id() + 1) % 3] = -1
            pos_p[(current_player.get_id() + 2) % 3] = 1
        else:
            pos_p[current_player.get_id()] = 0
            pos_p[(current_player.get_id() + 1) % 3] = 1
            pos_p[(current_player.get_id() + 2) % 3] = -1

        # get the current trump
        trump_enc = get_trump(self.game.game_variant.get_trump())

        # get the cards of the last trick and convert them to the card representation
        last_trick = [convert_card_to_vec(card) for card in self.game.get_last_trick_cards()] if self.game.round != 0 \
            else [[0] * act_dim, [0] * act_dim]

        # get the open cards and convert them to the card representation
        open_cards = [convert_card_to_vec(card) for card in self.game.trick.get_open_cards()]

        # update hand cards
        hand_cards = [convert_card_to_vec(card) for card in current_player.cards]

        game_state = np.concatenate([pos_p, trump_enc, last_trick, open_cards, hand_cards], axis=None)

        # Return state, reward
        return game_state, reward

    # def train(self):
    #     self.training = True
    #
    # def eval(self):
    #     self.training = False

