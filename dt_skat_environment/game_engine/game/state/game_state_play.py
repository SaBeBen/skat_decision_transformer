from dt_skat_environment.game_engine.exceptions import InvalidPlayerMove
from dt_skat_environment.game_engine.game.game import Game
from dt_skat_environment.game_engine.game.game_state_machine import GameState, PlayerAction, SurrenderAction
from dt_skat_environment.game_engine.game.state.game_state_end import GameStateEnd


# ------------------------------------------------------------
# Concrete game state class for playing the game
# ------------------------------------------------------------
class GameStatePlay(GameState):
    def __init__(self, game):
        super().__init__(game)
        self.game.round = 1

    def handle_action(self, action):
        if isinstance(action, PlayCardAction):
            self.play_card(action.player, action.card)
        elif isinstance(action, SurrenderAction):
            self.surrender(action.player)
        else:
            super().handle_action(action)

    def play_card(self, player, card):
        self.check_valid_card_play(player, card)

        # add the card to the trick
        self.game.trick.add(player, card)

        # remove the card from the players hand
        player.cards.remove(card)
        # print("Player " + player.name + " played " + str(card))

        if self.game.trick.is_complete():
            self.finish_trick()

    def check_valid_card_play(self, player, card):
        # check if player holding this card
        if not player.has_card(card):
            raise InvalidPlayerMove("Player " + player.name + " isn't holding the card " + str(card) + ".")
        # check if player already played a card to current trick
        if self.game.trick.has_already_played_card(player):
            raise InvalidPlayerMove("Player " + player.name + " already played a card to the trick.")
        # check if this is players turn or waiting for another player to play his card before
        if not self.game.trick.can_move(player):
            raise InvalidPlayerMove("It's not player " + player.name + "s move.")
        # check if player can play this card
        if not self.game.trick.is_valid_card_move(self.game.game_variant, player, card):
            raise InvalidPlayerMove("Card " + str(card) + " is not a valid move by player " + player.name)

    def finish_trick(self):
        self.game.finish_trick()
        # check if game is over
        if self.game.round is Game.MAX_ROUNDS:
            self.finish_game()
        else:
            # increase round
            self.game.round += 1

    def finish_game(self):
        self.handle_state_finished()

    def surrender(self, player):
        self.game.surrender(player)
        self.finish_game()

    def get_next_state(self):
        return GameStateEnd(self.game)


# ------------------------------------------------------------
# Concrete action classes
# ------------------------------------------------------------
class PlayCardAction(PlayerAction):
    def __init__(self, player, card):
        super().__init__(player)
        self.card = card


# class SurrenderAction(SurrenderAction):
#     def __init__(self, player: Player):
#         super().__init__(player)

