import timeit

setupSnippet = "from game.game import Game \n" \
               "from game.game_state_machine import GameStateMachine \n" \
               "from game.game_variant import GameVariantGrand \n" \
               "from game.state.game_state_bid import BidCallAction, BidAcceptAction, BidPassAction, PickUpSkatAction, PutDownSkatAction, DeclareGameVariantAction \n" \
               "from game.state.game_state_play import PlayCardAction \n" \
               "from game.state.game_state_start import GameStateStart, StartGameAction \n" \
               "from model.card import Card \n" \
               "from model.player import Player \n"

testedSnippet = '''
player1 = Player(1, "Alice")
player2 = Player(2, "Bob")
player3 = Player(3, "Carol")
game = Game([player1, player2, player3])
state_machine = GameStateMachine(GameStateStart(game))

state_machine.handle_action(StartGameAction())

state_machine.handle_action(BidCallAction(player3, 18))

state_machine.handle_action(BidAcceptAction(player2, 18))

state_machine.handle_action(BidCallAction(player3, 20))

state_machine.handle_action(BidPassAction(player2, 20))

state_machine.handle_action(BidPassAction(player1, 20))

state_machine.handle_action(PickUpSkatAction(player3))


state_machine.handle_action(PutDownSkatAction(player3, player3.cards[0:2]))

state_machine.handle_action(DeclareGameVariantAction(player3, GameVariantGrand()))


game.dealer = 1
game.trick.leader = player3
game.skat = [Card(Card.Suit.CLUB, Card.Face.EIGHT), Card(Card.Suit.CLUB, Card.Face.NINE)]
player3.cards = [Card(Card.Suit.SPADE, Card.Face.JACK), Card(Card.Suit.CLUB, Card.Face.TEN), Card(Card.Suit.DIAMOND, Card.Face.ACE), Card(Card.Suit.HEARTS, Card.Face.JACK), Card(Card.Suit.SPADE, Card.Face.ACE), Card(Card.Suit.HEARTS, Card.Face.ACE), Card(Card.Suit.HEARTS, Card.Face.TEN), Card(Card.Suit.SPADE, Card.Face.TEN), Card(Card.Suit.DIAMOND, Card.Face.KING), Card(Card.Suit.DIAMOND, Card.Face.EIGHT)]
player1.cards = [Card(Card.Suit.CLUB, Card.Face.JACK), Card(Card.Suit.CLUB, Card.Face.KING), Card(Card.Suit.DIAMOND, Card.Face.SEVEN), Card(Card.Suit.SPADE, Card.Face.SEVEN), Card(Card.Suit.SPADE, Card.Face.EIGHT), Card(Card.Suit.HEARTS, Card.Face.EIGHT), Card(Card.Suit.HEARTS, Card.Face.KING), Card(Card.Suit.SPADE, Card.Face.KING), Card(Card.Suit.CLUB, Card.Face.SEVEN), Card(Card.Suit.CLUB, Card.Face.QUEEN)]
player2.cards = [Card(Card.Suit.DIAMOND, Card.Face.JACK), Card(Card.Suit.CLUB, Card.Face.ACE), Card(Card.Suit.DIAMOND, Card.Face.TEN), Card(Card.Suit.HEARTS, Card.Face.SEVEN), Card(Card.Suit.SPADE, Card.Face.QUEEN), Card(Card.Suit.HEARTS, Card.Face.NINE), Card(Card.Suit.HEARTS, Card.Face.QUEEN), Card(Card.Suit.SPADE, Card.Face.NINE), Card(Card.Suit.DIAMOND, Card.Face.QUEEN), Card(Card.Suit.DIAMOND, Card.Face.NINE)]


state_machine.handle_action(PlayCardAction(player3, Card(Card.Suit.SPADE, Card.Face.JACK)))
state_machine.handle_action(PlayCardAction(player1, Card(Card.Suit.CLUB, Card.Face.JACK)))
state_machine.handle_action(PlayCardAction(player2, Card(Card.Suit.DIAMOND, Card.Face.JACK)))


state_machine.handle_action(PlayCardAction(player1, Card(Card.Suit.CLUB, Card.Face.KING)))
state_machine.handle_action(PlayCardAction(player2, Card(Card.Suit.CLUB, Card.Face.ACE)))
state_machine.handle_action(PlayCardAction(player3, Card(Card.Suit.CLUB, Card.Face.TEN)))

state_machine.handle_action(PlayCardAction(player2, Card(Card.Suit.DIAMOND, Card.Face.TEN)))
state_machine.handle_action(PlayCardAction(player3, Card(Card.Suit.DIAMOND, Card.Face.ACE)))
state_machine.handle_action(PlayCardAction(player1, Card(Card.Suit.DIAMOND, Card.Face.SEVEN)))

state_machine.handle_action(PlayCardAction(player3, Card(Card.Suit.HEARTS, Card.Face.JACK)))
state_machine.handle_action(PlayCardAction(player1, Card(Card.Suit.SPADE, Card.Face.SEVEN)))
state_machine.handle_action(PlayCardAction(player2, Card(Card.Suit.HEARTS, Card.Face.SEVEN)))

state_machine.handle_action(PlayCardAction(player3, Card(Card.Suit.SPADE, Card.Face.ACE)))
state_machine.handle_action(PlayCardAction(player1, Card(Card.Suit.SPADE, Card.Face.EIGHT)))
state_machine.handle_action(PlayCardAction(player2, Card(Card.Suit.SPADE, Card.Face.QUEEN)))

state_machine.handle_action(PlayCardAction(player3, Card(Card.Suit.HEARTS, Card.Face.ACE)))
state_machine.handle_action(PlayCardAction(player1, Card(Card.Suit.HEARTS, Card.Face.EIGHT)))
state_machine.handle_action(PlayCardAction(player2, Card(Card.Suit.HEARTS, Card.Face.NINE)))

state_machine.handle_action(PlayCardAction(player3, Card(Card.Suit.HEARTS, Card.Face.TEN)))
state_machine.handle_action(PlayCardAction(player1, Card(Card.Suit.HEARTS, Card.Face.KING)))
state_machine.handle_action(PlayCardAction(player2, Card(Card.Suit.HEARTS, Card.Face.QUEEN)))


state_machine.handle_action(PlayCardAction(player3, Card(Card.Suit.SPADE, Card.Face.TEN)))
state_machine.handle_action(PlayCardAction(player1, Card(Card.Suit.SPADE, Card.Face.KING)))
state_machine.handle_action(PlayCardAction(player2, Card(Card.Suit.SPADE, Card.Face.NINE)))


state_machine.handle_action(PlayCardAction(player3, Card(Card.Suit.DIAMOND, Card.Face.KING)))
state_machine.handle_action(PlayCardAction(player1, Card(Card.Suit.CLUB, Card.Face.SEVEN)))
state_machine.handle_action(PlayCardAction(player2, Card(Card.Suit.DIAMOND, Card.Face.QUEEN)))


state_machine.handle_action(PlayCardAction(player3, Card(Card.Suit.DIAMOND, Card.Face.EIGHT)))
state_machine.handle_action(PlayCardAction(player1, Card(Card.Suit.CLUB, Card.Face.QUEEN)))
state_machine.handle_action(PlayCardAction(player2, Card(Card.Suit.DIAMOND, Card.Face.NINE)))

'''

exTime = timeit.timeit(stmt=testedSnippet, setup=setupSnippet, number=10000)

print(f"Execution time of one run: {exTime/10000} ")
