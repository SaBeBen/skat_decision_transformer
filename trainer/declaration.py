"""
Return-conditioned bidding and game declaration.

A Decision Transformer with a single time step: given the dealt hand, the seat and a target return (the Seeger-Fabian
points the player wants to achieve), it predicts the declaration of the player (pass or a game variant, played from the
hand or with the Skat). As in the DT for the card play, conditioning on a high return selects the declarations that
lead to high scores with the given cards. The highest bid of a player is the value of its declaration.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from transformers import DecisionTransformerConfig, DecisionTransformerModel

from dt_skat_environment.card_representation_conversion import convert_numerical_to_card
from dt_skat_environment.environment import get_peaks
from dt_skat_environment.game_engine.game.game_variant import GameVariantNull
from dt_skat_environment.game_engine.model.card import Card

# pass, followed by (game, hand) with the game in the encoding of the data
DECLARATIONS: List[Optional[Tuple[int, bool]]] = [None] + [(game, hand) for game in (9, 10, 11, 12, 24, 23)
                                                           for hand in (False, True)]
DECLARATION_STATE_DIM = 32 + 3  # dealt hand + seat (fore-hand, middle-hand, rear-hand)

# the auction only allows these bids
BID_VALUES = [18, 20, 22, 23, 24, 27, 30, 33, 35, 36, 40, 44, 45, 46, 48, 50, 54, 55, 59, 60, 63, 66, 70, 72, 77, 80,
              81, 84, 88, 90, 96, 99, 100, 108, 110, 117, 120, 121, 126, 130, 132, 135, 140, 143, 144, 150, 153, 154,
              156, 160, 162, 165, 168, 170, 176, 180, 187, 192, 198, 204, 216, 240, 264]

# index of a card in the hand encoding (same numbering as the data)
CARD_INDEX = {convert_numerical_to_card(i): i for i in range(32)}


def declaration_index(game: int, hand: bool) -> int:
    # null ouvert is declared as null (hand)
    if game in (35, 59):
        game, hand = 23, True
    elif game == 46:
        game = 23
    return DECLARATIONS.index((game, bool(hand)))


def encode_hand(cards: List[Card], seat: int) -> np.ndarray:
    state = np.zeros(DECLARATION_STATE_DIM, dtype=np.float32)
    for card in cards:
        state[CARD_INDEX[card]] = 1
    state[32 + seat] = 1
    return state


def declaration_value(cards: List[Card], declaration: Optional[Tuple[int, bool]]) -> int:
    """The highest bid for a declaration: the game value with the dealt cards (the Skat is unknown)."""
    if declaration is None:
        return 0
    game, hand = declaration
    if game == 23:
        return GameVariantNull(hand=hand).get_value()
    if game == 24:
        base, peaks = 24, get_peaks(cards, None)
    else:
        suit = Card.Suit(game - 9)
        base, peaks = game, get_peaks(cards, suit)
    return base * (1 + peaks + hand)


def run_auction(limits: List[int]) -> Optional[Tuple[int, int]]:
    """
    Simulates the auction: middle-hand bids to fore-hand, afterwards rear-hand bids to the winner.
    A player keeps bidding (or holding) while the bid does not exceed its limit.

    :param limits: highest bid of fore-hand, middle-hand and rear-hand (0 to pass).
    :return: (declarer seat, bid) or None if all players pass.
    """

    def duel(listener, speaker, bid):
        for value in BID_VALUES:
            if value <= bid:
                continue
            if value > limits[speaker]:
                return listener, bid
            if value > limits[listener]:
                return speaker, value
            bid = value
        return listener, bid

    winner, bid = duel(0, 1, 0)
    winner, bid = duel(winner, 2, bid)

    if bid == 0:
        # nobody bid: the remaining player can play at 18
        if limits[winner] >= 18:
            return winner, 18
        return None
    return winner, bid


class DeclarationDT(DecisionTransformerModel):
    def __init__(self, config):
        super().__init__(config)

    @classmethod
    def create(cls, n_layer=1, n_head=1, hidden_size=64, dropout=0.1, return_scale=0.01):
        config = DecisionTransformerConfig(
            state_dim=DECLARATION_STATE_DIM,
            act_dim=len(DECLARATIONS),
            action_tanh=False,
            hidden_size=hidden_size,
            n_layer=n_layer,
            n_head=n_head,
            max_ep_len=1,
            n_positions=16,
            embd_pdrop=dropout,
            resid_pdrop=dropout,
            attn_pdrop=dropout,
        )
        # returns are scaled to keep the inputs small
        config.return_scale = return_scale
        # high return the model is conditioned on when declaring (set from the training data)
        config.target_return = 100.0
        return cls(config)

    def declaration_logits(self, states, returns):
        """
        :param states: (batch, DECLARATION_STATE_DIM) encoded hands and seats.
        :param returns: (batch,) returns to condition on (unscaled Seeger-Fabian points).
        :return: (batch, number of declarations) logits.
        """
        batch_size = states.shape[0]
        device = self.device
        _, action_preds, _ = super().forward(
            states=states.reshape(batch_size, 1, -1).to(device),
            actions=torch.zeros(batch_size, 1, self.config.act_dim, device=device),
            rewards=None,
            returns_to_go=(returns.reshape(batch_size, 1, 1) * self.config.return_scale).to(device).float(),
            timesteps=torch.zeros(batch_size, 1, dtype=torch.long, device=device),
            attention_mask=torch.ones(batch_size, 1, dtype=torch.long, device=device),
            return_dict=False,
        )
        return action_preds[:, 0]

    def loss(self, states, declarations, returns):
        logits = self.declaration_logits(states, returns)
        return nn.functional.cross_entropy(logits, declarations.to(self.device))

    @torch.no_grad()
    def declare(self, cards: List[Card], seat: int, target_return: Optional[float] = None, greedy: bool = True,
                temperature: float = 1.0, rng: Optional[np.random.Generator] = None) -> int:
        """:return: index of the declaration in DECLARATIONS."""
        target_return = self.config.target_return if target_return is None else target_return
        state = torch.from_numpy(encode_hand(cards, seat)).reshape(1, -1)
        logits = self.declaration_logits(state, torch.tensor([target_return], dtype=torch.float32))[0].cpu()
        if greedy:
            return int(torch.argmax(logits))
        probs = torch.softmax(logits / temperature, dim=0).numpy().astype(np.float64)
        rng = rng if rng is not None else np.random.default_rng()
        return int(rng.choice(len(DECLARATIONS), p=probs / probs.sum()))


def declarations_from_data(meta_and_cards: np.ndarray):
    """
    Training examples from the data: every seat of every game with its declaration (pass for the defenders) and the
    Seeger-Fabian points it achieved.

    :return: states (n, DECLARATION_STATE_DIM), declaration indices (n,), returns (n,)
    """
    states, declarations, returns = [], [], []
    for game in meta_and_cards:
        declarer_points = game[-5]
        for seat in range(3):
            cards = [convert_numerical_to_card(card) for card in game[4 + 10 * seat:14 + 10 * seat]]
            states.append(encode_hand(cards, seat))
            if game[1 + seat] == game[-13]:
                declarations.append(declaration_index(game[-12], game[-11]))
                returns.append(declarer_points)
            else:
                declarations.append(0)
                returns.append(40 if declarer_points < 0 else 0)
    return np.stack(states), np.asarray(declarations, dtype=np.int64), np.asarray(returns, dtype=np.float32)


def train_declaration_model(model: DeclarationDT, states, declarations, returns, epochs=20, batch_size=256,
                            learning_rate=1e-3, weight_decay=1e-4, target_quantile=0.9, log=print):
    """Offline training on (hand, declaration, return) examples with the cross-entropy of the declaration."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    states, declarations, returns = (torch.as_tensor(x) for x in (states, declarations, returns))
    n = len(states)

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n)
        losses = []
        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            loss = model.loss(states[idx], declarations[idx], returns[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        log(f"Declaration epoch {epoch + 1}: loss {np.mean(losses):.4f}")

    # condition on a high, but achievable return of declarers
    declarer_returns = returns[declarations > 0].numpy()
    if len(declarer_returns):
        model.config.target_return = float(np.quantile(declarer_returns, target_quantile))
    model.eval()
    return model


class DeclarationPolicy:
    """Bidding of all three seats with a declaration model, used to start games in Env.online_reset."""

    def __init__(self, model: DeclarationDT, greedy: bool = True, temperature: float = 1.0,
                 rng: Optional[np.random.Generator] = None):
        self.model = model
        self.greedy = greedy
        self.temperature = temperature
        self.rng = rng
        # (encoded hand, declaration index) of every seat in the last deal, to learn from the outcome
        self.last_choices = []

    def __call__(self, env):
        self.last_choices = []
        limits, chosen = [], []
        for seat, player in enumerate(env.game.players):
            index = self.model.declare(player.cards, seat, greedy=self.greedy, temperature=self.temperature,
                                       rng=self.rng)
            chosen.append(index)
            limits.append(declaration_value(player.cards, DECLARATIONS[index]))
            self.last_choices.append((encode_hand(player.cards, seat), index))

        result = run_auction(limits)
        if result is None:
            return None
        seat, bid = result
        game, hand = DECLARATIONS[chosen[seat]]
        # the other players effectively passed
        self.last_choices = [(state, index if pl == seat else 0) for pl, (state, index) in
                             enumerate(self.last_choices)]
        return {"declarer": seat, "game": game, "hand": hand, "bid": bid}
