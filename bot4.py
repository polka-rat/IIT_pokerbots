"""
Bot4: Rebuilt using bot1's equity-driven poker strategy with a new auction plan.
"""
from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot

import eval7
import random


# --------------------------
# Preflop equity buckets
# --------------------------
PREMIUM_PAIRS = {'AA', 'KK', 'QQ', 'JJ', 'TT'}
GOOD_PAIRS = {'99', '88', '77', '66'}
SMALL_PAIRS = {'55', '44', '33', '22'}
PREMIUM_SUITE = {'AKs', 'AQs', 'AJs', 'ATs', 'KQs', 'KJs', 'QJs', 'JTs'}
PREMIUM_OFF = {'AKo', 'AQo', 'AJo', 'KQo'}
GOOD_SUITE = {'A9s', 'A8s', 'A7s', 'A6s', 'A5s', 'A4s', 'A3s', 'A2s',
              'KTs', 'K9s', 'QTs', 'Q9s', 'J9s', 'T9s', '98s', '87s', '76s', '65s'}


def _hand_key(cards):
    c1, c2 = cards
    ranks = 'AKQJT98765432'
    r1, s1 = c1[0], c1[1]
    r2, s2 = c2[0], c2[1]
    if ranks.index(r1) > ranks.index(r2):
        r1, s1, r2, s2 = r2, s2, r1, s1
    if r1 == r2:
        return r1 + r2
    suited = 's' if s1 == s2 else 'o'
    return r1 + r2 + suited


def preflop_equity_estimate(cards):
    key = _hand_key(cards)
    if key in PREMIUM_PAIRS:
        return 0.82
    if key in GOOD_PAIRS:
        return 0.68
    if key in SMALL_PAIRS:
        return 0.56
    if key in PREMIUM_SUITE:
        return 0.67
    if key in PREMIUM_OFF:
        return 0.64
    if key in GOOD_SUITE:
        return 0.58
    if cards[0][0] == 'A' or cards[1][0] == 'A':
        return 0.54
    if cards[0][0] == 'K' or cards[1][0] == 'K':
        return 0.52
    return 0.46


def monte_carlo_equity(my_hand, board, opp_known=None, num_simulations=300):
    try:
        my_cards = [eval7.Card(c) for c in my_hand]
        brd_cards = [eval7.Card(c) for c in board]
        known_opp = [eval7.Card(c) for c in (opp_known or [])]

        deck = eval7.Deck()
        known_set = set(str(c) for c in my_cards + brd_cards + known_opp)
        remaining = [c for c in deck.cards if str(c) not in known_set]

        board_needed = 5 - len(brd_cards)
        opp_needed = 2 - len(known_opp)

        wins = 0.0
        for _ in range(num_simulations):
            random.shuffle(remaining)
            opp_full = known_opp + remaining[:opp_needed]
            full_board = brd_cards + remaining[opp_needed: opp_needed + board_needed]

            my_score = eval7.evaluate(my_cards + full_board)
            opp_score = eval7.evaluate(opp_full + full_board)

            if my_score > opp_score:
                wins += 1.0
            elif my_score == opp_score:
                wins += 0.5

        return wins / num_simulations
    except Exception:
        return 0.5


class Player(BaseBot):

    def __init__(self) -> None:
        self.hand_count = 0
        self.opp_fold_count = 0
        self.opp_raise_count = 0
        self.preflop_equity = 0.5
        self.prev_opp_wager = 0
        self.applied_reveal_adjustment = False

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        self.hand_count += 1
        self.prev_opp_wager = 0
        self.preflop_equity = preflop_equity_estimate(current_state.my_hand)
        self.applied_reveal_adjustment = False

    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        payoff = current_state.payoff
        if payoff > 0 and not current_state.opp_revealed_cards:
            self.opp_fold_count += 1
        if current_state.opp_wager > current_state.my_wager:
            self.opp_raise_count += 1

    # ---------------------- Equity Helpers ----------------------
    def _get_equity(self, current_state: PokerState) -> float:
        street = current_state.street
        board = current_state.board

        if street in ('preflop', 'pre-flop'):
            self._apply_auction_reveal_adjustment(current_state)
            return self.preflop_equity

        sims = 400 if street == 'flop' else 300
        return monte_carlo_equity(
            current_state.my_hand,
            board,
            opp_known=current_state.opp_revealed_cards,
            num_simulations=sims
        )

    def _apply_auction_reveal_adjustment(self, current_state: PokerState) -> None:
        revealed = current_state.opp_revealed_cards or []
        if not revealed or self.applied_reveal_adjustment:
            return
        rank_order = '23456789TJQKA'
        def rank_val(card: str) -> int:
            return rank_order.index(card[0])
        my_high = max(rank_val(c) for c in current_state.my_hand)
        opp_high = max(rank_val(c) for c in revealed)
        if opp_high > my_high:
            self.preflop_equity = max(0.0, self.preflop_equity * 0.8)
        elif my_high > opp_high:
            self.preflop_equity = min(1.0, self.preflop_equity * 1.2)
        self.applied_reveal_adjustment = True

    def _decide_raise_size(self, equity, pot, min_raise, max_raise) -> int:
        fraction = min(0.75, max(0.0, (equity - 0.5) * 1.5))
        amount = int(min_raise + fraction * (max_raise - min_raise))
        return max(min_raise, min(amount, max_raise))

    # ---------------------- Auction Strategy ----------------------
    def _auction_bid(self, current_state: PokerState) -> ActionBid:
        opp_bid = getattr(current_state, 'opp_bid', 0)
        my_bid = getattr(current_state, 'my_bid', 0)
        chips = current_state.my_chips

        # If opponent bid above our stack, shove what we have left
        if opp_bid >= chips:
            return ActionBid(chips)

        # Equity-informed opener: bet more with better hole cards
        base_eq = self.preflop_equity
        base_bid = int(min(chips, 30 + base_eq * 120))  # ranges roughly 30-150

        if opp_bid == 0:
            return ActionBid(base_bid)

        # If they bid, overcall slightly but respect a hard cap
        target = min(chips, opp_bid + 5)
        target = min(target, max(base_bid, 160))
        return ActionBid(target)

    # ---------------------- Main Decision ----------------------
    def get_move(self, game_info: GameInfo, current_state: PokerState):

        if current_state.street == 'auction':
            return self._auction_bid(current_state)

        equity = self._get_equity(current_state)

        pot = current_state.pot
        cost_to_call = current_state.cost_to_call if current_state.can_act(ActionCall) else 0
        pot_odds = cost_to_call / (pot + cost_to_call) if (pot + cost_to_call) > 0 else 0.0

        opp_fold_rate = self.opp_fold_count / max(self.hand_count, 1)

        can_raise = current_state.can_act(ActionRaise)
        can_call = current_state.can_act(ActionCall)
        can_check = current_state.can_act(ActionCheck)
        min_raise, max_raise = current_state.raise_bounds if can_raise else (0, 0)

        if equity > 0.72:
            if can_raise:
                return ActionRaise(self._decide_raise_size(equity, pot, min_raise, max_raise))
            if can_call:
                return ActionCall()
            return ActionCheck()

        if equity > 0.58:
            if can_raise:
                return ActionRaise(self._decide_raise_size(equity, pot, min_raise, max_raise))
            if can_call:
                return ActionCall()
            return ActionCheck()

        if equity > 0.42:
            if can_check:
                return ActionCheck()
            if can_call and equity >= pot_odds:
                return ActionCall()
            if can_raise and opp_fold_rate > 0.45 and random.random() < 0.18:
                return ActionRaise(min_raise)
            return ActionFold() if can_call else ActionCheck()

        if can_check:
            return ActionCheck()

        if can_raise and opp_fold_rate > 0.5 and equity > 0.32 and random.random() < 0.12:
            return ActionRaise(min_raise)

        if can_call and pot_odds < 0.2 and equity >= pot_odds:
            return ActionCall()

        return ActionFold() if can_call else ActionCheck()


if __name__ == '__main__':
    run_bot(Player(), parse_args())