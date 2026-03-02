'''
Bot5: Bot1 poker strategy + always bid 100 in auction.
If we win the auction (opponent card revealed), compute win probability x
via Monte Carlo with the revealed card, then override equity = min(1.0, 2*x)
for every street of that hand.
'''
from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot

import eval7
import random


# ---------------------------------------------------------------------------
# Preflop hand strength bucketing
# ---------------------------------------------------------------------------
PREMIUM_PAIRS  = {'AA', 'KK', 'QQ', 'JJ', 'TT'}
GOOD_PAIRS     = {'99', '88', '77', '66'}
SMALL_PAIRS    = {'55', '44', '33', '22'}
PREMIUM_SUITED = {'AKs', 'AQs', 'AJs', 'ATs', 'KQs', 'KJs', 'QJs', 'JTs'}
PREMIUM_OFF    = {'AKo', 'AQo', 'AJo', 'KQo'}
GOOD_SUITED    = {'A9s', 'A8s', 'A7s', 'A6s', 'A5s', 'A4s', 'A3s', 'A2s',
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
    if key in PREMIUM_PAIRS:   return 0.82
    if key in GOOD_PAIRS:      return 0.68
    if key in SMALL_PAIRS:     return 0.56
    if key in PREMIUM_SUITED:  return 0.67
    if key in PREMIUM_OFF:     return 0.64
    if key in GOOD_SUITED:     return 0.58
    if cards[0][0] == 'A' or cards[1][0] == 'A': return 0.54
    if cards[0][0] == 'K' or cards[1][0] == 'K': return 0.52
    return 0.46


RANK_ORDER = '23456789TJQKA'

def rank_val(card):
    return RANK_ORDER.index(card[0])


def monte_carlo_equity(my_hand, board, opp_known=None, num_simulations=300):
    """Monte Carlo equity (non-mutating deck, safe for repeated calls)."""
    try:
        my_cards  = [eval7.Card(c) for c in my_hand]
        brd_cards = [eval7.Card(c) for c in board]
        known_opp = [eval7.Card(c) for c in (opp_known or [])]

        deck = eval7.Deck()
        known_set = set(str(c) for c in my_cards + brd_cards + known_opp)
        remaining = [c for c in deck.cards if str(c) not in known_set]

        board_needed = 5 - len(brd_cards)
        opp_needed   = 2 - len(known_opp)

        wins = 0.0
        for _ in range(num_simulations):
            random.shuffle(remaining)
            opp_full   = known_opp + remaining[:opp_needed]
            full_board = brd_cards + remaining[opp_needed: opp_needed + board_needed]

            my_score  = eval7.evaluate(my_cards + full_board)
            opp_score = eval7.evaluate(opp_full  + full_board)

            if my_score > opp_score:
                wins += 1.0
            elif my_score == opp_score:
                wins += 0.5

        return wins / num_simulations
    except Exception:
        return 0.5


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------
class Player(BaseBot):

    def __init__(self) -> None:
        self.hand_count       = 0
        self.opp_fold_count   = 0
        self.opp_raise_count  = 0

        self.preflop_equity          = 0.5
        self.auction_equity_override = None   # min(1.0, 2*x) when auction won
        self.applied_reveal_adjust   = False

    # ------------------------------------------------------------------
    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        self.hand_count             += 1
        self.preflop_equity          = preflop_equity_estimate(current_state.my_hand)
        self.auction_equity_override = None
        self.applied_reveal_adjust   = False

    # ------------------------------------------------------------------
    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        if current_state.payoff > 0 and not current_state.opp_revealed_cards:
            self.opp_fold_count += 1
        if current_state.opp_wager > current_state.my_wager:
            self.opp_raise_count += 1

    # ------------------------------------------------------------------
    def _apply_auction_win(self, current_state: PokerState) -> None:
        """
        Called once when opponent card is revealed (auction win).
        Runs MC with revealed card to get x, sets override = min(1.0, 2*x).
        """
        if self.applied_reveal_adjust:
            return
        revealed = current_state.opp_revealed_cards or []
        if not revealed:
            return

        x = monte_carlo_equity(
            current_state.my_hand,
            current_state.board,
            opp_known=list(revealed),
            num_simulations=350
        )
        self.auction_equity_override = min(1.0, 2.0 * x)
        self.applied_reveal_adjust   = True

    # ------------------------------------------------------------------
    def _get_equity(self, current_state: PokerState) -> float:
        # Compute auction override once as soon as revealed cards appear
        if not self.applied_reveal_adjust and current_state.opp_revealed_cards:
            self._apply_auction_win(current_state)

        # Use auction-informed override for every street this hand
        if self.auction_equity_override is not None:
            return self.auction_equity_override

        street = current_state.street
        if street in ('preflop', 'pre-flop'):
            return self.preflop_equity

        sims = 400 if street == 'flop' else 300
        return monte_carlo_equity(
            current_state.my_hand,
            current_state.board,
            opp_known=current_state.opp_revealed_cards,
            num_simulations=sims
        )

    # ------------------------------------------------------------------
    def _decide_raise_size(self, equity, pot, min_raise, max_raise) -> int:
        fraction = min(0.75, max(0.0, (equity - 0.5) * 1.5))
        amount   = int(min_raise + fraction * (max_raise - min_raise))
        return max(min_raise, min(amount, max_raise))

    # ------------------------------------------------------------------
    def get_move(self, game_info: GameInfo, current_state: PokerState):

        # ── Auction: always bid 100 (or all-in if stack < 100) ─────────
        if current_state.street == 'auction':
            return ActionBid(min(100, current_state.my_chips))

        equity = self._get_equity(current_state)

        pot          = current_state.pot
        cost_to_call = current_state.cost_to_call if current_state.can_act(ActionCall) else 0
        pot_odds     = cost_to_call / (pot + cost_to_call) if (pot + cost_to_call) > 0 else 0.0

        opp_fold_rate = self.opp_fold_count / max(self.hand_count, 1)

        can_raise = current_state.can_act(ActionRaise)
        can_call  = current_state.can_act(ActionCall)
        can_check = current_state.can_act(ActionCheck)

        min_raise, max_raise = current_state.raise_bounds if can_raise else (0, 0)

        # ── Very strong (>0.72) → value bet big ─────────────────────────
        if equity > 0.72:
            if can_raise:
                return ActionRaise(self._decide_raise_size(equity, pot, min_raise, max_raise))
            if can_call:
                return ActionCall()
            return ActionCheck()

        # ── Strong (0.58–0.72) → raise moderate / call ───────────────────
        if equity > 0.58:
            if can_raise:
                return ActionRaise(self._decide_raise_size(equity, pot, min_raise, max_raise))
            if can_call:
                return ActionCall()
            return ActionCheck()

        # ── Marginal (0.42–0.58) → check / pot-odds call / squeeze bluff ─
        if equity > 0.42:
            if can_check:
                return ActionCheck()
            if can_call and equity >= pot_odds:
                return ActionCall()
            if can_raise and opp_fold_rate > 0.45 and random.random() < 0.18:
                return ActionRaise(min_raise)
            return ActionFold() if can_call else ActionCheck()

        # ── Weak (≤0.42) → fold / rare bluff ─────────────────────────────
        if can_check:
            return ActionCheck()

        if can_raise and opp_fold_rate > 0.5 and equity > 0.32 and random.random() < 0.12:
            return ActionRaise(min_raise)

        if can_call and pot_odds < 0.2 and equity >= pot_odds:
            return ActionCall()

        return ActionFold() if can_call else ActionCheck()


if __name__ == '__main__':
    run_bot(Player(), parse_args())