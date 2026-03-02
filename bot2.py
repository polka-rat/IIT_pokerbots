'''
Competitive poker bot using Monte Carlo equity estimation, opponent modeling,
and an adaptive auction strategy.

Auction strategy:
  - Base bid starts at 100 chips.
  - If opponent consistently wins the auction (overbids us), raise our bid to compete.
  - If opponent consistently loses / folds post-auction, lower bid to save chips.
  - After winning the auction and seeing the opponent's revealed card:
      * If opponent's card rank > our highest card rank → FOLD (we are dominated).
      * Otherwise → play aggressively (raise more).
  - After losing the auction (no reveal info), play normally with equity.
'''
from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot

import eval7
import random


# ---------------------------------------------------------------------------
# Card rank utilities
# ---------------------------------------------------------------------------
RANK_ORDER = 'AKQJT98765432'   # index 0 = highest

def rank_value(card_str: str) -> int:
    """Higher return value = stronger rank (A=12, K=11, …, 2=0)."""
    return len(RANK_ORDER) - 1 - RANK_ORDER.index(card_str[0])

def my_highest_rank(hand) -> int:
    return max(rank_value(c) for c in hand)

def opp_card_beats_my_best(opp_cards, my_hand) -> bool:
    """True if the best opponent revealed card outranks our best card."""
    if not opp_cards:
        return False
    opp_best = max(rank_value(c) for c in opp_cards)
    my_best  = my_highest_rank(my_hand)
    return opp_best > my_best


# ---------------------------------------------------------------------------
# Preflop hand strength lookup
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
    r1, s1 = c1[0], c1[1]
    r2, s2 = c2[0], c2[1]
    if RANK_ORDER.index(r1) > RANK_ORDER.index(r2):
        r1, s1, r2, s2 = r2, s2, r1, s1
    if r1 == r2:
        return r1 + r2
    return r1 + r2 + ('s' if s1 == s2 else 'o')

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


# ---------------------------------------------------------------------------
# Monte Carlo equity for post-flop streets
# ---------------------------------------------------------------------------
def monte_carlo_equity(my_hand, board, opp_known=None, num_simulations=300):
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
            full_board  = brd_cards + remaining[opp_needed: opp_needed + board_needed]
            my_score   = eval7.evaluate(my_cards + full_board)
            opp_score  = eval7.evaluate(opp_full  + full_board)
            if my_score > opp_score:
                wins += 1.0
            elif my_score == opp_score:
                wins += 0.5

        return wins / num_simulations
    except Exception:
        return 0.5


# ---------------------------------------------------------------------------
# Main Bot
# ---------------------------------------------------------------------------

AUCTION_BASE_BID   = 100   # starting auction bid
AUCTION_BID_MIN    = 10    # never bid less than this
AUCTION_BID_MAX    = 400   # cap so we don't over-invest
AUCTION_STEP       = 30    # chip adjustment per learning update


class Player(BaseBot):

    def __init__(self) -> None:
        # ── Persistent opponent model ────────────────────────────────────
        self.hand_count      = 0
        self.opp_fold_count  = 0    # folds to our raise
        self.opp_raise_count = 0    # times opp raised us

        # ── Auction model ────────────────────────────────────────────────
        self.auction_bid_amount  = AUCTION_BASE_BID
        self.auction_hands       = 0    # hands where auction occurred
        # Track how many times we appeared to WIN the auction (opp card revealed)
        self.auction_wins        = 0
        # Track how profitable auction wins were (positive = good, negative = bad)
        self.auction_win_payoffs : list[int] = []

        # ── Per-hand state ───────────────────────────────────────────────
        self.preflop_equity    = 0.5
        self.won_auction       = False  # did WE win the auction this hand?

    # ------------------------------------------------------------------
    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        self.hand_count    += 1
        self.won_auction    = False
        self.preflop_equity = preflop_equity_estimate(current_state.my_hand)

    # ------------------------------------------------------------------
    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        payoff = current_state.payoff

        # General opponent modeling
        if payoff > 0 and not current_state.opp_revealed_cards:
            self.opp_fold_count += 1
        if current_state.opp_wager > current_state.my_wager:
            self.opp_raise_count += 1

        # ── Auction learning ─────────────────────────────────────────────
        # If opp_revealed_cards is non-empty, we won the auction this hand
        opp_revealed = current_state.opp_revealed_cards
        if opp_revealed:
            self.auction_hands += 1
            self.auction_wins  += 1
            self.auction_win_payoffs.append(payoff)

            # Rolling window: last 20 auction wins
            recent = self.auction_win_payoffs[-20:]
            avg_payoff = sum(recent) / len(recent)

            if avg_payoff > 0:
                # Winning auction is profitable → keep bidding (modest increase to compete)
                self.auction_bid_amount = min(
                    AUCTION_BID_MAX,
                    self.auction_bid_amount + AUCTION_STEP // 2
                )
            else:
                # Winning auction is costing us chips → bid less
                self.auction_bid_amount = max(
                    AUCTION_BID_MIN,
                    self.auction_bid_amount - AUCTION_STEP
                )
        else:
            # We did NOT win the auction (or no auction happened).
            # If opponent never reveals cards they may be consistently outbidding us.
            # After enough hands, nudge bid up slightly to compete.
            if self.auction_hands > 0 and self.hand_count % 10 == 0:
                win_rate = self.auction_wins / max(self.auction_hands, 1)
                if win_rate < 0.4:
                    # We're losing the auction too often → bid more
                    self.auction_bid_amount = min(
                        AUCTION_BID_MAX,
                        self.auction_bid_amount + AUCTION_STEP
                    )

    # ------------------------------------------------------------------
    def _get_equity(self, current_state: PokerState) -> float:
        street = current_state.street
        if street == 'preflop':
            return self.preflop_equity
        sims = 400 if street == 'flop' else 300
        return monte_carlo_equity(
            current_state.my_hand,
            current_state.board,
            opp_known=current_state.opp_revealed_cards,
            num_simulations=sims
        )

    # ------------------------------------------------------------------
    def _decide_raise_size(self, equity, min_raise, max_raise, boost=False) -> int:
        """Scale raise between min and max based on equity. boost=True for post-auction aggression."""
        fraction = min(0.75, max(0.0, (equity - 0.5) * 1.5))
        if boost:
            fraction = min(1.0, fraction + 0.3)   # push raise size higher after auction info
        amount = int(min_raise + fraction * (max_raise - min_raise))
        return max(min_raise, min(amount, max_raise))

    # ------------------------------------------------------------------
    def get_move(self, game_info: GameInfo, current_state: PokerState):

        # ── AUCTION ──────────────────────────────────────────────────────
        if current_state.street == 'auction':
            bid = min(self.auction_bid_amount, current_state.my_chips)
            bid = max(0, bid)
            self.won_auction = True   # optimistic; corrected in on_hand_end if opp never reveals
            return ActionBid(bid)

        # ── POST-AUCTION CARD CHECK ───────────────────────────────────────
        # If we won the auction we have the opponent's revealed card(s).
        opp_revealed = current_state.opp_revealed_cards
        auction_dominated = False   # flag: opp has higher card than us

        if opp_revealed:
            if opp_card_beats_my_best(opp_revealed, current_state.my_hand):
                auction_dominated = True   # their top card beats ours → danger

        # ── If dominated by auction info → fold as soon as possible ──────
        if auction_dominated:
            if current_state.can_act(ActionFold):
                return ActionFold()
            if current_state.can_act(ActionCheck):
                return ActionCheck()
            # If we somehow can't fold/check, call (don't over-commit though)
            return ActionCall()

        # ── Normal equity-based play (with aggression boost if we won auction) ──
        equity = self._get_equity(current_state)

        pot          = current_state.pot
        cost_to_call = current_state.cost_to_call if current_state.can_act(ActionCall) else 0
        pot_odds     = cost_to_call / (pot + cost_to_call) if (pot + cost_to_call) > 0 else 0.0

        opp_fold_rate = self.opp_fold_count / max(self.hand_count, 1)
        opp_agg_rate  = self.opp_raise_count / max(self.hand_count, 1)

        can_raise = current_state.can_act(ActionRaise)
        can_call  = current_state.can_act(ActionCall)
        can_check = current_state.can_act(ActionCheck)
        min_raise, max_raise = current_state.raise_bounds if can_raise else (0, 0)

        # We won the auction and our card beats theirs → extra aggression boost
        auction_boost = bool(opp_revealed) and not auction_dominated

        # ── Very strong hand (equity > 0.72) → big value bet ─────────────
        if equity > 0.72:
            if can_raise:
                return ActionRaise(self._decide_raise_size(equity, min_raise, max_raise, boost=auction_boost))
            if can_call:
                return ActionCall()
            return ActionCheck()

        # ── Strong hand (0.58–0.72) → raise / call ───────────────────────
        if equity > 0.58:
            if can_raise:
                return ActionRaise(self._decide_raise_size(equity, min_raise, max_raise, boost=auction_boost))
            if can_call:
                return ActionCall()
            return ActionCheck()

        # ── Marginal hand (0.42–0.58) ─────────────────────────────────────
        if equity > 0.42:
            # With auction info confirming we have better card, play it stronger
            if auction_boost and can_raise:
                return ActionRaise(self._decide_raise_size(equity, min_raise, max_raise, boost=True))
            if can_check:
                return ActionCheck()
            if can_call and equity >= pot_odds:
                return ActionCall()
            if can_raise and opp_fold_rate > 0.45 and random.random() < 0.18:
                return ActionRaise(min_raise)
            return ActionFold() if can_call else ActionCheck()

        # ── Weak hand (equity ≤ 0.42) ─────────────────────────────────────
        if can_check:
            return ActionCheck()
        if can_raise and opp_fold_rate > 0.5 and equity > 0.32 and random.random() < 0.12:
            return ActionRaise(min_raise)
        if can_call and pot_odds < 0.2 and equity >= pot_odds:
            return ActionCall()
        return ActionFold() if can_call else ActionCheck()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
