'''
Bot7: Lessons from log analysis.

Core insights incorporated:
1. After winning auction: ALWAYS lead-bet the flop — never check first and hand
   aggression over to the opponent. Checking gives hyper-aggressive bots a free
   all-in bluff on the turn/river.
2. Multi-street value extraction: continue betting turn and river with strong
   equity (>0.60) using graduated sizing (~60% pot per street).
3. Opponent aggression tracking: measure avg opponent bet-to-pot ratio. Against
   hyper-aggressive opponents (random shoves), lower call threshold when we have
   precise MC equity from the revealed card.
4. Give up IMMEDIATELY on weak revealed-card equity: fold first bet, don't bleed
   chips through multiple streets.
5. Preflop discipline: fold junk to any meaningful raise.
'''
from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot

import eval7
import random


# ---------------------------------------------------------------------------
# Preflop hand strength buckets (same as bot1)
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


# ---------------------------------------------------------------------------
# Monte Carlo equity (non-mutating, safe for repeated calls)
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
# Player
# ---------------------------------------------------------------------------
class Player(BaseBot):

    def __init__(self) -> None:
        # ── Cross-hand opponent model ──
        self.hand_count       = 0
        self.opp_fold_count   = 0
        self.opp_raise_count  = 0

        # Aggression tracking: accumulate total (opp_bet / pot) ratios to detect bluff-pressure style
        self.opp_bet_ratio_sum   = 0.0   # sum of (bet / pot) for each opp bet seen
        self.opp_bet_sample_count = 0    # number of opp bets sampled

        # Auction tracking
        self.opp_auction_wins = 0
        self.auction_total    = 0

        # ── Per-hand state ──
        self.preflop_equity      = 0.5
        self.auction_equity      = None   # filled in after winning auction
        self.opp_has_info        = False  # opp won the auction this hand
        self.saw_auction         = False
        self.auction_result_seen = False

    # ------------------------------------------------------------------
    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        self.hand_count         += 1
        self.preflop_equity      = preflop_equity_estimate(current_state.my_hand)
        self.auction_equity      = None
        self.opp_has_info        = False
        self.saw_auction         = False
        self.auction_result_seen = False

    # ------------------------------------------------------------------
    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        if current_state.payoff > 0 and not current_state.opp_revealed_cards:
            self.opp_fold_count += 1
        if current_state.opp_wager > current_state.my_wager:
            self.opp_raise_count += 1

    # ------------------------------------------------------------------
    # Opponent aggression helpers
    # ------------------------------------------------------------------
    def _record_opp_bet(self, bet_size: int, pot: int) -> None:
        if pot > 0:
            self.opp_bet_ratio_sum    += bet_size / pot
            self.opp_bet_sample_count += 1

    def _opp_avg_bet_ratio(self) -> float:
        '''Average (opp_bet / pot). Hyper-aggressive bots: >2.0; normal: 0.5-1.0.'''
        if self.opp_bet_sample_count < 5:
            return 1.0   # default – assume normal until we have data
        return self.opp_bet_ratio_sum / self.opp_bet_sample_count

    def _opp_is_hyper_aggressive(self) -> bool:
        return self._opp_avg_bet_ratio() >= 2.5

    # ------------------------------------------------------------------
    # Auction bid sizing
    # ------------------------------------------------------------------
    def _smart_bid(self, current_state: PokerState) -> int:
        eq    = self.preflop_equity
        chips = current_state.my_chips

        # Info value peaks at equity=0.5 (most uncertain → info is most actionable)
        uncertainty = 1.0 - abs(eq - 0.5) * 2.0

        base  = 5
        bonus = int(uncertainty * 40)   # 0–40 chips
        bid   = base + bonus

        # If hand is very clear (strong or weak), info matters less
        if eq > 0.72 or eq < 0.32:
            bid = base

        max_bid = min(chips, int(chips * 0.08), 120)
        return max(0, min(bid, max_bid))

    # ------------------------------------------------------------------
    # Auction result processing (once per hand, on first post-auction call)
    # ------------------------------------------------------------------
    def _process_auction_result(self, current_state: PokerState) -> None:
        if not self.saw_auction or self.auction_result_seen:
            return
        self.auction_result_seen = True
        self.auction_total      += 1

        revealed = current_state.opp_revealed_cards or []
        if revealed:
            self.auction_equity = monte_carlo_equity(
                current_state.my_hand,
                current_state.board,
                opp_known=list(revealed),
                num_simulations=400
            )
        else:
            self.opp_has_info     = True
            self.opp_auction_wins += 1

    # ------------------------------------------------------------------
    # Equity computation
    # ------------------------------------------------------------------
    def _get_equity(self, current_state: PokerState) -> float:
        street = current_state.street
        self._process_auction_result(current_state)

        if self.auction_equity is not None:
            if street not in ('preflop', 'pre-flop'):
                sims = 350 if street == 'flop' else 280
                return monte_carlo_equity(
                    current_state.my_hand,
                    current_state.board,
                    opp_known=current_state.opp_revealed_cards,
                    num_simulations=sims
                )
            return self.auction_equity

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
    # Bet sizing helpers
    # ------------------------------------------------------------------
    def _pot_bet(self, fraction: float, pot: int, min_raise: int, max_raise: int) -> int:
        '''Bet a fraction of the pot, clamped to valid raise range.'''
        amount = int(pot * fraction)
        return max(min_raise, min(amount, max_raise))

    def _value_raise(self, equity: float, pot: int, min_raise: int, max_raise: int) -> int:
        '''Scale raise between min and max based on equity [0.5→1.0] → [0→75%].'''
        fraction = min(0.75, max(0.0, (equity - 0.5) * 1.5))
        amount   = int(min_raise + fraction * (max_raise - min_raise))
        return max(min_raise, min(amount, max_raise))

    # ------------------------------------------------------------------
    def get_move(self, game_info: GameInfo, current_state: PokerState):

        # ── Auction ─────────────────────────────────────────────────────
        if current_state.street == 'auction':
            self.saw_auction = True
            return ActionBid(self._smart_bid(current_state))

        equity   = self._get_equity(current_state)
        pot      = current_state.pot
        cost     = current_state.cost_to_call if current_state.can_act(ActionCall) else 0
        pot_odds = cost / (pot + cost) if (pot + cost) > 0 else 0.0

        # Track opponent bet size vs pot for aggression detection
        if cost > 0:
            self._record_opp_bet(cost, pot)

        opp_fold_rate         = self.opp_fold_count / max(self.hand_count, 1)
        opp_hyper_aggressive  = self._opp_is_hyper_aggressive()

        can_raise = current_state.can_act(ActionRaise)
        can_call  = current_state.can_act(ActionCall)
        can_check = current_state.can_act(ActionCheck)
        min_raise, max_raise = current_state.raise_bounds if can_raise else (0, 0)

        # ════════════════════════════════════════════════════════════════
        # AUCTION-WIN PATH: we have one revealed opponent card
        #
        # Sizing discipline matters: bot6 also has precise equity and calls
        # correctly. Thin-betting marginal hands (0.40–0.55) donates chips.
        # Only bet when we have a genuine edge worth extracting.
        # ════════════════════════════════════════════════════════════════
        if self.auction_equity is not None:
            street = current_state.street

            # ── Very strong (>0.65): value-bet all three streets ───────
            # Sizing: 50% pot (flop), 60% pot (turn), 70% pot (river)
            # Smaller than bot7 v1 — competent callers need tighter sizing.
            if equity > 0.65:
                if can_raise:
                    factor = {'flop': 0.50, 'turn': 0.60, 'river': 0.70}.get(street, 0.50)
                    return ActionRaise(self._pot_bet(factor, pot, min_raise, max_raise))
                if can_call:
                    # Call any bet when we're clearly ahead
                    if equity >= pot_odds:
                        return ActionCall()
                    return ActionFold()
                return ActionCheck()

            # ── Edge (0.55–0.65): single-street lead, then check/call ──
            # Bet the flop once to take pot control. On turn/river, check
            # and call only if equity >= pot odds (don't overcommit).
            if equity > 0.55:
                if can_raise:
                    return ActionRaise(self._pot_bet(0.45, pot, min_raise, max_raise))
                if can_call:
                    call_threshold = 0.42 if opp_hyper_aggressive else pot_odds
                    if equity >= call_threshold:
                        return ActionCall()
                    return ActionFold()
                return ActionCheck()

            # ── Marginal (0.42–0.55): check/call only — no thin bets ──
            # Betting marginal equity into a competent opponent is -EV.
            # Check to see more cards cheaply; call only with good odds.
            if equity >= 0.42:
                if can_check:
                    return ActionCheck()
                if can_call and equity >= pot_odds:
                    return ActionCall()
                return ActionFold() if can_call else ActionCheck()

            # ── Weak (<0.42): give up immediately ──────────────────────
            if can_check:
                return ActionCheck()
            return ActionFold()

        # ════════════════════════════════════════════════════════════════
        # STANDARD PATH (no auction info / we lost the auction)
        # ════════════════════════════════════════════════════════════════

        # If opponent has info about our cards, tighten bluff thresholds
        bluff_fold_thr = 0.50 if self.opp_has_info else 0.45
        bluff_rand_thr = 0.10 if self.opp_has_info else 0.18
        weak_fold_thr  = 0.55 if self.opp_has_info else 0.50
        weak_rand_thr  = 0.06 if self.opp_has_info else 0.12

        # ── Very strong (>0.72) ──────────────────────────────────────
        if equity > 0.72:
            if can_raise:
                return ActionRaise(self._value_raise(equity, pot, min_raise, max_raise))
            if can_call:
                return ActionCall()
            return ActionCheck()

        # ── Strong (0.58–0.72) ──────────────────────────────────────
        if equity > 0.58:
            if can_raise:
                return ActionRaise(self._value_raise(equity, pot, min_raise, max_raise))
            if can_call:
                return ActionCall()
            return ActionCheck()

        # ── Marginal (0.42–0.58) ────────────────────────────────────
        if equity > 0.42:
            if can_check:
                return ActionCheck()
            if can_call and equity >= pot_odds:
                return ActionCall()
            if can_raise and opp_fold_rate > bluff_fold_thr and random.random() < bluff_rand_thr:
                return ActionRaise(min_raise)
            return ActionFold() if can_call else ActionCheck()

        # ── Weak (≤0.42) ─────────────────────────────────────────────
        if can_check:
            return ActionCheck()
        if can_raise and opp_fold_rate > weak_fold_thr and equity > 0.32 and random.random() < weak_rand_thr:
            return ActionRaise(min_raise)
        if can_call and pot_odds < 0.2 and equity >= pot_odds:
            return ActionCall()
        return ActionFold() if can_call else ActionCheck()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
