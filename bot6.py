'''
Bot6: Bot1 logic with a proper auction strategy.

Auction bid sizing:
  - Bid is proportional to how uncertain our hand equity is (near 0.5 = most valuable
    to learn opponent's card) and the current pot size.
  - Capped to protect stack and avoid overbidding.

If we WIN the auction:
  - Run Monte Carlo with the revealed card for a precise equity estimate.
  - Use that MC result (not the bucket estimate) for all streets of this hand.

If we LOSE the auction (opponent paid more, so they see our card):
  - Set opp_has_info=True. On this hand we tighten bluff thresholds since opp
    can play more accurately against us.
  - Track how often opp wins the auction to calibrate future bids.
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
# Monte Carlo equity (non-mutating deck, safe for repeated calls)
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
        # Cross-hand opponent modeling
        self.hand_count          = 0
        self.opp_fold_count      = 0
        self.opp_raise_count     = 0
        self.opp_auction_wins    = 0   # times opp outbid us (they saw our card)
        self.auction_total       = 0   # total auctions played

        # Per-hand state (reset in on_hand_start)
        self.preflop_equity      = 0.5
        self.auction_equity      = None   # precise MC equity when we win auction
        self.opp_has_info        = False  # True when opp won the auction this hand
        self.auction_result_seen = False  # guard so we compute MC only once
        self.saw_auction         = False  # True after we submitted a bid this hand

    # ------------------------------------------------------------------
    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        self.hand_count         += 1
        self.preflop_equity      = preflop_equity_estimate(current_state.my_hand)
        self.auction_equity      = None
        self.opp_has_info        = False
        self.auction_result_seen = False
        self.saw_auction         = False

    # ------------------------------------------------------------------
    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        payoff = current_state.payoff

        if payoff > 0 and not current_state.opp_revealed_cards:
            self.opp_fold_count += 1
        if current_state.opp_wager > current_state.my_wager:
            self.opp_raise_count += 1

    # ------------------------------------------------------------------
    # AUCTION LOGIC
    # ------------------------------------------------------------------
    def _smart_bid(self, current_state: PokerState) -> int:
        '''
        Compute how much to bid for seeing one opponent card.

        Core idea:
          - The revealed card is worth more when our preflop equity is close to 0.5
            (marginal hand → info can flip our decision) and less when we are
            very strong or very weak (decision is already clear).
          - We scale by pot size: more chips at stake → info worth more.
          - We subtract a mild "overpay discount" to avoid paying too much.
          - Hard cap: never bid more than 12% of stack (stack protection).
        '''
        eq    = self.preflop_equity
        chips = current_state.my_chips

        # uncertainty peaks at 1.0 when eq=0.5, drops to 0 at eq=0 or eq=1
        uncertainty = 1.0 - abs(eq - 0.5) * 2.0

        # Base bid: flat minimum plus uncertainty-scaled bonus.
        # Minimum ensures we always win the auction (bot1 bids 0) at a small fixed cost.
        # Extra chips only when knowing opp's card is decision-relevant.
        base   = 5                               # always spend at least 5 to win vs 0-bidder
        bonus  = int(uncertainty * 40)           # up to +40 chips for marginal hands
        raw_value = base + bonus

        # Don't overpay on very strong or very weak hands where decision is clear
        if eq > 0.72 or eq < 0.32:
            raw_value = base   # info won't change our action much

        # Hard cap: never risk more than 8% of stack
        max_bid = min(chips, int(chips * 0.08), 120)
        bid = max(0, min(raw_value, max_bid))

        return bid

    # ------------------------------------------------------------------
    # POST-AUCTION INFO PROCESSING
    # ------------------------------------------------------------------
    def _process_auction_result(self, current_state: PokerState) -> None:
        '''
        Called once on the first post-auction street.
        - If we see opp_revealed_cards → we WON the auction:
            Run a precise MC with the revealed card and cache that equity.
        - If opp_revealed_cards is empty → we LOST the auction:
            Set opp_has_info=True so we tighten bluff thresholds.
        '''
        # Only process after we have actually gone through an auction this hand
        if not self.saw_auction or self.auction_result_seen:
            return
        self.auction_result_seen = True
        self.auction_total += 1

        revealed = current_state.opp_revealed_cards or []

        if revealed:
            # We won → compute precise equity using the known opponent card
            self.auction_equity = monte_carlo_equity(
                current_state.my_hand,
                current_state.board,
                opp_known=list(revealed),
                num_simulations=400
            )
        else:
            # We lost → opponent knows one of our cards
            self.opp_has_info = True
            self.opp_auction_wins += 1

    # ------------------------------------------------------------------
    def _get_equity(self, current_state: PokerState) -> float:
        street = current_state.street

        # Process auction outcome exactly once on the first post-auction call
        self._process_auction_result(current_state)

        # If we won the auction, use the precise MC estimate for every street
        if self.auction_equity is not None:
            # On later streets refine with board cards already dealt
            if street not in ('preflop', 'pre-flop'):
                return monte_carlo_equity(
                    current_state.my_hand,
                    current_state.board,
                    opp_known=current_state.opp_revealed_cards,
                    num_simulations=350 if street == 'flop' else 280
                )
            return self.auction_equity

        # No auction info → bot1 default
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

        # ── Auction: smart bid based on uncertainty ─────────────────────
        if current_state.street == 'auction':
            self.saw_auction = True
            return ActionBid(self._smart_bid(current_state))

        equity = self._get_equity(current_state)

        pot          = current_state.pot
        cost_to_call = current_state.cost_to_call if current_state.can_act(ActionCall) else 0
        pot_odds     = cost_to_call / (pot + cost_to_call) if (pot + cost_to_call) > 0 else 0.0

        opp_fold_rate = self.opp_fold_count / max(self.hand_count, 1)

        can_raise = current_state.can_act(ActionRaise)
        can_call  = current_state.can_act(ActionCall)
        can_check = current_state.can_act(ActionCheck)

        min_raise, max_raise = current_state.raise_bounds if can_raise else (0, 0)

        # ── AUCTION-WIN PATH: we know one of opp's cards → precise equity ─
        # The key insight from the log: after winning the auction, the bot was
        # checking the flop and ceding initiative, then folding to large turn/river
        # bets. With a known-accurate MC equity we should:
        #   (a) Lead-bet on the flop with any positive-edge hand to take control.
        #   (b) Call bets when equity >= pot_odds (trust the MC, it's precise).
        #   (c) Give up immediately with weak equity rather than check through.
        if self.auction_equity is not None:
            # Very strong: raise/call anything
            if equity > 0.72:
                if can_raise:
                    return ActionRaise(self._decide_raise_size(equity, pot, min_raise, max_raise))
                if can_call:
                    return ActionCall()
                return ActionCheck()

            # Edge hand with precise info: lead-bet instead of checking (take control)
            if equity > 0.52:
                if can_raise:
                    # Lead bet ~40-70% pot
                    fraction = (equity - 0.52) / 0.48   # 0 at 0.52, 1 at 1.0
                    return ActionRaise(self._decide_raise_size(equity, pot, min_raise, max_raise))
                if can_call:
                    # Call if equity exceeds pot odds (trust the MC)
                    if equity >= pot_odds:
                        return ActionCall()
                    return ActionFold()
                return ActionCheck()

            # Very weak hand: give up to any bet, don't bleed chips checking/calling
            if equity < 0.38:
                if can_check:
                    return ActionCheck()    # stay in for free to avoid wasting auction cost
                return ActionFold()         # fold to any bet immediately

            # Marginal with auction info: pot-odds call only
            if can_check:
                return ActionCheck()
            if can_call and equity >= pot_odds:
                return ActionCall()
            return ActionFold() if can_call else ActionCheck()

        # ── STANDARD PATH (no auction info, or we lost the auction) ──────

        # If opp has info about our hand, tighten bluff thresholds
        bluff_fold_threshold  = 0.50 if self.opp_has_info else 0.45
        bluff_rand_threshold  = 0.10 if self.opp_has_info else 0.18
        weak_bluff_threshold  = 0.55 if self.opp_has_info else 0.50
        weak_bluff_rand       = 0.06 if self.opp_has_info else 0.12

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
            if can_raise and opp_fold_rate > bluff_fold_threshold and random.random() < bluff_rand_threshold:
                return ActionRaise(min_raise)
            return ActionFold() if can_call else ActionCheck()

        # ── Weak (≤0.42) → fold / rare bluff ─────────────────────────────
        if can_check:
            return ActionCheck()

        if can_raise and opp_fold_rate > weak_bluff_threshold and equity > 0.32 and random.random() < weak_bluff_rand:
            return ActionRaise(min_raise)

        if can_call and pot_odds < 0.2 and equity >= pot_odds:
            return ActionCall()

        return ActionFold() if can_call else ActionCheck()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
