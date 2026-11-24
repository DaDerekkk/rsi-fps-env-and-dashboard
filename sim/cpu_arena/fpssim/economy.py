"""Platform economy (rsi-fps-5.5, BALANCED rules): cash cap, round settlement, loss-bonus levels, kill rewards.

Rules from the platform guide (economy section) and cs2-web server/game.js; validated transaction by transaction against the
recorded `cash` events in validate/economy.py.
"""
from .common import WEAPONS

CAP = 16000
START_CASH = 800
LOSS = (1400, 1900, 2400, 2900, 3400)
WIN = 3250
OBJECTIVE_WIN = 3500
OBJECTIVE_REASONS = ('bomb_explode', 'defuse')
PLANT_BONUS = 300
DEFUSE_BONUS = 300
PLANTED_LOSS = 600
CT_KILL_BONUS = 50


def clip(x):
    return max(0, min(CAP, x))


class Ledger:
    """Cash + loss levels for 10 seats (teams 0: seats 0-4, 1: seats 5-9)."""

    def __init__(self):
        self.cash = [0] * 10
        self.level = [1, 1]
        self.level_before = [1, 1]
        self.attack_team = None

    def apply(self, seat, requested):
        self.cash[seat] = clip(self.cash[seat] + requested)
        return self.cash[seat]

    def round_start(self, attack_team):
        if self.attack_team is not None and attack_team != self.attack_team:
            self.level = [1, 1]                         # halftime: loss levels reset
        self.attack_team = attack_team

    def settle(self, winner, reason):
        """Update loss levels after a round; returns (win amount, loss amount for the loser)."""
        loser = 1 - winner
        self.level_before = list(self.level)
        loss_amt = LOSS[self.level[loser]]
        self.level[loser] = min(4, self.level[loser] + 1)
        self.level[winner] = max(0, self.level[winner] - 1)
        return (OBJECTIVE_WIN if reason in OBJECTIVE_REASONS else WIN), loss_amt

    # ---- validation helper: expected `requested` for a recorded cash event
    def expect_round_end(self, winner, reason, attack_team, alive=None):
        self._win, self._loss = self.settle(winner, reason)

    def expected(self, seat, reason, e, item_kind):
        if reason == 'initial_cash':
            return START_CASH
        if reason == 'period_reset':
            return START_CASH - e['before']
        if reason == 'round_win':
            return self._win
        if reason == 'round_loss':
            return self._loss
        if reason == 'planted_loss':
            return PLANTED_LOSS
        if reason == 'plant':
            return PLANT_BONUS
        if reason == 'defuse':
            return DEFUSE_BONUS
        if reason == 'ct_kill_bonus':
            return CT_KILL_BONUS
        if reason == 'kill':
            k = item_kind.get(e.get('item_id'))
            return WEAPONS[k]['killReward'] if k in WEAPONS else None
        if reason in ('buy', 'refund'):
            return None                                 # checked against the catalogue in the arena tests
        return None
