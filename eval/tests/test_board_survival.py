from types import SimpleNamespace

from agents.board_survival import assess_board_survival, survival_option_score
from agents.harness import Candidate, DecisionHarness
from eval.board_wipe import BoardWipeStats


class Cards:
    def card(self, card_id):
        if card_id == 1:
            return SimpleNamespace(attacks=[], weakness=None, energyType=1)
        return SimpleNamespace(attacks=[10], weakness=None, energyType=2)

    def attack(self, attack_id):
        return SimpleNamespace(damage=80)


def test_threatened_active_prefers_legal_retreat():
    mine = SimpleNamespace(id=1, hp=60)
    theirs = SimpleNamespace(id=2, hp=100)
    me = SimpleNamespace(active=[mine], bench=[SimpleNamespace(id=3, hp=100)])
    opp = SimpleNamespace(active=[theirs], bench=[])
    parsed = SimpleNamespace(
        current=SimpleNamespace(yourIndex=0, players=[me, opp])
    )
    select = SimpleNamespace(
        option=[SimpleNamespace(type=12), SimpleNamespace(type=14)]
    )
    assessment = assess_board_survival(parsed, select, Cards())
    assert assessment.threatened
    assert assessment.reachable_damage == 80
    assert assessment.bench_count == 1
    assert survival_option_score(parsed, select, 0, Cards()) == 2.0
    assert survival_option_score(parsed, select, 1, Cards()) == 0.0


def test_healthy_active_does_not_change_option_scores():
    mine = SimpleNamespace(id=1, hp=100)
    theirs = SimpleNamespace(id=2, hp=100)
    parsed = SimpleNamespace(
        current=SimpleNamespace(
            yourIndex=0,
            players=[
                SimpleNamespace(active=[mine], bench=[SimpleNamespace(id=3)]),
                SimpleNamespace(active=[theirs], bench=[]),
            ],
        )
    )
    select = SimpleNamespace(option=[SimpleNamespace(type=12)])
    assert not assess_board_survival(parsed, select, Cards()).threatened
    assert survival_option_score(parsed, select, 0, Cards()) == 0.0


def test_board_wipe_report_rates():
    stats = BoardWipeStats(losses=4, board_wipes=1, risk_exposures=5, avoided=3)
    report = stats.report()
    assert report["board_wipe_rate_in_losses"] == 0.25
    assert report["board_wipe_avoidance_rate"] == 0.6


def test_survival_candidate_beats_policy_sample_when_threatened():
    sample = Candidate([1], "policy_sample", valid=True, total=0.0)
    sample.scores["board_survival"] = 0.0
    retreat = Candidate([0], "board_survival", valid=True, total=1.0)
    retreat.scores["board_survival"] = 2.0
    assert DecisionHarness._choose([sample, retreat]) is retreat
