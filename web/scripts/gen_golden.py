#!/usr/bin/env python3
"""
Generate golden expectation JSONs for the TypeScript rating/scheduler modules.

Strategy:
  - TrueSkill: import the real legacy/trueskill_utils.py module.
  - Scheduler: import the real legacy/random_utils.py module.
  - ELO doubles: the legacy main.py implementation contains a precedence bug
    (`change = K * (1 if win else 0 - expected)`). This script uses the
    corrected formula `Δ = K * (S - E)` where S is 1 for a win and 0 for a loss.

If scipy is missing from the system Python, we shim scipy.special.erfinv using
only the Python stdlib (statistics.NormalDist) so the legacy modules can still
be imported unchanged.
"""

import json
import math
import os
import sys
import statistics
from pathlib import Path

# Try to import scipy; if unavailable, install a minimal shim so that the
# legacy modules can still be imported unchanged.
try:
    from scipy.special import erfinv  # noqa: F401
except Exception:
    class _SciPyShim:
        @staticmethod
        def erfinv(y: float) -> float:
            # erfinv(y) = Phi^{-1}((y+1)/2) / sqrt(2)
            return statistics.NormalDist().inv_cdf((y + 1.0) / 2.0) / math.sqrt(2.0)

    class _SciPySpecial:
        erfinv = staticmethod(_SciPyShim.erfinv)

    class _SciPy:
        special = _SciPySpecial()

    sys.modules["scipy"] = _SciPy()
    sys.modules["scipy.special"] = _SciPySpecial()

# Add legacy directory to the path so we can import the original modules.
LEGACY_DIR = Path(__file__).resolve().parents[2] / "legacy"
sys.path.insert(0, str(LEGACY_DIR))

from trueskill_utils import Player, TrueSkill  # noqa: E402
from random_utils import optimize_schedule  # noqa: E402


INITIAL_RATING = 1000
K_DOUBLES = 16

MATCHES = [
    {"date": "2024-01-01", "a1": "p1", "a2": "p2", "b1": "p3", "b2": "p4", "scoreA": 21, "scoreB": 15},
    {"date": "2024-01-01", "a1": "p3", "a2": "p4", "b1": "p1", "b2": "p2", "scoreA": 18, "scoreB": 21},
    {"date": "2024-01-02", "a1": "p1", "a2": "p3", "b1": "p2", "b2": "p4", "scoreA": 21, "scoreB": 19},
    {"date": "2024-01-03", "a1": "p2", "a2": "p3", "b1": "p1", "b2": "p4", "scoreA": 12, "scoreB": 21},
]


def recompute_elo_fixed(matches):
    """Recompute doubles ELO using the corrected update formula."""
    ratings = {}
    snapshots = []
    last_date = None

    def record_boundary(date):
        if date is None:
            return
        for player, elo in ratings.items():
            snapshots.append({"date": date, "playerId": player, "elo": round(elo, 10)})

    for match in matches:
        if last_date is None or last_date != match["date"]:
            record_boundary(last_date)
            last_date = match["date"]

        for pid in [match["a1"], match["a2"], match["b1"], match["b2"]]:
            ratings.setdefault(pid, float(INITIAL_RATING))

        team_a_avg = (ratings[match["a1"]] + ratings[match["a2"]]) / 2
        team_b_avg = (ratings[match["b1"]] + ratings[match["b2"]]) / 2

        a_wins = match["scoreA"] > match["scoreB"]
        s_a = 1 if a_wins else 0
        s_b = 0 if a_wins else 1

        for pid in [match["a1"], match["a2"]]:
            expected = 1 / (1 + 10 ** ((team_b_avg - ratings[pid]) / 400))
            ratings[pid] += K_DOUBLES * (s_a - expected)

        for pid in [match["b1"], match["b2"]]:
            expected = 1 / (1 + 10 ** ((team_a_avg - ratings[pid]) / 400))
            ratings[pid] += K_DOUBLES * (s_b - expected)

    record_boundary(last_date)
    return ratings, snapshots


def recompute_trueskill(matches):
    """Recompute TrueSkill by replaying matches using the legacy module."""
    ts_util = TrueSkill(draw_probability=0.0)
    players = {}
    snapshots = []
    last_date = None

    def record_boundary(date):
        if date is None:
            return
        for player_id, player in players.items():
            snapshots.append({
                "date": date,
                "playerId": player_id,
                "mu": round(player.mu, 10),
                "sigma": round(player.sigma, 10),
            })

    for match in matches:
        if last_date is None or last_date != match["date"]:
            record_boundary(last_date)
            last_date = match["date"]

        for pid in [match["a1"], match["a2"], match["b1"], match["b2"]]:
            if pid not in players:
                players[pid] = Player()

        team_a = [players[match["a1"]], players[match["a2"]]]
        team_b = [players[match["b1"]], players[match["b2"]]]
        result = 1 if match["scoreA"] > match["scoreB"] else -1
        ts_util.rate_team(team_a, team_b, result)

    record_boundary(last_date)
    return players, snapshots


def gen_elo():
    ratings, snapshots = recompute_elo_fixed(MATCHES)
    return {
        "matches": MATCHES,
        "ratings": {k: round(v, 10) for k, v in ratings.items()},
        "snapshots": snapshots,
    }


def gen_trueskill():
    players, snapshots = recompute_trueskill(MATCHES)
    return {
        "matches": MATCHES,
        "players": {
            k: {"mu": round(v.mu, 10), "sigma": round(v.sigma, 10)}
            for k, v in players.items()
        },
        "snapshots": snapshots,
    }


def gen_predict():
    # Use a fixed set of ratings for ELO prediction.
    elo_ratings = {"p1": 1100, "p2": 1050, "p3": 1000, "p4": 950}
    elo_cases = []
    for a1, a2, b1, b2 in [
        ("p1", "p2", "p3", "p4"),
        ("p1", "p3", "p2", "p4"),
        ("p4", "p3", "p1", "p2"),
    ]:
        team_a_avg = (elo_ratings[a1] + elo_ratings[a2]) / 2
        team_b_avg = (elo_ratings[b1] + elo_ratings[b2]) / 2
        team_a_win = 1 / (1 + 10 ** ((team_b_avg - team_a_avg) / 400))
        elo_cases.append({
            "a1": a1,
            "a2": a2,
            "b1": b1,
            "b2": b2,
            "ratings": elo_ratings,
            "expected": {
                "teamAWin": round(team_a_win, 10),
                "teamBWin": round(1 - team_a_win, 10),
            },
        })

    ts_util = TrueSkill(draw_probability=0.0)
    ts_cases = []
    for team_a_mus, team_b_mus in [
        ([25.0, 25.0], [25.0, 25.0]),
        ([30.0, 28.0], [22.0, 20.0]),
        ([18.0, 20.0], [30.0, 32.0]),
    ]:
        team_a = [Player(mu=m, sigma=25.0 / 3.0) for m in team_a_mus]
        team_b = [Player(mu=m, sigma=25.0 / 3.0) for m in team_b_mus]
        outcome = ts_util.predict_team_outcome(team_a, team_b)
        ts_cases.append({
            "teamA": [{"mu": p.mu, "sigma": p.sigma} for p in team_a],
            "teamB": [{"mu": p.mu, "sigma": p.sigma} for p in team_b],
            "expected": {
                "win": round(outcome["win"], 10),
                "draw": round(outcome["draw"], 10),
                "loss": round(outcome["loss"], 10),
            },
        })

    return {"elo": elo_cases, "trueskill": ts_cases}


def gen_scheduler():
    player_ids = ["p1", "p2", "p3", "p4", "p5", "p6", "p7"]
    players = [
        Player(mu=25.0, sigma=25.0 / 3.0),
        Player(mu=30.5, sigma=25.0 / 3.0),
        Player(mu=20.0, sigma=25.0 / 3.0),
        Player(mu=35.0, sigma=8.0),
        Player(mu=15.5, sigma=8.5),
        Player(mu=32.0, sigma=7.5),
        Player(mu=16.5, sigma=9.0),
    ]
    seed = 42
    matches = 10
    lambda_weight = 0.5

    schedule, alpha_var, best_loss, mean_closeness, max_closeness, entropy = optimize_schedule(
        n=len(player_ids),
        m=matches,
        ts=TrueSkill(draw_probability=0.0),
        players=players,
        lambda_weight=lambda_weight,
        iters=5000,
        max_play_gap=2,
        seed=seed,
    )

    return {
        "input": {
            "playerIds": player_ids,
            "matches": matches,
            "players": [{"mu": p.mu, "sigma": p.sigma} for p in players],
            "seed": seed,
            "lambda": lambda_weight,
        },
        "output": {
            "schedule": [
                {"a1": player_ids[t1[0]], "a2": player_ids[t1[1]],
                 "b1": player_ids[t2[0]], "b2": player_ids[t2[1]]}
                for t1, t2 in schedule
            ],
            "metrics": {
                "alphaVar": alpha_var,
                "bestLoss": best_loss,
                "meanCloseness": mean_closeness,
                "maxCloseness": max_closeness,
                "entropy": entropy,
            },
        },
    }


def main():
    out_dir = Path(__file__).resolve().parent.parent / "test" / "golden"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "elo.json": gen_elo(),
        "trueskill.json": gen_trueskill(),
        "predict.json": gen_predict(),
        "scheduler.json": gen_scheduler(),
    }

    for name, data in files.items():
        path = out_dir / name
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
