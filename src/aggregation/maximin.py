from __future__ import annotations

from collections import Counter, defaultdict
from typing import TYPE_CHECKING

from aggregation.base import AggregationResult, AggregationRule, _normalize

if TYPE_CHECKING:
    from agents.value_agent import ValueAgent
    from dilemmas.scenarios import Dilemma


class Maximin(AggregationRule):
    """Select the choice that maximises the minimum welfare across value profiles.

    Each agent responds independently (solo_response). Welfare for a profile
    under a given choice is the fraction of that profile's agents who chose it.
    The rule picks the choice where the worst-off profile's welfare is highest —
    the spirit of MaxMin-RLHF applied to a group decision.

    Tie-breaking hierarchy (all deterministic with respect to dilemma.choices order):
      1. Highest min-welfare across profiles.
      2. Most distinct profiles with any nonzero support (broadest coalition).
      3. Most total votes (falls back to plurality when fully tied).
      4. Earliest position in dilemma.choices.

    The distribution field is the raw vote tally (same as MajorityVote), so
    callers can directly compare decision vs. plurality winner to detect
    maximin-selected choices that differ from the majority.
    """

    def aggregate(
        self,
        agents: list[ValueAgent],
        dilemma: Dilemma,
    ) -> AggregationResult:
        responses = [agent.solo_response(dilemma) for agent in agents]

        # Group each agent's choice by profile id.
        profile_choices: dict[str, list[str]] = defaultdict(list)
        for agent, resp in zip(agents, responses):
            profile_choices[agent.profile.id].append(resp.choice)

        # Per-choice welfare matrix: choice → {profile_id → fraction of that profile voting for it}.
        welfare: dict[str, dict[str, float]] = {
            choice: {
                pid: votes.count(choice) / len(votes)
                for pid, votes in profile_choices.items()
            }
            for choice in dilemma.choices
        }

        counts: Counter[str] = Counter(r.choice for r in responses)

        def _score(choice: str) -> tuple[float, int, int]:
            w = welfare[choice]
            min_welfare = min(w.values())
            breadth = sum(1 for v in w.values() if v > 0.0)
            total_votes = counts[choice]
            return (min_welfare, breadth, total_votes)

        # max() is stable on equal keys: first element of dilemma.choices wins ties.
        decision = max(dilemma.choices, key=_score)

        return AggregationResult(
            decision=decision,
            distribution=_normalize(counts, dilemma.choices),
            metadata={
                "welfare_by_choice": welfare,
                "profile_choices": {pid: list(v) for pid, v in profile_choices.items()},
                "agent_choices": {r.agent_id: r.choice for r in responses},
            },
        )
