from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from aggregation.base import AggregationResult, AggregationRule, _normalize

if TYPE_CHECKING:
    from agents.value_agent import ValueAgent
    from dilemmas.scenarios import Dilemma


class MajorityVote(AggregationRule):
    """Plurality vote over agents' independent solo responses.

    Each agent responds to the dilemma without seeing peers. The choice with
    the most votes wins; ties are broken by earliest position in dilemma.choices
    so the result is deterministic regardless of agent order.

    This is the baseline against which deliberation and maximin are compared:
    it applies no minority protection and no inter-agent influence.
    """

    def aggregate(
        self,
        agents: list[ValueAgent],
        dilemma: Dilemma,
    ) -> AggregationResult:
        responses = [agent.solo_response(dilemma) for agent in agents]
        counts: Counter[str] = Counter(r.choice for r in responses)
        distribution = _normalize(counts, dilemma.choices)
        # Tiebreak by dilemma.choices order: max() returns the first maximum found
        # when the key function is equal, and dilemma.choices is the canonical order.
        decision = max(dilemma.choices, key=lambda c: distribution.get(c, 0.0))
        return AggregationResult(
            decision=decision,
            distribution=distribution,
            metadata={
                "vote_counts": dict(counts),
                "agent_choices": {r.agent_id: r.choice for r in responses},
            },
        )
