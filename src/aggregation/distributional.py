from __future__ import annotations

from collections import Counter, defaultdict
from typing import TYPE_CHECKING

from aggregation.base import AggregationResult, AggregationRule, _normalize

if TYPE_CHECKING:
    from agents.value_agent import ValueAgent
    from dilemmas.scenarios import Dilemma


class Distributional(AggregationRule):
    """Report the full spread of value positions without committing to one choice.

    This is the Overton-style alternative from pluralistic alignment work:
    rather than collapsing to a single decision, the output is the distribution
    of positions across all agents and, in metadata, broken down by profile.

    decision is always None. distribution is the aggregate normalized vote
    histogram over dilemma.choices. The profile_distribution in metadata
    gives the per-profile breakdown so minority survival metrics can read
    which profiles endorsed which choices without re-running the LLM.
    """

    def aggregate(
        self,
        agents: list[ValueAgent],
        dilemma: Dilemma,
    ) -> AggregationResult:
        responses = [agent.solo_response(dilemma) for agent in agents]

        counts: Counter[str] = Counter(r.choice for r in responses)

        # Per-profile counts, then normalised within each profile.
        profile_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for agent, resp in zip(agents, responses):
            profile_counts[agent.profile.id][resp.choice] += 1

        profile_distribution: dict[str, dict[str, float]] = {
            pid: _normalize(counter, dilemma.choices)
            for pid, counter in profile_counts.items()
        }

        return AggregationResult(
            decision=None,
            distribution=_normalize(counts, dilemma.choices),
            metadata={
                "profile_distribution": profile_distribution,
                "agent_choices": {r.agent_id: r.choice for r in responses},
            },
        )
