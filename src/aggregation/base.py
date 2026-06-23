from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fidelity.classifier import FidelityScore

if TYPE_CHECKING:
    from agents.value_agent import ValueAgent
    from dilemmas.scenarios import Dilemma


@dataclass
class RoundSnapshot:
    """Complete state of one deliberation round.

    Stored in AggregationResult.round_history so the drift metric never needs
    to re-call classify() — fidelity scores are captured at write time.
    """
    round_number: int
    agent_responses: dict[str, str]           # agent_id → raw LLM output
    agent_choices: dict[str, str]             # agent_id → extracted choice label
    agent_fidelity: dict[str, FidelityScore]  # agent_id → fidelity score this round
    vote_distribution: dict[str, float]       # choice → normalized fraction this round


@dataclass
class AggregationResult:
    """Output of any AggregationRule.

    distribution is always the final-state normalized vote tally, making it
    directly comparable across all four rules:
      - majority_vote:    normalized vote histogram
      - deliberation:     final-round vote_distribution from round_history[-1]
      - maximin:          tally after the protected-choice selection
      - distributional:  the full reported spread (decision is None)

    round_history is non-empty only for deliberation. It carries the full
    trajectory — per-round choices, fidelity scores, and vote tallies — so
    value_drift and minority_survival can read it without touching the LLM again.
    """
    decision: str | None                         # plurality winner; None for distributional rule
    distribution: dict[str, float]               # choice → fraction (final state, all rules)
    round_history: list[RoundSnapshot] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class AggregationRule(ABC):
    @abstractmethod
    def aggregate(
        self,
        agents: list[ValueAgent],
        dilemma: Dilemma,
    ) -> AggregationResult:
        ...
