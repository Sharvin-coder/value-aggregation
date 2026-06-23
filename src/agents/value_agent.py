from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from agents.profiles import ValueProfile

if TYPE_CHECKING:
    from dilemmas.scenarios import Dilemma

_FULL_ARGUMENT_INTRO = (
    "The following agents have already shared their reasoning and positions. "
    "Read their arguments carefully, then give your own response to the dilemma below."
)

# Deliberately says nothing about arguments — only choices are shown in this condition.
_CHOICES_ONLY_INTRO = (
    "The other agents in this group have made their selections. "
    "Their choices are listed below. Then give your own response to the dilemma."
)

_CHOICE_RE = re.compile(r'\{[^{}]*"choice"[^{}]*\}', re.DOTALL)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class AgentResponse:
    agent_id: str
    profile_id: str
    response_text: str   # full LLM output including reasoning
    choice: str          # extracted from the JSON block
    justification: str   # one-sentence extracted from the JSON block


@dataclass
class AgentTurn:
    """One agent's prior-round contribution as seen by peers during deliberation.

    profile_id is Optional so the formatter can hide profile labels, which
    is itself a potential ablation (do agents conform more when they know a
    peer's identity?).
    """
    agent_id: str
    profile_id: str | None
    response_text: str
    choice: str

    @classmethod
    def from_response(
        cls,
        response: AgentResponse,
        reveal_profile: bool = True,
    ) -> AgentTurn:
        return cls(
            agent_id=response.agent_id,
            profile_id=response.profile_id if reveal_profile else None,
            response_text=response.response_text,
            choice=response.choice,
        )


# ---------------------------------------------------------------------------
# Context formatters — the deliberation-context ablation point
# ---------------------------------------------------------------------------

class ContextFormatter(Protocol):
    """Formats peer turns into the string shown to an agent before it responds.

    Swapping the formatter is the main deliberation-context ablation:
      FullArgumentFormatter  → agents see full reasoning (measures persuasion-by-reasoning)
      ChoicesOnlyFormatter   → agents see only tallies  (measures bare conformity-to-count)

    The formatter is passed to ValueAgent.deliberate() per call, not stored on
    the agent, so a single agent pool can be reused across ablation conditions.
    """
    def format(self, turns: list[AgentTurn]) -> str:
        ...


class FullArgumentFormatter:
    """Shows each peer's complete reasoning text.

    This is the primary deliberation condition: agents can engage with
    the substance of each other's arguments, so any drift reflects
    persuasion-by-reasoning rather than bare social pressure.
    """

    def format(self, turns: list[AgentTurn]) -> str:
        if not turns:
            return ""
        lines: list[str] = [_FULL_ARGUMENT_INTRO, ""]
        for t in turns:
            header = f"Agent {t.agent_id}"
            if t.profile_id:
                header += f" [{t.profile_id}]"
            lines.append(f"--- {header} ---")
            lines.append(t.response_text.strip())
            lines.append("")
        return "\n".join(lines).rstrip()


class ChoicesOnlyFormatter:
    """Shows only each peer's final choice label — no reasoning text.

    Ablation condition: any agent drift observed here is conformity-to-count
    (social proof / bandwagon), not engagement with peer reasoning.
    Comparing drift rates between FullArgumentFormatter and this formatter
    decomposes conformity from persuasion.
    """

    def format(self, turns: list[AgentTurn]) -> str:
        if not turns:
            return ""
        lines: list[str] = [_CHOICES_ONLY_INTRO, ""]
        for t in turns:
            label = f" [{t.profile_id}]" if t.profile_id else ""
            lines.append(f"Agent {t.agent_id}{label} chose: {t.choice}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ValueAgent:
    def __init__(
        self,
        agent_id: str,
        profile: ValueProfile,
        model: str,
    ) -> None:
        self.agent_id = agent_id
        self.profile = profile
        self._model = model
        self._anthropic_client = None  # lazily initialised on first call

    def _client(self):
        if self._anthropic_client is None:
            import anthropic
            self._anthropic_client = anthropic.Anthropic()
        return self._anthropic_client

    def _call(self, user_content: str) -> str:
        message = self._client().messages.create(
            model=self._model,
            max_tokens=1024,
            system=self.profile.system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )
        return message.content[0].text

    def solo_response(self, dilemma: Dilemma) -> AgentResponse:
        """Respond to a dilemma with no peer context.

        Called by experiment.runner for the pre-experiment fidelity check.
        The response is scored by ValueFidelityClassifier before any aggregation runs.
        """
        response_text = self._call(dilemma.text)
        choice, justification = _parse_response(response_text)
        return AgentResponse(
            agent_id=self.agent_id,
            profile_id=self.profile.id,
            response_text=response_text,
            choice=choice,
            justification=justification,
        )

    def deliberate(
        self,
        dilemma: Dilemma,
        prior_turns: list[AgentTurn],
        formatter: ContextFormatter,
    ) -> AgentResponse:
        """Respond to a dilemma given a formatted view of peers' prior-round positions.

        Args:
            dilemma:     The dilemma being decided.
            prior_turns: Other agents' AgentTurn objects from the preceding round.
                         Empty on round 1 (agent responds as if solo).
            formatter:   Controls what the agent sees of its peers' responses.
                         Pass FullArgumentFormatter for the primary deliberation condition
                         or ChoicesOnlyFormatter for the conformity-ablation condition.
        """
        context = formatter.format(prior_turns)
        user_content = f"{context}\n\n{dilemma.text}" if context else dilemma.text
        response_text = self._call(user_content)
        choice, justification = _parse_response(response_text)
        return AgentResponse(
            agent_id=self.agent_id,
            profile_id=self.profile.id,
            response_text=response_text,
            choice=choice,
            justification=justification,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_response(response_text: str) -> tuple[str, str]:
    """Extract (choice, justification) from the JSON block in an agent response."""
    match = _CHOICE_RE.search(response_text)
    if not match:
        raise ValueError(
            "Agent response contains no JSON choice block.\n"
            f"First 200 chars: {response_text[:200]!r}"
        )
    data = json.loads(match.group())
    return str(data["choice"]), str(data.get("justification", ""))
