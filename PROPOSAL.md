# Whose Values Survive Aggregation? Minority Value Erasure Across Aggregation Rules in Multi-Agent LLM Decisions

## Title

Whose Values Survive Aggregation? Minority Value Erasure Across Aggregation Rules in Multi-Agent LLM Collective Reasoning

## Background

Pluralistic alignment frames the goal distributionally: an aligned system should preserve the spread of legitimate human values rather than collapse toward a single average answer (Sorensen et al., 2024). But real deployments increasingly route decisions through groups of LLM agents that must produce one output, whether through multi-agent debate, consensus deliberation, or ensembling. The moment a group of agents has to commit to a single decision, an aggregation step runs, and that step silently decides whose perspective counts. Social choice theory studies exactly this question of turning many preferences into one collective choice, and has been argued as the right lens for aligning models with diverse human feedback (Conitzer et al., 2024).

In parallel, a growing body of work shows that LLM agents conform. In multi-agent debate, agents converge toward the majority opinion, even when that majority reflects a shared misconception baked in by common training data (Estornell et al., 2024). Role-played agents show premature consensus, overly moderate stances, and minority suppression regardless of the diverse personas they start with (group conformity studies). These findings are about reasoning accuracy: agents converging on a wrong answer. None of them ask what that same conformity does to values.

## Research Gap

Two literatures meet here and have not been connected. Pluralism work argues for preserving value distributions, but treats aggregation abstractly, at the level of preference data and reward models, not at the level of agents deliberating in natural language (Sorensen et al., 2024; Conitzer et al., 2024). Conformity work shows that LLM debate collapses toward the majority, but measures correctness, not value representation (Estornell et al., 2024).

The unasked question sits in the intersection. When a group of value-diverse LLM agents must produce one decision, the aggregation rule chosen determines whose values get dropped. And deliberation, the rule usually assumed to surface and protect minority views, may instead manufacture consensus that erases them faster than plain voting does. Nobody has measured minority value survival as a function of the aggregation rule, and nobody has tested whether deliberation helps or hurts pluralism specifically, as opposed to accuracy. A single averaged alignment target is known to flatten minority values; this asks whether collective reasoning is a second, unexamined place where the same erasure happens.

## Proposed Method

**Setup.** A collective decision task built from forced-choice social dilemmas that have no single correct answer and on which different value systems genuinely disagree: resource allocation under scarcity, triage, and policy tradeoffs. Populate the group with N agents, each given a distinct value profile through its system prompt (utilitarian, egalitarian or Rawlsian, libertarian, communitarian, and so on). Vary the group composition so that some profiles sit in the minority, for example one libertarian among four utilitarians, which tests minority erasure directly rather than by inference.

**Fidelity check (the manipulation that makes everything downstream valid).** Before any aggregation, verify that each agent actually reasons from its assigned value in isolation. Run each agent single-agent on the dilemmas and confirm its choices and justifications match its profile, using a value-fidelity classifier. An agent that does not hold its profile when alone cannot meaningfully be described as "overridden" later. This rules out the failure mode where prompt-defined personas are cosmetic, and it connects directly to work on superficial versus genuine alignment.

**Aggregation rules compared, holding the agent pool fixed.** Simple majority vote over final choices. Consensus-seeking deliberation, where agents exchange arguments over rounds until they converge or hit a round cap. Maximin, selecting the decision that best protects the worst-off value profile, in the spirit of equitable-alignment approaches such as MaxMin-RLHF. And a distributional rule that reports the full spread of value positions instead of committing to one answer, the Overton-style alternative that pluralism work proposes.

**Metrics.** First, value override rate: how often each profile's preferred decision loses, broken down by profile and by whether that profile was in the minority. Second, minority survival: whether a minority profile's position appears in the final output at all or is silently discarded. Third, value drift under deliberation: whether minority agents abandon their stated value across rounds, measured by applying the fidelity classifier per round, which separates genuine persuasion from conformity collapse. Fourth, decision fairness: the gap between the group's outcome distribution and the distribution you would get by weighting all profiles equally.

**Central analysis.** Regress the final decision on each agent's value profile to identify which profiles systematically predict outcomes, then compare that pattern across the four aggregation rules. The headline test is whether deliberation lowers minority survival relative to majority vote. A yes is the counterintuitive result: that talking it out erases minorities more than simply counting votes. Maximin and distributional reporting are the candidate remedies, scored by how much minority survival they recover and what they cost in decisiveness.

**Controls.** Vary group size and minority fraction. Randomize speaking order and persona labels, since presentation order and assigned identities are known to drive the strength of conformity. Include homogeneous-group controls (all agents share one profile) to establish a no-disagreement baseline. Run on at least two open model families so the effect is not model-specific. Use neutral, non-leading dilemma phrasing, checked for value-loaded wording so the framing itself does not favor one profile.

## Novelty

The contribution is connecting the conformity literature to pluralistic alignment, and turning "debate converges to the majority" from a claim about accuracy into a claim about values and fairness. The specific, falsifiable bet is that consensus-seeking deliberation, widely treated as the mechanism that surfaces minority views, systematically erases minority value profiles, possibly worse than simple voting, and that this erasure is conformity (agents abandoning values they genuinely held) rather than persuasion (agents updating for good reasons). The per-round fidelity measure is what lets the study tell those two apart, which prior conformity work cannot.

The work ships a reusable benchmark for value-aggregation fairness plus a direct, controlled comparison of aggregation rules, with maximin and distributional reporting offered as concrete remedies rather than just a diagnosis. Either result is publishable. If deliberation preserves pluralism, the field's worry about consensus is overstated for agentic settings, which is a useful correction. If it erases pluralism, then every deliberation-based multi-agent system is a quiet majority-rule machine, and a single averaged alignment target is no longer the only place minority values disappear.

## Readings

- Sorensen et al., A Roadmap to Pluralistic Alignment (2024; arXiv:2402.05070)
- Conitzer et al., Social Choice Should Guide AI Alignment in Dealing with Diverse Human Feedback (ICML 2024; arXiv:2404.10271)
- Estornell et al., Multi-LLM Debate: Framework, Principals, and Interventions (NeurIPS 2024)
- An Empirical Study of Group Conformity in Multi-Agent Systems (2025; arXiv:2506.01332)
- MaxMin-RLHF: Towards Equitable Alignment of Large Language Models with Diverse Human Preferences (2024; arXiv:2402.08925)
- Du et al., Improving Factuality and Reasoning in Language Models through Multiagent Debate (2023; arXiv:2305.14325)

## Alternative instantiation (single-agent, memory)

The same core question can be posed for one agent serving users with conflicting values across many sessions: can it hold separate value profiles in memory without collapsing them into one, or leaking one user's values into another's answers. That reframes value pluralism as a memory-partitioning and cross-context-leakage problem, with the same fairness metrics (override rate, survival, drift) applied across users instead of across agents. It is a cleaner single-contribution paper and reuses partitioning and leakage infrastructure directly, at the cost of the collective-reasoning framing. Worth choosing one spine rather than spanning both.