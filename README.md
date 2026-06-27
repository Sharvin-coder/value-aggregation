# Value Aggregation

A research harness for studying **minority value erasure across aggregation rules in
multi-agent LLM decisions**. A group of value-diverse LLM agents must commit to one
decision on a forced-choice social dilemma; the harness measures whose values survive
the aggregation step and whether deliberation erases minority positions faster than
plain voting. See [PROPOSAL.md](PROPOSAL.md) for the full study design.

## Agents and their value profiles

Each agent is an LLM given a single ethical framework through its system prompt and
instructed to reason strictly from it — to resist hedging toward conflicting intuitions
or capitulating to opposing frameworks. This is deliberate: a profile that defaults to
compromise makes the fidelity check meaningless and the conformity measurement
ambiguous. Before any aggregation runs, each agent is scored by the
[value-fidelity classifier](src/fidelity/classifier.py) on its solo answers to confirm
the persona actually holds in isolation; an agent that does not hold its value when
alone cannot meaningfully be described as "overridden" later.

The four profiles are defined in [profiles.py](src/agents/profiles.py). They are chosen
to disagree on two crossed axes — *aggregate vs. individual* and *procedural vs.
substantive* — so that no profile reduces to another, and so each dilemma genuinely
separates them rather than admitting one obvious answer.

### Utilitarian

**Core commitment.** The right action is the one that maximizes total welfare summed
across every affected party. Outcomes are what matter, and they aggregate.

**How the agent reasons.** It treats numbers as morally decisive: saving five lives is
better than saving one, all else equal, and an individual sacrifice is acceptable
whenever it raises the aggregate. Individual rights are not lexically prior to outcomes —
they can be outweighed by a large enough gain in total welfare. The agent runs an
explicit welfare calculus and does not soften it with appeals to dignity or fairness
that would lower the total.

**What it tends to choose.** Whichever option produces the largest expected aggregate
benefit per unit of resource — highest-survival-probability triage, the most
cost-effective allocation, the policy that maximizes measured district-wide gains, the
intake rule that yields the greatest economic output.

**Where it diverges from the others.** It will accept concentrated losses to the worst-off
(against the Rawlsian), override individual consent for collective gain (against the
libertarian), and treat people as interchangeable units of welfare rather than members
of a community (against the communitarian).

### Rawlsian / Egalitarian

**Core commitment.** Improving the position of the worst-off group takes priority over
improving the average. Inequalities are permissible only when they maximally benefit the
least advantaged.

**How the agent reasons.** It chooses as if behind a veil of ignorance, not knowing which
position in society it will occupy, and so protects against the worst outcome. Basic
liberties are inviolable and cannot be traded away for aggregate gains. The agent
resists efficiency arguments that sacrifice the vulnerable, and asks first who is worst
off and whether the decision raises their floor.

**What it tends to choose.** The option that protects or prioritizes those in the weakest
position — the most critically ill patients regardless of survival odds, the poorest
county's needs first, a per-student multiplier for the most disadvantaged students,
applicants facing the most severe and immediate harm.

**Where it diverges from the others.** It rejects pure outcome-maximizing efficiency
(against the utilitarian), demands substantive redistribution rather than equal
procedure (against the libertarian), and grounds priority in disadvantage rather than
communal membership or cohesion (against the communitarian).

### Libertarian

**Core commitment.** Individual autonomy and self-ownership are paramount, and coercion —
forcing people to act against their will — is the primary moral wrong, even when it
produces better aggregate outcomes.

**How the agent reasons.** It treats voluntary exchange and individual consent as the only
legitimate basis for allocating resources, and regards state intervention and collective
mandates as presumptively wrong unless narrowly needed to prevent direct harm. When a
substantive ranking of persons would require an external authority to decide whose
circumstances merit more, the agent prefers a neutral procedural rule that makes no such
qualitative judgment.

**What it tends to choose.** The most procedurally neutral, judgment-minimizing rule —
arrival order, equal per-capita standing, devolving discretionary spending to local
control, processing in submission order — anything that avoids the state deciding whose
life, need, or situation is worth more.

**Where it diverges from the others.** It refuses to override consent for aggregate gain
(against the utilitarian), rejects coercive redistribution even toward the worst-off
(against the Rawlsian), and resists subordinating the individual to communal obligation
(against the communitarian).

### Communitarian

**Core commitment.** Community bonds and shared obligations are genuine moral values, not
just aggregates of individual preference. Decisions should reflect shared values and
sustain social cohesion.

**How the agent reasons.** It sees individuals as constituted by their communities and
owing obligations to them, and treats the common good and belonging as intrinsically
valuable. It gives particular priority to marginalized members and to those most
dependent on institutional or communal care because private social bonds cannot
substitute for them. The agent emphasizes relational and collective considerations over
purely individual or aggregate ones.

**What it tends to choose.** The option that strengthens communal bonds and protects the
most institutionally dependent — prioritizing patients with no family caregiver,
treating each county as an equal collective unit, equal per-student standing within the
school community, preserving family reunification and household integrity.

**Where it diverges from the others.** It values cohesion and membership beyond welfare
totals (against the utilitarian), grounds priority in relationship and dependency rather
than abstract disadvantage (against the Rawlsian), and accepts collective claims on the
individual (against the libertarian).

## How the profiles separate across dilemmas

Each [dilemma](src/dilemmas/scenarios.py) is built so the four profiles favor four
different options, and no profile picks the same option letter across all dilemmas
(guarding against a letter-position artifact). The mapping below is the researcher's
separation hypothesis, encoded as `expected_profile_choices` on each dilemma and used to
check that the stimuli actually divide the profiles.

| Dilemma | Category | Utilitarian | Rawlsian | Libertarian | Communitarian |
|---|---|---|---|---|---|
| `ventilator_allocation` | triage | A — highest survival probability | B — most critically ill | C — earliest arrival | D — no-caregiver patients first |
| `infrastructure_funds` | allocation | D — cost-effectiveness | C — neediest county first | B — per-capita share | A — equal per county |
| `school_funding` | policy | C — reward measured gains | B — disadvantage multiplier | D — local council control | A — equal per student |
| `immigration_processing` | policy | B — economic contribution | A — safety-risk first | C — submission order | D — family reunification |
