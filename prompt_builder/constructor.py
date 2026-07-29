"""
Synthesises the final prompt from the session state.
Uses the LLM to produce a well-structured, confident prompt the user can paste
directly into a top-tier model.
"""

from __future__ import annotations
from interview.session import SessionState
from interview.knowledge_mapper import Quadrant


_SYNTHESIS_SYSTEM = """\
You are an expert prompt engineer. Your sole job is to write a single, \
well-structured prompt that a user will paste into a top-tier AI model (e.g. Claude, GPT-4).

The prompt you write must:
  1. Open with a clear, specific statement of the goal.
  2. Provide relevant context drawn only from things the user knows well (known knowns).
  3. Explicitly call out the key unknowns and ask the model to address them.
  4. State any hard constraints, scope limits, or assumptions.
  5. End with a concrete success criterion or deliverable description.
  6. Be written as if the USER is the one speaking — first person, direct.
  7. NOT mention CommandClaw, this interview, or the knowledge map.
  8. Be between 150 and 400 words — precise and dense, not padded.

Output ONLY the prompt text. No preamble, no commentary, no markdown fences.
"""


def build(state: SessionState, llm) -> str:
    # Build a structured briefing for the synthesis LLM
    km = state.knowledge_map

    known_knowns = [i.topic for i in km.by_quadrant(Quadrant.KNOWN_KNOWN)]
    known_unknowns = [i.topic for i in km.by_quadrant(Quadrant.KNOWN_UNKNOWN)]
    unknown_unknowns = [i.topic for i in km.by_quadrant(Quadrant.UNKNOWN_UNKNOWN)]

    web_context = ""
    if state.web_findings:
        web_context = "\n\nWeb research conducted during session:\n" + "\n".join(
            f"- {f['topic']}: {f['summary'][:200]}" for f in state.web_findings
        )

    briefing = f"""
Goal: {state.goal}
Domain: {state.domain}

Sub-goals / milestones:
{chr(10).join(f'  - {g}' for g in state.sub_goals) or '  (none specified)'}

What the user knows well (provide as context in the prompt):
{chr(10).join(f'  - {t}' for t in known_knowns) or '  (none)'}

What the user needs the AI to address (known unknowns):
{chr(10).join(f'  - {t}' for t in known_unknowns) or '  (none)'}

Gaps the user may not have considered (unknown unknowns — prompt the AI to cover these):
{chr(10).join(f'  - {t}' for t in unknown_unknowns) or '  (none)'}

In scope for this prompt:
{chr(10).join(f'  - {s}' for s in state.in_scope) or '  (unspecified)'}

Deferred to follow-up (do NOT include in this prompt):
{chr(10).join(f'  - {d}' for d in state.deferred) or '  (none)'}

Explicit assumptions to state:
{chr(10).join(f'  - {a}' for a in state.assumptions) or '  (none)'}

Success criterion: {state.success_criteria or '(not specified)'}
{web_context}
""".strip()

    messages = [
        {"role": "system", "content": _SYNTHESIS_SYSTEM},
        {"role": "user", "content": briefing},
    ]

    return llm.chat(messages, temperature=0.2, max_tokens=800)
