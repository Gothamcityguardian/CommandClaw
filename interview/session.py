"""
Interview session: runs the three conversation phases, maintains state,
coordinates LLM calls and UI output.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field

from interview.knowledge_mapper import KnowledgeMap, Quadrant, KnowledgeItem


ADVANCE_SIGNAL = "[READY_TO_ADVANCE]"


@dataclass
class SessionState:
    goal: str = ""
    domain: str = ""
    sub_goals: list = field(default_factory=list)
    knowledge_map: KnowledgeMap = field(default_factory=KnowledgeMap)
    in_scope: list = field(default_factory=list)
    deferred: list = field(default_factory=list)
    assumptions: list = field(default_factory=list)
    success_criteria: str = ""
    web_findings: list = field(default_factory=list)
    final_prompt: str = ""


# ─── System prompts ───────────────────────────────────────────────────────────

_GOAL_SYSTEM = """\
You are CommandClaw, an expert prompt engineer helping the user construct a precise, \
high-quality prompt they will send to a top-tier AI model.

PHASE: Goal Extraction

Your job is to understand exactly what the user wants to achieve.
Probe past the surface request to uncover:
  • The real end goal (not just the immediate task)
  • Intermediate milestones or phase goals
  • The primary domain or field
  • What they need NOW versus what can be deferred

Rules:
  – Ask one or two focused questions per turn, never a long list.
  – Follow up on vague or incomplete answers.
  – When you believe you have a clear enough picture, end your message with \
exactly: """ + ADVANCE_SIGNAL + """
  – The user may also type /next at any time to move on.
  – Do not explain the interview process; just conduct it naturally.
"""

_KNOWLEDGE_SYSTEM = """\
You are CommandClaw, an expert prompt engineer.

PHASE: Knowledge Assessment
Domain: {domain}
Goal: {goal}

Your job is to map the user's knowledge across four quadrants:
  • Known Knowns  — things they're confident about
  • Known Unknowns — gaps they already recognise
  • Unknown Knowns — things they know but haven't connected to this task
  • Unknown Unknowns — gaps they don't know they have

Probe each quadrant with targeted questions:
  – Ask them to explain key concepts to surface Known Knowns vs Unknown Knowns.
  – Ask what feels unclear or uncertain to surface Known Unknowns.
  – Ask about common pitfalls or adjacent concepts they may not have considered \
to surface Unknown Unknowns.

Rules:
  – One or two questions per turn.
  – Acknowledge what they know confidently; don't be condescending about gaps.
  – When you've built a clear picture, end your message with """ + ADVANCE_SIGNAL + """
  – The user may type /next to advance.
"""

_SCOPE_SYSTEM = """\
You are CommandClaw, an expert prompt engineer.

PHASE: Scope & Constraints
Goal: {goal}
Domain: {domain}

Your job is to nail down the scope so the final prompt is focused and actionable:
  • What must be included in the prompt right now?
  • What can be deferred to a follow-up?
  • What assumptions or constraints should be stated explicitly?
  • What does a successful response look like?

Rules:
  – One or two questions per turn.
  – Help the user think about what they DON'T need now — focus sharpens a prompt.
  – When scope is clear, end your message with """ + ADVANCE_SIGNAL + """
  – The user may type /next to advance.
"""

# ─── Extraction prompts ───────────────────────────────────────────────────────

_GOAL_EXTRACT = """\
Based on the conversation so far, extract the following as valid JSON \
(no markdown fences, no commentary):

{
  "goal": "one clear sentence describing the end goal",
  "domain": "primary domain or field",
  "sub_goals": ["milestone or phase goal 1", "..."],
  "urgency": "now | later | mixed"
}
"""

_KNOWLEDGE_EXTRACT = """\
Based on the conversation so far, extract the user's knowledge map as valid JSON \
(no markdown fences, no commentary):

{
  "known_knowns":     [{"topic": "...", "confidence": 0.0-1.0, "evidence": "brief reason"}],
  "known_unknowns":   [{"topic": "...", "confidence": 0.0-1.0, "evidence": "brief reason"}],
  "unknown_knowns":   [{"topic": "...", "confidence": 0.0-1.0, "evidence": "brief reason"}],
  "unknown_unknowns": [{"topic": "...", "confidence": 0.0-1.0, "evidence": "brief reason"}]
}

Confidence means: how well does the user understand this topic (1.0 = expert, 0.0 = unaware).
"""

_SCOPE_EXTRACT = """\
Based on the conversation so far, extract scope decisions as valid JSON \
(no markdown fences, no commentary):

{
  "in_scope":    ["item that must be in the prompt now"],
  "deferred":    ["item that can wait for a follow-up"],
  "assumptions": ["assumption that should be stated explicitly in the prompt"],
  "success_criteria": "what a good response looks like in one sentence"
}
"""


# ─── Session class ────────────────────────────────────────────────────────────

class InterviewSession:
    def __init__(self, llm, ui):
        self.llm = llm
        self.ui = ui
        self.state = SessionState()
        self.history: list[dict] = []

    # ── public entry point ────────────────────────────────────────────────────

    def run(self):
        self.ui.welcome()

        self.ui.phase_header("Phase 1 / 3 — Goal", "What are you trying to achieve?")
        self._run_phase(_GOAL_SYSTEM, _GOAL_EXTRACT, self._apply_goal)

        self.ui.phase_header("Phase 2 / 3 — Knowledge Assessment",
                             "Let's map what you know and what you don't.")
        knowledge_system = _KNOWLEDGE_SYSTEM.format(
            domain=self.state.domain or "general",
            goal=self.state.goal or "unspecified",
        )
        self._run_phase(knowledge_system, _KNOWLEDGE_EXTRACT, self._apply_knowledge)

        self.ui.phase_header("Phase 3 / 3 — Scope & Constraints",
                             "What goes in the prompt now, and what can wait?")
        scope_system = _SCOPE_SYSTEM.format(
            goal=self.state.goal,
            domain=self.state.domain or "general",
        )
        self._run_phase(scope_system, _SCOPE_EXTRACT, self._apply_scope)

    # ── phase runner ──────────────────────────────────────────────────────────

    def _run_phase(self, system_prompt: str, extract_prompt: str, apply_fn):
        phase_history: list[dict] = [{"role": "system", "content": system_prompt}]

        # Opening move from LLM
        opening = self._llm_turn(phase_history)
        phase_history.append({"role": "assistant", "content": opening})

        ready = ADVANCE_SIGNAL in opening

        while not ready:
            user_text = self.ui.user_input()
            if user_text.strip().lower() in ("/next", "next"):
                break
            phase_history.append({"role": "user", "content": user_text})

            reply = self._llm_turn(phase_history)
            phase_history.append({"role": "assistant", "content": reply})
            ready = ADVANCE_SIGNAL in reply

        # Extraction step (silent)
        self.ui.status("Analysing…")
        extract_messages = phase_history + [{"role": "user", "content": extract_prompt}]
        data = self.llm.extract_json(extract_messages)
        if data:
            apply_fn(data)

        # Carry forward into global history for context in later phases
        self.history.extend(phase_history[1:])  # skip per-phase system prompt

    # ── LLM wrapper with streaming ────────────────────────────────────────────

    def _llm_turn(self, messages: list[dict]) -> str:
        self.ui.assistant_start()
        full = ""
        for chunk in self.llm.stream_chat(messages):
            # Strip the advance signal from displayed text
            visible = chunk.replace(ADVANCE_SIGNAL, "")
            if visible:
                self.ui.assistant_chunk(visible)
            full += chunk
        self.ui.assistant_end()
        return full

    # ── extraction appliers ───────────────────────────────────────────────────

    def _apply_goal(self, data: dict):
        self.state.goal = data.get("goal", "")
        self.state.domain = data.get("domain", "")
        self.state.sub_goals = data.get("sub_goals", [])

    def _apply_knowledge(self, data: dict):
        self.state.knowledge_map.add_from_dict(data)

    def _apply_scope(self, data: dict):
        self.state.in_scope = data.get("in_scope", [])
        self.state.deferred = data.get("deferred", [])
        self.state.assumptions = data.get("assumptions", [])
        self.state.success_criteria = data.get("success_criteria", "")

    # ── optional gap fill ─────────────────────────────────────────────────────

    def run_gap_fill(self):
        from interview import web_searcher

        gaps = self.state.knowledge_map.by_quadrant(Quadrant.UNKNOWN_UNKNOWN) + \
               self.state.knowledge_map.by_quadrant(Quadrant.KNOWN_UNKNOWN)

        if not gaps:
            self.ui.info("No gaps found to research.")
            return

        self.ui.info("Identified gaps for research:")
        for i, item in enumerate(gaps, 1):
            self.ui.info(f"  {i}. {item.topic}")

        choice = self.ui.user_input(
            "Enter gap numbers to research (e.g. 1,3) or press Enter to skip: "
        ).strip()
        if not choice:
            return

        indices = []
        for part in choice.split(","):
            try:
                idx = int(part.strip()) - 1
                if 0 <= idx < len(gaps):
                    indices.append(idx)
            except ValueError:
                pass

        for idx in indices:
            item = gaps[idx]
            query = f"{item.topic} {self.state.domain} overview"
            self.ui.status(f"Searching: {query}")
            results = web_searcher.search(query)
            summary = web_searcher.summarise_for_llm(results)

            # Ask LLM to distil for the user
            distil_messages = [
                {"role": "system", "content": (
                    f"You are a helpful researcher. The user wants to understand "
                    f"'{item.topic}' in the context of: {self.state.goal}. "
                    f"Summarise the following search results in plain language, "
                    f"focusing on what's most relevant. Keep it under 200 words."
                )},
                {"role": "user", "content": summary},
            ]
            self.ui.assistant_start()
            full = ""
            for chunk in self.llm.stream_chat(distil_messages, max_tokens=400):
                self.ui.assistant_chunk(chunk)
                full += chunk
            self.ui.assistant_end()

            self.state.web_findings.append({"topic": item.topic, "summary": full})

            # Raise confidence for this item
            item.confidence = min(item.confidence + 0.3, 0.7)
            item.web_researched = True
