#!/usr/bin/env python3
"""
Simulation test — runs a full CommandClaw session with pre-canned user answers.
Used to verify the pipeline end-to-end without needing interactive input.

Topic: How to think about validating an AI system (like SeekClaw) used in
FDA-regulated Design History File reviews — something genuinely worth figuring out.
"""

import sys
import json
from pathlib import Path
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent))

from llm.runner import LLMRunner, find_model
from interview.session import InterviewSession, SessionState
from interview.knowledge_mapper import Quadrant
from prompt_builder import confidence_map as cmap
from prompt_builder import constructor

# ── simulated user answers per phase ─────────────────────────────────────────
# Each list is the sequence of replies the "user" gives in that phase.
# The last entry in each phase is "/next" to advance.

PHASE1_ANSWERS = [
    # Goal: what am I trying to achieve?
    "I've built an offline AI tool called SeekClaw that reads medical device design "
    "documents and automatically fills in a Design History File checklist using a "
    "local LLM. I want to understand how to validate it properly before using it on "
    "real product submissions — specifically what a rigorous validation framework looks "
    "like for AI-assisted DHF review in an FDA-regulated context.",
    "/next",
]

PHASE2_ANSWERS = [
    # Knowledge: what do I know, what don't I know?
    "I know AI models hallucinate — they can sound confident while being wrong. "
    "I know software validation for medical devices generally involves IQ/OQ/PQ "
    "(installation, operational, performance qualification). I know DHF documents "
    "include things like design inputs, risk analysis, verification and validation records.",

    "I'm honestly not sure how FDA views AI-assisted review specifically. I don't know "
    "if there's a specific guidance document for AI in regulatory submissions. I also "
    "don't know how to construct a ground truth dataset for DHF checklist answers — "
    "who decides the right answer?",

    "/next",
]

PHASE3_ANSWERS = [
    # Scope: what goes in the prompt now vs later?
    "Right now I need to understand the validation approach — what test cases to build, "
    "how to measure accuracy, and what documentation I'd need. I don't need the actual "
    "FDA submission strategy yet. I also don't need code — just the conceptual framework.",
    "/next",
]


class SimulatedUI:
    """UI that replays canned answers and prints to console for inspection."""

    def __init__(self):
        self.console = Console()
        self._phase_answers = [PHASE1_ANSWERS, PHASE2_ANSWERS, PHASE3_ANSWERS]
        self._phase_idx = -1  # incremented to 0 on first phase_header call
        self._turn_idx = 0

    def welcome(self):
        self.console.rule("[bold cyan]CommandClaw — Simulation Test[/bold cyan]")
        self.console.print("[dim]Topic: Validating AI for FDA-regulated DHF reviews[/dim]\n")

    def phase_header(self, title, subtitle=""):
        self.console.print()
        self.console.rule(f"[bold cyan]{title}[/bold cyan]")
        if subtitle:
            self.console.print(f"[dim]{subtitle}[/dim]")
        self.console.print()
        # Reset turn index for new phase
        self._turn_idx = 0
        self._phase_idx = min(self._phase_idx + 1, len(self._phase_answers) - 1)

    def assistant_start(self):
        self._first_chunk = True
        self.console.print(f"\n[bold cyan]CommandClaw[/bold cyan] ", end="")

    def assistant_chunk(self, text):
        clean = text.replace("[READY_TO_ADVANCE]", "")
        if self._first_chunk:
            clean = clean.lstrip("\n")
            if not clean:
                return
            self._first_chunk = False
        if clean:
            self.console.print(clean, end="")

    def assistant_end(self):
        self.console.print()

    def user_input(self, prompt=""):
        answers = self._phase_answers[self._phase_idx]
        if self._turn_idx < len(answers):
            answer = answers[self._turn_idx]
            self._turn_idx += 1
        else:
            answer = "/next"
        self.console.print(f"\n[bold white]You[/bold white] › {answer}")
        return answer

    def status(self, msg):
        self.console.print(f"[dim italic]{msg}[/dim italic]")

    def info(self, msg):
        self.console.print(f"[yellow]{msg}[/yellow]")

    def ask_yes_no(self, question):
        self.console.print(f"\n[yellow]{question}[/yellow] → [dim]No (simulation)[/dim]")
        return False

    def show_final_prompt(self, prompt_text, out_path=None):
        self.console.print()
        self.console.rule("[bold green]Constructed Prompt[/bold green]")
        from rich.panel import Panel
        self.console.print(Panel(
            prompt_text,
            border_style="green",
            title="[bold green]Ready to paste[/bold green]",
        ))
        if out_path:
            self.console.print(f"\n[dim]Saved to: {out_path}[/dim]")


def main():
    console = Console()

    model_info = find_model()
    if not model_info:
        console.print("[bold red]No models found. Start Ollama first.[/bold red]")
        sys.exit(1)

    console.print(f"[dim]Model: {model_info['name']} ({model_info['backend']})[/dim]")

    llm = LLMRunner(model_info)
    ui = SimulatedUI()

    session = InterviewSession(llm=llm, ui=ui)

    # Patch phase_header to not double-increment phase index
    # (welcome() is called first, then phase_header for each phase)
    ui._phase_idx = 0

    try:
        session.run()
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        sys.exit(0)

    state = session.state

    console.print()
    cmap.render(state.knowledge_map, console)

    console.print("\n[dim]Building final prompt…[/dim]")
    final_prompt = constructor.build(state, llm)
    state.final_prompt = final_prompt

    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(__file__).parent / "sessions" / f"sim_prompt_{ts}.txt"
    out.write_text(final_prompt)

    ui.show_final_prompt(final_prompt, out_path=out)

    console.print("\n[bold green]Simulation complete.[/bold green]")

    # Print state summary for debugging
    console.print(f"\n[dim]Goal: {state.goal}[/dim]")
    console.print(f"[dim]Domain: {state.domain}[/dim]")
    console.print(f"[dim]Knowledge items: {len(state.knowledge_map.items)}[/dim]")
    console.print(f"[dim]In scope: {len(state.in_scope)} items[/dim]")


if __name__ == "__main__":
    main()
