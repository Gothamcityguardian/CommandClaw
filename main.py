#!/usr/bin/env python3
"""
CommandClaw — entry point.
Loads model, runs the interview, shows confidence map, builds prompt.
"""

import sys
import json
import datetime
from pathlib import Path

import config
from llm.runner import LLMRunner, find_model
from interview.session import InterviewSession
from interview.knowledge_mapper import Quadrant
from prompt_builder import confidence_map as cmap
from prompt_builder import constructor
from gui.terminal_ui import TerminalUI


def main():
    ui = TerminalUI()

    # ── find interview model ──────────────────────────────────────────────────
    model_path = find_model(config.INTERVIEW_MODEL_PRIORITY)
    if not model_path:
        ui.console.print(
            f"[bold red]No GGUF models found in {config.MODELS_DIR}[/bold red]\n"
            "Run [bold]python download_models.py[/bold] first."
        )
        sys.exit(1)

    ui.console.print(f"[dim]Loading model: {model_path.name}[/dim]")
    llm = LLMRunner(model_path)

    # ── run interview ─────────────────────────────────────────────────────────
    session = InterviewSession(llm=llm, ui=ui)
    try:
        session.run()
    except KeyboardInterrupt:
        ui.console.print("\n[dim]Session interrupted.[/dim]")
        sys.exit(0)

    state = session.state

    # ── confidence map ────────────────────────────────────────────────────────
    cmap.render(state.knowledge_map, ui.console)

    # ── optional gap fill ─────────────────────────────────────────────────────
    gaps = (state.knowledge_map.by_quadrant(Quadrant.UNKNOWN_UNKNOWN) +
            state.knowledge_map.by_quadrant(Quadrant.KNOWN_UNKNOWN))
    if gaps and ui.ask_yes_no("Would you like to research any knowledge gaps before building the prompt?"):
        session.run_gap_fill()
        ui.console.print("\n[dim]Updated confidence map:[/dim]")
        cmap.render(state.knowledge_map, ui.console)

    # ── synthesis model (optional swap) ──────────────────────────────────────
    if not config.USE_SINGLE_MODEL:
        synthesis_path = find_model(config.SYNTHESIS_MODEL_PRIORITY)
        if synthesis_path and synthesis_path != model_path:
            ui.status(f"Loading synthesis model: {synthesis_path.name}")
            llm.unload()
            llm = LLMRunner(synthesis_path)

    # ── build prompt ──────────────────────────────────────────────────────────
    ui.status("Building your prompt…")
    final_prompt = constructor.build(state, llm)
    state.final_prompt = final_prompt

    # ── save session ──────────────────────────────────────────────────────────
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_file = config.SESSIONS_DIR / f"session_{timestamp}.json"
    prompt_file = config.SESSIONS_DIR / f"prompt_{timestamp}.txt"

    session_data = {
        "goal": state.goal,
        "domain": state.domain,
        "sub_goals": state.sub_goals,
        "in_scope": state.in_scope,
        "deferred": state.deferred,
        "assumptions": state.assumptions,
        "success_criteria": state.success_criteria,
        "web_findings": state.web_findings,
        "knowledge_map": [
            {
                "topic": i.topic,
                "quadrant": i.quadrant.value,
                "confidence": i.confidence,
                "evidence": i.evidence,
                "web_researched": i.web_researched,
            }
            for i in state.knowledge_map.items
        ],
        "final_prompt": final_prompt,
    }
    session_file.write_text(json.dumps(session_data, indent=2))
    prompt_file.write_text(final_prompt)

    ui.show_final_prompt(final_prompt, out_path=prompt_file)


if __name__ == "__main__":
    main()
