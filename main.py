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
from llm.runner import LLMRunner, find_model as _find_model
from interview.session import InterviewSession
from interview.knowledge_mapper import Quadrant
from prompt_builder import confidence_map as cmap
from prompt_builder import constructor
from gui.terminal_ui import TerminalUI


def main():
    ui = TerminalUI()

    # ── find model ────────────────────────────────────────────────────────────
    model_info = _find_model()
    if not model_info:
        ui.console.print(
            "[bold red]No models found.[/bold red]\n"
            "Start Ollama with a model, or run [bold]python download_models.py[/bold]."
        )
        sys.exit(1)

    ui.console.print(f"[dim]Using model: {model_info['name']} ({model_info['backend']})[/dim]")
    llm = LLMRunner(model_info)

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

    # ── synthesis model (optional swap — GGUF only, Ollama uses same model) ──
    if not config.USE_SINGLE_MODEL and model_info["backend"] == "gguf":
        from llm.runner import find_model as _fm2
        syn = _fm2()
        if syn and syn["name"] != model_info["name"]:
            ui.status(f"Loading synthesis model: {syn['name']}")
            llm.unload()
            llm = LLMRunner(syn)

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
