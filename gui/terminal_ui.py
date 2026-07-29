"""
Rich-based terminal UI for CommandClaw.
All user-facing output goes through this class.
"""

from __future__ import annotations
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.prompt import Prompt, Confirm
from rich.text import Text
from rich import print as rprint


BRAND = "[bold cyan]Command[/bold cyan][bold white]Claw[/bold white]"


class TerminalUI:
    def __init__(self):
        self.console = Console()
        self._assistant_buf = ""

    # ── structural ──────────────────────────────────────────────────────────

    def welcome(self):
        self.console.print()
        self.console.print(Panel(
            f"{BRAND}\n[dim]Prompt construction through structured interview[/dim]\n\n"
            "[white]You will be guided through three phases:[/white]\n"
            "  1. [cyan]Goal[/cyan]          — what you're trying to achieve\n"
            "  2. [cyan]Knowledge[/cyan]     — mapping what you know and don't\n"
            "  3. [cyan]Scope[/cyan]         — what goes in the prompt now vs later\n\n"
            "[dim]Type [bold]/next[/bold] at any time to advance to the next phase.[/dim]",
            title=BRAND,
            border_style="cyan",
            expand=False,
        ))
        self.console.print()

    def phase_header(self, title: str, subtitle: str = ""):
        self.console.print()
        self.console.rule(f"[bold cyan]{title}[/bold cyan]")
        if subtitle:
            self.console.print(f"[dim]{subtitle}[/dim]")
        self.console.print()

    def status(self, msg: str):
        self.console.print(f"[dim italic]{msg}[/dim italic]")

    def info(self, msg: str):
        self.console.print(f"[yellow]{msg}[/yellow]")

    # ── conversation ────────────────────────────────────────────────────────

    def assistant_start(self):
        self._assistant_buf = ""
        self.console.print(f"\n[bold cyan]CommandClaw[/bold cyan] ", end="")

    def assistant_chunk(self, text: str):
        # Strip [READY_TO_ADVANCE] if it leaks through
        clean = text.replace("[READY_TO_ADVANCE]", "")
        if clean:
            self.console.print(clean, end="")
            self._assistant_buf += clean

    def assistant_end(self):
        self.console.print()  # newline after streamed reply

    def user_input(self, prompt: str = "") -> str:
        self.console.print()
        label = prompt or "[bold white]You[/bold white] › "
        return self.console.input(label)

    # ── decisions ───────────────────────────────────────────────────────────

    def ask_yes_no(self, question: str) -> bool:
        self.console.print()
        return Confirm.ask(f"[yellow]{question}[/yellow]", default=False)

    # ── final output ────────────────────────────────────────────────────────

    def show_final_prompt(self, prompt_text: str, out_path=None):
        self.console.print()
        self.console.rule("[bold green]Your Constructed Prompt[/bold green]")
        self.console.print(Panel(
            prompt_text,
            border_style="green",
            title="[bold green]Ready to paste[/bold green]",
            subtitle="[dim]Copy the text above[/dim]",
        ))
        if out_path:
            self.console.print(f"\n[dim]Also saved to: {out_path}[/dim]")
        self.console.print()
