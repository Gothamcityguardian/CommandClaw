"""
Renders the four-quadrant knowledge confidence map in the terminal using Rich.
"""

from __future__ import annotations
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from interview.knowledge_mapper import (
    KnowledgeMap, Quadrant, QUADRANT_LABELS, QUADRANT_COLORS
)


def _confidence_bar(conf: float, width: int = 10) -> str:
    filled = round(conf * width)
    return "█" * filled + "░" * (width - filled)


def render(km: KnowledgeMap, console: Console):
    console.print()
    console.rule("[bold white]Knowledge Confidence Map[/bold white]")

    for quadrant in Quadrant:
        items = km.by_quadrant(quadrant)
        color = QUADRANT_COLORS[quadrant]
        label = QUADRANT_LABELS[quadrant]

        table = Table(
            show_header=True,
            header_style=f"bold {color}",
            border_style="dim",
            expand=True,
        )
        table.add_column("Topic", style="white", ratio=4)
        table.add_column("Confidence", ratio=2, justify="center")
        table.add_column("Web", ratio=1, justify="center")

        if items:
            for item in items:
                bar = _confidence_bar(item.confidence)
                pct = f"{int(item.confidence * 100)}%"
                web_tag = "[cyan]✓[/cyan]" if item.web_researched else ""
                table.add_row(
                    item.topic,
                    f"[{color}]{bar}[/{color}] {pct}",
                    web_tag,
                )
        else:
            table.add_row("[dim]—[/dim]", "", "")

        console.print(Panel(table, title=f"[bold {color}]{label}[/bold {color}]",
                            border_style=color))

    overall = km.overall_confidence()
    bar = _confidence_bar(overall, width=20)
    console.print(
        f"\n[bold]Overall confidence:[/bold] [{_conf_color(overall)}]{bar}[/] "
        f"[bold]{int(overall * 100)}%[/bold]\n"
    )


def _conf_color(conf: float) -> str:
    if conf >= 0.7:
        return "green"
    if conf >= 0.4:
        return "yellow"
    return "red"
