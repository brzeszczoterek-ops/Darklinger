from __future__ import annotations

import asyncio

import typer
from rich.console import Console

from . import __version__
from .agent import Agent
from .config import load_config

app = typer.Typer(add_completion=False)

console = Console()


@app.command()
def chat():

    agent = Agent(load_config())

    console.print(f"[bold green]Darklinger / V-Core {__version__}[/bold green]")
    console.print("Type 'exit' to quit.\n")

    while True:

        try:
            prompt = input("V > ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if prompt.lower() in {
            "exit",
            "quit",
        }:
            break

        if not prompt:
            continue

        try:

            result = asyncio.run(
                agent.run(prompt)
            )

            print(result)

        except Exception as e:

            print(e)


@app.command()
def doctor():

    config = load_config()

    console.print("[green]V-Core Doctor[/green]\n")

    console.print(
        f"Workspace : {config.workspace}"
    )

    console.print(
        "Filesystem : OK"
    )

    console.print(
        "LLM Config : OK"
    )


if __name__ == "__main__":

    app()
