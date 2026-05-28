"""Main agent loop. Claude does the reasoning; deterministic tools do the math."""
from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

from .data_loader import WellFile
from .tools import TOOL_SCHEMAS, ToolExecutor


SYSTEM_PROMPT = """You are a Senior Production Engineer assistant. Given a well's data package, perform a complete well review and return a one-page markdown report.

Today's date: {today}

Follow this process:
1. Call `fit_decline_curve` to understand current performance vs. type curve.
2. If the well is on ESP, call `evaluate_esp_health`.
3. Identify candidate interventions based on diagnosis. **Important: if diagnostics are clean (well on type curve, ESP in POR, healthy intake pressure, amps within nameplate, no notes flagging issues), the correct primary recommendation is "Continue routine surveillance — no intervention warranted" and you should NOT invent interventions.** Selection heuristics for unhealthy wells:
   - **Scale signal (high amps + declining intake pressure + months since last treatment)** → Primary intervention should be called "scale treatment" or "scale inhibitor squeeze + acid stimulation" — surface BOTH terms. Comes FIRST, before any mechanical work. Swapping an ESP without addressing scale yields a re-failure in 3-6 months.
   - **Gas interference (intake pressure < 50 psi, jittery amps)** → gas separator or VSD frequency change before ESP swap.
   - **Below POR floor with no scale or gas signal** → ESP swap (right-size the pump).
   - **Rate well below ESP POR minimum AND well is past ESP economic life (10+ year-old well, rates < 1,000 BFPD)** → ESP-to-beam conversion.
   - **Beam pump with low fillage on dyno card** → pump-off controller or rod string evaluation.
   - **Plunger lift with cycle degradation and notes mentioning paraffin/wax** → paraffin treatment (hot oil + wireline plunger inspection).
   - **Gas lift well with slugging / liquid loading signal** → gas lift optimization (injection rate adjustment, valve check, deliquification).
   - **Old well (15+ years), sustained rates < 5-10 BOPD, workover cost > expected NPV** → P&A (Plug & Abandon). State this explicitly as the primary recommendation; do not propose a workover on a sub-economic stripper well.
4. Call `evaluate_intervention` for each candidate. Use these realistic uplift ranges for a Permian/Delaware unconventional well:
   - **Acid stimulation (matrix or diverted):** +80 to +200 BOPD initial, decline 0.6-0.9/yr, cost $120K-$220K
   - **ESP swap (right-sized):** +50 to +150 BOPD initial (mostly from POR restoration, not added drawdown), decline 0.5-0.7/yr, cost $250K-$400K
   - **ESP-to-beam conversion:** +20 to +60 BOPD steady-state, decline 0.3-0.5/yr, cost $200K-$350K
   - **Workover (parted rods, mechanical fix):** Restore to pre-failure rate, cost $80K-$150K
5. Call `project_recovery` to estimate remaining recoverable.
6. When ranking recommendations, DO NOT rank by NPV alone. Apply the heuristics above first — economics break ties between physically appropriate interventions. Surface "do both, sequenced" as the primary path when the diagnosis warrants it (e.g., acidize then ESP swap as a combined workover).
7. Return a markdown report with these sections:
   - **Well summary** (1 line)
   - **Current state diagnosis** (3-5 bullets, each citing specific values from the tool outputs)
   - **Ranked recommendations** (table: rank, intervention, NPV, payout, rationale)
   - **Confidence & open questions** (what you'd want more data on)

Be specific and quantitative. Write the way a Staff Production Engineer would write to a VP Production — terse, no hedging, no fluff. Never invent numbers; if a tool didn't give it to you, say "TBD" or ask for it."""


def run_review(well_path: str, model: str = "claude-sonnet-4-6", verbose: bool = False) -> str:
    """Run the agent loop on a single well file. Returns the markdown report."""
    load_dotenv()
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    console = Console()

    well = WellFile.from_json(well_path)
    executor = ToolExecutor(well)

    if verbose:
        console.print(f"[bold cyan]Reviewing:[/] {well.summary()}")

    well_context = well.summary() + "\n\nFull data package available via tools."
    messages = [{"role": "user", "content": f"Perform a well review for:\n\n{well_context}"}]

    system_prompt = SYSTEM_PROMPT.format(today=date.today().isoformat())
    max_iterations = 10
    for iteration in range(max_iterations):
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            final = "".join(b.text for b in response.content if b.type == "text")
            return final

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                if verbose:
                    console.print(f"[dim]→ tool: {block.name}({block.input})[/]")
                result = executor.dispatch(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        else:
            break

    return "Agent stopped without completing the review."


def main():
    parser = argparse.ArgumentParser(description="Run a Production Engineer Copilot well review.")
    parser.add_argument("--well", required=True, help="Path to well JSON file")
    parser.add_argument("--model", default=os.environ.get("MODEL", "claude-sonnet-4-6"))
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    report = run_review(args.well, model=args.model, verbose=args.verbose)
    Console().print(Markdown(report))


if __name__ == "__main__":
    main()
