"""Run the agent against the eval set and compute agreement rate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

from src.agent import run_review


CASES_FILE = Path(__file__).parent / "cases.yaml"
RESULTS_DIR = Path(__file__).parent / "results"


# Synonym groups — a recommendation matches if ANY of the listed phrases appear in the report.
# Keys are the strings used in cases.yaml's expected_primary_recommendation field.
RECOMMENDATION_SYNONYMS = {
    "acid_stimulation": ["acid stimulation", "acid stim", "matrix acidiz", "diverted acid"],
    "scale_treatment": ["scale treatment", "scale inhibitor", "scale squeeze", "acid stim", "acidize", "chemical treatment"],
    "esp_swap": ["esp swap", "pump swap", "right-siz", "replace the esp", "esp replacement"],
    "esp-to-beam_conversion": ["esp-to-beam", "esp to beam", "beam pump conversion", "convert to beam"],
    "gas_separator": ["gas separator", "downhole separator", "vsd frequency", "gas-lock mitigation"],
    "gas_lift_optimization": ["gas lift optimization", "injection rate", "lift optimization", "valve check", "deliquification"],
    "pump-off_controller": ["pump-off controller", "pump off controller", "poc", "rod string evaluation"],
    "paraffin_treatment": ["paraffin", "hot oil", "wax treatment", "wireline plunger"],
    "workover": ["workover", "rig", "rod replacement", "well intervention"],
    "monitor": ["continue routine surveillance", "no intervention", "monitor", "routine surveillance", "no action warranted"],
    "p&a": ["p&a", "plug and abandon", "abandonment", "abandon the well"],
}

KEYWORD_SYNONYMS = {
    "low intake pressure": ["intake pressure", "low intake", "intake = ", "intake at "],
    "below POR": ["below por", "outside por", "below the por", "below the preferred operating range"],
    "high amps": ["high amps", "over-amp", "amp overload", "amperage above", "high amperage"],
    "fluid pound": ["fluid pound", "incomplete fillage", "low fillage", "pump-off"],
    "parted rods": ["parted rod", "rod string failure", "broken rod"],
    "liquid loading": ["liquid loading", "loading up", "slugging", "turner velocity"],
    "p&a candidate": ["p&a", "plug and abandon", "uneconomic", "stripper well"],
}


def _matches_with_synonyms(text: str, term: str, synonym_map: dict[str, list[str]]) -> bool:
    text_l = text.lower()
    term_l = term.lower()
    if term_l in text_l:
        return True
    for synonym in synonym_map.get(term, []):
        if synonym.lower() in text_l:
            return True
    return False


def keyword_hit_rate(report: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    hits = sum(1 for kw in keywords if _matches_with_synonyms(report, kw, KEYWORD_SYNONYMS))
    return hits / len(keywords)


def recommendation_matches(report: str, expected: str) -> bool:
    """Match expected recommendation (with underscores) against report using synonyms."""
    return _matches_with_synonyms(report, expected.replace("_", " "), RECOMMENDATION_SYNONYMS) or \
           _matches_with_synonyms(report, expected, RECOMMENDATION_SYNONYMS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only the first N cases (useful for quick checks)")
    parser.add_argument("--case", type=str, default=None,
                        help="Run a single case by id (e.g., case_005)")
    args = parser.parse_args()

    console = Console()
    RESULTS_DIR.mkdir(exist_ok=True)

    with CASES_FILE.open() as f:
        cases = yaml.safe_load(f)["cases"]

    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
    if args.limit:
        cases = cases[: args.limit]

    table = Table(title="Eval results")
    table.add_column("Case", style="cyan")
    table.add_column("Pattern")
    table.add_column("Keyword", justify="right")
    table.add_column("Recommendation", justify="center")

    total_keyword = 0.0
    total_recommend = 0
    summary_rows = []

    for case in cases:
        console.print(f"\n[bold cyan]Running {case['id']} ({case.get('notes', '')})...[/]")
        try:
            report = run_review(case["well_file"])
        except Exception as e:
            console.print(f"[red]FAILED: {e}[/]")
            table.add_row(case["id"], case.get("notes", ""), "ERR", "ERR")
            summary_rows.append({"id": case["id"], "error": str(e)})
            continue

        out_path = RESULTS_DIR / f"{case['id']}.md"
        out_path.write_text(report)

        kw_rate = keyword_hit_rate(report, case.get("expected_diagnosis_keywords", []))
        expected_raw = case["expected_primary_recommendation"]
        expected = expected_raw.replace("_", " ").lower()
        rec_match = recommendation_matches(report, expected_raw)

        table.add_row(
            case["id"],
            case.get("notes", "")[:30],
            f"{kw_rate:.0%}",
            "✓" if rec_match else "✗",
        )
        total_keyword += kw_rate
        total_recommend += int(rec_match)
        summary_rows.append({
            "id": case["id"],
            "notes": case.get("notes", ""),
            "expected": expected,
            "keyword_hit_rate": kw_rate,
            "recommendation_match": rec_match,
        })

    n = len(cases)
    console.print(table)
    console.print(
        f"\n[bold]Overall:[/] keyword {total_keyword / n:.0%} · "
        f"recommendation {total_recommend}/{n} ({total_recommend / n:.0%})"
    )

    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary_rows, indent=2))
    console.print(f"\n[dim]Reports saved to {RESULTS_DIR}/, summary in summary.json[/]")


if __name__ == "__main__":
    main()
