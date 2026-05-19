from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = typer.Typer(help="LLM Output Eval Suite — evaluate and compare LLM responses.")
eval_app = typer.Typer(help="Evaluation commands.")
regression_app = typer.Typer(help="Regression testing commands.")
app.add_typer(eval_app, name="eval")
app.add_typer(regression_app, name="regression")


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def _to_markdown(report) -> str:
    from agent.models import EvalReport
    r: EvalReport = report
    lines = []
    lines.append(f"# LLM Eval Report")
    lines.append(f"**Task type:** {r.task_type}  |  **Mode:** {r.eval_mode}  |  **Timestamp:** {r.timestamp}")
    lines.append("")

    if r.winner:
        lines.append(f"## Winner: {r.winner}")
        if r.winner_reasoning:
            lines.append(r.winner_reasoning)
        lines.append("")

    if r.ranking:
        lines.append("## Ranking")
        for i, label in enumerate(r.ranking, 1):
            lines.append(f"{i}. {label}")
        lines.append("")

    lines.append("## Response Evaluations")
    for ev in r.evaluations:
        lines.append(f"### {ev.label} — {ev.overall_score:.1f}/10")
        lines.append(f"*{ev.one_line_verdict}*")
        lines.append("")
        lines.append("| Dimension | Score | Reasoning |")
        lines.append("|-----------|-------|-----------|")
        for d in ev.dimensions:
            quote_note = f' (quote: "{d.quote}")' if d.quote else ""
            lines.append(f"| {d.dimension} | {d.score:.1f} | {d.reasoning}{quote_note} |")
        lines.append("")
        if ev.strengths:
            lines.append("**Strengths:** " + "; ".join(ev.strengths))
        if ev.weaknesses:
            lines.append("**Weaknesses:** " + "; ".join(ev.weaknesses))
        lines.append("")

    if r.pairwise:
        lines.append("## Pairwise Comparisons")
        for p in r.pairwise:
            lines.append(f"- **{p.response_a} vs {p.response_b}**: Winner = {p.winner} ({p.margin}). {p.reasoning}")
        lines.append("")

    lines.append("## Key Insight")
    lines.append(r.key_insight)
    lines.append("")
    lines.append("## Recommendation")
    lines.append(r.recommendation)

    if r.efficiency_note:
        lines.append("")
        lines.append("## Efficiency")
        lines.append(r.efficiency_note)

    return "\n".join(lines)


def _to_html(report) -> str:
    from agent.models import EvalReport
    r: EvalReport = report

    rows = ""
    for ev in r.evaluations:
        dims_html = "".join(
            f"<tr><td>{d.dimension}</td><td>{d.score:.1f}</td><td>{d.reasoning}</td></tr>"
            for d in ev.dimensions
        )
        rows += f"""
        <div class="response-card">
            <h2>{ev.label} — {ev.overall_score:.1f}/10</h2>
            <p><em>{ev.one_line_verdict}</em></p>
            <table>
                <thead><tr><th>Dimension</th><th>Score</th><th>Reasoning</th></tr></thead>
                <tbody>{dims_html}</tbody>
            </table>
            <p><strong>Strengths:</strong> {"; ".join(ev.strengths)}</p>
            <p><strong>Weaknesses:</strong> {"; ".join(ev.weaknesses)}</p>
        </div>
        """

    winner_block = ""
    if r.winner:
        winner_block = f"<div class='winner-banner'>🏆 Winner: {r.winner}</div>"

    ranking_block = ""
    if r.ranking:
        ranking_items = "".join(f"<li>{label}</li>" for label in r.ranking)
        ranking_block = f"<h2>Ranking</h2><ol>{ranking_items}</ol>"

    pairwise_block = ""
    if r.pairwise:
        p_rows = "".join(
            f"<tr><td>{p.response_a}</td><td>{p.response_b}</td><td>{p.winner}</td><td>{p.margin}</td><td>{p.reasoning}</td></tr>"
            for p in r.pairwise
        )
        pairwise_block = f"""
        <h2>Pairwise Comparisons</h2>
        <table>
            <thead><tr><th>A</th><th>B</th><th>Winner</th><th>Margin</th><th>Reasoning</th></tr></thead>
            <tbody>{p_rows}</tbody>
        </table>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>LLM Eval Report</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 1100px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }}
h1 {{ border-bottom: 2px solid #0071e3; padding-bottom: .5rem; }}
.winner-banner {{ background: #e8f5e9; border: 1px solid #66bb6a; padding: 1rem; border-radius: 6px; font-size: 1.2rem; font-weight: bold; margin: 1rem 0; }}
.response-card {{ border: 1px solid #ddd; border-radius: 8px; padding: 1.5rem; margin: 1rem 0; background: #fafafa; }}
table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
th, td {{ border: 1px solid #ddd; padding: .6rem .9rem; text-align: left; }}
th {{ background: #f0f0f0; }}
.insight-box {{ background: #fff8e1; border-left: 4px solid #ffc107; padding: 1rem; margin: 1rem 0; border-radius: 0 6px 6px 0; }}
</style>
</head>
<body>
<h1>LLM Eval Report</h1>
<p><strong>Task:</strong> {r.task_type} &nbsp;|&nbsp; <strong>Mode:</strong> {r.eval_mode} &nbsp;|&nbsp; <strong>Timestamp:</strong> {r.timestamp}</p>
{winner_block}
{ranking_block}
{rows}
{pairwise_block}
<div class="insight-box">
    <p><strong>Key Insight:</strong> {r.key_insight}</p>
    <p><strong>Recommendation:</strong> {r.recommendation}</p>
    {"<p><strong>Efficiency:</strong> " + r.efficiency_note + "</p>" if r.efficiency_note else ""}
</div>
</body>
</html>"""


def _to_junit(report) -> str:
    import xml.etree.ElementTree as ET
    from agent.models import EvalReport
    r: EvalReport = report

    threshold = float(os.environ.get("EVAL_PASS_THRESHOLD", "7.0"))

    suite = ET.Element("testsuite", {
        "name": f"llm-eval-{r.task_type}",
        "timestamp": r.timestamp,
    })

    total = 0
    failures = 0

    for ev in r.evaluations:
        for d in ev.dimensions:
            total += 1
            tc = ET.SubElement(suite, "testcase", {
                "classname": ev.label,
                "name": d.dimension,
                "score": f"{d.score:.2f}",
            })
            if d.score < threshold:
                failures += 1
                failure = ET.SubElement(tc, "failure", {
                    "message": f"Score {d.score:.1f} below threshold {threshold}",
                })
                failure.text = d.reasoning
            else:
                tc.set("status", "pass")

    suite.set("tests", str(total))
    suite.set("failures", str(failures))

    tree = ET.ElementTree(suite)
    ET.indent(tree)
    import io
    buf = io.StringIO()
    buf.write("<?xml version='1.0' encoding='utf-8'?>\n")
    tree.write(buf, encoding="unicode", xml_declaration=False)
    return buf.getvalue()


def _format_report(report, fmt: str) -> str:
    if fmt == "json":
        return report.model_dump_json(indent=2)
    elif fmt == "markdown":
        return _to_markdown(report)
    elif fmt == "html":
        return _to_html(report)
    elif fmt == "junit":
        return _to_junit(report)
    else:
        return report.model_dump_json(indent=2)


def _write_output(content: str, output: Optional[str]) -> None:
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(content, encoding="utf-8")
        typer.echo(f"Report written to {output}")
    else:
        typer.echo(content)


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

@eval_app.callback(invoke_without_command=True)
def eval_command(
    ctx: typer.Context,
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="Path to eval input JSON file"),
    batch: Optional[Path] = typer.Option(None, "--batch", "-b", help="Directory of eval JSON files for batch eval"),
    prompt: Optional[str] = typer.Option(None, "--prompt", help="Inline prompt text (for single quick eval)"),
    response: Optional[str] = typer.Option(None, "--response", help="Inline response text (for single quick eval)"),
    label: str = typer.Option("Response", "--label", help="Label for inline response"),
    task_type: str = typer.Option("general", "--task-type", "-t", help="Task type for inline eval"),
    mode: str = typer.Option("score", "--mode", "-m", help="Eval mode: score | compare | rank"),
    dimensions: Optional[str] = typer.Option(None, "--dimensions", "-d", help="Comma-separated dimension names to use"),
    fmt: str = typer.Option("json", "--format", help="Output format: json | markdown | html | junit"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path (default: stdout)"),
) -> None:
    if ctx.invoked_subcommand is not None:
        return

    from agent.evaluator import evaluate
    from agent.comparator import attach_pairwise_to_report
    from agent.models import EvalInput, LLMResponse

    if batch:
        _run_batch(batch, mode, fmt, output)
        return

    if file:
        raw = json.loads(Path(file).read_text())
        if mode:
            raw["eval_mode"] = mode
        if dimensions:
            raw["dimensions"] = [d.strip() for d in dimensions.split(",")]
        eval_input = EvalInput.model_validate(raw)

    elif prompt and response:
        dims = [d.strip() for d in dimensions.split(",")] if dimensions else None
        eval_input = EvalInput(
            prompt=prompt,
            task_type=task_type,
            responses=[LLMResponse(label=label, response=response)],
            eval_mode=mode,
            dimensions=dims,
        )
    else:
        typer.echo("Provide --file or both --prompt and --response.", err=True)
        raise typer.Exit(code=1)

    report = evaluate(eval_input)

    if eval_input.eval_mode in ("compare", "rank") and len(eval_input.responses) > 1:
        report = attach_pairwise_to_report(report, eval_input)

    _write_output(_format_report(report, fmt), output)


def _run_batch(batch_dir: Path, mode: str, fmt: str, output: Optional[str]) -> None:
    from agent.evaluator import evaluate
    from agent.comparator import attach_pairwise_to_report
    from agent.models import EvalInput, EvalReport

    json_files = sorted(batch_dir.glob("*.json"))
    if not json_files:
        typer.echo(f"No JSON files found in {batch_dir}", err=True)
        raise typer.Exit(code=1)

    reports: list[EvalReport] = []
    for jf in json_files:
        typer.echo(f"Evaluating {jf.name}...")
        raw = json.loads(jf.read_text())
        raw["eval_mode"] = mode
        eval_input = EvalInput.model_validate(raw)
        report = evaluate(eval_input)
        if eval_input.eval_mode in ("compare", "rank") and len(eval_input.responses) > 1:
            report = attach_pairwise_to_report(report, eval_input)
        reports.append(report)

    if fmt == "markdown":
        sections = [f"# Batch Eval — {len(reports)} evaluations\n"]
        for r in reports:
            sections.append(_to_markdown(r))
            sections.append("\n---\n")
        content = "\n".join(sections)
    elif fmt == "json":
        content = json.dumps([json.loads(r.model_dump_json()) for r in reports], indent=2)
    else:
        content = json.dumps([json.loads(r.model_dump_json()) for r in reports], indent=2)

    _write_output(content, output)


@app.command("hallucination")
def hallucination_command(
    response: str = typer.Option(..., "--response", "-r", help="Path to response text file"),
    source: str = typer.Option(..., "--source", "-s", help="Path to source document file"),
    label: str = typer.Option("response", "--label", "-l", help="Label for the response"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path (default: stdout)"),
    fmt: str = typer.Option("json", "--format", help="Output format: json | markdown"),
) -> None:
    from agent.hallucination import detect_hallucinations

    response_text = Path(response).read_text(encoding="utf-8")
    source_text = Path(source).read_text(encoding="utf-8")

    report = detect_hallucinations(response_text=response_text, source=source_text, response_label=label)

    if fmt == "markdown":
        lines = [
            f"# Hallucination Report — {report.response_label}",
            f"**Risk Level:** {report.risk_level}  |  **Hallucination Rate:** {report.hallucination_rate:.0%}  |  **Safe to Use:** {report.safe_to_use}",
            "",
            f"**Summary:** {report.summary}",
            "",
        ]
        if report.suspicious_claims:
            lines.append("## Suspicious Claims")
            for c in report.suspicious_claims:
                supported = "Supported" if c.is_supported else "Unsupported"
                lines.append(f"- [{c.severity.upper()}] {c.claim} — {supported}. {c.evidence_or_gap}")
        content = "\n".join(lines)
    else:
        content = report.model_dump_json(indent=2)

    _write_output(content, output)


@regression_app.command("save")
def regression_save(
    report_file: Path = typer.Argument(..., help="Path to EvalReport JSON"),
    baseline_id: str = typer.Option(..., "--id", help="Baseline identifier"),
    description: str = typer.Option("", "--description", "-d", help="Description of this baseline"),
) -> None:
    from agent.regression import save_baseline
    from agent.models import EvalReport

    data = json.loads(report_file.read_text())
    report = EvalReport.model_validate(data)
    record = save_baseline(report, baseline_id, description)
    typer.echo(f"Baseline '{record.baseline_id}' saved at {record.created_at}")


@regression_app.command("list")
def regression_list() -> None:
    from agent.regression import list_baselines
    baselines = list_baselines()
    if not baselines:
        typer.echo("No baselines saved.")
    else:
        for b in baselines:
            typer.echo(f"  - {b}")


@regression_app.command("run")
def regression_run(
    report_file: Path = typer.Argument(..., help="Path to new EvalReport JSON"),
    baseline_id: str = typer.Option(..., "--id", help="Baseline identifier to compare against"),
    thresholds: Optional[str] = typer.Option(None, "--thresholds", help="JSON string of per-dimension thresholds"),
    output: Optional[str] = typer.Option(None, "--output", "-o"),
    fmt: str = typer.Option("json", "--format"),
) -> None:
    from agent.regression import run_regression
    from agent.models import EvalReport

    data = json.loads(report_file.read_text())
    report = EvalReport.model_validate(data)
    threshold_dict = json.loads(thresholds) if thresholds else None
    result = run_regression(report, baseline_id, threshold_dict)

    if fmt == "markdown":
        status = "PASSED" if result.passed_overall else "FAILED"
        lines = [
            f"# Regression Result — {status}",
            f"**Baseline:** {result.baseline_id}",
            "",
            result.summary,
            "",
            "| Dimension | Baseline | New | Delta | Passed |",
            "|-----------|----------|-----|-------|--------|",
        ]
        for c in result.checks:
            tick = "Yes" if c.passed else "No"
            lines.append(f"| {c.dimension} | {c.baseline_score} | {c.new_score} | {c.delta:+.3f} | {tick} |")
        if result.improvements:
            lines.append("\n**Improvements:** " + "; ".join(result.improvements))
        content = "\n".join(lines)
    else:
        content = result.model_dump_json(indent=2)

    _write_output(content, output)
    if not result.passed_overall:
        raise typer.Exit(code=1)


@app.command("sensitivity")
def sensitivity_command(
    prompts_file: Path = typer.Argument(..., help="JSON file with list of {label, prompt} objects"),
    response_file: Path = typer.Argument(..., help="Path to the fixed response text file"),
    task_type: str = typer.Option("general", "--task-type", "-t"),
    output: Optional[str] = typer.Option(None, "--output", "-o"),
    fmt: str = typer.Option("json", "--format"),
) -> None:
    from agent.sensitivity import analyse_prompt_sensitivity
    from agent.models import PromptVariant

    raw_prompts = json.loads(prompts_file.read_text())
    variants = [PromptVariant.model_validate(p) for p in raw_prompts]
    fixed_response = response_file.read_text(encoding="utf-8")

    report = analyse_prompt_sensitivity(variants, fixed_response, task_type)

    if fmt == "markdown":
        lines = [
            f"# Prompt Sensitivity Report",
            f"**Recommended Prompt:** {report.recommended_prompt}",
            f"**Stability:** {report.stability_summary}",
            "",
            "## Prompt Scores",
            "| Prompt | Overall |",
            "|--------|---------|",
        ]
        for ps in report.prompt_scores:
            lines.append(f"| {ps.label} | {ps.overall_score:.2f} |")
        lines.append("\n## Dimension Sensitivities")
        lines.append("| Dimension | Mean | Variance | Sensitivity |")
        lines.append("|-----------|------|----------|-------------|")
        for d in report.dimension_sensitivities:
            lines.append(f"| {d.dimension} | {d.mean_score} | {d.variance} | {d.sensitivity} |")
        content = "\n".join(lines)
    else:
        content = report.model_dump_json(indent=2)

    _write_output(content, output)


@eval_app.command("panel")
def panel_command(
    file: Path = typer.Argument(..., help="Path to eval input JSON file"),
    num_judges: int = typer.Option(3, "--judges", "-j", help="Number of judge iterations (2-5)"),
    output: Optional[str] = typer.Option(None, "--output", "-o"),
    fmt: str = typer.Option("json", "--format"),
) -> None:
    from agent.evaluator import evaluate_with_panel
    from agent.models import EvalInput

    raw = json.loads(file.read_text())
    eval_input = EvalInput.model_validate(raw)
    report = evaluate_with_panel(eval_input, num_judges)

    if fmt == "markdown":
        lines = [
            f"# Panel Eval Report ({report.num_judges} judges)",
            f"**Task:** {report.task_type}  |  **Mode:** {report.eval_mode}  |  **Timestamp:** {report.timestamp}",
            "",
        ]
        if report.winner:
            lines.append(f"**Winner:** {report.winner}")
        lines.append(f"\n{report.panel_key_insight}")
        lines.append("\n## Response Consensus Scores")
        for rc in report.response_consensus:
            flag = " [REVIEW NEEDED]" if rc.needs_human_review else ""
            lines.append(f"\n### {rc.label} — {rc.consensus_score:.2f}/10{flag}")
            if rc.reviewer_note:
                lines.append(f"*{rc.reviewer_note}*")
            lines.append("\n| Dimension | Mean | Variance | Disagreement |")
            lines.append("|-----------|------|----------|--------------|")
            for d in rc.dimension_consensus:
                flag_mark = "YES" if d.disagreement_flag else "-"
                lines.append(f"| {d.dimension} | {d.mean_score} | {d.variance} | {flag_mark} |")
        if report.flagged_for_review:
            lines.append(f"\n**Flagged for human review:** {', '.join(report.flagged_for_review)}")
        content = "\n".join(lines)
    else:
        content = report.model_dump_json(indent=2)

    _write_output(content, output)


if __name__ == "__main__":
    app()
