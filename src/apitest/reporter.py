from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict


def generate_report(results, provider_name, output_dir="reports"):
    """Generate a Markdown report from test results.

    results: list of dicts with keys:
        model, test_name, verdict (PASS/WARN/FAIL), reason, details
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = output_dir / f"report-{timestamp}.md"

    # Group by model
    by_model = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r)

    # Build summary table
    test_names = sorted(set(r["test_name"] for r in results))
    lines = [
        f"# API Quality Report — {provider_name}",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Summary",
        "",
        "| Model | " + " | ".join(test_names) + " |",
        "|-------|" + "------|" * len(test_names),
    ]

    for model, tests in sorted(by_model.items()):
        row = [model]
        for tn in test_names:
            match = [t for t in tests if t["test_name"] == tn]
            if match:
                v = match[0]["verdict"]
                emoji = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(v, "❓")
                row.append(f"{emoji} {v}")
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")

    # Detail sections
    lines.extend(["", "## Details", ""])
    for model in sorted(by_model):
        lines.append(f"### {model}")
        for t in by_model[model]:
            v = t["verdict"]
            emoji = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(v, "❓")
            lines.append(f"- **{t['test_name']}**: {emoji} {v}")
            if t.get("reason"):
                lines.append(f"  - {t['reason']}")
            if t.get("details"):
                lines.append(f"  - {t['details']}")
        lines.append("")

    # Budget
    budget_info = results[0].get("budget", {}) if results else {}
    if budget_info:
        lines.extend([
            "## Budget",
            f"- Limit: ${budget_info.get('limit', 0):.4f}",
            f"- Spent: ${budget_info.get('spent', 0):.4f}",
            f"- Remaining: ${budget_info.get('remaining', 0):.4f}",
            "",
        ])

    content = "\n".join(lines)
    path.write_text(content, encoding="utf-8")
    return path
