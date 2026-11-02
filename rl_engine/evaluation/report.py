"""Research plots and text report generation."""

from pathlib import Path


def write_research_report(
    path: str | Path,
    dataset_manifest: str,
    results: dict[str, object],
    notes: list[str],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        stream.write("# StreamIQ Research Validation Report\n\n")
        stream.write(f"Dataset manifest: `{dataset_manifest}`\n\n")
        stream.write("## Policy results\n\n")
        for name, metrics in results.items():
            stream.write(f"- **{name}**: {metrics}\n")
        stream.write("\n## Notes and limitations\n\n")
        for note in notes:
            stream.write(f"- {note}\n")
    return destination
