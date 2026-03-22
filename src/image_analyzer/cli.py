"""CLI interface using Click."""

import sys
from pathlib import Path
from typing import Optional

import click
import structlog

# Configure structlog before anything else
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO
)

logger = structlog.get_logger()


@click.group()
@click.version_option(version="0.1.0", prog_name="uteamup-image-analyzer")
def cli():
    """UteamUP Image Analyzer — batch-analyze inventory photos with Gemini Vision AI."""
    pass


@cli.command()
@click.option("--config", "config_path", default="config.yaml", help="Path to config YAML file")
@click.option("--folder", help="Override image folder from config")
@click.option("--output", help="Override output folder from config")
@click.option("--dry-run", is_flag=True, help="Estimate cost without making API calls")
@click.option("--resume", is_flag=True, help="Resume from checkpoint")
@click.option("--no-rename", is_flag=True, help="Skip image renaming")
@click.option("--max-cost", type=float, help="Maximum API cost budget in USD")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def analyze(
    config_path: str,
    folder: Optional[str],
    output: Optional[str],
    dry_run: bool,
    resume: bool,
    no_rename: bool,
    max_cost: Optional[float],
    verbose: bool,
):
    """Analyze images in a folder and export structured CSVs."""
    if verbose:
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(10),  # DEBUG
        )

    from image_analyzer.config import load_config

    config = load_config(
        config_path=config_path,
        folder_override=folder,
        output_override=output,
        dry_run=dry_run,
        no_rename=no_rename,
        max_cost=max_cost,
    )

    # Validate config
    errors = config.validate()
    if errors:
        for error in errors:
            click.echo(f"Config error: {error}", err=True)
        sys.exit(1)

    from image_analyzer.pipeline import Pipeline

    pipeline = Pipeline(config)
    pipeline.run()


@cli.command()
@click.option(
    "--checkpoint",
    default=".checkpoint.json",
    help="Path to checkpoint file",
)
def status(checkpoint: str):
    """Show processing progress from checkpoint file."""
    from image_analyzer.utils.checkpoint import Checkpoint

    cp_path = Path(checkpoint).resolve()
    if not cp_path.exists():
        click.echo("No checkpoint file found. No processing in progress.")
        return

    cp = Checkpoint.load(str(cp_path))
    status_data = cp.get_status()

    click.echo("\n=== Processing Status ===")
    click.echo(f"Processed:     {status_data['processed_count']} images")
    click.echo(f"Started:       {status_data['started_at']}")
    click.echo(f"Last updated:  {status_data['last_updated']}")
    click.echo(f"Flagged:       {status_data['flagged_for_review']} for review")
    click.echo("\nType breakdown:")
    for entity_type, count in sorted(status_data["type_breakdown"].items()):
        click.echo(f"  {entity_type}: {count}")
    click.echo("=========================\n")


if __name__ == "__main__":
    cli()
