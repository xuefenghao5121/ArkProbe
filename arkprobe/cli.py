"""CLI interface for ArkProbe.

Commands:
    collect   - Collect performance data for specified scenarios
    analyze   - Analyze collected data and extract feature vectors
    compare   - Compare multiple workload feature vectors
    report    - Generate comprehensive HTML report
    full-run  - End-to-end: collect + analyze + report
    list      - List available scenario configurations
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

console = Console()


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.option("--data-dir", "-d", type=click.Path(), default="./data",
              help="Data output directory")
@click.pass_context
def cli(ctx, verbose, data_dir):
    """ArkProbe: Workload characterization for chip design space exploration."""
    setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["data_dir"] = Path(data_dir)
    ctx.obj["data_dir"].mkdir(parents=True, exist_ok=True)


@cli.command("list")
def list_scenarios_cmd():
    """List available scenario configurations."""
    from .scenarios.loader import load_all_scenarios

    scenarios = load_all_scenarios()

    table = Table(title="Available Scenarios")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Cores", style="yellow")
    table.add_column("Scalability", style="magenta")

    for s in scenarios:
        table.add_row(
            s.name,
            s.type.value,
            str(s.platform.recommended_cores),
            "Yes" if s.scalability.enabled else "No",
        )

    console.print(table)


@cli.command()
@click.option("--scenario", "-s", multiple=True,
              help='Scenario name(s) or "all"')
@click.option("--duration", "-t", type=int, default=60,
              help="Collection duration per phase (seconds)")
@click.option("--skip-ebpf", is_flag=True,
              help="Skip eBPF collection (runs without root)")
@click.option("--skip-scalability", is_flag=True,
              help="Skip multi-core scalability sweep")
@click.option("--kunpeng-model", type=click.Choice(["920", "930"]),
              default="920", help="Kunpeng processor model")
@click.pass_context
def collect(ctx, scenario, duration, skip_ebpf, skip_scalability, kunpeng_model):
    """Collect performance data for specified scenarios."""
    from .collectors.collector_orchestrator import (
        CollectorOrchestrator,
        ScenarioCollectionConfig,
    )
    from .scenarios.loader import load_all_scenarios, load_scenario

    data_dir = ctx.obj["data_dir"]

    # Load scenarios
    if "all" in scenario or not scenario:
        scenarios = load_all_scenarios()
    else:
        from .scenarios.loader import get_scenario_by_name, CONFIGS_DIR
        scenarios = []
        for name in scenario:
            # Try as file path first
            p = Path(name)
            if p.exists():
                scenarios.append(load_scenario(p))
            else:
                s = get_scenario_by_name(name)
                if s:
                    scenarios.append(s)
                else:
                    console.print(f"[red]Scenario not found: {name}[/red]")

    if not scenarios:
        console.print("[red]No scenarios to collect. Use 'list' to see available scenarios.[/red]")
        return

    console.print(f"\n[bold]Collecting {len(scenarios)} scenario(s)[/bold]\n")

    for sc in scenarios:
        console.print(f"[cyan]>>> {sc.name}[/cyan]")
        config = ScenarioCollectionConfig(
            scenario_name=sc.name.lower().replace(" ", "_"),
            workload_command=sc.workload.command.format(
                thread_count=sc.platform.recommended_cores,
                duration=duration,
            ),
            kunpeng_model=kunpeng_model,
            perf_duration_sec=duration,
            ebpf_duration_sec=min(duration, sc.collection.ebpf_duration_sec),
            warmup_sec=sc.collection.warmup_sec,
            ebpf_probes=sc.collection.ebpf_probes,
            skip_ebpf=skip_ebpf,
            skip_scalability=skip_scalability,
        )

        orchestrator = CollectorOrchestrator(config, data_dir)
        result = orchestrator.run()

        if result.errors:
            console.print(f"  [yellow]Warnings: {len(result.errors)}[/yellow]")
            for err in result.errors[:3]:
                console.print(f"    - {err}")
        console.print(f"  [green]Done in {result.collection_duration_sec:.1f}s[/green]\n")

    console.print(f"[bold green]Collection complete. Data saved to {data_dir}[/bold green]")


@cli.command()
@click.option("--input", "-i", "input_dir", type=click.Path(exists=True), required=True,
              help="Raw collection data directory")
@click.option("--output", "-o", "output_dir", type=click.Path(), default="./data",
              help="Output directory for feature vectors")
@click.option("--kunpeng-model", type=click.Choice(["920", "930"]),
              default="920")
@click.pass_context
def analyze(ctx, input_dir, output_dir, kunpeng_model):
    """Analyze collected data and extract feature vectors."""
    from .analysis.feature_extractor import FeatureExtractor
    from .collectors.collector_orchestrator import FullCollectionResult
    from .model.feature_vector import save_feature_vector
    from .scenarios.loader import load_all_scenarios

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    extractor = FeatureExtractor(kunpeng_model)
    scenarios = {s.name.lower().replace(" ", "_"): s for s in load_all_scenarios()}

    # Find all raw collection results
    raw_files = list(input_path.rglob("*_raw.json"))
    if not raw_files:
        console.print("[red]No raw collection files found[/red]")
        return

    console.print(f"\n[bold]Analyzing {len(raw_files)} collection(s)[/bold]\n")

    for raw_file in raw_files:
        console.print(f"[cyan]>>> {raw_file.stem}[/cyan]")
        raw = FullCollectionResult.load(raw_file)

        # Match to scenario config
        scenario = scenarios.get(raw.scenario_name)
        if scenario is None:
            # Use a default scenario config
            from .scenarios.loader import ScenarioConfig, WorkloadConfig
            from .model.enums import ScenarioType
            scenario = ScenarioConfig(
                name=raw.scenario_name,
                type=ScenarioType.MICROSERVICE,
                workload=WorkloadConfig(command="unknown"),
            )

        fv = extractor.extract(raw, scenario)
        fv_path = output_path / f"{raw.scenario_name}_features.json"
        save_feature_vector(fv, fv_path)
        console.print(f"  [green]Saved: {fv_path}[/green]")
        console.print(f"  IPC={fv.compute.ipc:.2f}, L3 MPKI={fv.cache.l3_mpki:.1f}, "
                       f"Branch MPKI={fv.branch.branch_mpki:.1f}")

    console.print(f"\n[bold green]Analysis complete.[/bold green]")


@cli.command()
@click.option("--feature-vectors", "-f", multiple=True, required=True,
              help="Feature vector JSON files to compare")
@click.pass_context
def compare(ctx, feature_vectors):
    """Compare multiple workload feature vectors."""
    from .analysis.comparator import WorkloadComparator
    from .model.feature_vector import load_feature_vector

    fvs = [load_feature_vector(Path(p)) for p in feature_vectors]
    comparator = WorkloadComparator()
    result = comparator.compare(fvs)

    # Print comparison table
    table = Table(title="Workload Comparison")
    table.add_column("Scenario", style="cyan")
    table.add_column("IPC")
    table.add_column("L3 MPKI")
    table.add_column("Branch MPKI")
    table.add_column("FE Bound")
    table.add_column("BE Bound")
    table.add_column("Retiring")

    for fv in fvs:
        table.add_row(
            fv.scenario_name,
            f"{fv.compute.ipc:.2f}",
            f"{fv.cache.l3_mpki:.1f}",
            f"{fv.branch.branch_mpki:.1f}",
            f"{fv.compute.topdown_l1.frontend_bound:.0%}",
            f"{fv.compute.topdown_l1.backend_bound:.0%}",
            f"{fv.compute.topdown_l1.retiring:.0%}",
        )

    console.print(table)

    # Clusters
    if result.clusters:
        console.print("\n[bold]Workload Clusters (by micro-arch similarity):[/bold]")
        for cid, members in result.clusters.items():
            console.print(f"  Cluster {cid}: {', '.join(members)}")


@cli.command()
@click.option("--feature-vectors", "-f", multiple=True, required=True,
              help="Feature vector JSON files")
@click.option("--output", "-o", type=click.Path(), default="./report.html",
              help="Output HTML file path")
@click.option("--title", type=str, default="Workload Characterization Report")
@click.pass_context
def report(ctx, feature_vectors, output, title):
    """Generate comprehensive HTML report."""
    from .model.feature_vector import load_feature_vector
    from .reports.generator import ReportGenerator

    fvs = [load_feature_vector(Path(p)) for p in feature_vectors]

    generator = ReportGenerator(output_dir=Path(output).parent)
    report_path = generator.generate_full_report(
        fvs, title=title, output_file=Path(output)
    )

    console.print(f"\n[bold green]Report generated: {report_path}[/bold green]")
    console.print(f"Open in browser: file://{report_path.resolve()}")


@cli.command("full-run")
@click.option("--scenario", "-s", multiple=True, default=("all",),
              help='Scenario name(s) or "all"')
@click.option("--duration", "-t", type=int, default=60)
@click.option("--output", "-o", type=click.Path(), default="./report.html")
@click.option("--skip-ebpf", is_flag=True)
@click.option("--kunpeng-model", type=click.Choice(["920", "930"]), default="920")
@click.pass_context
def full_run(ctx, scenario, duration, output, skip_ebpf, kunpeng_model):
    """End-to-end: collect + analyze + report."""
    data_dir = ctx.obj["data_dir"]

    # Step 1: Collect
    console.print("[bold]Step 1/3: Collecting performance data...[/bold]")
    ctx.invoke(collect, scenario=scenario, duration=duration,
               skip_ebpf=skip_ebpf, kunpeng_model=kunpeng_model)

    # Step 2: Analyze
    console.print("\n[bold]Step 2/3: Analyzing...[/bold]")
    ctx.invoke(analyze, input_dir=str(data_dir), output_dir=str(data_dir),
               kunpeng_model=kunpeng_model)

    # Step 3: Report
    console.print("\n[bold]Step 3/3: Generating report...[/bold]")
    fv_files = list(data_dir.rglob("*_features.json"))
    if fv_files:
        ctx.invoke(report, feature_vectors=tuple(str(f) for f in fv_files),
                   output=output)
    else:
        console.print("[red]No feature vectors found to report on[/red]")


@cli.command()
@click.option("--feature-vectors", "-f", multiple=True, required=True)
@click.pass_context
def sensitivity(ctx, feature_vectors):
    """Show design parameter sensitivity analysis."""
    from .analysis.design_space import DesignSpaceExplorer
    from .model.feature_vector import load_feature_vector

    fvs = [load_feature_vector(Path(p)) for p in feature_vectors]
    explorer = DesignSpaceExplorer()
    report = explorer.full_analysis(fvs)

    # Sensitivity matrix
    if report.matrix is not None:
        console.print("\n[bold]Design Parameter Sensitivity Matrix:[/bold]")
        table = Table()
        table.add_column("Workload", style="cyan")
        for col in report.matrix.columns:
            table.add_column(col[:12], justify="center")

        for idx, row in report.matrix.iterrows():
            vals = []
            for v in row:
                if v > 0.7:
                    vals.append(f"[red]{v:.2f}[/red]")
                elif v > 0.4:
                    vals.append(f"[yellow]{v:.2f}[/yellow]")
                else:
                    vals.append(f"[green]{v:.2f}[/green]")
            table.add_row(str(idx), *vals)

        console.print(table)

    # Top recommendations
    console.print("\n[bold]Top Design Recommendations:[/bold]")
    for i, rec in enumerate(report.recommendations[:5]):
        console.print(
            f"  {i+1}. [bold]{rec.parameter}[/bold] (priority={rec.priority:.3f}, "
            f"cost={rec.area_cost})"
        )
        console.print(f"     {rec.justification}")


@cli.command()
@click.option("--feature-vectors", "-f", multiple=True, required=True,
              help="Feature vector JSON files")
@click.pass_context
def optimize(ctx, feature_vectors):
    """Show platform optimization recommendations for workloads."""
    from .analysis.optimization_analyzer import OptimizationAnalyzer
    from .model.feature_vector import load_feature_vector

    fvs = [load_feature_vector(Path(p)) for p in feature_vectors]
    analyzer = OptimizationAnalyzer()

    for fv in fvs:
        report = analyzer.analyze(fv)
        score_style = "green" if report.optimization_score >= 80 else \
                      "yellow" if report.optimization_score >= 50 else "red"
        console.print(
            f"\n[bold cyan]{fv.scenario_name}[/bold cyan] "
            f"([{score_style}]{report.optimization_score:.0f}/100[/{score_style}])"
        )

        for layer_name in ("os", "bios", "driver"):
            layer = report.layers.get(layer_name)
            if not layer or layer.gaps_found == 0:
                continue
            console.print(
                f"\n  [bold]{layer_name.upper()}[/bold] "
                f"({layer.gaps_found} gaps)"
            )

            table = Table(show_header=True)
            table.add_column("Parameter", style="cyan")
            table.add_column("Current")
            table.add_column("Recommended")
            table.add_column("Impact", justify="center")
            table.add_column("Command")

            for rec in layer.recommendations:
                if not rec.gap_detected:
                    continue
                impact_style = "red" if rec.impact_score > 0.6 else \
                               "yellow" if rec.impact_score > 0.3 else "green"
                cmd = rec.apply_commands[0] if rec.apply_commands else "-"
                table.add_row(
                    rec.display_name,
                    rec.current_value,
                    rec.recommended_value,
                    f"[{impact_style}]{rec.impact_score:.0%}[/{impact_style}]",
                    cmd,
                )
            console.print(table)

    # Cross-scenario summary
    if len(fvs) >= 2:
        cross = analyzer.cross_scenario_analysis(fvs)
        if cross.universal_recommendations:
            console.print("\n[bold]Universal Recommendations (benefit all workloads):[/bold]")
            for i, rec in enumerate(cross.universal_recommendations[:5]):
                cmd = rec.apply_commands[0] if rec.apply_commands else "-"
                console.print(
                    f"  {i+1}. [bold]{rec.display_name}[/bold] → {cmd}"
                )

        if cross.conflicting_parameters:
            console.print("\n[bold yellow]Conflicting Parameters:[/bold yellow]")
            for conflict in cross.conflicting_parameters:
                scenarios = ", ".join(
                    f"{k}={v}" for k, v in conflict["scenarios_disagree"].items()
                )
                console.print(f"  {conflict['parameter']}: {scenarios}")


def main():
    cli(obj={})


if __name__ == "__main__":
    main()
