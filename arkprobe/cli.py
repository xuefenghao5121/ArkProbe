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
from typing import List, Optional

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
@click.option("--check", "do_check", is_flag=True,
              help="Check dependency availability for each scenario")
@click.option("--builtin-names", "builtin_only", is_flag=True,
              help="Show builtin scenario short names and exit")
def list_scenarios_cmd(do_check, builtin_only):
    """List available scenario configurations."""
    from .deps.checker import check_dependencies
    from .scenarios.loader import list_scenarios_lightweight, BUILTIN_DIR, load_builtin_scenarios

    # Quick lookup for builtin short names
    if builtin_only:
        table = Table(title="Builtin Scenario Short Names")
        table.add_column("Short", style="cyan", width=10)
        table.add_column("Full Name", style="green")
        builtin_map = {
            "compute": "Compute Intensive",
            "memory": "Memory Intensive",
            "mixed": "Mixed Workload",
            "stream": "STREAM Bandwidth",
            "random": "Random Access",
            "crypto": "Cryptography",
            "compress": "Compression",
            "video": "Video Encoding",
            "ml": "ML Inference",
            "oltp": "Database OLTP",
            "kv": "KV Store",
            "web": "Web Server",
            "jvm": "JVM General",
        }
        for short, full in builtin_map.items():
            table.add_row(short, full)
        console.print(table)
        console.print("\n[dim]Usage: arkprobe collect -b <short-name>[/dim]")
        console.print("[dim]Example: arkprobe collect -b compute[/dim]")
        return

    items = list_scenarios_lightweight()

    if not items:
        console.print("[yellow]No scenario configurations found.[/yellow]")
        return

    table = Table(title="Available Scenarios")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Builtin", style="magenta")
    if do_check:
        table.add_column("Status", style="yellow")
        table.add_column("Missing", style="red")

    for item in items:
        builtin_mark = "Yes" if item["builtin"] else ""
        if do_check:
            deps = item.get("dependencies", [])
            if not deps:
                status = "[green]Ready[/green]"
                missing = ""
            else:
                results = check_dependencies(deps)
                missing_list = [r.binary for r in results if not r.available]
                if missing_list:
                    status = "[red]Missing deps[/red]"
                    missing = ", ".join(missing_list)
                else:
                    status = "[green]Ready[/green]"
                    missing = ""
            table.add_row(item["name"], item["type"], builtin_mark, status, missing)
        else:
            table.add_row(item["name"], item["type"], builtin_mark)

    console.print(table)
    if not do_check:
        console.print("\n[dim]Tip: Use --check to verify dependency availability.[/dim]")


@cli.command()
@click.option("--scenario", "-s", multiple=True,
              help='Scenario name(s), "builtin", "all", or use --bin for custom binary')
@click.option("--builtin", "-b", multiple=True,
              help="Builtin scenario name(s) by short name: compute/memory/mixed/stream/random/crypto/compress/video/ml/oltp/kv/web/jvm")
@click.option("--binary", "--bin", "binary_path", type=click.Path(exists=True),
              default=None, help="Direct path to workload binary (skip scenario config)")
@click.option("--duration", "-t", type=int, default=60,
              help="Collection duration per phase (seconds)")
@click.option("--skip-ebpf", is_flag=True,
              help="Skip eBPF collection (runs without root)")
@click.option("--skip-scalability", is_flag=True,
              help="Skip multi-core scalability sweep")
@click.option("--jfr/--no-jfr", default=False,
              help="Enable JFR collection for JVM applications")
@click.option("--jvm-pid", type=int, default=None,
              help="Target JVM process PID for JFR collection")
@click.option("--jfr-events", multiple=True,
              help="JFR event groups (gc/jit/thread/memory)")
@click.option("--kunpeng-model", type=click.Choice(["920", "930"]),
              default="920", help="Kunpeng processor model")
@click.option("--cache-ttl", type=int, default=0,
              help="Cache TTL in seconds (0=no cache, re-collect always)")
@click.option("--force", is_flag=True,
              help="Force re-collection, ignoring cache")
@click.pass_context
def collect(ctx, scenario, builtin, binary_path, duration, skip_ebpf, skip_scalability,
            jfr, jvm_pid, jfr_events, kunpeng_model, cache_ttl, force):
    """Collect performance data for specified scenarios."""
    from .collectors.collector_orchestrator import (
        CollectorOrchestrator,
        ScenarioCollectionConfig,
    )
    from .deps.checker import check_dependencies, format_missing_deps
    from .scenarios.loader import load_all_scenarios, load_builtin_scenarios, load_scenario
    from .workloads.build import resolve_builtin_command

    data_dir = ctx.obj["data_dir"]

    # Validate options: --binary is exclusive with --scenario/--builtin
    if binary_path and (scenario or builtin):
        console.print("[red]Error: --binary cannot be used with --scenario or --builtin[/red]")
        return

    scenarios: List[ScenarioConfig] = []

    # Handle direct binary path
    if binary_path:
        from .scenarios.loader import ScenarioConfig, WorkloadConfig, CollectionConfig, PlatformConfig
        from .model.enums import ScenarioType
        bin_name = Path(binary_path).stem
        scenarios.append(ScenarioConfig(
            name=bin_name,
            type=ScenarioType.MICROSERVICE,
            builtin=False,
            workload=WorkloadConfig(
                command=binary_path,
                target_process=bin_name,
            ),
            collection=CollectionConfig(
                perf_duration_sec=duration,
                ebpf_duration_sec=0 if skip_ebpf else min(duration, 30),
                warmup_sec=5,
            ),
        ))

    # Load scenarios from YAML configs
    else:
        from .scenarios.loader import load_all_scenarios, load_builtin_scenarios, load_scenario, get_scenario_by_name

        # Resolve --builtin short names to full scenario objects
        builtin_scenarios = []
        for bname in builtin:
            sc = get_scenario_by_name(bname)
            if sc is None:
                console.print(f"[red]Builtin scenario not found: {bname}[/red]")
                console.print("Use 'arkprobe list' to see available builtin scenarios.")
                return
            builtin_scenarios.append(sc)

        # Resolve --scenario names
        # Only load all scenarios as default when neither --scenario nor --builtin is given
        if "builtin" in scenario or "all" in scenario:
            scenarios.extend(load_all_scenarios())
        elif not scenario and not builtin:
            scenarios.extend(load_all_scenarios())
        else:
            for name in scenario:
                p = Path(name)
                if p.exists():
                    scenarios.append(load_scenario(p))
                else:
                    sc = get_scenario_by_name(name)
                    if sc:
                        scenarios.append(sc)
                    else:
                        console.print(f"[red]Scenario not found: {name}[/red]")
                        return

        # Combine with builtin scenarios
        scenarios.extend(builtin_scenarios)

    if not scenarios:
        console.print("[red]No scenarios to collect. Use 'list' to see available scenarios.[/red]")
        return

    # Pre-flight dependency check
    blocked = []
    for sc in scenarios:
        if sc.dependencies:
            results = check_dependencies(sc.dependencies)
            missing = [r for r in results if not r.available]
            if missing:
                blocked.append((sc.name, format_missing_deps(results)))

    if blocked:
        console.print("[red]Some scenarios have missing dependencies:[/red]\n")
        for name, msg in blocked:
            console.print(f"[cyan]{name}[/cyan]")
            console.print(msg)
            console.print()
        console.print("[yellow]Install the missing tools and retry, or use "
                       "'arkprobe collect -s builtin' for zero-dependency testing.[/yellow]")
        return

    console.print(f"\n[bold]Collecting {len(scenarios)} scenario(s)[/bold]\n")

    for sc in scenarios:
        console.print(f"[cyan]>>> {sc.name}[/cyan]")

        # Resolve builtin workload command to actual binary path if needed
        workload_cmd = sc.workload.command
        if sc.builtin:
            from .workloads.build import resolve_builtin_command
            workload_cmd = resolve_builtin_command(workload_cmd)

        # Format command with thread_count and duration
        formatted_cmd = workload_cmd.format(
            thread_count=sc.platform.recommended_cores,
            duration=duration,
        )

        config = ScenarioCollectionConfig(
            scenario_name=sc.name.lower().replace(" ", "_"),
            workload_command=formatted_cmd,
            kunpeng_model=kunpeng_model,
            perf_duration_sec=duration,
            ebpf_duration_sec=min(duration, sc.collection.ebpf_duration_sec),
            warmup_sec=sc.collection.warmup_sec,
            ebpf_probes=sc.collection.ebpf_probes,
            skip_ebpf=skip_ebpf or not sc.collection.ebpf_probes,
            skip_scalability=skip_scalability,
            cache_ttl_sec=cache_ttl,
            force=force,
            skip_jfr=not (jfr or sc.collection.jfr_enabled),
            jfr_duration_sec=sc.collection.jfr_duration_sec,
            jfr_events=list(jfr_events) if jfr_events else sc.collection.jfr_events,
            jvm_pid=jvm_pid,
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
def check():
    """Check dependency availability for all scenarios."""
    from .deps.checker import check_binary, check_dependencies
    from .scenarios.loader import list_scenarios_lightweight

    items = list_scenarios_lightweight()
    if not items:
        console.print("[yellow]No scenario configurations found.[/yellow]")
        return

    builtin = [i for i in items if i["builtin"]]
    external = [i for i in items if not i["builtin"]]

    if builtin:
        console.print("\n[bold]Builtin Scenarios[/bold] (no external tools required)")
        for item in builtin:
            console.print(f"  [green]Ready[/green]  {item['name']}")

    if external:
        console.print("\n[bold]External Scenarios[/bold]")
        table = Table(show_header=True)
        table.add_column("Scenario", style="cyan")
        table.add_column("Status")
        table.add_column("Missing")
        table.add_column("Install hint", style="dim")

        for item in external:
            deps = item.get("dependencies", [])
            if not deps:
                table.add_row(item["name"], "[green]Ready[/green]", "", "")
            else:
                results = check_dependencies(deps)
                missing = [r for r in results if not r.available]
                if missing:
                    names = ", ".join(r.binary for r in missing)
                    hints = "; ".join(r.install_hint for r in missing)
                    table.add_row(item["name"], "[red]Missing[/red]", names, hints)
                else:
                    table.add_row(item["name"], "[green]Ready[/green]", "", "")

        console.print(table)

    console.print("\n[bold]System Tools[/bold]")
    for tool in ["perf", "bpftrace", "gcc"]:
        r = check_binary(tool)
        if r.available:
            console.print(f"  [green]OK[/green]     {tool} ({r.path})")
        else:
            console.print(f"  [red]Missing[/red]  {tool} -> {r.install_hint}")

    ready_count = sum(1 for i in items
                      if i["builtin"] or not i.get("dependencies"))
    console.print(f"\n[bold]{ready_count} scenario(s) ready to run.[/bold]")


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

        for layer_name in ("os", "bios", "driver", "jvm"):
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


@cli.command()
@click.option("--scenario", "-s", required=True,
              help="Scenario name to run")
@click.option("--config", "-c", multiple=True,
              help="Tuning config name (can specify multiple)")
@click.option("--duration", "-t", type=int, default=60,
              help="Collection duration per config (seconds)")
@click.option("--baseline", type=str, default="default",
              help="Baseline configuration name")
@click.option("--dry-run", is_flag=True,
              help="Preview changes without applying")
@click.option("--output", "-o", type=click.Path(), default="./tuning_results",
              help="Output directory for results")
@click.pass_context
def tune(ctx, scenario, config, duration, baseline, dry_run, output):
    """Run workload under different hardware tuning configurations.

    Examples:
        # Compare performance and power configs
        arkprobe tune -s compute -c performance -c power

        # Use database-optimized config
        arkprobe tune -s database_oltp -c database

        # Dry run to preview changes
        arkprobe tune -s memory -c latency --dry-run
    """
    from .collectors.collector_orchestrator import CollectorOrchestrator, ScenarioCollectionConfig
    from .analysis.feature_extractor import FeatureExtractor
    from .model.feature_vector import save_feature_vector
    from .scenarios.loader import get_scenario_by_name
    from .workloads.build import resolve_builtin_command
    from .tuner.hardware_tuner import HardwareTuner, TUNING_PRESETS, TuningConfig
    from .tuner.comparator import TuningComparator

    data_dir = ctx.obj["data_dir"]
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load scenario
    sc = get_scenario_by_name(scenario)
    if not sc:
        console.print(f"[red]Scenario not found: {scenario}[/red]")
        console.print("Use 'arkprobe list' to see available scenarios.")
        return

    # Resolve configs
    configs_to_run = list(config) if config else ["performance", "power"]
    if baseline not in configs_to_run:
        configs_to_run.insert(0, baseline)

    # Validate config names
    for cfg_name in configs_to_run:
        if cfg_name not in TUNING_PRESETS:
            console.print(f"[red]Unknown config: {cfg_name}[/red]")
            console.print(f"Available: {', '.join(TUNING_PRESETS.keys())}")
            return

    console.print(f"\n[bold]Tuning Experiment: {sc.name}[/bold]")
    console.print(f"Configs: {', '.join(configs_to_run)}")
    console.print(f"Duration: {duration}s per config\n")

    tuner = HardwareTuner(dry_run=dry_run)
    extractor = FeatureExtractor()
    results = []

    for cfg_name in configs_to_run:
        config = TUNING_PRESETS[cfg_name]
        console.print(f"[cyan]>>> Config: {cfg_name}[/cyan]")
        console.print(f"    {config.description}")

        # Apply tuning
        result = tuner.apply(config)
        if not result.success:
            console.print(f"    [red]Failed to apply config: {result.errors}[/red]")
            continue

        if result.errors:
            for err in result.errors:
                console.print(f"    [yellow]Warning: {err}[/yellow]")

        # Run workload
        workload_cmd = sc.workload.command
        if sc.builtin:
            workload_cmd = resolve_builtin_command(workload_cmd)

        # Wrap with numactl/taskset if configured
        wrapped_cmd = tuner.wrap_command(config, workload_cmd.split())

        collection_config = ScenarioCollectionConfig(
            scenario_name=f"{sc.name.lower().replace(' ', '_')}_{cfg_name}",
            workload_command=" ".join(wrapped_cmd),
            kunpeng_model="920",
            perf_duration_sec=duration,
            ebpf_duration_sec=0,
            warmup_sec=sc.collection.warmup_sec,
            skip_ebpf=True,
            skip_scalability=True,
        )

        orchestrator = CollectorOrchestrator(collection_config, data_dir)
        collection_result = orchestrator.run()

        if collection_result.errors:
            console.print(f"    [yellow]Collection warnings: {len(collection_result.errors)}[/yellow]")

        # Extract features
        fv = extractor.extract(collection_result, sc)
        fv_path = output_dir / f"{sc.name.lower().replace(' ', '_')}_{cfg_name}_features.json"
        save_feature_vector(fv, fv_path)

        console.print(f"    IPC={fv.compute.ipc:.2f}, L3 MPKI={fv.cache.l3_mpki:.1f}")
        console.print(f"    [green]Saved: {fv_path}[/green]\n")

        results.append((cfg_name, fv))

        # Restore original state
        tuner.restore()

    # Compare results
    if len(results) >= 2:
        console.print("\n[bold]Comparison Results[/bold]\n")

        comparator = TuningComparator()
        baseline_name, baseline_fv = results[0]

        # Comparison table
        table = Table(title=f"Configuration Impact (baseline: {baseline_name})")
        table.add_column("Config", style="cyan")
        table.add_column("IPC")
        table.add_column("IPC Δ%")
        table.add_column("L3 MPKI")
        table.add_column("L3 Δ%")
        table.add_column("Branch MPKI")
        table.add_column("Overall")

        for cfg_name, fv in results:
            ipc = fv.compute.ipc
            l3_mpki = fv.cache.l3_mpki
            branch_mpki = fv.branch.branch_mpki

            if cfg_name == baseline_name:
                ipc_delta = "-"
                l3_delta = "-"
                overall = "-"
            else:
                report = comparator.compare(baseline_fv, fv, cfg_name)
                ipc_change = next((c for c in report.metric_changes if c.name == "ipc"), None)
                l3_change = next((c for c in report.metric_changes if c.name == "l3_mpki"), None)

                ipc_delta = f"{ipc_change.percent_change:+.1f}%" if ipc_change else "-"
                l3_delta = f"{l3_change.percent_change:+.1f}%" if l3_change else "-"

                if report.overall_improvement > 0.1:
                    overall = f"[green]+{report.overall_improvement:.2f}[/green]"
                elif report.overall_improvement < -0.1:
                    overall = f"[red]{report.overall_improvement:.2f}[/red]"
                else:
                    overall = f"{report.overall_improvement:.2f}"

            table.add_row(
                cfg_name,
                f"{ipc:.2f}",
                ipc_delta,
                f"{l3_mpki:.1f}",
                l3_delta,
                f"{branch_mpki:.1f}",
                overall,
            )

        console.print(table)

        # Key findings
        for cfg_name, fv in results[1:]:
            report = comparator.compare(baseline_fv, fv, cfg_name)
            if report.key_findings:
                console.print(f"\n[bold]{cfg_name}[/bold]:")
                for finding in report.key_findings:
                    console.print(f"  • {finding}")

        # Save comparison report
        comparison_path = output_dir / "tuning_comparison.json"
        comparison_data = {
            "scenario": sc.name,
            "baseline": baseline_name,
            "configs_tested": [r[0] for r in results],
            "results": [
                {
                    "config": cfg_name,
                    "ipc": fv.compute.ipc,
                    "l3_mpki": fv.cache.l3_mpki,
                    "branch_mpki": fv.branch.branch_mpki,
                    "backend_bound": fv.compute.topdown_l1.backend_bound,
                }
                for cfg_name, fv in results
            ],
        }
        comparison_path.write_text(json.dumps(comparison_data, indent=2))
        console.print(f"\n[green]Comparison saved to: {comparison_path}[/green]")


@cli.command("tune-configs")
def list_tune_configs():
    """List available tuning configurations."""
    from .tuner.hardware_tuner import TUNING_PRESETS

    table = Table(title="Available Tuning Configurations")
    table.add_column("Name", style="cyan")
    table.add_column("Governor")
    table.add_column("SMT")
    table.add_column("C-state")
    table.add_column("THP")
    table.add_column("Description")

    for name, config in TUNING_PRESETS.items():
        table.add_row(
            name,
            config.cpu_governor.value,
            "on" if config.smt_enabled else "off",
            f"C{config.cstate_limit.value}" if config.cstate_limit.value >= 0 else "unlimited",
            config.thp_setting.value,
            config.description,
        )

    console.print(table)
    console.print("\n[dim]Usage: arkprobe tune -s <scenario> -c <config_name>[/dim]")


@cli.command("gem5-configs")
def list_gem5_configs():
    """List available gem5 simulation configurations."""
    from .tuner.gem5_tuner import GEM5_PRESETS

    table = Table(title="Available gem5 Configurations")
    table.add_column("Name", style="cyan")
    table.add_column("L1I", justify="right")
    table.add_column("L1D", justify="right")
    table.add_column("L2", justify="right")
    table.add_column("Issue Width", justify="center")
    table.add_column("ROB", justify="right")
    table.add_column("Description")

    for name, config in GEM5_PRESETS.items():
        l1i = f"{config.l1i_cache.size_kb}KB"
        l1d = f"{config.l1d_cache.size_kb}KB"
        l2 = f"{config.l2_cache.size_kb}KB" if config.l2_cache else "-"
        issue = str(config.cpu_config.issue_width)
        rob = str(config.cpu_config.rob_entries)

        table.add_row(name, l1i, l1d, l2, issue, rob, config.description[:30])

    console.print(table)
    console.print("\n[dim]Usage: arkprobe simulate -s <scenario> -c <config_name>[/dim]")
    console.print("[dim]Requires gem5 to be installed and configured.[/dim]")


@cli.command()
@click.option("--scenario", "-s", required=True,
              help="Scenario name to simulate")
@click.option("--config", "-c", multiple=True,
              help="gem5 config name (can specify multiple)")
@click.option("--gem5-path", type=click.Path(exists=True),
              help="Path to gem5 installation")
@click.option("--sim-time", type=float, default=0.1,
              help="Simulation time in seconds (default: 0.1s)")
@click.option("--output", "-o", type=click.Path(), default="./gem5_results",
              help="Output directory for results")
@click.pass_context
def simulate(ctx, scenario, config, gem5_path, sim_time, output):
    """Run gem5 simulation with different microarchitectural configurations.

    This command simulates workloads in gem5 to explore the impact of
    microarchitectural parameters that cannot be changed on real hardware.

    Examples:
        # Simulate with default config
        arkprobe simulate -s compute

        # Compare multiple configurations
        arkprobe simulate -s memory -c default -c large_cache -c deep_rob

        # Specify gem5 path
        arkprobe simulate -s compute --gem5-path ~/gem5
    """
    from .tuner.gem5_tuner import Gem5Tuner, Gem5Config, GEM5_PRESETS
    from .workloads.build import get_workload_binary
    from pathlib import Path

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check gem5 availability
    tuner = Gem5Tuner(gem5_path=Path(gem5_path) if gem5_path else None)
    if tuner.gem5_path is None:
        console.print("[red]gem5 not found.[/red]")
        console.print("\nTo install gem5:")
        console.print("  1. Clone: git clone https://github.com/gem5/gem5.git")
        console.print("  2. Build: scons build/ARM/gem5.opt -j$(nproc)")
        console.print("  3. Set path: export GEM5_PATH=/path/to/gem5")
        console.print("\nOr use --gem5-path to specify location.")
        return

    # Load scenario
    from .scenarios.loader import get_scenario_by_name
    sc = get_scenario_by_name(scenario)
    if not sc:
        console.print(f"[red]Scenario not found: {scenario}[/red]")
        return

    # Get workload binary
    if not sc.builtin:
        console.print("[red]Only builtin scenarios are supported for gem5 simulation.[/red]")
        console.print("Use 'arkprobe list' to see builtin scenarios.")
        return

    binary_path = get_workload_binary(scenario.split()[0].lower())
    if binary_path is None or not binary_path.exists():
        console.print(f"[red]Workload binary not found for: {scenario}[/red]")
        console.print("Run 'arkprobe collect' first to build the binary.")
        return

    # Resolve configs
    configs_to_run = list(config) if config else ["default", "large_cache", "wide_issue"]
    if "default" not in configs_to_run:
        configs_to_run.insert(0, "default")

    # Validate config names
    for cfg_name in configs_to_run:
        if cfg_name not in GEM5_PRESETS:
            console.print(f"[red]Unknown gem5 config: {cfg_name}[/red]")
            console.print(f"Available: {', '.join(GEM5_PRESETS.keys())}")
            return

    console.print(f"\n[bold]gem5 Simulation: {sc.name}[/bold]")
    console.print(f"Binary: {binary_path}")
    console.print(f"Configs: {', '.join(configs_to_run)}")
    console.print(f"Simulation time: {sim_time}s\n")

    results = []

    for cfg_name in configs_to_run:
        preset_config = GEM5_PRESETS[cfg_name]
        # Update simulation time
        config = Gem5Config(
            name=preset_config.name,
            cpu_config=preset_config.cpu_config,
            l1i_cache=preset_config.l1i_cache,
            l1d_cache=preset_config.l1d_cache,
            l2_cache=preset_config.l2_cache,
            l3_cache=preset_config.l3_cache,
            cpu_freq=preset_config.cpu_freq,
            mem_size=preset_config.mem_size,
            mem_type=preset_config.mem_type,
            simulation_time=sim_time,
            description=preset_config.description,
        )

        console.print(f"[cyan]>>> Simulating: {cfg_name}[/cyan]")
        console.print(f"    {config.description}")

        stats = tuner.simulate(config, binary_path)

        if stats.instructions == 0:
            console.print(f"    [red]Simulation failed or no instructions executed[/red]")
            continue

        console.print(f"    IPC={stats.ipc:.4f}, Instructions={stats.instructions:,}")
        console.print(f"    L1D MPKI={stats.l1d_mpki:.2f}, Branch MPKI={stats.branch_mpki:.2f}")
        console.print(f"    Sim time: {stats.sim_seconds:.4f}s\n")

        results.append((cfg_name, stats))

        # Save individual result
        result_path = output_dir / f"{scenario}_{cfg_name}_stats.json"
        result_path.write_text(json.dumps(tuner.stats_to_feature_dict(stats), indent=2))

    # Compare results
    if len(results) >= 2:
        console.print("\n[bold]Simulation Comparison[/bold]\n")

        table = Table(title="gem5 Configuration Impact")
        table.add_column("Config", style="cyan")
        table.add_column("IPC")
        table.add_column("IPC Δ%")
        table.add_column("L1D MPKI")
        table.add_column("Branch MPKI")

        baseline_name, baseline_stats = results[0]
        baseline_ipc = baseline_stats.ipc

        for cfg_name, stats in results:
            if cfg_name == baseline_name:
                ipc_delta = "-"
            else:
                delta = ((stats.ipc - baseline_ipc) / baseline_ipc * 100) if baseline_ipc > 0 else 0
                ipc_delta = f"{delta:+.1f}%"

            table.add_row(
                cfg_name,
                f"{stats.ipc:.4f}",
                ipc_delta,
                f"{stats.l1d_mpki:.2f}",
                f"{stats.branch_mpki:.2f}",
            )

        console.print(table)

        # Save comparison
        comparison_path = output_dir / "gem5_comparison.json"
        comparison_data = {
            "scenario": scenario,
            "baseline": baseline_name,
            "simulation_time": sim_time,
            "results": [
                {
                    "config": cfg_name,
                    **tuner.stats_to_feature_dict(stats),
                }
                for cfg_name, stats in results
            ],
        }
        comparison_path.write_text(json.dumps(comparison_data, indent=2))
        console.print(f"\n[green]Comparison saved to: {comparison_path}[/green]")


def main():
    cli(obj={})


if __name__ == "__main__":
    main()
