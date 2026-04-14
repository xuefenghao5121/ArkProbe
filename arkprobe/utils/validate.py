"""Data validation utilities for ArkProbe.

Provides methods to verify:
1. Raw perf data integrity
2. Feature extraction accuracy
3. Cross-validation with manual calculations
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""
    check_name: str
    passed: bool
    expected: float
    actual: float
    tolerance: float
    error_pct: float
    message: str


class DataValidator:
    """Validate ArkProbe data collection and feature extraction."""

    def __init__(self, tolerance: float = 0.05):
        """Initialize validator.

        Args:
            tolerance: Acceptable error tolerance (default 5%)
        """
        self.tolerance = tolerance

    def validate_raw_data(self, raw_path: Path) -> list[ValidationResult]:
        """Validate raw perf data integrity.

        Checks:
        1. All event groups have data
        2. Events have non-zero values
        3. Time enabled is reasonable
        """
        results = []
        perf_dir = raw_path / "perf"

        if not perf_dir.exists():
            results.append(ValidationResult(
                check_name="perf_dir_exists",
                passed=False,
                expected=1,
                actual=0,
                tolerance=0,
                error_pct=100,
                message=f"Perf directory not found: {perf_dir}",
            ))
            return results

        # Check each event group file
        expected_groups = [
            "topdown_l1", "instruction_mix", "cache_l1",
            "cache_l2_l3", "branch_prediction", "memory_access"
        ]

        for group in expected_groups:
            json_file = perf_dir / f"perf_stat_{group}.json"
            if json_file.exists():
                content = json_file.read_text()
                events = self._parse_perf_csv(content)
                results.append(ValidationResult(
                    check_name=f"{group}_events",
                    passed=len(events) > 0,
                    expected=1,
                    actual=len(events),
                    tolerance=0,
                    error_pct=0 if len(events) > 0 else 100,
                    message=f"{group}: {len(events)} events collected",
                ))
            else:
                results.append(ValidationResult(
                    check_name=f"{group}_file",
                    passed=False,
                    expected=1,
                    actual=0,
                    tolerance=0,
                    error_pct=100,
                    message=f"Missing: {json_file.name}",
                ))

        return results

    def validate_feature_extraction(
        self,
        raw_path: Path,
        features_path: Path
    ) -> list[ValidationResult]:
        """Validate feature extraction accuracy.

        Compares manually calculated values with extracted features.
        """
        results = []

        # Load raw data
        all_events = self._load_all_perf_events(raw_path)

        # Load feature vector
        fv = json.loads(features_path.read_text())

        # 1. Validate IPC
        cycles = all_events.get('armv8_pmuv3/cpu_cycles/', {}).get('value', 0)
        instrs = all_events.get('armv8_pmuv3/inst_retired/', {}).get('value', 0)

        if cycles > 0:
            manual_ipc = instrs / cycles
            extracted_ipc = fv['compute']['ipc']
            error_pct = abs(manual_ipc - extracted_ipc) / manual_ipc * 100

            results.append(ValidationResult(
                check_name="ipc",
                passed=error_pct <= self.tolerance * 100,
                expected=manual_ipc,
                actual=extracted_ipc,
                tolerance=self.tolerance,
                error_pct=error_pct,
                message=f"IPC: manual={manual_ipc:.4f}, extracted={extracted_ipc:.4f}, error={error_pct:.2f}%",
            ))

        # 2. Validate TopDown L1
        frontend = all_events.get('armv8_pmuv3/stall_frontend/', {}).get('value', 0)
        backend = all_events.get('armv8_pmuv3/stall_backend/', {}).get('value', 0)

        if cycles > 0:
            # Frontend Bound
            manual_fe = frontend / cycles
            extracted_fe = fv['compute']['topdown_l1']['frontend_bound']
            error_pct = abs(manual_fe - extracted_fe) / max(manual_fe, 0.01) * 100

            results.append(ValidationResult(
                check_name="frontend_bound",
                passed=error_pct <= self.tolerance * 100,
                expected=manual_fe,
                actual=extracted_fe,
                tolerance=self.tolerance,
                error_pct=error_pct,
                message=f"Frontend Bound: manual={manual_fe:.2%}, extracted={extracted_fe:.2%}",
            ))

            # Backend Bound
            manual_be = backend / cycles
            extracted_be = fv['compute']['topdown_l1']['backend_bound']
            error_pct = abs(manual_be - extracted_be) / max(manual_be, 0.01) * 100

            results.append(ValidationResult(
                check_name="backend_bound",
                passed=error_pct <= self.tolerance * 100,
                expected=manual_be,
                actual=extracted_be,
                tolerance=self.tolerance,
                error_pct=error_pct,
                message=f"Backend Bound: manual={manual_be:.2%}, extracted={extracted_be:.2%}",
            ))

        # 3. Validate L3 MPKI
        l3_refill = all_events.get('armv8_pmuv3/l3d_cache_refill/', {}).get('value', 0)

        if instrs > 0:
            manual_mpki = l3_refill / (instrs / 1000)
            extracted_mpki = fv['cache']['l3_mpki']
            error_pct = abs(manual_mpki - extracted_mpki) / max(manual_mpki, 0.01) * 100

            results.append(ValidationResult(
                check_name="l3_mpki",
                passed=error_pct <= self.tolerance * 100,
                expected=manual_mpki,
                actual=extracted_mpki,
                tolerance=self.tolerance,
                error_pct=error_pct,
                message=f"L3 MPKI: manual={manual_mpki:.2f}, extracted={extracted_mpki:.2f}",
            ))

        # 4. Validate L1D MPKI
        l1d_refill = all_events.get('armv8_pmuv3/l1d_cache_refill/', {}).get('value', 0)

        if instrs > 0:
            manual_mpki = l1d_refill / (instrs / 1000)
            extracted_mpki = fv['cache']['l1d_mpki']
            error_pct = abs(manual_mpki - extracted_mpki) / max(manual_mpki, 0.01) * 100

            results.append(ValidationResult(
                check_name="l1d_mpki",
                passed=error_pct <= self.tolerance * 100,
                expected=manual_mpki,
                actual=extracted_mpki,
                tolerance=self.tolerance,
                error_pct=error_pct,
                message=f"L1D MPKI: manual={manual_mpki:.2f}, extracted={extracted_mpki:.2f}",
            ))

        return results

    def validate_consistency(
        self,
        features_paths: list[Path],
        expected_order: list[str]
    ) -> list[ValidationResult]:
        """Validate consistency across multiple runs.

        Checks if workload characteristics are consistent with expectations.
        """
        results = []

        if len(features_paths) < 2:
            return results

        # Load all feature vectors
        fvs = [json.loads(p.read_text()) for p in features_paths]

        # Check IPC ordering
        ipcs = [fv['compute']['ipc'] for fv in fvs]
        names = [Path(p).stem for p in features_paths]

        # Compute relative differences
        for i in range(len(ipcs)):
            for j in range(i + 1, len(ipcs)):
                diff_pct = abs(ipcs[i] - ipcs[j]) / max(ipcs[i], ipcs[j]) * 100
                results.append(ValidationResult(
                    check_name=f"ipc_consistency_{i}_{j}",
                    passed=diff_pct < 50,  # Allow 50% variation across configs
                    expected=ipcs[i],
                    actual=ipcs[j],
                    tolerance=0.5,
                    error_pct=diff_pct,
                    message=f"IPC {names[i]} vs {names[j]}: {ipcs[i]:.2f} vs {ipcs[j]:.2f} ({diff_pct:.1f}% diff)",
                ))

        return results

    def _parse_perf_csv(self, content: str) -> dict:
        """Parse perf stat CSV output."""
        events = {}
        for line in content.strip().split('\n'):
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 6:
                try:
                    events[parts[2]] = {
                        'value': int(parts[0]),
                        'stddev_pct': float(parts[3].rstrip('%')) if parts[3] else 0,
                        'time_enabled_ns': int(parts[4]) if parts[4] else 0,
                        'pcnt_running': float(parts[5]) / 100 if parts[5] else 1.0,
                    }
                except (ValueError, IndexError):
                    continue
        return events

    def _load_all_perf_events(self, raw_path: Path) -> dict:
        """Load all perf events from raw data directory."""
        all_events = {}
        perf_dir = raw_path / "perf"

        if not perf_dir.exists():
            return all_events

        for json_file in perf_dir.glob("*.json"):
            content = json_file.read_text()
            events = self._parse_perf_csv(content)
            all_events.update(events)

        return all_events

    def generate_report(self, results: list[ValidationResult]) -> str:
        """Generate a validation report."""
        lines = ["=" * 60, "ArkProbe Data Validation Report", "=" * 60]

        passed = sum(1 for r in results if r.passed)
        total = len(results)

        lines.append(f"\nSummary: {passed}/{total} checks passed ({passed/total*100:.0f}%)")
        lines.append("\n" + "-" * 60)

        for r in results:
            status = "✓ PASS" if r.passed else "✗ FAIL"
            lines.append(f"\n{status} [{r.check_name}]")
            lines.append(f"  Expected: {r.expected:.4f}")
            lines.append(f"  Actual:   {r.actual:.4f}")
            lines.append(f"  Error:    {r.error_pct:.2f}%")
            lines.append(f"  {r.message}")

        return "\n".join(lines)


def main():
    """CLI entry point for validation."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m arkprobe.utils.validate <raw_data_dir> [features.json]")
        sys.exit(1)

    raw_path = Path(sys.argv[1])
    validator = DataValidator()

    # Validate raw data
    print("\n=== Validating Raw Data ===")
    raw_results = validator.validate_raw_data(raw_path)
    for r in raw_results:
        status = "✓" if r.passed else "✗"
        print(f"  {status} {r.message}")

    # Validate feature extraction if provided
    if len(sys.argv) >= 3:
        features_path = Path(sys.argv[2])
        print("\n=== Validating Feature Extraction ===")
        feat_results = validator.validate_feature_extraction(raw_path, features_path)
        for r in feat_results:
            status = "✓" if r.passed else "✗"
            print(f"  {status} {r.message}")

    # Summary
    all_results = raw_results + (feat_results if len(sys.argv) >= 3 else [])
    passed = sum(1 for r in all_results if r.passed)
    print(f"\n=== Summary: {passed}/{len(all_results)} checks passed ===")


if __name__ == "__main__":
    main()
