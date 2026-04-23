#!/bin/bash
# ArkProbe Hotspot C++ Acceleration Benchmark
# Full pipeline: compile Java → baseline → profiling → C++ codegen → compile .so → accelerated → compare
#
# Usage:
#   ./hotspot_benchmark.sh [--skip-profiling] [--java-only]
#
# Prerequisites:
#   JDK 21+ (openjdk-21-jdk), cmake, g++

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JAVA_DIR="${SCRIPT_DIR}/java"
JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-21-openjdk-amd64}"
OUTPUT_DIR="/tmp/arkprobe_benchmark_$$"

SKIP_PROFILING=false
JAVA_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --skip-profiling) SKIP_PROFILING=true ;;
        --java-only) JAVA_ONLY=true ;;
        -h|--help)
            echo "Usage: $0 [--skip-profiling] [--java-only]"
            echo "  --skip-profiling  Skip JFR profiling step, use pre-built .so"
            echo "  --java-only       Only run Java baseline, no C++ acceleration"
            exit 0
            ;;
    esac
done

echo "=============================================="
echo " ArkProbe Hotspot C++ Acceleration Benchmark"
echo "=============================================="
echo "Java dir: ${JAVA_DIR}"
echo "Output:   ${OUTPUT_DIR}"
echo ""

mkdir -p "${OUTPUT_DIR}"

# ---- Phase 1: Compile Java benchmark ----
echo "[Phase 1] Compiling Java benchmark..."
cd "${JAVA_DIR}"
export JAVA_HOME
javac HotspotBench.java
echo "  Done."

# ---- Phase 2: Run pure Java baseline ----
echo "[Phase 2] Running pure Java baseline..."
java HotspotBench > "${OUTPUT_DIR}/baseline.json" 2>"${OUTPUT_DIR}/baseline_stderr.txt"
echo "  Results: ${OUTPUT_DIR}/baseline.json"

if [ "$JAVA_ONLY" = true ]; then
    echo ""
    echo "Java-only mode. Baseline results:"
    cat "${OUTPUT_DIR}/baseline.json"
    exit 0
fi

# ---- Phase 3: Build C++ native library ----
if [ "$SKIP_PROFILING" = false ]; then
    echo "[Phase 3] Running JFR profiling + ArkProbe pipeline..."
    # Start Java benchmark in profile mode (long-running loop)
    java -XX:StartFlightRecording=duration=10s,settings=profile,filename="${OUTPUT_DIR}/bench.jfr" \
         HotspotBench > /dev/null 2>&1 || true
    echo "  JFR recording saved: ${OUTPUT_DIR}/bench.jfr"

    # Run ArkProbe hotspot pipeline (if available)
    if command -v python3 &>/dev/null && python3 -c "import arkprobe" 2>/dev/null; then
        echo "  Running ArkProbe hotspot pipeline..."
        python3 -m arkprobe.cli hotspot \
            --jfr-file "${OUTPUT_DIR}/bench.jfr" \
            --output "${OUTPUT_DIR}/hotspot_output" \
            2>"${OUTPUT_DIR}/pipeline_stderr.txt" || {
            echo "  [WARN] ArkProbe pipeline failed, falling back to manual C++ build"
            SKIP_PROFILING=true
        }
    else
        echo "  [WARN] ArkProbe not installed, using pre-built C++ implementation"
        SKIP_PROFILING=true
    fi
else
    echo "[Phase 3] Skipping profiling (using pre-built C++)"
fi

# ---- Phase 4: Compile C++ shared library ----
echo "[Phase 4] Compiling C++ native library..."
cd "${JAVA_DIR}"

if [ "$SKIP_PROFILING" = true ]; then
    # Use the hand-crafted C++ implementation in the benchmark dir
    CPP_SRC="${JAVA_DIR}/arkprobe_hotspot.cpp"
else
    # Use ArkProbe-generated C++ (if available)
    GEN_CPP="${OUTPUT_DIR}/hotspot_output/build"
    if [ -d "${GEN_CPP}" ]; then
        CPP_SRC="${GEN_CPP}"
    else
        CPP_SRC="${JAVA_DIR}/arkprobe_hotspot.cpp"
    fi
fi

g++ -shared -fPIC -O3 -march=native \
    -o "${OUTPUT_DIR}/libarkprobe_hotspot.so" \
    "${JAVA_DIR}/arkprobe_hotspot.cpp" \
    -I"${JAVA_HOME}/include" -I"${JAVA_HOME}/include/linux"

echo "  Built: ${OUTPUT_DIR}/libarkprobe_hotspot.so"

# ---- Phase 5: Run accelerated benchmark ----
echo "[Phase 5] Running C++ accelerated benchmark..."
java -Djava.library.path="${OUTPUT_DIR}" HotspotBench > "${OUTPUT_DIR}/accelerated.json" 2>"${OUTPUT_DIR}/accelerated_stderr.txt"
echo "  Results: ${OUTPUT_DIR}/accelerated.json"

# ---- Phase 6: Compare results ----
echo "[Phase 6] Comparing results..."
echo ""
python3 "${SCRIPT_DIR}/compare_results.py" \
    "${OUTPUT_DIR}/baseline.json" \
    "${OUTPUT_DIR}/accelerated.json"

echo ""
echo "Output files in ${OUTPUT_DIR}/"
echo "  baseline.json      — Pure Java timing"
echo "  accelerated.json   — Java + C++ timing"
echo "Done."
