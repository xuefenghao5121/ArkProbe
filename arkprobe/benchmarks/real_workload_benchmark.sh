#!/bin/bash
# ArkProbe Real Workload Benchmark — Java compute-intensive applications
# with C++ acceleration via JNI.
#
# Workloads: FFT, SOR, Sparse Matvec, LU Decomposition, KMeans
#
# Usage:
#   ./real_workload_benchmark.sh [--java-only]
#
# Prerequisites:
#   JDK 21+ (openjdk-21-jdk), g++

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JAVA_DIR="${SCRIPT_DIR}/java"
JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-21-openjdk-amd64}"
OUTPUT_DIR="/tmp/arkprobe_realworkload_$$"

JAVA_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --java-only) JAVA_ONLY=true ;;
        -h|--help)
            echo "Usage: $0 [--java-only]"
            echo "  --java-only  Only run Java baseline, no C++ acceleration"
            exit 0
            ;;
    esac
done

echo "=========================================================="
echo " ArkProbe Real Workload Benchmark (Java vs C++)"
echo "=========================================================="
echo "Java dir: ${JAVA_DIR}"
echo "Output:   ${OUTPUT_DIR}"
echo ""

mkdir -p "${OUTPUT_DIR}"

# ---- Phase 1: Compile Java benchmark ----
echo "[Phase 1] Compiling RealWorkloadBench.java..."
cd "${JAVA_DIR}"
export JAVA_HOME
javac RealWorkloadBench.java
echo "  Done."

# ---- Phase 2: Run pure Java baseline ----
echo "[Phase 2] Running pure Java baseline..."
java RealWorkloadBench > "${OUTPUT_DIR}/baseline.json" 2>"${OUTPUT_DIR}/baseline_stderr.txt"
echo "  Results: ${OUTPUT_DIR}/baseline.json"

if [ "$JAVA_ONLY" = true ]; then
    echo ""
    echo "Java-only mode. Baseline results:"
    cat "${OUTPUT_DIR}/baseline.json"
    exit 0
fi

# ---- Phase 3: Compile C++ shared library ----
echo "[Phase 3] Compiling C++ native library..."
cd "${JAVA_DIR}"

g++ -shared -fPIC -O3 -march=native \
    -o "${OUTPUT_DIR}/libarkprobe_realworkload.so" \
    "${JAVA_DIR}/arkprobe_realworkload.cpp" \
    -I"${JAVA_HOME}/include" -I"${JAVA_HOME}/include/linux"

echo "  Built: ${OUTPUT_DIR}/libarkprobe_realworkload.so"

# ---- Phase 4: Run accelerated benchmark ----
echo "[Phase 4] Running C++ accelerated benchmark..."
java -Djava.library.path="${OUTPUT_DIR}" RealWorkloadBench > "${OUTPUT_DIR}/accelerated.json" 2>"${OUTPUT_DIR}/accelerated_stderr.txt"
echo "  Results: ${OUTPUT_DIR}/accelerated.json"

# ---- Phase 5: Compare results ----
echo "[Phase 5] Comparing results..."
echo ""
python3 "${SCRIPT_DIR}/compare_results.py" \
    "${OUTPUT_DIR}/baseline.json" \
    "${OUTPUT_DIR}/accelerated.json"

echo ""
echo "Output files in ${OUTPUT_DIR}/"
echo "  baseline.json      — Pure Java timing"
echo "  accelerated.json   — Java + C++ timing"
echo "Done."
