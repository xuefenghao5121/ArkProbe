# ArkProbe Benchmark Scripts

Official benchmark and validation scripts for ArkProbe.

---

## JVM Load Test

`jvm_load_test.sh` — Generates a sustained JVM workload with configurable heap size and GC algorithm. Used to validate JVM application analysis features.

### Usage

```bash
# Basic usage (defaults: 60s duration, 2GB heap, G1GC)
cd /path/to/arkprobe
./arkprobe/benchmarks/jvm_load_test.sh

# Custom parameters
./arkprobe/benchmarks/jvm_load_test.sh \
  --duration 120 \
  --heap-size 4g \
  --gc ZGC
```

### Parameters

| Option | Description | Default |
|--------|-------------|---------|
| `--duration SECS` | Test duration in seconds | `60` |
| `--heap-size SIZE` | JVM heap size (e.g., 2g, 4g) | `2g` |
| `--gc ALGO` | GC algorithm: G1GC, ZGC, Parallel, Serial | `G1GC` |
| `--help` | Show help message | — |

### Output

The script:
1. Compiles `JvmLoadTest.java` in a temporary workspace
2. Starts the JVM with specified parameters
3. Prints the PID for use with ArkProbe
4. Runs until completion (or Ctrl+C to stop)
5. Prints final GC statistics

### With ArkProbe

Once the JVM is running:

```bash
# Collect JFR data for the running JVM
arkprobe collect --jfr --jvm-pid <PID> -b jvm --duration 60

# Analyze collected data
arkprobe analyze --input ./data --output ./data

# View feature vector
cat ./data/jvm_general_features.json | jq '.jvm'

# Get optimization recommendations
arkprobe optimize -f ./data/jvm_general_features.json
```

### Notes

- Requires JDK 11+ (JFR built-in) or JDK 8 with fallback to jstat
- Test generates both CPU load (compute) and GC pressure (memory allocation)
- Safe to run multiple times (creates isolated workspace under `benchmarks/jvm_load_test_workspace/`)

---

## Hotspot C++ Acceleration Benchmark

`hotspot_benchmark.sh` — End-to-end benchmark comparing pure Java vs. C++-accelerated (via JNI) performance for 11 hotspot method patterns.

### Quick Start

```bash
cd /path/to/arkprobe/arkprobe/benchmarks

# Full pipeline: compile Java → baseline → compile C++ → accelerated → compare
./hotspot_benchmark.sh

# Java baseline only (no C++ acceleration)
./hotspot_benchmark.sh --java-only

# Skip JFR profiling, use pre-built C++ implementation
./hotspot_benchmark.sh --skip-profiling
```

### Prerequisites

- JDK 21+ (`openjdk-21-jdk` or equivalent)
- `g++` with C++17 support
- `python3` (for `compare_results.py`)

### Parameters

| Option | Description |
|--------|-------------|
| `--skip-profiling` | Skip JFR profiling step, use pre-built .so |
| `--java-only` | Only run Java baseline, no C++ acceleration |
| `-h` / `--help` | Show help message |

### Benchmark Methods (11 methods, 4 pattern types)

| Method | Pattern | Java Implementation | C++ Optimization |
|--------|---------|-------------------|-----------------|
| `vector_map` | vector_expr | float[] element-wise multiply | NEON vmulq_f32 / AVX2 _mm256_mul_ps |
| `vector_reduce` | vector_expr | float[] sum | NEON vaddq_f32 tree-reduce |
| `vector_filter` | vector_expr | float[] conditional filter | NEON vcgeq_f32 + compact |
| `math_sigmoid` | math | 1/(1+exp(-x)) | NEON vrecpe / libm |
| `math_relu` | math | max(0, x) | NEON vmaxq_f32 / AVX2 _mm256_max_ps |
| `matmul` | math | 64x64 matrix multiply | Cache-blocked GEMM + SIMD |
| `array_copy` | memory_bandwidth | System.arraycopy | memcpy / SIMD stream |
| `array_scale` | memory_bandwidth | element-wise scalar multiply | NEON vmulq_n_f32 |
| `prefetch` | memory_bandwidth | sequential access + prefetch | __builtin_prefetch / _mm_prefetch |
| `string_parse` | string | Integer.parseInt batch | sscanf / strtol |
| `string_search` | string | String.indexOf batch | memchr / SIMD strstr |

### Output

The script outputs to `/tmp/arkprobe_benchmark_$$/`:

| File | Description |
|------|-------------|
| `baseline.json` | Pure Java timing results |
| `accelerated.json` | Java + C++ timing results |
| `bench.jfr` | JFR recording (if profiling enabled) |
| `libarkprobe_hotspot.so` | Compiled C++ native library |

`compare_results.py` prints a comparison table:

```
Method               Java (ms)    C++ (ms)    Speedup
----------------------------------------------------------
array_copy               13.456       13.389      1.00x
array_scale              55.823       55.412      1.01x
math_relu                46.721       33.482      1.40x
math_sigmoid             48.153       41.047      1.17x
matmul                  104.638       95.102      1.10x
prefetch                 72.891       72.534      1.00x
string_parse             85.234      209.102      0.41x
string_search            42.312      263.841      0.16x
vector_filter            91.523       82.341      1.11x
vector_map               83.920       58.705      1.43x
vector_reduce            79.812       14.301      5.58x
```

### Key Findings

| Category | Speedup | Recommendation |
|----------|---------|---------------|
| Numeric vector (map/reduce/filter) | 1.1-5.6x | Recommended for C++ acceleration |
| Math functions (sigmoid/relu) | 1.2-1.4x | Recommended |
| Memory bandwidth (copy/scale/prefetch) | ~1.0x | Not recommended — memory-bound |
| String processing (parse/search) | 0.16-0.4x | Not recommended — JNI string conversion overhead |

### Files

| File | Description |
|------|-------------|
| `java/HotspotBench.java` | Self-contained Java benchmark (warmup + measurement + JSON output) |
| `java/arkprobe_hotspot.cpp` | C++ JNI implementation (ARM NEON / x86 AVX2 / scalar) |
| `hotspot_benchmark.sh` | Orchestration script |
| `compare_results.py` | Result comparison tool |

### Notes

- Java benchmark auto-detects native library via `System.loadLibrary("arkprobe_hotspot")`
- Two-run approach: pure Java first, then with `-Djava.library.path` pointing to .so
- C++ compiled with `-O3 -march=native -fPIC -shared`
- `GetPrimitiveArrayCritical` used for numeric arrays (zero-copy JNI access)
- String methods are slower in C++ due to per-string JNI conversion overhead — this is a fundamental JNI limitation
