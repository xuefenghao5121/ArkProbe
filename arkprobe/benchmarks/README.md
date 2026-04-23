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

---

## Real Workload Benchmark

`real_workload_benchmark.sh` — End-to-end benchmark comparing pure Java vs. C++-accelerated performance for 5 real compute-intensive algorithms (not micro-benchmarks).

### Quick Start

```bash
cd /path/to/arkprobe/arkprobe/benchmarks

# Full pipeline: compile Java → baseline → compile C++ → accelerated → compare
./real_workload_benchmark.sh

# Java baseline only (no C++ acceleration)
./real_workload_benchmark.sh --java-only
```

### Prerequisites

- JDK 21+ (`openjdk-21-jdk` or equivalent)
- `g++` with C++17 support
- `python3` (for `compare_results.py`)

### Parameters

| Option | Description |
|--------|-------------|
| `--java-only` | Only run Java baseline, no C++ acceleration |
| `-h` / `--help` | Show help message |

### Benchmark Methods (5 real algorithms)

| Method | Algorithm | Data Size | C++ Optimization |
|--------|-----------|-----------|-----------------|
| `fftTransform` | Cooley-Tukey radix-2 1D FFT | 2^20 (1M) points, 50 iters | SIMD butterfly expansion |
| `sorIteration` | Red-black SOR (PDE solver) | 500x500, 50 iters | Scalar loop (irregular access) |
| `sparseMatvec` | CSR sparse matrix-vector multiply | 100Kx100K, 10 nnz/row, 200 iters | SIMD dot product per row |
| `luDecompose` | LU decomposition with partial pivoting | 512x512, 10 iters | SIMD row elimination |
| `kmeansAssign` | KMeans distance calculation | 100K pts x 32 dim x 64 centroids, 50 iters | SIMD distance (dx^2+dy^2+...) |

### x86 Measured Results

| Method | Java (ms) | C++ (ms) | Speedup | Notes |
|--------|-----------|----------|---------|-------|
| `luDecompose` | 270 | 134 | **2.0x** | Contiguous SIMD row elimination |
| `sparseMatvec` | 210 | 156 | **1.3x** | SIMD per-row, gather-scatter limits |
| `kmeansAssign` | 4850 | 3710 | **1.3x** | SIMD distance, n*k*dim triple loop |
| `sorIteration` | 143 | 118 | **1.2x** | Compiler optimization (not SIMD) |
| `fftTransform` | 2000 | 1950 | **1.0x** | Sequential twiddle rotation limits SIMD |

### Key Findings

1. **Real algorithms show more nuanced results** than micro-benchmarks — data dependencies and irregular access patterns reduce SIMD effectiveness
2. **LU decomposition benefits most** (2.0x) — contiguous row elimination is ideal for SIMD
3. **FFT benefits least** (1.0x) — butterfly twiddle rotation is sequential, SIMD overhead not worth it
4. **Larger data sizes reduce JNI overhead** — 512x512 LU gets 2.0x vs 64x64 matmul's 1.1x in micro-benchmarks
5. **Even non-SIMD methods benefit** — SOR gets 1.2x from compiler optimization alone

### Files

| File | Description |
|------|-------------|
| `java/RealWorkloadBench.java` | Self-contained Java benchmark with 5 real algorithms |
| `java/arkprobe_realworkload.cpp` | C++ JNI implementation (ARM NEON / x86 AVX2 / scalar) |
| `real_workload_benchmark.sh` | Orchestration script |

### Notes

- Library named `libarkprobe_realworkload.so` (separate from HotspotBench's `libarkprobe_hotspot.so`)
- All methods use `GetPrimitiveArrayCritical` for zero-copy array access
- SOR C++ uses scalar loop matching Java stride-2 access pattern (SIMD would be incorrect for red-black)
- Benchmark includes 5-iteration warmup before measurement

---

## Application Workload Benchmark

`app_workload_benchmark.sh` — Real Java application workloads: ML feature engineering pipeline + Flink-style stream processing kernels.

### Quick Start

```bash
cd /path/to/arkprobe/arkprobe/benchmarks

# Full pipeline
./app_workload_benchmark.sh

# Java baseline only
./app_workload_benchmark.sh --java-only
```

### Prerequisites

- JDK 21+ (`openjdk-21-jdk` or equivalent)
- `g++` with C++17 support
- `python3` (for `compare_results.py`)
- 4GB+ available memory (`-Xmx4g`)

### Scenario 1: ML Feature Engineering (mirrors Spark MLlib)

| Method | Equivalent | Data Size | C++ Optimization |
|--------|-----------|-----------|-----------------|
| `zscoreNormalize` | StandardScaler | 1M rows × 64 features | SIMD broadcast-subtract-divide per row |
| `minMaxScale` | MinMaxScaler | 1M rows × 64 features | SIMD broadcast-subtract-divide with div-by-zero guard |
| `tfidfMultiply` | HashingTF + IDF | 100K docs, 50K vocab | Gather-multiply for sparse TF × dense IDF |
| `featureHash` | FeatureHasher | 1M features, 2^16 buckets | C++ MurmurHash3 on contiguous byte[] |
| `bucketize` | Bucketizer | 1M values, 100 splits | Same binary search (branch-heavy, not SIMD-friendly) |

### Scenario 2: Flink-Style Stream Processing

| Method | Equivalent Flink API | Data Size | C++ Optimization |
|--------|---------------------|-----------|-----------------|
| `windowedAggregate` | keyBy().window(Tumbling).aggregate(Sum) | 10M events, 10K keys | Direct index accumulate (no HashMap) |
| `windowedTopN` | keyBy().window().sort().limit(N) | 10M events, 10K keys, top-10 | nth_element + partial sort |
| `sessionWindowMerge` | keyBy().window(EventTimeSessionGap) | 10M events, 10K keys | C++ sort + merge (no HashMap/ArrayList) |

### x86 Measured Results (3 runs)

| Method | Java (ms) | C++ (ms) | Speedup | Notes |
|--------|-----------|----------|---------|-------|
| `windowedTopN` | 23,500 | 920 | **25x** | Java HashMap + sort per key vs C++ nth_element |
| `sessionWindowMerge` | 2,000 | 790 | **2.5x** | Java HashMap/ArrayList vs C++ sort + merge |
| `minMaxScale` | 1,000 | 540 | **1.8x** | SIMD broadcast-subtract-divide |
| `zscoreNormalize` | 940 | 620 | **1.5x** | SIMD broadcast-subtract-divide (div more costly) |
| `featureHash` | 440 | 340 | **1.3x** | C++ MurmurHash3 on contiguous byte[] |
| `tfidfMultiply` | 215 | 160 | **1.3x** | Gather-multiply for sparse vectors |
| `windowedAggregate` | 760 | 640 | **1.2x** | Direct index vs HashMap, but computation is simple |
| `bucketize` | 1,700 | 1,740 | **1.0x** | Binary search is branch-heavy, C++ no advantage |

### Key Findings

1. **Data structure replacement is the biggest win**: `windowedTopN` 25x — replacing Java HashMap + ArrayList + sort with C++ nth_element + direct arrays eliminates GC pressure and container overhead
2. **SIMD normalization effective**: `minMaxScale` 1.8x, `zscoreNormalize` 1.5x — broadcast-subtract-divide pattern matches SIMD well
3. **TF-IDF and FeatureHash moderate gains** (1.3x): gather-scatter and hash computation have inherent irregularity
4. **Bucketize shows no gain** (1.0x): binary search is branch-heavy and data-dependent, C++ compiler can't improve over JIT
5. **Flink-style hotspots are often containers, not compute**: the 25x topN win comes from eliminating Java container overhead, not SIMD

### Files

| File | Description |
|------|-------------|
| `java/AppWorkloadBench.java` | Self-contained Java benchmark with ML + Flink workloads |
| `java/arkprobe_appworkload.cpp` | C++ JNI implementation (ARM NEON / x86 AVX2 / scalar) |
| `app_workload_benchmark.sh` | Orchestration script |

### Notes

- Library named `libarkprobe_appworkload.so`
- FeatureHash uses contiguous `byte[]` + `offsets[]` + `lengths[]` (not `String[]` or `byte[][]`) to avoid JNI string/array-of-array overhead
- ML workloads operate on row-major `double[]` with column parameters (means/stds/mins/maxes) — matches Spark's `Vector` representation
- Flink-style workloads use pre-sorted event streams (simulating Kafka source with event-time ordering)
- `-Xmx4g` recommended due to 10M event stream data
