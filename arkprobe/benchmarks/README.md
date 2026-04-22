# ArkProbe Benchmark Scripts

Official benchmark and validation scripts for ArkProbe.

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
