#!/bin/bash
# JVM Load Test for ArkProbe
# Platform: Kunpeng 920/930, JDK 11+
#
# Usage:
#   ./jvm_load_test.sh [--duration SECS] [--heap-size SIZE] [--gc ALGO]
#
# Examples:
#   ./jvm_load_test.sh                        # Default: 60s, 2GB heap, G1GC
#   ./jvm_load_test.sh --duration 120        # Run for 2 minutes
#   ./jvm_load_test.sh --heap-size 4g        # Use 4GB heap
#   ./jvm_load_test.sh --gc ZGC              # Use ZGC

set -e

# Default parameters
DURATION=60
HEAP_SIZE="2g"
GC_ALGO="G1GC"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --duration)
            DURATION="$2"
            shift 2
            ;;
        --heap-size)
            HEAP_SIZE="$2"
            shift 2
            ;;
        --gc)
            GC_ALGO="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--duration SECS] [--heap-size SIZE] [--gc ALGO]"
            echo "  --duration   Collection duration in seconds (default: 60)"
            echo "  --heap-size  JVM heap size, e.g. 2g, 4g (default: 2g)"
            echo "  --gc         GC algorithm: G1GC, ZGC, Parallel, Serial (default: G1GC)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Determine GC options
case "$GC_ALGO" in
    G1GC)    GC_OPTS="-XX:+UseG1GC" ;;
    ZGC)      GC_OPTS="-XX:+UseZGC -XX:+ZGenerational" ;;
    Parallel) GC_OPTS="-XX:+UseParallelGC" ;;
    Serial)   GC_OPTS="-XX:+UseSerialGC" ;;
    *)        GC_OPTS="-XX:+UseG1GC" ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="${SCRIPT_DIR}/jvm_load_test_workspace"
mkdir -p "$TEST_DIR"
cd "$TEST_DIR"

echo "=========================================="
echo "ArkProbe JVM Load Test"
echo "=========================================="
echo "Heap Size: $HEAP_SIZE"
echo "GC Algorithm: $GC_ALGO"
echo "Duration: ${DURATION}s"
echo "Working Directory: $TEST_DIR"
echo ""

# Create Java source
cat > JvmLoadTest.java << 'JAVA_EOF'
import java.util.concurrent.*;
import java.util.*;
import java.lang.management.*;

public class JvmLoadTest {
    private static List<byte[]> allocations = new ArrayList<>();
    private static volatile long computeResult = 0;
    private static ScheduledExecutorService scheduler;

    public static void main(String[] args) throws Exception {
        System.out.println("=== JVM Load Test Started ===");
        System.out.println("JDK: " + System.getProperty("java.version"));
        System.out.println("GC: " + System.getProperty("java.vm.name"));
        System.out.println("Max Heap: " + Runtime.getRuntime().maxMemory() / (1024*1024) + " MB");
        System.out.println("Available CPUs: " + Runtime.getRuntime().availableProcessors());

        scheduler = Executors.newScheduledThreadPool(4);

        // Memory allocation task - triggers GC
        scheduler.scheduleAtFixedRate(() -> {
            try {
                for (int i = 0; i < 50; i++) {
                    allocations.add(new byte[1024 * 512]); // 512KB each
                }
                // Clear when we have too many to trigger GC
                if (allocations.size() > 500) {
                    allocations.clear();
                }
            } catch (OutOfMemoryError e) {
                allocations.clear();
                System.gc();
            }
        }, 0, 50, TimeUnit.MILLISECONDS);

        // CPU-intensive JIT compilation trigger
        scheduler.scheduleAtFixedRate(() -> {
            for (int i = 0; i < 1000; i++) {
                computeResult += compute(i);
            }
        }, 0, 10, TimeUnit.MILLISECONDS);

        // Phase markers
        System.out.println("[PHASE] WARMUP_START");
        Thread.sleep(5000);
        System.out.println("[PHASE] WARMUP_END");

        System.out.println("[PHASE] MEASUREMENT_START");
        Thread.sleep((${DURATION} - 10) * 1000);
        System.out.println("[PHASE] MEASUREMENT_END");

        // Print final stats
        System.out.println("=== Final Statistics ===");
        System.out.println("Allocations created: " + (allocations.size() * 50));
        System.out.println("Compute result: " + computeResult);

        // Print GC stats
        List<GarbageCollectorMXBean> gcBeans = ManagementFactory.getGarbageCollectorMXBeans();
        long totalGC = 0, totalTime = 0;
        for (GarbageCollectorMXBean gc : gcBeans) {
            System.out.println("GC: " + gc.getName() +
                ", count=" + gc.getCollectionCount() +
                ", time=" + gc.getCollectionTime() + "ms");
            totalGC += gc.getCollectionCount();
            totalTime += gc.getCollectionTime();
        }
        System.out.println("Total: " + totalGC + " collections, " + totalTime + "ms");

        System.out.println("=== JVM Load Test Complete ===");

        scheduler.shutdown();
        scheduler.awaitTermination(5, TimeUnit.SECONDS);
    }

    private static long compute(int x) {
        long sum = 0;
        for (int i = 0; i < 100; i++) {
            sum += x * x + i * i;
        }
        return sum;
    }
}
JAVA_EOF

# Compile
echo "Compiling..."
javac JvmLoadTest.java

# Start JVM
echo "Starting JVM with heap=$HEAP_SIZE, GC=$GC_ALGO..."
java $GC_OPTS -Xms${HEAP_SIZE} -Xmx${HEAP_SIZE} -XX:+PrintGCDetails -Xloggc:gc.log JvmLoadTest &
JVM_PID=$!

echo "JVM PID: $JVM_PID"

# Wait for JVM to start
sleep 3

# Verify process is running
if ! ps -p $JVM_PID > /dev/null 2>&1; then
    echo "ERROR: JVM process not found"
    exit 1
fi

echo ""
echo "=========================================="
echo "JVM is running. Use this PID with ArkProbe:"
echo ""
echo "  arkprobe collect --jfr --jvm-pid $JVM_PID -b jvm"
echo ""
echo "Or run the full test with arkprobe:"
echo "  arkprobe collect --jfr --jvm-pid $JVM_PID -b jvm --duration $DURATION"
echo ""
echo "Press Ctrl+C to stop the JVM when done."
echo "=========================================="

# Wait for JVM to complete
wait $JVM_PID
EXIT_CODE=$?

echo ""
echo "JVM exited with code: $EXIT_CODE"

# Show GC log if available
if [ -f gc.log ]; then
    echo ""
    echo "=== Recent GC Log (last 20 lines) ==="
    tail -20 gc.log
fi

exit 0
