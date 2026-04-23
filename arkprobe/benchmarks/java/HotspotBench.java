/**
 * ArkProbe Hotspot Benchmark — Java compute/memory-intensive workload
 * with native (C++) acceleration via JNI.
 *
 * Usage:
 *   java HotspotBench                  — pure Java baseline
 *   java -Djava.library.path=<dir> HotspotBench  — with native acceleration
 *
 * Output: JSON with timing for each benchmark method.
 */
import java.util.*;

public class HotspotBench {

    // ========== native method declarations (C++ .so provides these) ==========
    public static native float[] vectorMap(float[] data, float factor);
    public static native float vectorReduce(float[] data);
    public static native float[] vectorFilter(float[] data, float threshold);
    public static native double[] mathSigmoid(double[] data);
    public static native double[] mathRelu(double[] data);
    public static native double[][] matmul(double[][] a, double[][] b);
    public static native float[] arrayCopy(float[] src);
    public static native float[] arrayScale(float[] src, float factor);
    public static native float prefetch(float[] data, int stride);
    public static native int stringParse(String[] data);
    public static native int stringSearch(String[] data, String pattern);

    // ========== Java implementations ==========

    public static float[] vectorMapJava(float[] data, float factor) {
        float[] out = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            out[i] = data[i] * factor;
        }
        return out;
    }

    public static float vectorReduceJava(float[] data) {
        float sum = 0.0f;
        for (int i = 0; i < data.length; i++) {
            sum += data[i];
        }
        return sum;
    }

    public static float[] vectorFilterJava(float[] data, float threshold) {
        int count = 0;
        for (float v : data) {
            if (v > threshold) count++;
        }
        float[] out = new float[count];
        int idx = 0;
        for (float v : data) {
            if (v > threshold) out[idx++] = v;
        }
        return out;
    }

    public static double[] mathSigmoidJava(double[] data) {
        double[] out = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            out[i] = 1.0 / (1.0 + Math.exp(-data[i]));
        }
        return out;
    }

    public static double[] mathReluJava(double[] data) {
        double[] out = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            out[i] = Math.max(0.0, data[i]);
        }
        return out;
    }

    public static double[][] matmulJava(double[][] a, double[][] b) {
        int n = a.length;
        double[][] c = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                double aik = a[i][k];
                for (int j = 0; j < n; j++) {
                    c[i][j] += aik * b[k][j];
                }
            }
        }
        return c;
    }

    public static float[] arrayCopyJava(float[] src) {
        float[] dst = new float[src.length];
        System.arraycopy(src, 0, dst, 0, src.length);
        return dst;
    }

    public static float[] arrayScaleJava(float[] src, float factor) {
        float[] out = new float[src.length];
        for (int i = 0; i < src.length; i++) {
            out[i] = src[i] * factor;
        }
        return out;
    }

    public static float prefetchJava(float[] data, int stride) {
        float sum = 0.0f;
        for (int i = 0; i < data.length; i += stride) {
            sum += data[i];
        }
        return sum;
    }

    public static int stringParseJava(String[] data) {
        int count = 0;
        for (String s : data) {
            int len = s.length();
            for (int i = 0; i < len; i++) {
                char c = s.charAt(i);
                if (c >= '0' && c <= '9') count++;
            }
        }
        return count;
    }

    public static int stringSearchJava(String[] data, String pattern) {
        int count = 0;
        int plen = pattern.length();
        for (String s : data) {
            int idx = 0;
            while ((idx = s.indexOf(pattern, idx)) != -1) {
                count++;
                idx += plen;
            }
        }
        return count;
    }

    // ========== benchmark harness ==========

    private static final int WARMUP_ITERS = 10;

    static long measure(Runnable r, int iterations) {
        for (int i = 0; i < WARMUP_ITERS; i++) r.run();
        long start = System.nanoTime();
        for (int i = 0; i < iterations; i++) r.run();
        return System.nanoTime() - start;
    }

    public static void main(String[] args) throws Exception {
        boolean nativeLoaded = false;
        try {
            System.loadLibrary("arkprobe_hotspot");
            nativeLoaded = true;
            System.err.println("[INFO] Native library loaded.");
        } catch (UnsatisfiedLinkError e) {
            System.err.println("[INFO] Native library not found, Java-only mode.");
        }

        // data sizes
        int vecSize = 1_000_000;
        int matSize = 256;
        int memSize = 4_000_000;
        int strSize = 100_000;
        int prefetchSize = 4_000_000;

        // generate test data
        Random rng = new Random(42);
        float[] vecData = new float[vecSize];
        for (int i = 0; i < vecSize; i++) vecData[i] = rng.nextFloat() * 100.0f;

        double[] mathData = new double[vecSize];
        for (int i = 0; i < vecSize; i++) mathData[i] = rng.nextDouble() * 20.0 - 10.0;

        double[][] matA = new double[matSize][matSize];
        double[][] matB = new double[matSize][matSize];
        for (int i = 0; i < matSize; i++) {
            for (int j = 0; j < matSize; j++) {
                matA[i][j] = rng.nextDouble();
                matB[i][j] = rng.nextDouble();
            }
        }

        float[] memData = new float[memSize];
        for (int i = 0; i < memSize; i++) memData[i] = rng.nextFloat();

        float[] prefetchData = new float[prefetchSize];
        for (int i = 0; i < prefetchSize; i++) prefetchData[i] = rng.nextFloat();

        String[] strData = new String[strSize];
        for (int i = 0; i < strSize; i++) {
            StringBuilder sb = new StringBuilder();
            int len = 20 + rng.nextInt(30);
            for (int j = 0; j < len; j++) {
                sb.append((char) ('a' + rng.nextInt(26)));
            }
            if (rng.nextFloat() < 0.3) {
                int pos = rng.nextInt(sb.length());
                sb.setCharAt(pos, (char) ('0' + rng.nextInt(10)));
            }
            strData[i] = sb.toString();
        }

        String searchPattern = "abc";

        // iteration counts per benchmark
        int vecIters = 100;
        int mathIters = 100;
        int matIters = 10;
        int memIters = 100;
        int prefetchIters = 50;
        int strIters = 100;

        StringBuilder json = new StringBuilder();
        json.append("{\n");

        // --- vector_map ---
        appendResult(json, "vector_map_java", measure(() -> vectorMapJava(vecData, 2.5f), vecIters));
        if (nativeLoaded)
            appendResult(json, "vector_map_cpp", measure(() -> vectorMap(vecData, 2.5f), vecIters));

        // --- vector_reduce ---
        appendResult(json, "vector_reduce_java", measure(() -> vectorReduceJava(vecData), vecIters));
        if (nativeLoaded)
            appendResult(json, "vector_reduce_cpp", measure(() -> vectorReduce(vecData), vecIters));

        // --- vector_filter ---
        appendResult(json, "vector_filter_java", measure(() -> vectorFilterJava(vecData, 50.0f), vecIters));
        if (nativeLoaded)
            appendResult(json, "vector_filter_cpp", measure(() -> vectorFilter(vecData, 50.0f), vecIters));

        // --- math_sigmoid ---
        appendResult(json, "math_sigmoid_java", measure(() -> mathSigmoidJava(mathData), mathIters));
        if (nativeLoaded)
            appendResult(json, "math_sigmoid_cpp", measure(() -> mathSigmoid(mathData), mathIters));

        // --- math_relu ---
        appendResult(json, "math_relu_java", measure(() -> mathReluJava(mathData), mathIters));
        if (nativeLoaded)
            appendResult(json, "math_relu_cpp", measure(() -> mathRelu(mathData), mathIters));

        // --- matmul ---
        appendResult(json, "matmul_java", measure(() -> matmulJava(matA, matB), matIters));
        if (nativeLoaded)
            appendResult(json, "matmul_cpp", measure(() -> matmul(matA, matB), matIters));

        // --- array_copy ---
        appendResult(json, "array_copy_java", measure(() -> arrayCopyJava(memData), memIters));
        if (nativeLoaded)
            appendResult(json, "array_copy_cpp", measure(() -> arrayCopy(memData), memIters));

        // --- array_scale ---
        appendResult(json, "array_scale_java", measure(() -> arrayScaleJava(memData, 1.5f), memIters));
        if (nativeLoaded)
            appendResult(json, "array_scale_cpp", measure(() -> arrayScale(memData, 1.5f), memIters));

        // --- prefetch ---
        appendResult(json, "prefetch_java", measure(() -> prefetchJava(prefetchData, 16), prefetchIters));
        if (nativeLoaded)
            appendResult(json, "prefetch_cpp", measure(() -> prefetch(prefetchData, 16), prefetchIters));

        // --- string_parse ---
        appendResult(json, "string_parse_java", measure(() -> stringParseJava(strData), strIters));
        if (nativeLoaded)
            appendResult(json, "string_parse_cpp", measure(() -> stringParse(strData), strIters));

        // --- string_search ---
        appendResult(json, "string_search_java", measure(() -> stringSearchJava(strData, searchPattern), strIters));
        if (nativeLoaded)
            appendResult(json, "string_search_cpp", measure(() -> stringSearch(strData, searchPattern), strIters));

        // remove trailing comma
        if (json.length() > 2 && json.charAt(json.length() - 2) == ',') {
            json.deleteCharAt(json.length() - 2);
        }
        json.append("}\n");

        System.out.println(json.toString());
    }

    private static void appendResult(StringBuilder json, String key, long nanos) {
        double ms = nanos / 1_000_000.0;
        json.append(String.format("  \"%s_ms\": %.3f,\n", key, ms));
    }
}
