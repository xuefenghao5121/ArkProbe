/**
 * ArkProbe Real Application Workload Benchmark.
 *
 * Two real-world Java application scenarios:
 *
 *   Scenario 1 — ML Feature Engineering Pipeline
 *     Mirrors Spark MLlib / scikit-learn preprocessing:
 *     1. zscoreNormalize   — StandardScaler: per-column (x-mean)/std
 *     2. minMaxScale       — MinMaxScaler: per-column (x-min)/(max-min)
 *     3. tfidfMultiply     — HashingTF + IDF: sparse TF * dense IDF weights
 *     4. featureHash       — FeatureHasher: MurmurHash3 on byte[] → fixed vector
 *     5. bucketize         — Bucketizer: binary-search binning of continuous values
 *
 *   Scenario 2 — Flink-Style Stream Processing Kernel
 *     Mirrors Apache Flink windowed aggregations (UDF compute kernel only):
 *     6. windowedAggregate — keyBy().window(Tumbling).aggregate(Sum)
 *     7. windowedTopN      — keyBy().window().sort().limit(N)
 *     8. sessionWindowMerge— keyBy().window(EventTimeSessionGap)
 *
 * Usage:
 *   java AppWorkloadBench                           — pure Java baseline
 *   java -Djava.library.path=<dir> AppWorkloadBench — with C++ acceleration
 */
import java.util.*;

public class AppWorkloadBench {

    // ========== Native method declarations ==========

    // ML Feature Engineering
    public static native void zscoreNormalize(double[] data, int rows, int cols,
                                              double[] means, double[] stds);
    public static native void minMaxScale(double[] data, int rows, int cols,
                                          double[] mins, double[] maxes);
    public static native void tfidfMultiply(int[] docOffsets, int[] termIds,
                                            double[] tfValues, double[] idfWeights,
                                            double[] output, int nDocs, int vocabSize);
    public static native void featureHash(byte[] allNames, int[] offsets, int[] lengths,
                                          double[] values, double[] output, int numFeatures, int n);
    public static native void bucketize(double[] values, double[] splits,
                                        int[] buckets, int n, int numSplits);

    // Flink-Style Stream Processing
    public static native void windowedAggregate(long[] timestamps, int[] keys,
                                                double[] values, double[] results,
                                                int n, int windowSizeMs, int numKeys);
    public static native void windowedTopN(long[] timestamps, int[] keys,
                                           double[] values, double[] topNOutput,
                                           int n, int windowSizeMs, int numKeys, int topK);
    public static native void sessionWindowMerge(long[] timestamps, int[] keys,
                                                 double[] values, int[] sessionCounts,
                                                 int n, long gapMs, int numKeys);

    // ========== ML Feature Engineering — Java implementations ==========

    /** StandardScaler: per-column (x - mean) / std. data is row-major [rows * cols]. */
    public static void zscoreNormalizeJava(double[] data, int rows, int cols,
                                           double[] means, double[] stds) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int idx = i * cols + j;
                data[idx] = (data[idx] - means[j]) / stds[j];
            }
        }
    }

    /** MinMaxScaler: per-column (x - min) / (max - min). data is row-major [rows * cols]. */
    public static void minMaxScaleJava(double[] data, int rows, int cols,
                                       double[] mins, double[] maxes) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int idx = i * cols + j;
                double range = maxes[j] - mins[j];
                data[idx] = range > 0 ? (data[idx] - mins[j]) / range : 0.0;
            }
        }
    }

    /** TF-IDF: sparse TF * dense IDF. docOffsets[nDocs+1] delimits each doc's terms. */
    public static void tfidfMultiplyJava(int[] docOffsets, int[] termIds,
                                         double[] tfValues, double[] idfWeights,
                                         double[] output, int nDocs, int vocabSize) {
        Arrays.fill(output, 0.0);
        for (int d = 0; d < nDocs; d++) {
            int start = docOffsets[d];
            int end = docOffsets[d + 1];
            for (int j = start; j < end; j++) {
                int term = termIds[j];
                if (term >= 0 && term < vocabSize) {
                    output[term] += tfValues[j] * idfWeights[term];
                }
            }
        }
    }

    /**
     * FeatureHasher: MurmurHash3 on contiguous byte[] feature names.
     * allNames[offsets[i]..offsets[i]+lengths[i]-1] is the i-th feature name.
     * Output: fixed-size feature vector (numFeatures buckets).
     */
    public static void featureHashJava(byte[] allNames, int[] offsets, int[] lengths,
                                       double[] values, double[] output, int numFeatures, int n) {
        Arrays.fill(output, 0.0);
        for (int i = 0; i < n; i++) {
            int hash = murmurHash3(allNames, offsets[i], lengths[i], 0x9747b28c);
            int idx = (hash & 0x7FFFFFFF) % numFeatures;
            output[idx] += values[i];
        }
    }

    /** Bucketizer: binary search for each value in sorted splits array. */
    public static void bucketizeJava(double[] values, double[] splits,
                                     int[] buckets, int n, int numSplits) {
        for (int i = 0; i < n; i++) {
            double v = values[i];
            int lo = 0, hi = numSplits;
            while (lo < hi) {
                int mid = (lo + hi) >>> 1;
                if (splits[mid] <= v) lo = mid + 1;
                else hi = mid;
            }
            buckets[i] = lo;
        }
    }

    // ========== Flink-Style — Java implementations ==========

    /**
     * Tumbling window SUM aggregation.
     * Events sorted by timestamp. Each (key, window) accumulates sum.
     * results[numKeys * numWindows]: results[key * numWindows + winIdx] = sum.
     */
    public static void windowedAggregateJava(long[] timestamps, int[] keys,
                                             double[] values, double[] results,
                                             int n, int windowSizeMs, int numKeys) {
        Arrays.fill(results, 0.0);
        int numWindows = 0;
        if (n > 0) {
            long baseTs = timestamps[0];
            long maxTs = timestamps[n - 1];
            numWindows = (int)((maxTs - baseTs) / windowSizeMs) + 1;
        }
        for (int i = 0; i < n; i++) {
            long baseTs = timestamps[0];
            int winIdx = (int)((timestamps[i] - baseTs) / windowSizeMs);
            int k = keys[i];
            if (k >= 0 && k < numKeys) {
                results[k * numWindows + winIdx] += values[i];
            }
        }
    }

    /**
     * Windowed Top-N per key.
     * Collects all values per (key, window), then partial sorts to find top-N.
     * Output: topNOutput[key * topK + rank] for each key's top values.
     */
    public static void windowedTopNJava(long[] timestamps, int[] keys,
                                        double[] values, double[] topNOutput,
                                        int n, int windowSizeMs, int numKeys, int topK) {
        Arrays.fill(topNOutput, Double.NaN);
        // Group by (key, window)
        Map<Long, List<Double>>[] windowData = new HashMap[numKeys];
        for (int k = 0; k < numKeys; k++) windowData[k] = new HashMap<>();
        long baseTs = n > 0 ? timestamps[0] : 0;
        for (int i = 0; i < n; i++) {
            int winIdx = (int)((timestamps[i] - baseTs) / windowSizeMs);
            int k = keys[i];
            if (k >= 0 && k < numKeys) {
                windowData[k].computeIfAbsent((long)winIdx, x -> new ArrayList<>()).add(values[i]);
            }
        }
        // For each key, find global top-N across all windows
        for (int k = 0; k < numKeys; k++) {
            List<Double> allVals = new ArrayList<>();
            for (List<Double> wl : windowData[k].values()) allVals.addAll(wl);
            allVals.sort(Collections.reverseOrder());
            for (int r = 0; r < Math.min(topK, allVals.size()); r++) {
                topNOutput[k * topK + r] = allVals.get(r);
            }
        }
    }

    /**
     * Session window merge: group events by key, sort by time,
     * merge overlapping/adjacent sessions (gap < gapMs).
     * sessionCounts[key] = number of merged sessions for that key.
     */
    public static void sessionWindowMergeJava(long[] timestamps, int[] keys,
                                              double[] values, int[] sessionCounts,
                                              int n, long gapMs, int numKeys) {
        Arrays.fill(sessionCounts, 0);
        // Group events by key
        List<long[]>[] perKey = new List[numKeys];
        for (int k = 0; k < numKeys; k++) perKey[k] = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            int k = keys[i];
            if (k >= 0 && k < numKeys) {
                perKey[k].add(new long[]{timestamps[i], i});
            }
        }
        // Sort by timestamp, merge sessions
        for (int k = 0; k < numKeys; k++) {
            if (perKey[k].isEmpty()) continue;
            perKey[k].sort(Comparator.comparingLong(a -> a[0]));
            int sessions = 1;
            long lastTs = perKey[k].get(0)[0];
            for (int j = 1; j < perKey[k].size(); j++) {
                long ts = perKey[k].get(j)[0];
                if (ts - lastTs > gapMs) sessions++;
                lastTs = ts;
            }
            sessionCounts[k] = sessions;
        }
    }

    // ========== Utility: MurmurHash3 (32-bit, x86 variant) ==========

    static int murmurHash3(byte[] data, int offset, int len, int seed) {
        int h1 = seed;
        int nBlocks = len / 4;
        // body
        for (int i = 0; i < nBlocks; i++) {
            int k1 = (data[offset + i * 4] & 0xFF)
                   | ((data[offset + i * 4 + 1] & 0xFF) << 8)
                   | ((data[offset + i * 4 + 2] & 0xFF) << 16)
                   | ((data[offset + i * 4 + 3] & 0xFF) << 24);
            k1 *= 0xcc9e2d51; k1 = Integer.rotateLeft(k1, 15); k1 *= 0x1b873593;
            h1 ^= k1; h1 = Integer.rotateLeft(h1, 13);
            h1 = h1 * 5 + 0xe6546b64;
        }
        // tail
        int tail = offset + nBlocks * 4;
        int k1 = 0;
        switch (len & 3) {
            case 3: k1 ^= (data[tail + 2] & 0xFF) << 16;
            case 2: k1 ^= (data[tail + 1] & 0xFF) << 8;
            case 1: k1 ^= (data[tail] & 0xFF);
                    k1 *= 0xcc9e2d51; k1 = Integer.rotateLeft(k1, 15); k1 *= 0x1b873593;
                    h1 ^= k1;
        }
        // finalization
        h1 ^= len;
        h1 ^= h1 >>> 16; h1 *= 0x85ebca6b;
        h1 ^= h1 >>> 13; h1 *= 0xc2b2ae35;
        h1 ^= h1 >>> 16;
        return h1;
    }

    // ========== Data generators ==========

    static double[] generateFeatureMatrix(int rows, int cols, Random rng) {
        double[] data = new double[rows * cols];
        for (int i = 0; i < data.length; i++) data[i] = rng.nextGaussian() * 100.0;
        return data;
    }

    static double[] computeMeans(double[] data, int rows, int cols) {
        double[] means = new double[cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                means[j] += data[i * cols + j];
        for (int j = 0; j < cols; j++) means[j] /= rows;
        return means;
    }

    static double[] computeStds(double[] data, int rows, int cols, double[] means) {
        double[] stds = new double[cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++) {
                double d = data[i * cols + j] - means[j];
                stds[j] += d * d;
            }
        for (int j = 0; j < cols; j++) stds[j] = Math.sqrt(stds[j] / rows);
        return stds;
    }

    static double[] computeMins(double[] data, int rows, int cols) {
        double[] mins = new double[cols];
        Arrays.fill(mins, Double.MAX_VALUE);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                mins[j] = Math.min(mins[j], data[i * cols + j]);
        return mins;
    }

    static double[] computeMaxes(double[] data, int rows, int cols) {
        double[] maxes = new double[cols];
        Arrays.fill(maxes, -Double.MAX_VALUE);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                maxes[j] = Math.max(maxes[j], data[i * cols + j]);
        return maxes;
    }

    static Object[] generateTfidfData(int nDocs, int maxTermsPerDoc, int vocabSize, Random rng) {
        List<Integer> termIdList = new ArrayList<>();
        List<Double> tfList = new ArrayList<>();
        int[] docOffsets = new int[nDocs + 1];
        int offset = 0;
        for (int d = 0; d < nDocs; d++) {
            docOffsets[d] = offset;
            int nTerms = 5 + rng.nextInt(maxTermsPerDoc - 5);
            Set<Integer> terms = new TreeSet<>();
            while (terms.size() < nTerms) terms.add(rng.nextInt(vocabSize));
            for (int t : terms) {
                termIdList.add(t);
                tfList.add(1.0 + rng.nextInt(10));
                offset++;
            }
        }
        docOffsets[nDocs] = offset;
        int[] termIds = termIdList.stream().mapToInt(Integer::intValue).toArray();
        double[] tfValues = tfList.stream().mapToDouble(Double::doubleValue).toArray();
        double[] idfWeights = new double[vocabSize];
        for (int j = 0; j < vocabSize; j++) idfWeights[j] = Math.log(1.0 + nDocs / (1.0 + rng.nextInt(nDocs)));
        return new Object[]{docOffsets, termIds, tfValues, idfWeights};
    }

    static Object[] generateFeatureHashData(int n, int avgNameLen, Random rng) {
        // Pack all feature names into contiguous byte[]
        int totalBytes = n * avgNameLen;
        byte[] allNames = new byte[totalBytes];
        int[] offsets = new int[n];
        int[] lengths = new int[n];
        double[] values = new double[n];
        int pos = 0;
        for (int i = 0; i < n; i++) {
            offsets[i] = pos;
            int len = 4 + rng.nextInt(avgNameLen - 4);
            lengths[i] = len;
            for (int j = 0; j < len; j++) allNames[pos++] = (byte)('a' + rng.nextInt(26));
            values[i] = rng.nextDouble();
        }
        return new Object[]{allNames, offsets, lengths, values};
    }

    static double[] generateBucketSplits(int numSplits, Random rng) {
        double[] splits = new double[numSplits];
        for (int i = 0; i < numSplits; i++) splits[i] = rng.nextDouble() * 1000.0;
        Arrays.sort(splits);
        return splits;
    }

    static double[] generateBucketValues(int n, double maxVal, Random rng) {
        double[] values = new double[n];
        for (int i = 0; i < n; i++) values[i] = rng.nextDouble() * maxVal;
        return values;
    }

    static Object[] generateStreamData(int nEvents, int numKeys, long durationMs, Random rng) {
        long[] timestamps = new long[nEvents];
        int[] keys = new int[nEvents];
        double[] values = new double[nEvents];
        for (int i = 0; i < nEvents; i++) {
            timestamps[i] = (long)(rng.nextDouble() * durationMs);
            keys[i] = rng.nextInt(numKeys);
            values[i] = rng.nextDouble() * 100.0;
        }
        // Sort by timestamp (Flink processes in event-time order)
        Integer[] idx = new Integer[nEvents];
        for (int i = 0; i < nEvents; i++) idx[i] = i;
        Arrays.sort(idx, Comparator.comparingLong(a -> timestamps[a]));
        long[] ts2 = new long[nEvents]; int[] k2 = new int[nEvents]; double[] v2 = new double[nEvents];
        for (int i = 0; i < nEvents; i++) { ts2[i] = timestamps[idx[i]]; k2[i] = keys[idx[i]]; v2[i] = values[idx[i]]; }
        return new Object[]{ts2, k2, v2};
    }

    // ========== Benchmark harness ==========

    private static final int WARMUP_ITERS = 3;

    static long measure(Runnable r, int iterations) {
        for (int i = 0; i < WARMUP_ITERS; i++) r.run();
        long start = System.nanoTime();
        for (int i = 0; i < iterations; i++) r.run();
        return System.nanoTime() - start;
    }

    // ========== Main ==========

    public static void main(String[] args) throws Exception {
        boolean nativeLoaded = false;
        try {
            System.loadLibrary("arkprobe_appworkload");
            nativeLoaded = true;
            System.err.println("[INFO] Native library loaded.");
        } catch (UnsatisfiedLinkError e) {
            System.err.println("[INFO] Native library not found, Java-only mode.");
        }

        Random rng = new Random(42);

        // ---- Data sizes ----
        int mlRows = 1_000_000;
        int mlCols = 64;
        int tfidfDocs = 100_000;
        int tfidfMaxTerms = 30;
        int tfidfVocab = 50_000;
        int hashN = 1_000_000;
        int hashBuckets = 65536; // 2^16
        int bucketN = 1_000_000;
        int bucketSplits = 100;
        int streamEvents = 10_000_000;
        int streamKeys = 10_000;
        long streamDuration = 3600_000L; // 1 hour
        int windowMs = 1000;
        int topK = 10;
        long sessionGap = 30_000L;

        // ---- Generate data ----
        double[] mlData = generateFeatureMatrix(mlRows, mlCols, rng);
        double[] mlMeans = computeMeans(mlData, mlRows, mlCols);
        double[] mlStds = computeStds(mlData, mlRows, mlCols, mlMeans);
        double[] mlMins = computeMins(mlData, mlRows, mlCols);
        double[] mlMaxes = computeMaxes(mlData, mlRows, mlCols);

        Object[] tfidf = generateTfidfData(tfidfDocs, tfidfMaxTerms, tfidfVocab, rng);
        int[] tfDocOffsets = (int[]) tfidf[0];
        int[] tfTermIds = (int[]) tfidf[1];
        double[] tfValues = (double[]) tfidf[2];
        double[] tfIdfWeights = (double[]) tfidf[3];
        double[] tfOutput = new double[tfidfVocab];

        Object[] fhData = generateFeatureHashData(hashN, 12, rng);
        byte[] fhAllNames = (byte[]) fhData[0];
        int[] fhOffsets = (int[]) fhData[1];
        int[] fhLengths = (int[]) fhData[2];
        double[] fhValues = (double[]) fhData[3];
        double[] fhOutput = new double[hashBuckets];

        int numBucketSplits = 100;
        double[] bkSplits = generateBucketSplits(numBucketSplits, rng);
        double maxBucketVal = bkSplits[bkSplits.length - 1] + 1.0;
        double[] bkValues = generateBucketValues(bucketN, maxBucketVal, rng);
        int[] bkBuckets = new int[bucketN];

        Object[] streamData = generateStreamData(streamEvents, streamKeys, streamDuration, rng);
        long[] sTimestamps = (long[]) streamData[0];
        int[] sKeys = (int[]) streamData[1];
        double[] sValues = (double[]) streamData[2];
        int numWindows = (int)(streamDuration / windowMs) + 1;
        double[] aggResults = new double[streamKeys * numWindows];
        double[] topNOutput = new double[streamKeys * topK];
        int[] sessionCounts = new int[streamKeys];

        // ---- Iteration counts ----
        int mlIters = 10;
        int tfidfIters = 50;
        int hashIters = 20;
        int bucketIters = 50;
        int streamIters = 5;

        StringBuilder json = new StringBuilder();
        json.append("{\n");

        // ========== ZScore Normalize ==========
        double[] zData1 = mlData.clone();
        appendResult(json, "zscore_java",
            measure(() -> zscoreNormalizeJava(zData1, mlRows, mlCols, mlMeans, mlStds), mlIters));
        if (nativeLoaded) {
            double[] zData2 = mlData.clone();
            appendResult(json, "zscore_cpp",
                measure(() -> zscoreNormalize(zData2, mlRows, mlCols, mlMeans, mlStds), mlIters));
        }

        // ========== MinMax Scale ==========
        double[] mmData1 = mlData.clone();
        appendResult(json, "minmax_java",
            measure(() -> minMaxScaleJava(mmData1, mlRows, mlCols, mlMins, mlMaxes), mlIters));
        if (nativeLoaded) {
            double[] mmData2 = mlData.clone();
            appendResult(json, "minmax_cpp",
                measure(() -> minMaxScale(mmData2, mlRows, mlCols, mlMins, mlMaxes), mlIters));
        }

        // ========== TF-IDF Multiply ==========
        appendResult(json, "tfidf_java",
            measure(() -> tfidfMultiplyJava(tfDocOffsets, tfTermIds, tfValues, tfIdfWeights,
                                            tfOutput, tfidfDocs, tfidfVocab), tfidfIters));
        if (nativeLoaded) {
            appendResult(json, "tfidf_cpp",
                measure(() -> tfidfMultiply(tfDocOffsets, tfTermIds, tfValues, tfIdfWeights,
                                            tfOutput, tfidfDocs, tfidfVocab), tfidfIters));
        }

        // ========== Feature Hash ==========
        appendResult(json, "featurehash_java",
            measure(() -> featureHashJava(fhAllNames, fhOffsets, fhLengths, fhValues,
                                          fhOutput, hashBuckets, hashN), hashIters));
        if (nativeLoaded) {
            appendResult(json, "featurehash_cpp",
                measure(() -> featureHash(fhAllNames, fhOffsets, fhLengths, fhValues,
                                          fhOutput, hashBuckets, hashN), hashIters));
        }

        // ========== Bucketize ==========
        appendResult(json, "bucketize_java",
            measure(() -> bucketizeJava(bkValues, bkSplits, bkBuckets, bucketN, numBucketSplits), bucketIters));
        if (nativeLoaded) {
            appendResult(json, "bucketize_cpp",
                measure(() -> bucketize(bkValues, bkSplits, bkBuckets, bucketN, numBucketSplits), bucketIters));
        }

        // ========== Windowed Aggregation ==========
        appendResult(json, "windowed_agg_java",
            measure(() -> windowedAggregateJava(sTimestamps, sKeys, sValues, aggResults,
                                                streamEvents, windowMs, streamKeys), streamIters));
        if (nativeLoaded) {
            appendResult(json, "windowed_agg_cpp",
                measure(() -> windowedAggregate(sTimestamps, sKeys, sValues, aggResults,
                                                streamEvents, windowMs, streamKeys), streamIters));
        }

        // ========== Windowed Top-N ==========
        appendResult(json, "windowed_topn_java",
            measure(() -> windowedTopNJava(sTimestamps, sKeys, sValues, topNOutput,
                                           streamEvents, windowMs, streamKeys, topK), streamIters));
        if (nativeLoaded) {
            appendResult(json, "windowed_topn_cpp",
                measure(() -> windowedTopN(sTimestamps, sKeys, sValues, topNOutput,
                                           streamEvents, windowMs, streamKeys, topK), streamIters));
        }

        // ========== Session Window Merge ==========
        appendResult(json, "session_window_java",
            measure(() -> sessionWindowMergeJava(sTimestamps, sKeys, sValues, sessionCounts,
                                                 streamEvents, sessionGap, streamKeys), streamIters));
        if (nativeLoaded) {
            appendResult(json, "session_window_cpp",
                measure(() -> sessionWindowMerge(sTimestamps, sKeys, sValues, sessionCounts,
                                                 streamEvents, sessionGap, streamKeys), streamIters));
        }

        // Remove trailing comma
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
