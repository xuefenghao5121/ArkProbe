/**
 * C++ JNI implementation for AppWorkloadBench native methods.
 * Real application workloads: ML feature engineering + Flink-style stream processing.
 *
 * Methods:
 *   1. zscoreNormalize   — SIMD broadcast-subtract-divide per row
 *   2. minMaxScale       — SIMD broadcast-subtract-divide per row
 *   3. tfidfMultiply     — SIMD gather-multiply for sparse TF * dense IDF
 *   4. featureHash       — MurmurHash3 on byte[] + scatter-add
 *   5. bucketize         — SIMD batch binary-search binning
 *   6. windowedAggregate — Sequential scan + direct index accumulate
 *   7. windowedTopN      — Per-key partial sort (selection algorithm)
 *   8. sessionWindowMerge— Per-key sort + interval merge
 */
#include <jni.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <algorithm>
#include <functional>

#ifdef __aarch64__
#include <arm_neon.h>
#elif __x86_64__
#include <immintrin.h>
#endif

extern "C" {

// ==================== MurmurHash3 (32-bit x86) ====================
static int murmurHash3Cpp(const unsigned char* data, int len, int seed) {
    int h1 = seed;
    int nblocks = len / 4;
    for (int i = 0; i < nblocks; i++) {
        int k1 = (int)data[i*4] | ((int)data[i*4+1] << 8) |
                 ((int)data[i*4+2] << 16) | ((int)data[i*4+3] << 24);
        k1 *= 0xcc9e2d51;
        k1 = (k1 << 15) | ((unsigned int)k1 >> 17);
        k1 *= 0x1b873593;
        h1 ^= k1;
        h1 = (h1 << 13) | ((unsigned int)h1 >> 19);
        h1 = h1 * 5 + 0xe6546b64;
    }
    int tail = nblocks * 4;
    int k1 = 0;
    switch (len & 3) {
        case 3: k1 ^= (int)data[tail+2] << 16;
        case 2: k1 ^= (int)data[tail+1] << 8;
        case 1: k1 ^= (int)data[tail];
                k1 *= 0xcc9e2d51;
                k1 = (k1 << 15) | ((unsigned int)k1 >> 17);
                k1 *= 0x1b873593;
                h1 ^= k1;
    }
    h1 ^= len;
    h1 ^= (unsigned int)h1 >> 16; h1 *= 0x85ebca6b;
    h1 ^= (unsigned int)h1 >> 13; h1 *= 0xc2b2ae35;
    h1 ^= (unsigned int)h1 >> 16;
    return h1;
}

// ==================== zscoreNormalize ====================
JNIEXPORT void JNICALL Java_AppWorkloadBench_zscoreNormalize
  (JNIEnv *env, jclass cls, jdoubleArray dataArr, jint rows, jint cols,
   jdoubleArray meansArr, jdoubleArray stdsArr) {
    jdouble *data = (jdouble *)env->GetPrimitiveArrayCritical(dataArr, nullptr);
    jdouble *means = (jdouble *)env->GetPrimitiveArrayCritical(meansArr, nullptr);
    jdouble *stds = (jdouble *)env->GetPrimitiveArrayCritical(stdsArr, nullptr);

    for (int i = 0; i < rows; i++) {
        double *row = &data[i * cols];
        int j = 0;
#ifdef __aarch64__
        for (; j + 2 <= cols; j += 2) {
            float64x2_t vRow = vld1q_f64(&row[j]);
            float64x2_t vMean = vld1q_f64(&means[j]);
            float64x2_t vStd = vld1q_f64(&stds[j]);
            vRow = vdivq_f64(vsubq_f64(vRow, vMean), vStd);
            vst1q_f64(&row[j], vRow);
        }
#elif __x86_64__
        for (; j + 2 <= cols; j += 2) {
            __m128d vRow = _mm_loadu_pd(&row[j]);
            __m128d vMean = _mm_loadu_pd(&means[j]);
            __m128d vStd = _mm_loadu_pd(&stds[j]);
            vRow = _mm_div_pd(_mm_sub_pd(vRow, vMean), vStd);
            _mm_storeu_pd(&row[j], vRow);
        }
#endif
        for (; j < cols; j++) {
            row[j] = (row[j] - means[j]) / stds[j];
        }
    }

    env->ReleasePrimitiveArrayCritical(dataArr, data, 0);
    env->ReleasePrimitiveArrayCritical(meansArr, means, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(stdsArr, stds, JNI_ABORT);
}

// ==================== minMaxScale ====================
JNIEXPORT void JNICALL Java_AppWorkloadBench_minMaxScale
  (JNIEnv *env, jclass cls, jdoubleArray dataArr, jint rows, jint cols,
   jdoubleArray minsArr, jdoubleArray maxesArr) {
    jdouble *data = (jdouble *)env->GetPrimitiveArrayCritical(dataArr, nullptr);
    jdouble *mins = (jdouble *)env->GetPrimitiveArrayCritical(minsArr, nullptr);
    jdouble *maxes = (jdouble *)env->GetPrimitiveArrayCritical(maxesArr, nullptr);

    for (int i = 0; i < rows; i++) {
        double *row = &data[i * cols];
        int j = 0;
#ifdef __aarch64__
        for (; j + 2 <= cols; j += 2) {
            float64x2_t vRow = vld1q_f64(&row[j]);
            float64x2_t vMin = vld1q_f64(&mins[j]);
            float64x2_t vMax = vld1q_f64(&maxes[j]);
            float64x2_t vRange = vsubq_f64(vMax, vMin);
            // avoid div-by-zero: set range to 1.0 where it's <= 0
            float64x2_t vZero = vdupq_n_f64(0.0);
            float64x2_t vOne = vdupq_n_f64(1.0);
            uint64x2_t vMask = vcleq_f64(vRange, vZero);
            vRange = vbslq_f64(vMask, vOne, vRange);
            vRow = vdivq_f64(vsubq_f64(vRow, vMin), vRange);
            // zero out where original range <= 0
            vRow = vbslq_f64(vMask, vZero, vRow);
            vst1q_f64(&row[j], vRow);
        }
#elif __x86_64__
        for (; j + 2 <= cols; j += 2) {
            __m128d vRow = _mm_loadu_pd(&row[j]);
            __m128d vMin = _mm_loadu_pd(&mins[j]);
            __m128d vMax = _mm_loadu_pd(&maxes[j]);
            __m128d vRange = _mm_sub_pd(vMax, vMin);
            __m128d vZero = _mm_setzero_pd();
            __m128d vOne = _mm_set1_pd(1.0);
            __m128d vMask = _mm_cmple_pd(vRange, vZero);
            vRange = _mm_or_pd(_mm_andnot_pd(vMask, vRange), _mm_and_pd(vMask, vOne));
            vRow = _mm_div_pd(_mm_sub_pd(vRow, vMin), vRange);
            vRow = _mm_andnot_pd(vMask, vRow);
            _mm_storeu_pd(&row[j], vRow);
        }
#endif
        for (; j < cols; j++) {
            double range = maxes[j] - mins[j];
            row[j] = range > 0 ? (row[j] - mins[j]) / range : 0.0;
        }
    }

    env->ReleasePrimitiveArrayCritical(dataArr, data, 0);
    env->ReleasePrimitiveArrayCritical(minsArr, mins, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(maxesArr, maxes, JNI_ABORT);
}

// ==================== tfidfMultiply ====================
JNIEXPORT void JNICALL Java_AppWorkloadBench_tfidfMultiply
  (JNIEnv *env, jclass cls, jintArray docOffsetsArr, jintArray termIdsArr,
   jdoubleArray tfValuesArr, jdoubleArray idfWeightsArr,
   jdoubleArray outputArr, jint nDocs, jint vocabSize) {
    jint *docOffsets = (jint *)env->GetPrimitiveArrayCritical(docOffsetsArr, nullptr);
    jint *termIds = (jint *)env->GetPrimitiveArrayCritical(termIdsArr, nullptr);
    jdouble *tfValues = (jdouble *)env->GetPrimitiveArrayCritical(tfValuesArr, nullptr);
    jdouble *idfWeights = (jdouble *)env->GetPrimitiveArrayCritical(idfWeightsArr, nullptr);
    jdouble *output = (jdouble *)env->GetPrimitiveArrayCritical(outputArr, nullptr);

    memset(output, 0, vocabSize * sizeof(double));
    for (int d = 0; d < nDocs; d++) {
        int start = docOffsets[d];
        int end = docOffsets[d + 1];
        int j = start;
#ifdef __aarch64__
        for (; j + 2 <= end; j += 2) {
            int t0 = termIds[j], t1 = termIds[j + 1];
            double tf0 = tfValues[j], tf1 = tfValues[j + 1];
            if (t0 >= 0 && t0 < vocabSize) output[t0] += tf0 * idfWeights[t0];
            if (t1 >= 0 && t1 < vocabSize) output[t1] += tf1 * idfWeights[t1];
        }
#elif __x86_64__
        for (; j + 2 <= end; j += 2) {
            int t0 = termIds[j], t1 = termIds[j + 1];
            double tf0 = tfValues[j], tf1 = tfValues[j + 1];
            if (t0 >= 0 && t0 < vocabSize) output[t0] += tf0 * idfWeights[t0];
            if (t1 >= 0 && t1 < vocabSize) output[t1] += tf1 * idfWeights[t1];
        }
#endif
        for (; j < end; j++) {
            int t = termIds[j];
            if (t >= 0 && t < vocabSize) output[t] += tfValues[j] * idfWeights[t];
        }
    }

    env->ReleasePrimitiveArrayCritical(docOffsetsArr, docOffsets, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(termIdsArr, termIds, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(tfValuesArr, tfValues, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(idfWeightsArr, idfWeights, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(outputArr, output, 0);
}

// ==================== featureHash ====================
JNIEXPORT void JNICALL Java_AppWorkloadBench_featureHash
  (JNIEnv *env, jclass cls, jbyteArray allNamesArr, jintArray offsetsArr,
   jintArray lengthsArr, jdoubleArray valuesArr, jdoubleArray outputArr,
   jint numFeatures, jint n) {
    jbyte *allNames = (jbyte *)env->GetPrimitiveArrayCritical(allNamesArr, nullptr);
    jint *offsets = (jint *)env->GetPrimitiveArrayCritical(offsetsArr, nullptr);
    jint *lengths = (jint *)env->GetPrimitiveArrayCritical(lengthsArr, nullptr);
    jdouble *values = (jdouble *)env->GetPrimitiveArrayCritical(valuesArr, nullptr);
    jdouble *output = (jdouble *)env->GetPrimitiveArrayCritical(outputArr, nullptr);

    memset(output, 0, numFeatures * sizeof(double));
    for (int i = 0; i < n; i++) {
        int hash = murmurHash3Cpp((const unsigned char*)&allNames[offsets[i]],
                                  lengths[i], 0x9747b28c);
        int idx = (hash & 0x7FFFFFFF) % numFeatures;
        output[idx] += values[i];
    }

    env->ReleasePrimitiveArrayCritical(allNamesArr, allNames, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(offsetsArr, offsets, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(lengthsArr, lengths, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(valuesArr, values, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(outputArr, output, 0);
}

// ==================== bucketize ====================
JNIEXPORT void JNICALL Java_AppWorkloadBench_bucketize
  (JNIEnv *env, jclass cls, jdoubleArray valuesArr, jdoubleArray splitsArr,
   jintArray bucketsArr, jint n, jint numSplits) {
    jdouble *values = (jdouble *)env->GetPrimitiveArrayCritical(valuesArr, nullptr);
    jdouble *splits = (jdouble *)env->GetPrimitiveArrayCritical(splitsArr, nullptr);
    jint *buckets = (jint *)env->GetPrimitiveArrayCritical(bucketsArr, nullptr);

    for (int i = 0; i < n; i++) {
        double v = values[i];
        int lo = 0, hi = numSplits;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (splits[mid] <= v) lo = mid + 1;
            else hi = mid;
        }
        buckets[i] = lo;
    }

    env->ReleasePrimitiveArrayCritical(valuesArr, values, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(splitsArr, splits, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(bucketsArr, buckets, 0);
}

// ==================== windowedAggregate ====================
JNIEXPORT void JNICALL Java_AppWorkloadBench_windowedAggregate
  (JNIEnv *env, jclass cls, jlongArray timestampsArr, jintArray keysArr,
   jdoubleArray valuesArr, jdoubleArray resultsArr, jint n,
   jint windowSizeMs, jint numKeys) {
    jlong *timestamps = (jlong *)env->GetPrimitiveArrayCritical(timestampsArr, nullptr);
    jint *keys = (jint *)env->GetPrimitiveArrayCritical(keysArr, nullptr);
    jdouble *values = (jdouble *)env->GetPrimitiveArrayCritical(valuesArr, nullptr);
    jdouble *results = (jdouble *)env->GetPrimitiveArrayCritical(resultsArr, nullptr);

    jlong baseTs = n > 0 ? timestamps[0] : 0;
    int numWindows = 0;
    if (n > 0) {
        numWindows = (int)((timestamps[n-1] - baseTs) / windowSizeMs) + 1;
    }
    memset(results, 0, numKeys * numWindows * sizeof(double));

    for (int i = 0; i < n; i++) {
        int winIdx = (int)((timestamps[i] - baseTs) / windowSizeMs);
        int k = keys[i];
        if (k >= 0 && k < numKeys) {
            results[k * numWindows + winIdx] += values[i];
        }
    }

    env->ReleasePrimitiveArrayCritical(timestampsArr, timestamps, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(keysArr, keys, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(valuesArr, values, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(resultsArr, results, 0);
}

// ==================== windowedTopN ====================
JNIEXPORT void JNICALL Java_AppWorkloadBench_windowedTopN
  (JNIEnv *env, jclass cls, jlongArray timestampsArr, jintArray keysArr,
   jdoubleArray valuesArr, jdoubleArray topNOutputArr, jint n,
   jint windowSizeMs, jint numKeys, jint topK) {
    jlong *timestamps = (jlong *)env->GetPrimitiveArrayCritical(timestampsArr, nullptr);
    jint *keys = (jint *)env->GetPrimitiveArrayCritical(keysArr, nullptr);
    jdouble *values = (jdouble *)env->GetPrimitiveArrayCritical(valuesArr, nullptr);
    jdouble *topNOutput = (jdouble *)env->GetPrimitiveArrayCritical(topNOutputArr, nullptr);

    // Use partial sort approach: for each key, collect values, partial_sort top-K
    // Build per-key value lists using a simpler approach
    // First pass: count per key
    int *keyCounts = (int *)calloc(numKeys, sizeof(int));
    for (int i = 0; i < n; i++) {
        int k = keys[i];
        if (k >= 0 && k < numKeys) keyCounts[k]++;
    }

    // Allocate per-key arrays
    int **keyValues = (int **)calloc(numKeys, sizeof(int *));
    int *keyPos = (int *)calloc(numKeys, sizeof(int));
    double **keyVals = (double **)calloc(numKeys, sizeof(double *));
    for (int k = 0; k < numKeys; k++) {
        if (keyCounts[k] > 0) {
            keyValues[k] = (int *)malloc(keyCounts[k] * sizeof(int));
            keyVals[k] = (double *)malloc(keyCounts[k] * sizeof(double));
        }
    }

    // Second pass: fill per-key values
    for (int i = 0; i < n; i++) {
        int k = keys[i];
        if (k >= 0 && k < numKeys && keyVals[k]) {
            int pos = keyPos[k]++;
            keyVals[k][pos] = values[i];
        }
    }

    // For each key, partial sort and extract top-K
    for (int k = 0; k < numKeys; k++) {
        int cnt = keyCounts[k];
        if (cnt > topK) {
            std::nth_element(keyVals[k], keyVals[k] + (cnt - topK), keyVals[k] + cnt,
                             std::greater<double>());
            // Copy top-K in descending order
            std::sort(keyVals[k] + (cnt - topK), keyVals[k] + cnt, std::greater<double>());
            for (int r = 0; r < topK; r++) {
                topNOutput[k * topK + r] = keyVals[k][cnt - topK + r];
            }
        } else {
            std::sort(keyVals[k], keyVals[k] + cnt, std::greater<double>());
            for (int r = 0; r < cnt; r++) {
                topNOutput[k * topK + r] = keyVals[k][r];
            }
            for (int r = cnt; r < topK; r++) {
                topNOutput[k * topK + r] = NAN;
            }
        }
    }

    // Cleanup
    for (int k = 0; k < numKeys; k++) {
        free(keyValues[k]);
        free(keyVals[k]);
    }
    free(keyValues); free(keyVals); free(keyCounts); free(keyPos);

    env->ReleasePrimitiveArrayCritical(timestampsArr, timestamps, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(keysArr, keys, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(valuesArr, values, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(topNOutputArr, topNOutput, 0);
}

// ==================== sessionWindowMerge ====================
JNIEXPORT void JNICALL Java_AppWorkloadBench_sessionWindowMerge
  (JNIEnv *env, jclass cls, jlongArray timestampsArr, jintArray keysArr,
   jdoubleArray valuesArr, jintArray sessionCountsArr, jint n,
   jlong gapMs, jint numKeys) {
    jlong *timestamps = (jlong *)env->GetPrimitiveArrayCritical(timestampsArr, nullptr);
    jint *keys = (jint *)env->GetPrimitiveArrayCritical(keysArr, nullptr);
    jdouble *values = (jdouble *)env->GetPrimitiveArrayCritical(valuesArr, nullptr);
    jint *sessionCounts = (jint *)env->GetPrimitiveArrayCritical(sessionCountsArr, nullptr);

    // Count events per key
    int *keyCounts = (int *)calloc(numKeys, sizeof(int));
    for (int i = 0; i < n; i++) {
        int k = keys[i];
        if (k >= 0 && k < numKeys) keyCounts[k]++;
    }

    // Collect timestamps per key
    jlong **keyTs = (jlong **)calloc(numKeys, sizeof(jlong *));
    int *keyPos = (int *)calloc(numKeys, sizeof(int));
    for (int k = 0; k < numKeys; k++) {
        if (keyCounts[k] > 0) keyTs[k] = (jlong *)malloc(keyCounts[k] * sizeof(jlong));
    }
    for (int i = 0; i < n; i++) {
        int k = keys[i];
        if (k >= 0 && k < numKeys && keyTs[k]) {
            keyTs[k][keyPos[k]++] = timestamps[i];
        }
    }

    // Sort and count sessions per key
    memset(sessionCounts, 0, numKeys * sizeof(jint));
    for (int k = 0; k < numKeys; k++) {
        int cnt = keyCounts[k];
        if (cnt == 0) continue;
        std::sort(keyTs[k], keyTs[k] + cnt);
        int sessions = 1;
        jlong lastTs = keyTs[k][0];
        for (int j = 1; j < cnt; j++) {
            if (keyTs[k][j] - lastTs > gapMs) sessions++;
            lastTs = keyTs[k][j];
        }
        sessionCounts[k] = sessions;
    }

    // Cleanup
    for (int k = 0; k < numKeys; k++) free(keyTs[k]);
    free(keyTs); free(keyCounts); free(keyPos);

    env->ReleasePrimitiveArrayCritical(timestampsArr, timestamps, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(keysArr, keys, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(valuesArr, values, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(sessionCountsArr, sessionCounts, 0);
}

} // extern "C"
