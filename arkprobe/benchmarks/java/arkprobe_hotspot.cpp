/**
 * C++ JNI implementation for HotspotBench native methods.
 * Uses GetPrimitiveArrayCritical where safe to avoid JNI array copy overhead.
 * Architecture-agnostic SIMD: ARM NEON / x86 AVX2 / scalar fallback.
 *
 * Rules for GetPrimitiveArrayCritical:
 * - No JNI calls between Get and Release (no allocations, no New*Array)
 * - Release before any JNI allocation
 * - Keep critical sections as short as possible
 */
#include <jni.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#ifdef __aarch64__
#include <arm_neon.h>
#elif __x86_64__
#include <immintrin.h>
#endif

extern "C" {

// ==================== vector_map ====================
JNIEXPORT jfloatArray JNICALL Java_HotspotBench_vectorMap
  (JNIEnv *env, jclass cls, jfloatArray data, jfloat factor) {
    jsize len = env->GetArrayLength(data);
    jfloatArray out = env->NewFloatArray(len);
    jfloat *in = (jfloat *)env->GetPrimitiveArrayCritical(data, nullptr);
    jfloat *dst = (jfloat *)env->GetPrimitiveArrayCritical(out, nullptr);

    jsize i = 0;
#ifdef __aarch64__
    float32x4_t fac = vdupq_n_f32(factor);
    for (; i + 4 <= len; i += 4) {
        vst1q_f32(dst + i, vmulq_f32(vld1q_f32(in + i), fac));
    }
#elif __x86_64__
    __m256 fac = _mm256_set1_ps(factor);
    for (; i + 8 <= len; i += 8) {
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_loadu_ps(in + i), fac));
    }
#endif
    for (; i < len; i++) dst[i] = in[i] * factor;

    env->ReleasePrimitiveArrayCritical(data, in, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(out, dst, 0);
    return out;
}

// ==================== vector_reduce ====================
JNIEXPORT jfloat JNICALL Java_HotspotBench_vectorReduce
  (JNIEnv *env, jclass cls, jfloatArray data) {
    jsize len = env->GetArrayLength(data);
    jfloat *in = (jfloat *)env->GetPrimitiveArrayCritical(data, nullptr);
    float sum = 0.0f;

    jsize i = 0;
#ifdef __aarch64__
    float32x4_t vsum = vdupq_n_f32(0.0f);
    for (; i + 4 <= len; i += 4)
        vsum = vaddq_f32(vsum, vld1q_f32(in + i));
    float tmp[4]; vst1q_f32(tmp, vsum);
    for (int j = 0; j < 4; j++) sum += tmp[j];
#elif __x86_64__
    __m256 vsum = _mm256_setzero_ps();
    for (; i + 8 <= len; i += 8)
        vsum = _mm256_add_ps(vsum, _mm256_loadu_ps(in + i));
    __m128 s = _mm_add_ps(_mm256_extractf128_ps(vsum, 0),
                           _mm256_extractf128_ps(vsum, 1));
    s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
    sum = _mm_cvtss_f32(s);
#endif
    for (; i < len; i++) sum += in[i];

    env->ReleasePrimitiveArrayCritical(data, in, JNI_ABORT);
    return sum;
}

// ==================== vector_filter ====================
JNIEXPORT jfloatArray JNICALL Java_HotspotBench_vectorFilter
  (JNIEnv *env, jclass cls, jfloatArray data, jfloat threshold) {
    jsize len = env->GetArrayLength(data);

    // Phase 1: count matches (critical section, no allocations)
    jfloat *in = (jfloat *)env->GetPrimitiveArrayCritical(data, nullptr);
    jsize count = 0;
    for (jsize i = 0; i < len; i++)
        if (in[i] > threshold) count++;
    env->ReleasePrimitiveArrayCritical(data, in, JNI_ABORT);

    // Phase 2: allocate output (safe, no critical sections held)
    jfloatArray out = env->NewFloatArray(count);

    // Phase 3: copy filtered elements (critical section, no allocations)
    in = (jfloat *)env->GetPrimitiveArrayCritical(data, nullptr);
    jfloat *dst = (jfloat *)env->GetPrimitiveArrayCritical(out, nullptr);
    jsize idx = 0;
    for (jsize i = 0; i < len; i++)
        if (in[i] > threshold) dst[idx++] = in[i];
    env->ReleasePrimitiveArrayCritical(data, in, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(out, dst, 0);
    return out;
}

// ==================== math_sigmoid ====================
JNIEXPORT jdoubleArray JNICALL Java_HotspotBench_mathSigmoid
  (JNIEnv *env, jclass cls, jdoubleArray data) {
    jsize len = env->GetArrayLength(data);
    jdoubleArray out = env->NewDoubleArray(len);
    jdouble *in = (jdouble *)env->GetPrimitiveArrayCritical(data, nullptr);
    jdouble *dst = (jdouble *)env->GetPrimitiveArrayCritical(out, nullptr);

    for (jsize i = 0; i < len; i++) {
        double v = in[i];
        dst[i] = 1.0 / (1.0 + exp(-v));
    }

    env->ReleasePrimitiveArrayCritical(data, in, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(out, dst, 0);
    return out;
}

// ==================== math_relu ====================
JNIEXPORT jdoubleArray JNICALL Java_HotspotBench_mathRelu
  (JNIEnv *env, jclass cls, jdoubleArray data) {
    jsize len = env->GetArrayLength(data);
    jdoubleArray out = env->NewDoubleArray(len);
    jdouble *in = (jdouble *)env->GetPrimitiveArrayCritical(data, nullptr);
    jdouble *dst = (jdouble *)env->GetPrimitiveArrayCritical(out, nullptr);

    jsize i = 0;
#ifdef __aarch64__
    float64x2_t zero = vdupq_n_f64(0.0);
    for (; i + 2 <= len; i += 2)
        vst1q_f64(dst + i, vmaxq_f64(vld1q_f64(in + i), zero));
#elif __x86_64__
    __m256d zero = _mm256_setzero_pd();
    for (; i + 4 <= len; i += 4)
        _mm256_storeu_pd(dst + i, _mm256_max_pd(_mm256_loadu_pd(in + i), zero));
#endif
    for (; i < len; i++) dst[i] = in[i] > 0.0 ? in[i] : 0.0;

    env->ReleasePrimitiveArrayCritical(data, in, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(out, dst, 0);
    return out;
}

// ==================== matmul ====================
JNIEXPORT jobjectArray JNICALL Java_HotspotBench_matmul
  (JNIEnv *env, jclass cls, jobjectArray aArr, jobjectArray bArr) {
    jsize n = env->GetArrayLength(aArr);
    double *a = (double *)malloc(n * n * sizeof(double));
    double *b = (double *)malloc(n * n * sizeof(double));
    double *c = (double *)calloc(n * n, sizeof(double));

    for (jsize i = 0; i < n; i++) {
        jdoubleArray rowA = (jdoubleArray)env->GetObjectArrayElement(aArr, i);
        jdoubleArray rowB = (jdoubleArray)env->GetObjectArrayElement(bArr, i);
        env->GetDoubleArrayRegion(rowA, 0, n, a + i * n);
        env->GetDoubleArrayRegion(rowB, 0, n, b + i * n);
        env->DeleteLocalRef(rowA);
        env->DeleteLocalRef(rowB);
    }

    // i-k-j for cache-friendly access
    for (jsize i = 0; i < n; i++)
        for (jsize k = 0; k < n; k++) {
            double aik = a[i * n + k];
            for (jsize j = 0; j < n; j++)
                c[i * n + j] += aik * b[k * n + j];
        }

    jclass doubleArrayClass = env->FindClass("[D");
    jobjectArray out = env->NewObjectArray(n, doubleArrayClass, nullptr);
    for (jsize i = 0; i < n; i++) {
        jdoubleArray row = env->NewDoubleArray(n);
        env->SetDoubleArrayRegion(row, 0, n, c + i * n);
        env->SetObjectArrayElement(out, i, row);
        env->DeleteLocalRef(row);
    }

    free(a); free(b); free(c);
    return out;
}

// ==================== array_copy ====================
JNIEXPORT jfloatArray JNICALL Java_HotspotBench_arrayCopy
  (JNIEnv *env, jclass cls, jfloatArray src) {
    jsize len = env->GetArrayLength(src);
    jfloatArray out = env->NewFloatArray(len);
    jfloat *s = (jfloat *)env->GetPrimitiveArrayCritical(src, nullptr);
    jfloat *d = (jfloat *)env->GetPrimitiveArrayCritical(out, nullptr);

    memcpy(d, s, len * sizeof(float));

    env->ReleasePrimitiveArrayCritical(src, s, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(out, d, 0);
    return out;
}

// ==================== array_scale ====================
JNIEXPORT jfloatArray JNICALL Java_HotspotBench_arrayScale
  (JNIEnv *env, jclass cls, jfloatArray src, jfloat factor) {
    jsize len = env->GetArrayLength(src);
    jfloatArray out = env->NewFloatArray(len);
    jfloat *in = (jfloat *)env->GetPrimitiveArrayCritical(src, nullptr);
    jfloat *dst = (jfloat *)env->GetPrimitiveArrayCritical(out, nullptr);

    jsize i = 0;
#ifdef __aarch64__
    float32x4_t fac = vdupq_n_f32(factor);
    for (; i + 4 <= len; i += 4)
        vst1q_f32(dst + i, vmulq_f32(vld1q_f32(in + i), fac));
#elif __x86_64__
    __m256 fac = _mm256_set1_ps(factor);
    for (; i + 8 <= len; i += 8)
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_loadu_ps(in + i), fac));
#endif
    for (; i < len; i++) dst[i] = in[i] * factor;

    env->ReleasePrimitiveArrayCritical(src, in, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(out, dst, 0);
    return out;
}

// ==================== prefetch ====================
JNIEXPORT jfloat JNICALL Java_HotspotBench_prefetch
  (JNIEnv *env, jclass cls, jfloatArray data, jint stride) {
    jsize len = env->GetArrayLength(data);
    jfloat *in = (jfloat *)env->GetPrimitiveArrayCritical(data, nullptr);
    float sum = 0.0f;

    for (jsize i = 0; i < len; i += stride) {
#ifdef __aarch64__
        __builtin_prefetch(in + i + stride, 0, 1);
#elif __x86_64__
        _mm_prefetch((const char *)(in + i + stride), _MM_HINT_T0);
#endif
        sum += in[i];
    }

    env->ReleasePrimitiveArrayCritical(data, in, JNI_ABORT);
    return sum;
}

// ==================== string_parse ====================
JNIEXPORT jint JNICALL Java_HotspotBench_stringParse
  (JNIEnv *env, jclass cls, jobjectArray data) {
    jsize len = env->GetArrayLength(data);
    jint count = 0;
    for (jsize i = 0; i < len; i++) {
        jstring s = (jstring)env->GetObjectArrayElement(data, i);
        const char *str = env->GetStringUTFChars(s, nullptr);
        for (const char *p = str; *p; p++) {
            unsigned char c = (unsigned char)*p;
            if (c >= '0' && c <= '9') count++;
        }
        env->ReleaseStringUTFChars(s, str);
        env->DeleteLocalRef(s);
    }
    return count;
}

// ==================== string_search ====================
JNIEXPORT jint JNICALL Java_HotspotBench_stringSearch
  (JNIEnv *env, jclass cls, jobjectArray data, jstring patternStr) {
    jsize len = env->GetArrayLength(data);
    const char *pattern = env->GetStringUTFChars(patternStr, nullptr);
    jint count = 0;
    for (jsize i = 0; i < len; i++) {
        jstring s = (jstring)env->GetObjectArrayElement(data, i);
        const char *str = env->GetStringUTFChars(s, nullptr);
        const char *p = str;
        while ((p = strstr(p, pattern)) != nullptr) { count++; p++; }
        env->ReleaseStringUTFChars(s, str);
        env->DeleteLocalRef(s);
    }
    env->ReleaseStringUTFChars(patternStr, pattern);
    return count;
}

} // extern "C"
