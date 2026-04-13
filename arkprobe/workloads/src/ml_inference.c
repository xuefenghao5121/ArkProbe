/*
 * arkprobe_ml_inference — Machine Learning Inference micro-kernel workload.
 *
 * Models key characteristics of ML inference workloads:
 *   1. Matrix multiplication: Dense GEMM for fully connected layers
 *   2. Convolution: 2D convolution for CNN layers
 *   3. Activation functions: ReLU, sigmoid, tanh
 *   4. Pooling: Max/Average pooling
 *   5. Batch normalization: Scale and shift operations
 *
 * This workload produces:
 *   - High compute intensity (GEMM, convolution)
 *   - SIMD-friendly operations (vector operations)
 *   - Sequential memory access (weight matrices)
 *   - Low branch misprediction (predictable loops)
 *
 * Usage: arkprobe_ml_inference --threads N --duration S [--batch B]
 * Output: prints throughput (inferences/sec) and FLOPS.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <getopt.h>

#define INPUT_SIZE 784      /* 28x28 for MNIST-like input */
#define HIDDEN_SIZE 512
#define OUTPUT_SIZE 10
#define CONV_IN_SIZE 32
#define CONV_KERNEL 3
#define CONV_OUT_SIZE 30    /* 32 - 3 + 1 */
#define POOL_SIZE 2
#define MAX_BATCH 128

/* Worker statistics */
typedef struct {
    long inferences;
    long gemm_ops;
    long conv_ops;
    long activation_ops;
    long pooling_ops;
    double flops;
} worker_stats_t;

/* Global state */
static float *g_weights_ih = NULL;  /* Input to hidden weights */
static float *g_weights_ho = NULL;  /* Hidden to output weights */
static float *g_bias_h = NULL;
static float *g_bias_o = NULL;
static float *g_conv_weights = NULL;
static float *g_conv_bias = NULL;
static volatile int g_running = 1;

/* Activation functions */
static inline float relu(float x) {
    return x > 0 ? x : 0;
}

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static inline float tanh_fast(float x) {
    return tanhf(x);
}

/* Vectorized ReLU */
static void relu_vector(float *data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = relu(data[i]);
    }
}

/* Matrix-vector multiplication (GEMV) */
static void gemv(const float *weights, const float *input, float *output,
                 int out_size, int in_size, const float *bias) {
    for (int i = 0; i < out_size; i++) {
        float sum = bias ? bias[i] : 0.0f;
        for (int j = 0; j < in_size; j++) {
            sum += weights[i * in_size + j] * input[j];
        }
        output[i] = sum;
    }
}

/* Matrix-matrix multiplication (GEMM) for batch */
static void gemm(const float *weights, const float *input, float *output,
                 int batch_size, int out_size, int in_size, const float *bias) {
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < out_size; i++) {
            float sum = bias ? bias[i] : 0.0f;
            for (int j = 0; j < in_size; j++) {
                sum += weights[i * in_size + j] * input[b * in_size + j];
            }
            output[b * out_size + i] = sum;
        }
    }
}

/* 2D Convolution (simplified, single channel) */
static void conv2d(const float *input, const float *weights, float *output,
                   int in_size, int kernel_size, int out_size) {
    for (int oy = 0; oy < out_size; oy++) {
        for (int ox = 0; ox < out_size; ox++) {
            float sum = 0.0f;
            for (int ky = 0; ky < kernel_size; ky++) {
                for (int kx = 0; kx < kernel_size; kx++) {
                    int iy = oy + ky;
                    int ix = ox + kx;
                    sum += input[iy * in_size + ix] * weights[ky * kernel_size + kx];
                }
            }
            output[oy * out_size + ox] = sum;
        }
    }
}

/* 2D Convolution with multiple channels */
static void conv2d_multi(const float *input, const float *weights, float *output,
                         int in_size, int kernel_size, int out_size,
                         int in_channels, int out_channels) {
    for (int oc = 0; oc < out_channels; oc++) {
        for (int oy = 0; oy < out_size; oy++) {
            for (int ox = 0; ox < out_size; ox++) {
                float sum = 0.0f;
                for (int ic = 0; ic < in_channels; ic++) {
                    for (int ky = 0; ky < kernel_size; ky++) {
                        for (int kx = 0; kx < kernel_size; kx++) {
                            int iy = oy + ky;
                            int ix = ox + kx;
                            int i_idx = ic * in_size * in_size + iy * in_size + ix;
                            int w_idx = oc * in_channels * kernel_size * kernel_size +
                                       ic * kernel_size * kernel_size + ky * kernel_size + kx;
                            sum += input[i_idx] * weights[w_idx];
                        }
                    }
                }
                output[oc * out_size * out_size + oy * out_size + ox] = sum;
            }
        }
    }
}

/* Max pooling */
static void maxpool2d(const float *input, float *output,
                      int in_size, int pool_size, int out_size) {
    for (int oy = 0; oy < out_size; oy++) {
        for (int ox = 0; ox < out_size; ox++) {
            float max_val = -1e30f;
            for (int py = 0; py < pool_size; py++) {
                for (int px = 0; px < pool_size; px++) {
                    int iy = oy * pool_size + py;
                    int ix = ox * pool_size + px;
                    float val = input[iy * in_size + ix];
                    if (val > max_val) max_val = val;
                }
            }
            output[oy * out_size + ox] = max_val;
        }
    }
}

/* Average pooling */
static void avgpool2d(const float *input, float *output,
                      int in_size, int pool_size, int out_size) {
    float scale = 1.0f / (pool_size * pool_size);
    for (int oy = 0; oy < out_size; oy++) {
        for (int ox = 0; ox < out_size; ox++) {
            float sum = 0.0f;
            for (int py = 0; py < pool_size; py++) {
                for (int px = 0; px < pool_size; px++) {
                    int iy = oy * pool_size + py;
                    int ix = ox * pool_size + px;
                    sum += input[iy * in_size + ix];
                }
            }
            output[oy * out_size + ox] = sum * scale;
        }
    }
}

/* Batch normalization */
static void batch_norm(float *data, int size, float mean, float var, float gamma, float beta) {
    float inv_std = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < size; i++) {
        data[i] = gamma * (data[i] - mean) * inv_std + beta;
    }
}

/* Softmax */
static void softmax(float *data, int size) {
    float max_val = data[0];
    for (int i = 1; i < size; i++) {
        if (data[i] > max_val) max_val = data[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        data[i] = expf(data[i] - max_val);
        sum += data[i];
    }

    for (int i = 0; i < size; i++) {
        data[i] /= sum;
    }
}

/* Fully connected layer forward pass */
static void fc_layer(const float *input, float *output,
                     const float *weights, const float *bias,
                     int in_size, int out_size,
                     worker_stats_t *stats) {
    gemv(weights, input, output, out_size, in_size, bias);
    relu_vector(output, out_size);
    stats->gemm_ops++;
    stats->activation_ops++;
    stats->flops += 2.0 * in_size * out_size;  /* Multiply-add */
}

/* CNN block forward pass */
static void cnn_block(const float *input, float *output,
                      const float *weights, const float *bias,
                      int in_size, int kernel_size, int out_size,
                      worker_stats_t *stats) {
    conv2d(input, weights, output, in_size, kernel_size, out_size);
    relu_vector(output, out_size * out_size);
    stats->conv_ops++;
    stats->activation_ops++;
    stats->flops += 2.0 * kernel_size * kernel_size * out_size * out_size;
}

/* Run inference */
static void run_inference(worker_stats_t *stats, float *input) {
    float hidden[HIDDEN_SIZE];
    float output[OUTPUT_SIZE];

    /* Input -> Hidden layer */
    fc_layer(input, hidden, g_weights_ih, g_bias_h, INPUT_SIZE, HIDDEN_SIZE, stats);

    /* Hidden -> Output layer */
    gemv(g_weights_ho, hidden, output, OUTPUT_SIZE, HIDDEN_SIZE, g_bias_o);
    stats->gemm_ops++;
    stats->flops += 2.0 * HIDDEN_SIZE * OUTPUT_SIZE;

    /* Softmax for classification */
    softmax(output, OUTPUT_SIZE);
    stats->activation_ops++;

    stats->inferences++;
}

/* Run CNN inference */
static void run_cnn_inference(worker_stats_t *stats) {
    float input[CONV_IN_SIZE * CONV_IN_SIZE];
    float conv_out[CONV_OUT_SIZE * CONV_OUT_SIZE];
    float pool_out[(CONV_OUT_SIZE / POOL_SIZE) * (CONV_OUT_SIZE / POOL_SIZE)];

    /* Initialize random input */
    for (int i = 0; i < CONV_IN_SIZE * CONV_IN_SIZE; i++) {
        input[i] = (float)rand() / RAND_MAX;
    }

    /* Convolution layer */
    cnn_block(input, conv_out, g_conv_weights, g_conv_bias,
              CONV_IN_SIZE, CONV_KERNEL, CONV_OUT_SIZE, stats);

    /* Pooling layer */
    maxpool2d(conv_out, pool_out, CONV_OUT_SIZE, POOL_SIZE, CONV_OUT_SIZE / POOL_SIZE);
    stats->pooling_ops++;

    stats->inferences++;
}

static void *worker(void *arg) {
    worker_stats_t *stats = (worker_stats_t *)arg;
    float input[INPUT_SIZE];

    while (g_running) {
        /* Generate random input */
        for (int i = 0; i < INPUT_SIZE; i++) {
            input[i] = (float)rand() / RAND_MAX;
        }

        /* Run FC inference */
        run_inference(stats, input);

        /* Run CNN inference */
        run_cnn_inference(stats);
    }

    return NULL;
}

static void *timer_thread(void *arg) {
    int seconds = *(int *)arg;
    struct timespec ts = {seconds, 0};
    nanosleep(&ts, NULL);
    g_running = 0;
    return NULL;
}

/* Initialize weights with Xavier initialization */
static void init_weights(float *weights, int fan_in, int fan_out) {
    float scale = sqrtf(2.0f / (fan_in + fan_out));
    int size = fan_in * fan_out;
    for (int i = 0; i < size; i++) {
        weights[i] = scale * ((float)rand() / RAND_MAX * 2.0f - 1.0f);
    }
}

int main(int argc, char *argv[]) {
    int threads = 1;
    int duration = 60;

    static struct option long_opts[] = {
        {"threads",  required_argument, 0, 't'},
        {"duration", required_argument, 0, 'd'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "t:d:", long_opts, NULL)) != -1) {
        switch (opt) {
            case 't': threads = atoi(optarg); break;
            case 'd': duration = atoi(optarg); break;
        }
    }

    if (threads < 1) threads = 1;
    if (duration < 1) duration = 1;

    /* Initialize weights */
    g_weights_ih = malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    g_weights_ho = malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    g_bias_h = malloc(HIDDEN_SIZE * sizeof(float));
    g_bias_o = malloc(OUTPUT_SIZE * sizeof(float));
    g_conv_weights = malloc(CONV_KERNEL * CONV_KERNEL * sizeof(float));
    g_conv_bias = malloc(sizeof(float));

    init_weights(g_weights_ih, INPUT_SIZE, HIDDEN_SIZE);
    init_weights(g_weights_ho, HIDDEN_SIZE, OUTPUT_SIZE);
    for (int i = 0; i < HIDDEN_SIZE; i++) g_bias_h[i] = 0.01f;
    for (int i = 0; i < OUTPUT_SIZE; i++) g_bias_o[i] = 0.01f;
    init_weights(g_conv_weights, CONV_KERNEL * CONV_KERNEL, 1);
    *g_conv_bias = 0.0f;

    /* Start workers */
    worker_stats_t *stats = calloc(threads, sizeof(worker_stats_t));
    pthread_t *tids = malloc(sizeof(pthread_t) * (threads + 1));

    pthread_create(&tids[threads], NULL, timer_thread, &duration);

    for (int i = 0; i < threads; i++) {
        pthread_create(&tids[i], NULL, worker, &stats[i]);
    }

    for (int i = 0; i <= threads; i++) {
        pthread_join(tids[i], NULL);
    }

    /* Aggregate results */
    long total_inf = 0, total_gemm = 0, total_conv = 0;
    long total_act = 0, total_pool = 0;
    double total_flops = 0.0;

    for (int i = 0; i < threads; i++) {
        total_inf += stats[i].inferences;
        total_gemm += stats[i].gemm_ops;
        total_conv += stats[i].conv_ops;
        total_act += stats[i].activation_ops;
        total_pool += stats[i].pooling_ops;
        total_flops += stats[i].flops;
    }

    double inf_per_sec = (double)total_inf / duration;
    double gflops = total_flops / duration / 1e9;

    printf("Inferences: %.2f inf/sec\n", inf_per_sec);
    printf("GFLOPS: %.2f\n", gflops);
    printf("GEMM ops: %ld\n", total_gemm);
    printf("Conv ops: %ld\n", total_conv);
    printf("Activations: %ld\n", total_act);

    /* Cleanup */
    free(g_weights_ih);
    free(g_weights_ho);
    free(g_bias_h);
    free(g_bias_o);
    free(g_conv_weights);
    free(g_conv_bias);
    free(stats);
    free(tids);

    return 0;
}
