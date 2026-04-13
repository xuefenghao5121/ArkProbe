/*
 * arkprobe_videoenc — Video Encoding micro-kernel workload.
 *
 * Models key characteristics of video encoding workloads (H.264/H.265 style):
 *   1. DCT transform: 8x8 block DCT for intra prediction residual
 *   2. Motion estimation: Integer pixel search with SAD computation
 *   3. Intra prediction: Mode selection from neighboring pixels
 *   4. Quantization: Division-based coefficient quantization
 *
 * This workload produces:
 *   - High compute intensity (DCT, SAD)
 *   - SIMD-friendly operations (block processing)
 *   - Mixed memory access (sequential + random for motion search)
 *   - Branch-heavy code paths (mode decisions)
 *
 * Usage: arkprobe_videoenc --threads N --duration S [--frames F]
 * Output: prints throughput (frames/sec) and encoding statistics.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <getopt.h>

#define BLOCK_SIZE 8
#define FRAME_WIDTH 1920
#define FRAME_HEIGHT 1080
#define SEARCH_RANGE 32
#define NUM_INTRA_MODES 9

/* DCT basis functions (precomputed) */
static const float DCT_BASIS[8][8] = {
    {0.353553f, 0.353553f, 0.353553f, 0.353553f, 0.353553f, 0.353553f, 0.353553f, 0.353553f},
    {0.490393f, 0.415735f, 0.277785f, 0.097545f, -0.097545f, -0.277785f, -0.415735f, -0.490393f},
    {0.461940f, 0.191342f, -0.191342f, -0.461940f, -0.461940f, -0.191342f, 0.191342f, 0.461940f},
    {0.415735f, -0.097545f, -0.490393f, -0.277785f, 0.277785f, 0.490393f, 0.097545f, -0.415735f},
    {0.353553f, -0.353553f, -0.353553f, 0.353553f, 0.353553f, -0.353553f, -0.353553f, 0.353553f},
    {0.277785f, -0.490393f, 0.097545f, 0.415735f, -0.415735f, -0.097545f, 0.490393f, -0.277785f},
    {0.191342f, -0.461940f, 0.461940f, -0.191342f, -0.191342f, 0.461940f, -0.461940f, 0.191342f},
    {0.097545f, -0.277785f, 0.415735f, -0.490393f, 0.490393f, -0.415735f, 0.277785f, -0.097545f}
};

/* Quantization matrix (example) */
static const int QMATRIX[8][8] = {
    {16, 11, 10, 16, 24, 40, 51, 61},
    {12, 12, 14, 19, 26, 58, 60, 55},
    {14, 13, 16, 24, 40, 57, 69, 56},
    {14, 17, 22, 29, 51, 87, 80, 62},
    {18, 22, 37, 56, 68, 109, 103, 77},
    {24, 35, 55, 64, 81, 104, 113, 92},
    {49, 64, 78, 87, 103, 121, 120, 101},
    {72, 92, 95, 98, 112, 100, 103, 99}
};

/* Worker statistics */
typedef struct {
    long frames;
    long blocks_processed;
    long dct_ops;
    long sad_ops;
    long intra_modes;
    long quantized_coeffs;
} worker_stats_t;

/* Global state */
static uint8_t *g_frame_curr = NULL;
static uint8_t *g_frame_ref = NULL;
static volatile int g_running = 1;

/* Compute 8x8 DCT */
static void dct_8x8(const uint8_t *block, int stride, float output[8][8]) {
    float temp[8][8];

    /* Row-wise DCT */
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 8; k++) {
                sum += block[i * stride + k] * DCT_BASIS[j][k];
            }
            temp[i][j] = sum;
        }
    }

    /* Column-wise DCT */
    for (int j = 0; j < 8; j++) {
        for (int i = 0; i < 8; i++) {
            float sum = 0.0f;
            for (int k = 0; k < 8; k++) {
                sum += temp[k][j] * DCT_BASIS[i][k];
            }
            output[i][j] = sum;
        }
    }
}

/* Compute inverse DCT */
static void idct_8x8(const float input[8][8], uint8_t *block, int stride) {
    float temp[8][8];

    /* Row-wise IDCT */
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 8; k++) {
                sum += input[i][k] * DCT_BASIS[k][j];
            }
            temp[i][j] = sum;
        }
    }

    /* Column-wise IDCT */
    for (int j = 0; j < 8; j++) {
        for (int i = 0; i < 8; i++) {
            float sum = 0.0f;
            for (int k = 0; k < 8; k++) {
                sum += temp[k][j] * DCT_BASIS[k][i];
            }
            int val = (int)(sum + 128.5f);
            block[i * stride + j] = val < 0 ? 0 : (val > 255 ? 255 : val);
        }
    }
}

/* Quantize DCT coefficients */
static void quantize(float dct[8][8], int qscale, int output[8][8]) {
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            int q = QMATRIX[i][j] * qscale / 16;
            output[i][j] = (int)roundf(dct[i][j] / q);
        }
    }
}

/* Dequantize coefficients */
static void dequantize(const int quant[8][8], int qscale, float output[8][8]) {
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            int q = QMATRIX[i][j] * qscale / 16;
            output[i][j] = quant[i][j] * q;
        }
    }
}

/* Compute SAD (Sum of Absolute Differences) */
static uint32_t compute_sad(const uint8_t *ref, const uint8_t *cur,
                           int ref_stride, int cur_stride) {
    uint32_t sad = 0;
    for (int i = 0; i < BLOCK_SIZE; i++) {
        for (int j = 0; j < BLOCK_SIZE; j++) {
            int diff = ref[i * ref_stride + j] - cur[i * cur_stride + j];
            sad += diff < 0 ? -diff : diff;
        }
    }
    return sad;
}

/* Motion estimation (integer pixel search) */
static int motion_estimate(const uint8_t *ref_frame, const uint8_t *cur_frame,
                           int mb_x, int mb_y, int *best_mv_x, int *best_mv_y) {
    int cur_x = mb_x * BLOCK_SIZE;
    int cur_y = mb_y * BLOCK_SIZE;

    uint32_t best_sad = UINT32_MAX;
    *best_mv_x = 0;
    *best_mv_y = 0;

    /* Search within range */
    for (int dy = -SEARCH_RANGE; dy <= SEARCH_RANGE; dy += 2) {
        for (int dx = -SEARCH_RANGE; dx <= SEARCH_RANGE; dx += 2) {
            int ref_x = cur_x + dx;
            int ref_y = cur_y + dy;

            /* Boundary check */
            if (ref_x < 0 || ref_x + BLOCK_SIZE > FRAME_WIDTH) continue;
            if (ref_y < 0 || ref_y + BLOCK_SIZE > FRAME_HEIGHT) continue;

            uint32_t sad = compute_sad(
                ref_frame + ref_y * FRAME_WIDTH + ref_x,
                cur_frame + cur_y * FRAME_WIDTH + cur_x,
                FRAME_WIDTH, FRAME_WIDTH
            );

            if (sad < best_sad) {
                best_sad = sad;
                *best_mv_x = dx;
                *best_mv_y = dy;
            }
        }
    }

    return best_sad;
}

/* Intra prediction modes */
static void intra_predict_dc(const uint8_t *above, const uint8_t *left,
                             uint8_t *pred, int stride) {
    int dc = 0;
    for (int i = 0; i < BLOCK_SIZE; i++) {
        dc += above[i] + left[i];
    }
    dc = dc / (2 * BLOCK_SIZE);

    for (int i = 0; i < BLOCK_SIZE; i++) {
        for (int j = 0; j < BLOCK_SIZE; j++) {
            pred[i * stride + j] = dc;
        }
    }
}

static void intra_predict_h(const uint8_t *left, uint8_t *pred, int stride) {
    for (int i = 0; i < BLOCK_SIZE; i++) {
        for (int j = 0; j < BLOCK_SIZE; j++) {
            pred[i * stride + j] = left[i];
        }
    }
}

static void intra_predict_v(const uint8_t *above, uint8_t *pred, int stride) {
    for (int i = 0; i < BLOCK_SIZE; i++) {
        for (int j = 0; j < BLOCK_SIZE; j++) {
            pred[i * stride + j] = above[j];
        }
    }
}

/* Process a macroblock */
static void process_block(worker_stats_t *stats, int mb_x, int mb_y, int qscale) {
    int cur_x = mb_x * BLOCK_SIZE;
    int cur_y = mb_y * BLOCK_SIZE;

    uint8_t *cur_block = g_frame_curr + cur_y * FRAME_WIDTH + cur_x;

    /* Motion estimation */
    int mv_x, mv_y;
    uint32_t sad = motion_estimate(g_frame_ref, g_frame_curr, mb_x, mb_y, &mv_x, &mv_y);
    stats->sad_ops++;

    /* DCT of residual */
    float dct[8][8];
    dct_8x8(cur_block, FRAME_WIDTH, dct);
    stats->dct_ops++;

    /* Quantize */
    int quant[8][8];
    quantize(dct, qscale, quant);
    stats->quantized_coeffs++;

    /* Dequantize and IDCT */
    float dct_rec[8][8];
    dequantize(quant, qscale, dct_rec);

    uint8_t rec_block[64];
    idct_8x8(dct_rec, rec_block, 8);

    /* Intra prediction (for comparison) */
    uint8_t above[8], left[8];
    for (int i = 0; i < 8; i++) {
        above[i] = cur_y > 0 ? g_frame_curr[(cur_y - 1) * FRAME_WIDTH + cur_x + i] : 128;
        left[i] = cur_x > 0 ? g_frame_curr[(cur_y + i) * FRAME_WIDTH + cur_x - 1] : 128;
    }

    uint8_t pred[64];
    intra_predict_dc(above, left, pred, 8);
    stats->intra_modes++;

    stats->blocks_processed++;
}

/* Process a frame */
static void process_frame(worker_stats_t *stats, int qscale) {
    int mb_w = FRAME_WIDTH / BLOCK_SIZE;
    int mb_h = FRAME_HEIGHT / BLOCK_SIZE;

    for (int mb_y = 0; mb_y < mb_h; mb_y++) {
        for (int mb_x = 0; mb_x < mb_w; mb_x++) {
            if (!g_running) return;
            process_block(stats, mb_x, mb_y, qscale);
        }
    }

    stats->frames++;
}

static void *worker(void *arg) {
    worker_stats_t *stats = (worker_stats_t *)arg;
    int qscale = 16 + (rand() % 16);  /* Variable quantization */

    while (g_running) {
        process_frame(stats, qscale);
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

/* Generate test frame */
static void generate_frame(uint8_t *frame, int width, int height, int frame_num) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            /* Create patterns that simulate video content */
            int val = ((x + y + frame_num * 3) % 256);
            /* Add some texture */
            val = (val + ((x * y) % 64)) % 256;
            frame[y * width + x] = val;
        }
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

    /* Initialize frames */
    g_frame_curr = malloc(FRAME_WIDTH * FRAME_HEIGHT);
    g_frame_ref = malloc(FRAME_WIDTH * FRAME_HEIGHT);

    generate_frame(g_frame_curr, FRAME_WIDTH, FRAME_HEIGHT, 0);
    generate_frame(g_frame_ref, FRAME_WIDTH, FRAME_HEIGHT, 1);

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
    long total_frames = 0, total_blocks = 0, total_dct = 0;
    long total_sad = 0, total_intra = 0, total_quant = 0;

    for (int i = 0; i < threads; i++) {
        total_frames += stats[i].frames;
        total_blocks += stats[i].blocks_processed;
        total_dct += stats[i].dct_ops;
        total_sad += stats[i].sad_ops;
        total_intra += stats[i].intra_modes;
        total_quant += stats[i].quantized_coeffs;
    }

    double fps = (double)total_frames / duration;
    double blocks_per_sec = (double)total_blocks / duration;

    printf("FPS: %.2f frames/sec\n", fps);
    printf("Blocks: %.2f blocks/sec\n", blocks_per_sec);
    printf("DCT ops: %ld\n", total_dct);
    printf("SAD ops: %ld\n", total_sad);
    printf("Intra modes: %ld\n", total_intra);

    /* Cleanup */
    free(g_frame_curr);
    free(g_frame_ref);
    free(stats);
    free(tids);

    return 0;
}
