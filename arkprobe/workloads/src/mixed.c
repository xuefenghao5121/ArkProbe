/*
 * arkprobe_mixed — Mixed compute + memory workload.
 *
 * Each thread alternates between:
 *   - Compute phase: small matrix multiply (fits in L1/L2)
 *   - Memory phase: large buffer streaming copy (stresses memory subsystem)
 *
 * This creates a realistic mixed micro-architectural footprint with
 * moderate IPC, mixed cache behavior, and backend-bound characteristics.
 *
 * Usage: arkprobe_mixed --threads N --duration S
 * Output: prints "XXXXX.XX ops/sec" to stdout.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <getopt.h>

#define COMPUTE_N 128
#define MEMORY_MB 32

typedef struct {
    int duration_sec;
    long ops;
} worker_arg_t;

static volatile int g_running = 1;

static void matmul_small(double *A, double *B, double *C, int N) {
    memset(C, 0, sizeof(double) * N * N);
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++)
            for (int j = 0; j < N; j++)
                C[i * N + j] += A[i * N + k] * B[k * N + j];
}

static void fill_random(double *M, int N) {
    for (int i = 0; i < N * N; i++)
        M[i] = (double)rand() / RAND_MAX;
}

static void *worker(void *arg) {
    worker_arg_t *wa = (worker_arg_t *)arg;
    int N = COMPUTE_N;
    size_t mem_bytes = (size_t)MEMORY_MB * 1024 * 1024;

    /* Compute buffers */
    double *A = malloc(sizeof(double) * N * N);
    double *B = malloc(sizeof(double) * N * N);
    double *C = malloc(sizeof(double) * N * N);
    fill_random(A, N);
    fill_random(B, N);

    /* Memory buffers */
    char *src = malloc(mem_bytes);
    char *dst = malloc(mem_bytes);
    memset(src, 0xBB, mem_bytes);

    long count = 0;
    while (g_running) {
        /* Compute phase */
        matmul_small(A, B, C, N);

        /* Memory phase */
        memcpy(dst, src, mem_bytes);

        count++;
    }
    wa->ops = count;

    free(A); free(B); free(C);
    free(src); free(dst);
    return NULL;
}

static void *timer_thread(void *arg) {
    int seconds = *(int *)arg;
    struct timespec ts = {seconds, 0};
    nanosleep(&ts, NULL);
    g_running = 0;
    return NULL;
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

    worker_arg_t *args = calloc(threads, sizeof(worker_arg_t));
    pthread_t *tids = malloc(sizeof(pthread_t) * (threads + 1));

    pthread_create(&tids[threads], NULL, timer_thread, &duration);

    for (int i = 0; i < threads; i++) {
        args[i].duration_sec = duration;
        pthread_create(&tids[i], NULL, worker, &args[i]);
    }

    for (int i = 0; i <= threads; i++)
        pthread_join(tids[i], NULL);

    long total_ops = 0;
    for (int i = 0; i < threads; i++)
        total_ops += args[i].ops;

    printf("%.2f ops/sec\n", (double)total_ops / duration);

    free(args);
    free(tids);
    return 0;
}
