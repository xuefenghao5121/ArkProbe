/*
 * arkprobe_compute — Dense matrix multiplication workload.
 *
 * Pure compute-bound: high IPC, low cache miss, saturates ALU/FPU.
 * Uses ikj loop order for good spatial locality in the inner loop.
 *
 * Usage: arkprobe_compute --threads N --duration S
 * Output: prints "XXXXX.XX Mflops" to stdout on exit.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <getopt.h>

#define MATRIX_N 256

typedef struct {
    int duration_sec;
    long ops;
} worker_arg_t;

static volatile int g_running = 1;

static void matmul(double *A, double *B, double *C, int N) {
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
    int N = MATRIX_N;
    double *A = malloc(sizeof(double) * N * N);
    double *B = malloc(sizeof(double) * N * N);
    double *C = malloc(sizeof(double) * N * N);
    fill_random(A, N);
    fill_random(B, N);

    long count = 0;
    while (g_running) {
        matmul(A, B, C, N);
        count++;
    }
    wa->ops = count;

    free(A); free(B); free(C);
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

    /* Start timer thread */
    pthread_create(&tids[threads], NULL, timer_thread, &duration);

    /* Start worker threads */
    for (int i = 0; i < threads; i++) {
        args[i].duration_sec = duration;
        pthread_create(&tids[i], NULL, worker, &args[i]);
    }

    /* Wait for all workers */
    for (int i = 0; i <= threads; i++)
        pthread_join(tids[i], NULL);

    /* Compute total Mflops: each matmul is 2*N^3 flops */
    long total_ops = 0;
    for (int i = 0; i < threads; i++)
        total_ops += args[i].ops;

    double flops = (double)total_ops * 2.0 * MATRIX_N * MATRIX_N * MATRIX_N;
    double mflops = flops / (duration * 1e6);
    printf("%.2f Mflops\n", mflops);
    printf("%.2f ops/sec\n", (double)total_ops / duration);

    free(args);
    free(tids);
    return 0;
}
