/*
 * arkprobe_random — Random memory access pattern workload.
 *
 * Measures latency and bandwidth for random memory access patterns.
 * Uses pointer chasing through a randomized linked list to force
 * cache misses and measure memory latency.
 *
 * Usage: arkprobe_random --threads N --duration S [--buffer-mb M]
 * Output: prints average latency in ns and access rate in Mops/s.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <getopt.h>

#define DEFAULT_BUFFER_MB 256

typedef struct {
    int duration_sec;
    size_t buffer_bytes;
    long access_count;
    double total_latency_ns;
} worker_arg_t;

static volatile int g_running = 1;

/* Build a randomized pointer-chase chain.
 * Each element points to a random next element, creating
 * a single cycle that visits all elements exactly once.
 */
static void **build_random_chain(size_t count) {
    void **chain = malloc(count * sizeof(void *));

    /* Create indices 0..count-1 */
    size_t *indices = malloc(count * sizeof(size_t));
    for (size_t i = 0; i < count; i++)
        indices[i] = i;

    /* Fisher-Yates shuffle */
    for (size_t i = count - 1; i > 0; i--) {
        size_t j = (size_t)rand() % (i + 1);
        size_t tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }

    /* Build chain: chain[i] points to chain[indices[i]] */
    for (size_t i = 0; i < count; i++) {
        chain[i] = &chain[indices[i]];
    }

    free(indices);
    return chain;
}

static double get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

static void *worker(void *arg) {
    worker_arg_t *wa = (worker_arg_t *)arg;

    /* Build random chain */
    size_t count = wa->buffer_bytes / sizeof(void *);
    void **chain = build_random_chain(count);

    /* Start at beginning */
    void **p = chain;
    long accesses = 0;

    double start_time = get_time_ns();

    while (g_running) {
        /* Chase through chain - 1000 hops per batch */
        for (int i = 0; i < 1000; i++) {
            p = (void **)*p;
        }
        accesses += 1000;
    }

    double end_time = get_time_ns();

    wa->access_count = accesses;
    wa->total_latency_ns = end_time - start_time;

    free(chain);
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
    int buffer_mb = DEFAULT_BUFFER_MB;

    static struct option long_opts[] = {
        {"threads",   required_argument, 0, 't'},
        {"duration",  required_argument, 0, 'd'},
        {"buffer-mb", required_argument, 0, 'b'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "t:d:b:", long_opts, NULL)) != -1) {
        switch (opt) {
            case 't': threads = atoi(optarg); break;
            case 'd': duration = atoi(optarg); break;
            case 'b': buffer_mb = atoi(optarg); break;
        }
    }

    if (threads < 1) threads = 1;
    if (duration < 1) duration = 1;
    if (buffer_mb < 1) buffer_mb = 1;

    size_t buffer_bytes = (size_t)buffer_mb * 1024 * 1024;

    worker_arg_t *args = calloc(threads, sizeof(worker_arg_t));
    pthread_t *tids = malloc(sizeof(pthread_t) * (threads + 1));

    pthread_create(&tids[threads], NULL, timer_thread, &duration);

    for (int i = 0; i < threads; i++) {
        args[i].duration_sec = duration;
        args[i].buffer_bytes = buffer_bytes;
        pthread_create(&tids[i], NULL, worker, &args[i]);
    }

    for (int i = 0; i <= threads; i++)
        pthread_join(tids[i], NULL);

    /* Aggregate results */
    long total_accesses = 0;
    double total_time_ns = 0;
    for (int i = 0; i < threads; i++) {
        total_accesses += args[i].access_count;
        total_time_ns += args[i].total_latency_ns;
    }

    double avg_latency_ns = total_time_ns / total_accesses;
    double mops_per_sec = total_accesses / (total_time_ns / threads) / 1e6 * 1e9;

    printf("Avg latency: %.2f ns\n", avg_latency_ns);
    printf("Access rate: %.2f Mops/s\n", mops_per_sec);

    free(args);
    free(tids);
    return 0;
}
