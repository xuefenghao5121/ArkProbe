/*
 * arkprobe_memory — Memory bandwidth and latency workload.
 *
 * Two phases per iteration:
 *   1. Streaming copy: sequential memcpy of large buffers (bandwidth-bound)
 *   2. Pointer chase: random linked-list traversal (latency-bound)
 *
 * Usage: arkprobe_memory --threads N --duration S [--buffer-mb M]
 * Output: prints "XXXXX.XX MB/s" and "XXXXX.XX ops/sec" to stdout.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <getopt.h>

#define DEFAULT_BUFFER_MB 64

typedef struct {
    int duration_sec;
    size_t buffer_bytes;
    long stream_bytes;
    long chase_ops;
} worker_arg_t;

static volatile int g_running = 1;

/* Build a randomized pointer-chase chain through the buffer.
 * Each element is a pointer to another random location. */
static void build_chase_chain(void **buf, size_t count) {
    /* Fisher-Yates shuffle to create a full-cycle permutation */
    for (size_t i = 0; i < count; i++)
        buf[i] = (void *)(uintptr_t)i;

    for (size_t i = count - 1; i > 0; i--) {
        size_t j = (size_t)rand() % (i + 1);
        void *tmp = buf[i];
        buf[i] = buf[j];
        buf[j] = tmp;
    }

    /* Convert indices to pointers */
    void **copy = malloc(count * sizeof(void *));
    memcpy(copy, buf, count * sizeof(void *));
    for (size_t i = 0; i < count; i++)
        buf[i] = &buf[(uintptr_t)copy[i]];
    free(copy);
}

static void *worker(void *arg) {
    worker_arg_t *wa = (worker_arg_t *)arg;
    size_t nbytes = wa->buffer_bytes;
    char *src = malloc(nbytes);
    char *dst = malloc(nbytes);
    memset(src, 0xAA, nbytes);

    /* Pointer chase buffer */
    size_t chase_count = nbytes / sizeof(void *);
    void **chase_buf = (void **)malloc(chase_count * sizeof(void *));
    build_chase_chain(chase_buf, chase_count);

    long stream_total = 0;
    long chase_total = 0;

    while (g_running) {
        /* Phase 1: streaming copy */
        memcpy(dst, src, nbytes);
        stream_total += nbytes;

        /* Phase 2: pointer chase — 1024 hops per iteration */
        void **p = &chase_buf[0];
        for (int i = 0; i < 1024; i++)
            p = (void **)*p;
        chase_total += 1024;

        /* Prevent the compiler from optimizing away the chase */
        if ((uintptr_t)p == 1) abort();
    }

    wa->stream_bytes = stream_total;
    wa->chase_ops = chase_total;

    free(src); free(dst); free(chase_buf);
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

    long total_stream = 0;
    long total_chase = 0;
    for (int i = 0; i < threads; i++) {
        total_stream += args[i].stream_bytes;
        total_chase += args[i].chase_ops;
    }

    double mb_per_sec = (double)total_stream / (duration * 1024.0 * 1024.0);
    double ops_per_sec = (double)total_chase / duration;
    printf("%.2f MB/s\n", mb_per_sec);
    printf("%.2f ops/sec\n", ops_per_sec);

    free(args);
    free(tids);
    return 0;
}
