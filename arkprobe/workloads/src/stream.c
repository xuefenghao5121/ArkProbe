/*
 * arkprobe_stream — STREAM benchmark for memory bandwidth measurement.
 *
 * Implements the standard STREAM benchmark operations:
 *   COPY:    a(i) = b(i)
 *   SCALE:   a(i) = q * b(i)
 *   ADD:     a(i) = b(i) + c(i)
 *   TRIAD:   a(i) = b(i) + q * c(i)
 *
 * Usage: arkprobe_stream --threads N --duration S [--array-size M]
 * Output: prints bandwidth in MB/s for each operation.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <getopt.h>

#define DEFAULT_ARRAY_SIZE 10000000  /* 10M elements = 80MB per array */
#define NTIMES 10

typedef struct {
    double *a, *b, *c;
    size_t array_size;
    double copy_bw, scale_bw, add_bw, triad_bw;
} stream_data_t;

static volatile int g_running = 1;
static int g_duration = 60;

static double mysecond(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1.0e-9;
}

static void *stream_worker(void *arg) {
    stream_data_t *sd = (stream_data_t *)arg;
    size_t n = sd->array_size;
    double scalar = 3.0;

    double times[4][NTIMES];
    int iter = 0;

    while (g_running && iter < NTIMES * 100) {
        double t0, t1;

        /* COPY */
        t0 = mysecond();
        for (size_t j = 0; j < n; j++)
            sd->a[j] = sd->b[j];
        t1 = mysecond();
        if (iter < NTIMES) times[0][iter % NTIMES] = t1 - t0;

        /* SCALE */
        t0 = mysecond();
        for (size_t j = 0; j < n; j++)
            sd->a[j] = scalar * sd->b[j];
        t1 = mysecond();
        if (iter < NTIMES) times[1][iter % NTIMES] = t1 - t0;

        /* ADD */
        t0 = mysecond();
        for (size_t j = 0; j < n; j++)
            sd->a[j] = sd->b[j] + sd->c[j];
        t1 = mysecond();
        if (iter < NTIMES) times[2][iter % NTIMES] = t1 - t0;

        /* TRIAD */
        t0 = mysecond();
        for (size_t j = 0; j < n; j++)
            sd->a[j] = sd->b[j] + scalar * sd->c[j];
        t1 = mysecond();
        if (iter < NTIMES) times[3][iter % NTIMES] = t1 - t0;

        iter++;
    }

    /* Calculate average bandwidth (MB/s) */
    /* Each operation moves 2 or 3 arrays of 8 bytes each */
    double bytes[4] = {
        2 * n * sizeof(double),  /* COPY: read b, write a */
        2 * n * sizeof(double),  /* SCALE: read b, write a */
        3 * n * sizeof(double),  /* ADD: read b,c, write a */
        3 * n * sizeof(double),  /* TRIAD: read b,c, write a */
    };

    /* Find minimum time for each operation */
    double min_time[4] = {times[0][0], times[1][0], times[2][0], times[3][0]};
    for (int i = 0; i < 4; i++) {
        for (int j = 1; j < NTIMES && j < iter; j++) {
            if (times[i][j] < min_time[i])
                min_time[i] = times[i][j];
        }
    }

    sd->copy_bw = bytes[0] / min_time[0] / 1e6;
    sd->scale_bw = bytes[1] / min_time[1] / 1e6;
    sd->add_bw = bytes[2] / min_time[2] / 1e6;
    sd->triad_bw = bytes[3] / min_time[3] / 1e6;

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
    size_t array_size = DEFAULT_ARRAY_SIZE;

    static struct option long_opts[] = {
        {"threads",     required_argument, 0, 't'},
        {"duration",    required_argument, 0, 'd'},
        {"array-size",  required_argument, 0, 's'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "t:d:s:", long_opts, NULL)) != -1) {
        switch (opt) {
            case 't': threads = atoi(optarg); break;
            case 'd': duration = atoi(optarg); break;
            case 's': array_size = atol(optarg); break;
        }
    }

    if (threads < 1) threads = 1;
    if (duration < 1) duration = 1;
    if (array_size < 1000) array_size = 1000;

    g_duration = duration;

    /* Allocate arrays for each thread */
    stream_data_t *sd = calloc(threads, sizeof(stream_data_t));
    for (int i = 0; i < threads; i++) {
        sd[i].array_size = array_size;
        sd[i].a = calloc(array_size, sizeof(double));
        sd[i].b = calloc(array_size, sizeof(double));
        sd[i].c = calloc(array_size, sizeof(double));
        /* Initialize with random data */
        for (size_t j = 0; j < array_size; j++) {
            sd[i].b[j] = (double)rand() / RAND_MAX;
            sd[i].c[j] = (double)rand() / RAND_MAX;
        }
    }

    pthread_t *tids = malloc(sizeof(pthread_t) * (threads + 1));

    pthread_create(&tids[threads], NULL, timer_thread, &duration);

    for (int i = 0; i < threads; i++)
        pthread_create(&tids[i], NULL, stream_worker, &sd[i]);

    for (int i = 0; i <= threads; i++)
        pthread_join(tids[i], NULL);

    /* Aggregate results */
    double total_copy = 0, total_scale = 0, total_add = 0, total_triad = 0;
    for (int i = 0; i < threads; i++) {
        total_copy += sd[i].copy_bw;
        total_scale += sd[i].scale_bw;
        total_add += sd[i].add_bw;
        total_triad += sd[i].triad_bw;
    }

    printf("COPY:    %.2f MB/s\n", total_copy);
    printf("SCALE:   %.2f MB/s\n", total_scale);
    printf("ADD:     %.2f MB/s\n", total_add);
    printf("TRIAD:   %.2f MB/s\n", total_triad);

    /* Cleanup */
    for (int i = 0; i < threads; i++) {
        free(sd[i].a);
        free(sd[i].b);
        free(sd[i].c);
    }
    free(sd);
    free(tids);

    return 0;
}
